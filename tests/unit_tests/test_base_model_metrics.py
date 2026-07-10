"""Unit tests for the extended evaluation metrics in models/base_model.py (#410).

These exercise the pure `compute_epoch_metrics()` helper, so no Trainer, GPU, or
dataset is required. The tiers are env-gated, so each test constructs a fresh
classifier after setting the environment.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchmetrics")
pytest.importorskip("pytorch_lightning")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from conftest import MODELS_DIR, import_module_from_path  # noqa: E402

BASE_MODEL_PATH = MODELS_DIR / "base_model.py"
NUM_CLASSES = 3

# Perfectly separable logits -> ACC/AUROC/F1/Recall/Specificity/PR_AUC all == 1.0
LOGITS = torch.tensor([
    [3.0, 0.0, 0.0],
    [0.0, 3.0, 0.0],
    [0.0, 0.0, 3.0],
    [2.5, 0.2, 0.1],
    [0.1, 2.5, 0.2],
    [0.1, 0.2, 2.5],
])
TARGETS = torch.tensor([0, 1, 2, 0, 1, 2])

ENV_KEYS = ("ODELIA_EVAL_METRICS", "ODELIA_EVAL_BOOTSTRAP", "ODELIA_EVAL_BOOTSTRAP_N")
COMPAT_KEYS = ("val/ACC", "val/AUC_ROC")
DEFAULT_KEYS = ("val/F1", "val/Recall", "val/Specificity", "val/PR_AUC")


@pytest.fixture(scope="module")
def base_model():
    return import_module_from_path("_test_base_model", BASE_MODEL_PATH)


def _make(base_model, monkeypatch, **env):
    """Build a classifier with a clean, explicitly-set environment."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return base_model.BasicClassifier(in_ch=1, out_ch=NUM_CLASSES, spatial_dims=3)


def _feed(model, state):
    key = state + "_"
    model.acc[key].update(LOGITS, TARGETS)
    model.auc_roc[key].update(LOGITS, TARGETS)
    if key in model.extra_metrics:
        for metric in model.extra_metrics[key].values():
            metric.update(LOGITS, TARGETS)
    return model.compute_epoch_metrics(state)


def test_default_tier_keeps_compat_keys_and_adds_macro_scalars(base_model, monkeypatch):
    values = _feed(_make(base_model, monkeypatch), "val")
    for key in COMPAT_KEYS + DEFAULT_KEYS:
        assert key in values, f"missing {key}"
    # perfectly separable inputs
    assert values["val/ACC"].item() == pytest.approx(1.0)
    assert values["val/AUC_ROC"].item() == pytest.approx(1.0)
    assert values["val/F1"].item() == pytest.approx(1.0)
    assert values["val/Specificity"].item() == pytest.approx(1.0)
    # every value is a scalar so Lightning can log it
    assert all(v.ndim == 0 for v in values.values())


def test_default_tier_excludes_per_class_calibration_and_ci(base_model, monkeypatch):
    values = _feed(_make(base_model, monkeypatch), "val")
    assert not any("_class" in k for k in values)
    assert "val/ECE" not in values
    assert not any(k.endswith(("_ci_low", "_ci_high")) for k in values)


def test_train_state_has_no_extended_metrics(base_model, monkeypatch):
    """Training steps must keep their original cost -- compat keys only."""
    model = _make(base_model, monkeypatch)
    assert "train_" not in model.extra_metrics
    values = _feed(model, "train")
    assert set(values) == {"train/ACC", "train/AUC_ROC"}


def test_full_tier_adds_per_class_and_calibration(base_model, monkeypatch):
    values = _feed(_make(base_model, monkeypatch, ODELIA_EVAL_METRICS="full"), "val")
    assert "val/ECE" in values
    for base in ("AUC_ROC", "F1", "Recall", "Specificity", "PR_AUC"):
        for index in range(NUM_CLASSES):
            assert f"val/{base}_class{index}" in values
    # compat keys survive the full tier
    for key in COMPAT_KEYS:
        assert key in values
    assert all(v.ndim == 0 for v in values.values())


def test_bootstrap_adds_confidence_intervals_on_test_only(base_model, monkeypatch):
    model = _make(base_model, monkeypatch, ODELIA_EVAL_BOOTSTRAP="1", ODELIA_EVAL_BOOTSTRAP_N="20")

    test_values = _feed(model, "test")
    for base in ("ACC", "AUC_ROC"):
        assert f"test/{base}_ci_low" in test_values
        assert f"test/{base}_ci_high" in test_values
        assert test_values[f"test/{base}_ci_low"] <= test_values[f"test/{base}_ci_high"]

    # val must stay cheap: bootstrap is restricted to the test split
    val_values = _feed(model, "val")
    assert not any(k.endswith(("_ci_low", "_ci_high")) for k in val_values)


def test_bootstrap_off_by_default(base_model, monkeypatch):
    model = _make(base_model, monkeypatch)
    assert model.eval_bootstrap is False
    assert not any("boot" in name for name in model.extra_metrics["test_"])
