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
    model._update_support(state, TARGETS)
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
    # Support counts are per-class at every tier on purpose -- a per-class metric
    # must never be read without its n (docs/EVALUATION_PITFALLS.md, E1). The
    # tier gate is about per-class *metrics*.
    assert not any("_class" in k and not k.startswith("val/support") for k in values)
    assert "val/ECE" not in values
    assert not any(k.endswith(("_ci_low", "_ci_high")) for k in values)


def test_train_state_has_no_extended_metrics(base_model, monkeypatch):
    """Training steps must keep their original cost -- compat keys only."""
    model = _make(base_model, monkeypatch)
    assert "train_" not in model.extra_metrics
    values = _feed(model, "train")
    support = {f"train/support_class{i}" for i in range(NUM_CLASSES)} | {"train/n"}
    assert set(values) == {"train/ACC", "train/AUC_ROC"} | support


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


def _logits_for(labels, num_classes=NUM_CLASSES):
    """Confident, correct logits for each label -- metric values are not the
    point in these tests, only that `compute()` has state to work from."""
    valid = [int(x) for x in labels.tolist() if 0 <= int(x) < num_classes]
    out = torch.zeros(len(valid), num_classes)
    for row, label in enumerate(valid):
        out[row][label] = 3.0
    return out, torch.tensor(valid)


def _feed_labels(model, state, labels):
    """Update every metric and the support counter, then compute.

    Labels outside [0, num_classes) are passed to the support counter (which
    must ignore them) but withheld from torchmetrics, which would raise.
    """
    key = state + "_"
    logits, valid = _logits_for(labels)
    model.acc[key].update(logits, valid)
    model.auc_roc[key].update(logits, valid)
    if key in model.extra_metrics:
        for metric in model.extra_metrics[key].values():
            metric.update(logits, valid)
    model._update_support(state, labels)
    return model.compute_epoch_metrics(state)



# --- support counts (#441) -------------------------------------------------
# A per-class metric read without its support is what produced a wrong
# conclusion about UMCU: macro AUROC ranked centres by how many benign cases
# their split happened to hold (0, 1, 2, 2, 12), not by performance.
# See docs/EVALUATION_PITFALLS.md (E1).

def test_support_counts_are_emitted_at_every_tier(base_model, monkeypatch):
    for env in ({}, {"ODELIA_EVAL_METRICS": "full"}):
        values = _feed(_make(base_model, monkeypatch, **env), "val")
        for index in range(NUM_CLASSES):
            assert f"val/support_class{index}" in values, f"missing support at tier {env}"
        assert "val/n" in values


def test_support_counts_match_the_labels_seen(base_model, monkeypatch):
    values = _feed(_make(base_model, monkeypatch), "val")
    # TARGETS is [0, 1, 2, 0, 1, 2] -> two of each class, six in total
    for index in range(NUM_CLASSES):
        assert values[f"val/support_class{index}"].item() == pytest.approx(2.0)
    assert values["val/n"].item() == pytest.approx(6.0)


def test_support_counts_are_scalar_tensors_like_every_other_value(base_model, monkeypatch):
    values = _feed(_make(base_model, monkeypatch, ODELIA_EVAL_METRICS="full"), "val")
    assert all(v.ndim == 0 for v in values.values())


def test_support_reflects_imbalance_not_just_totals(base_model, monkeypatch):
    """The case that matters: one class barely represented."""
    model = _make(base_model, monkeypatch)
    values = _feed_labels(model, "val", torch.tensor([0, 0, 0, 0, 0, 1, 2, 2]))
    assert values["val/support_class0"].item() == pytest.approx(5.0)
    assert values["val/support_class1"].item() == pytest.approx(1.0)
    assert values["val/support_class2"].item() == pytest.approx(2.0)
    assert values["val/n"].item() == pytest.approx(8.0)


def test_support_resets_between_epochs(base_model, monkeypatch):
    """Counts must not accumulate across epochs, or n grows without bound."""
    model = _make(base_model, monkeypatch)
    model._update_support("val", TARGETS)
    model._support["val_"] = [0] * model.num_classes      # what _epoch_end does
    values = _feed_labels(model, "val", TARGETS)
    assert values["val/n"].item() == pytest.approx(6.0)


def test_support_ignores_out_of_range_labels(base_model, monkeypatch):
    model = _make(base_model, monkeypatch)
    values = _feed_labels(model, "val", torch.tensor([0, 1, 2, 7, -1]))
    assert values["val/n"].item() == pytest.approx(3.0)
