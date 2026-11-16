"""STAMP's validation AUROC must be computed on probabilities, and a job can demand a GPU.

STAMP 2.4.0 passes raw logits to its torchmetrics MulticlassAUROC (``valid_auroc``),
one validation sample per update. torchmetrics applies a softmax only when a value of
an update lies outside [0, 1], so samples with small logits are scored on the raw
logits and the result is not an AUROC. In a swarm run that value (``validation_auroc``)
picks the best global model. ``use_probabilities_for_validation_auroc`` makes every
update score ``softmax(logits)``.

``require_gpu_if_configured`` turns a silent CPU fallback into an error when the job
sets STAMP_REQUIRE_GPU=1.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
torchmetrics_classification = pytest.importorskip("torchmetrics.classification")
MulticlassAUROC = torchmetrics_classification.MulticlassAUROC

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_CUSTOM_DIR = (
    REPO_ROOT / "application" / "jobs" / "STAMP_classification" / "app" / "custom"
)
if str(STAMP_CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(STAMP_CUSTOM_DIR))

import stamp_training as st  # noqa: E402


class _TinyClassifier(torch.nn.Module):
    """Stands in for STAMP's LitTileClassifier: only the metric attribute matters."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.valid_auroc = MulticlassAUROC(num_classes)


def _auroc_one_by_one(model, logits, labels):
    """Feed one sample per update, as STAMP's validation loop does (batch size 1)."""
    model.valid_auroc.reset()
    for row, label in zip(logits, labels):
        model.valid_auroc.update(row.unsqueeze(0), torch.tensor([label]))
    return float(model.valid_auroc.compute())


def _pairwise_auroc(scores, labels):
    """Binary AUROC as the probability that a positive outranks a negative (ties 1/2)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# A positive whose logits lie inside [0, 1] (stored raw, unpatched) next to a negative
# whose logits are negative (softmaxed): the raw score 0.5 ranks below the negative's
# probability 0.599 although the positive's real probability (0.62) ranks above it.
LOGITS = torch.tensor([[0.0, 0.5], [-3.0, -2.6]])
LABELS = [1, 0]


def test_unpatched_metric_mixes_logits_and_probabilities():
    model = _TinyClassifier()
    assert _auroc_one_by_one(model, LOGITS, LABELS) == pytest.approx(0.5)


def test_patched_metric_scores_probabilities():
    model = _TinyClassifier()
    assert st.use_probabilities_for_validation_auroc(model) is True
    assert _auroc_one_by_one(model, LOGITS, LABELS) == pytest.approx(1.0)


def test_patched_metric_matches_the_auroc_of_the_predictions():
    generator = torch.Generator().manual_seed(7)
    labels = [int(v) for v in torch.randint(0, 2, (60,), generator=generator)]
    # Small logits, as from an untrained head: many samples lie inside [0, 1].
    logits = torch.rand(60, 2, generator=generator) * 1.4 - 0.2
    logits[:, 1] += 0.15 * torch.tensor(labels, dtype=torch.float32)
    expected = _pairwise_auroc(torch.softmax(logits, dim=1)[:, 1].tolist(), labels)

    model = _TinyClassifier()
    unpatched = _auroc_one_by_one(model, logits, labels)
    st.use_probabilities_for_validation_auroc(model)
    patched = _auroc_one_by_one(model, logits, labels)

    assert patched == pytest.approx(expected, abs=1e-6)
    assert unpatched != pytest.approx(expected, abs=1e-3)


def test_patch_is_idempotent():
    model = _TinyClassifier()
    assert st.use_probabilities_for_validation_auroc(model) is True
    first = model.valid_auroc.update
    assert st.use_probabilities_for_validation_auroc(model) is False
    assert model.valid_auroc.update is first
    assert _auroc_one_by_one(model, LOGITS, LABELS) == pytest.approx(1.0)


def test_model_without_validation_auroc_is_left_alone():
    model = torch.nn.Linear(2, 2)
    assert st.use_probabilities_for_validation_auroc(model) is False


def test_patched_metric_accepts_half_precision_logits():
    model = _TinyClassifier()
    st.use_probabilities_for_validation_auroc(model)
    assert _auroc_one_by_one(model, LOGITS.half(), LABELS) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# STAMP_REQUIRE_GPU
# ---------------------------------------------------------------------------

def test_gpu_not_required_by_default(monkeypatch):
    monkeypatch.delenv("STAMP_REQUIRE_GPU", raising=False)
    st.require_gpu_if_configured(cuda_available=False)


@pytest.mark.parametrize("value", ["0", "", "no"])
def test_gpu_not_required_when_switched_off(monkeypatch, value):
    monkeypatch.setenv("STAMP_REQUIRE_GPU", value)
    st.require_gpu_if_configured(cuda_available=False)


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_missing_gpu_fails_when_required(monkeypatch, value):
    monkeypatch.setenv("STAMP_REQUIRE_GPU", value)
    with pytest.raises(RuntimeError, match="must train on a GPU"):
        st.require_gpu_if_configured(cuda_available=False)


def test_present_gpu_passes_when_required(monkeypatch):
    monkeypatch.setenv("STAMP_REQUIRE_GPU", "1")
    st.require_gpu_if_configured(cuda_available=True)
