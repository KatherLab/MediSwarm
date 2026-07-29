"""Unit tests for the AUROC helpers in ``stamp_metrics_callback`` (#492).

STAMP does not log AUROC into Lightning's ``callback_metrics`` under any key we
can rely on, so ``val_auroc`` in the metrics summary was always blank. We now
compute AUROC directly from the validation predictions. These tests validate
that computation, including tie handling, against scikit-learn where available.
"""

import sys
from pathlib import Path

import pytest

# stamp_metrics_callback imports torch + lightning at module import time.
pytest.importorskip("torch")
pytest.importorskip("lightning")

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_CUSTOM_DIR = (
    REPO_ROOT / "application" / "jobs" / "STAMP_classification" / "app" / "custom"
)
if str(STAMP_CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(STAMP_CUSTOM_DIR))

import stamp_metrics_callback as smc  # noqa: E402


# ---------------------------------------------------------------------------
# _binary_auroc
# ---------------------------------------------------------------------------

def test_binary_auroc_perfect_separation():
    assert smc._binary_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_binary_auroc_perfectly_wrong():
    assert smc._binary_auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_binary_auroc_all_ties_is_half():
    # Every positive/negative pair is a tie -> 0.5.
    assert smc._binary_auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)


def test_binary_auroc_single_class_returns_none():
    assert smc._binary_auroc([0.1, 0.2, 0.3], [1, 1, 1]) is None
    assert smc._binary_auroc([0.1, 0.2, 0.3], [0, 0, 0]) is None
    assert smc._binary_auroc([], []) is None


def test_binary_auroc_matches_sklearn_random():
    sk = pytest.importorskip("sklearn.metrics")
    import random
    rng = random.Random(0)
    labels = [rng.randint(0, 1) for _ in range(200)]
    if sum(labels) in (0, len(labels)):  # guarantee both classes present
        labels[0], labels[1] = 0, 1
    scores = [l * 0.3 + rng.random() for l in labels]
    assert smc._binary_auroc(scores, labels) == pytest.approx(
        sk.roc_auc_score(labels, scores), abs=1e-9
    )


def test_binary_auroc_matches_sklearn_with_ties():
    sk = pytest.importorskip("sklearn.metrics")
    labels = [0, 0, 1, 1, 0, 1, 1, 0]
    scores = [0.2, 0.2, 0.2, 0.9, 0.5, 0.5, 0.5, 0.1]  # deliberate ties
    assert smc._binary_auroc(scores, labels) == pytest.approx(
        sk.roc_auc_score(labels, scores), abs=1e-9
    )


# ---------------------------------------------------------------------------
# _true_class_index
# ---------------------------------------------------------------------------

def test_true_class_index_scalar_and_onehot():
    assert smc._true_class_index(1) == 1
    assert smc._true_class_index(0) == 0
    assert smc._true_class_index([1.0, 0.0]) == 0
    assert smc._true_class_index([0.0, 1.0]) == 1
    assert smc._true_class_index([0.1, 0.2, 0.7]) == 2


# ---------------------------------------------------------------------------
# macro_ovr_auroc
# ---------------------------------------------------------------------------

def test_macro_ovr_auroc_binary_onehot_ground_truth():
    # Ground truth as one-hot [neg, pos], prob rows [p0, p1] — the exact layout
    # the prediction callback produces.
    gts = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    probs = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]]
    assert smc.macro_ovr_auroc(gts, probs) == pytest.approx(1.0)


def test_macro_ovr_auroc_binary_scalar_ground_truth():
    gts = [0, 0, 1, 1]
    probs = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]]
    assert smc.macro_ovr_auroc(gts, probs) == pytest.approx(1.0)


def test_macro_ovr_auroc_empty_returns_none():
    assert smc.macro_ovr_auroc([], []) is None


def test_macro_ovr_auroc_single_class_returns_none():
    # Only negatives present for every class -> no computable class AUROC.
    gts = [0, 0, 0]
    probs = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
    assert smc.macro_ovr_auroc(gts, probs) is None


def test_macro_ovr_auroc_matches_sklearn_multiclass():
    sk = pytest.importorskip("sklearn.metrics")
    import random
    rng = random.Random(1)
    n, k = 180, 3
    gts, probs = [], []
    for _ in range(n):
        true = rng.randint(0, k - 1)
        row = [rng.random() for _ in range(k)]
        row[true] += 1.5  # bias toward the true class so it's learnable
        s = sum(row)
        probs.append([x / s for x in row])
        gts.append(true)
    expected = sk.roc_auc_score(gts, probs, multi_class="ovr", average="macro")
    assert smc.macro_ovr_auroc(gts, probs) == pytest.approx(expected, abs=1e-9)
