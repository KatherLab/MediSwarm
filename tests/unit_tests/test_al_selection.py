"""Unit tests for application/jobs/_shared/custom/al_selection.py (D3.2 selection rules; pure Python).

Pins what the D3.2 arms rely on: every rule is independent of row order and nests across budgets, the
random arms depend only on (pool, seed), class_matched_random reproduces the entropy arm's class counts,
and the reported statistics carry aggregates only. Also checks that the offline study
(scripts/active_learning/selection.py) now runs the same implementation.
"""

import random
import sys
from pathlib import Path

import pytest

from conftest import SHARED_CUSTOM_DIR, import_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sel():
    return import_module_from_path("al_selection_under_test", SHARED_CUSTOM_DIR / "al_selection.py")


def _pool(n=60, seed=7, classes=(0, 0, 0, 0, 1, 2)):
    """n rows with varied probabilities and an imbalanced label mix (class 0 common, 1/2 rare)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        raw = [rng.random() + 0.05 for _ in range(3)]
        total = sum(raw)
        uid = f"SITE_{i:03d}_{'left' if i % 2 else 'right'}"
        rows.append((uid, [p / total for p in raw], classes[i % len(classes)]))
    return rows


STRATEGIES = ("entropy", "margin", "random", "class_matched_random")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_selection_does_not_depend_on_row_order(sel, strategy):
    rows = _pool()
    shuffled = list(rows)
    random.Random(123).shuffle(shuffled)
    for budget in (1, 10, 25):
        a = sel.select_uids(rows, budget, strategy, seed=3)
        b = sel.select_uids(list(reversed(rows)), budget, strategy, seed=3)
        c = sel.select_uids(shuffled, budget, strategy, seed=3)
        assert a == b == c
        assert len(a) == budget


def test_random_arm_is_order_independent_where_the_old_rule_was_not(sel):
    """The readiness check: 100 rows, budget 20, seed 0, reversed order kept 4 of 20 with the old rule."""
    rows = [(f"u{i:03d}", [1 / 3] * 3, 0) for i in range(100)]
    forward = sel.select_uids(rows, 20, "random", seed=0)
    backward = sel.select_uids(list(reversed(rows)), 20, "random", seed=0)
    assert forward == backward
    # ... and it is "sort by UID, then shuffle with the seed", as the contract specifies.
    expected = sorted(r[0] for r in rows)
    random.Random(0).shuffle(expected)
    assert forward == expected[:20]


@pytest.mark.parametrize("strategy", ["random", "class_matched_random"])
def test_random_arms_depend_on_the_seed(sel, strategy):
    rows = _pool(n=200)
    picks = {seed: frozenset(sel.select_uids(rows, 30, strategy, seed=seed)) for seed in range(5)}
    assert len(set(picks.values())) == 5
    assert sel.select_uids(rows, 30, strategy, seed=2) == sel.select_uids(rows, 30, strategy, seed=2)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_budgets_nest(sel, strategy):
    rows = _pool(n=120)
    previous = set()
    for budget in (5, 10, 20, 40, 80):
        current = set(sel.select_uids(rows, budget, strategy, seed=11))
        assert previous <= current, f"{strategy}: budget {budget} dropped cases chosen at a smaller budget"
        previous = current


def test_entropy_and_margin_rank_by_uncertainty_with_uid_tiebreak(sel):
    rows = [
        ("b_tie", [1 / 3, 1 / 3, 1 / 3], 0),
        ("a_tie", [1 / 3, 1 / 3, 1 / 3], 0),
        ("certain", [0.98, 0.01, 0.01], 0),
        ("torn", [0.45, 0.44, 0.11], 2),
    ]
    assert sel.select_uids(rows, 3, "entropy") == ["a_tie", "b_tie", "torn"]
    assert sel.select_uids(rows, 1, "margin")[0] in ("a_tie", "b_tie")
    assert sel.select_uids(rows, 4, "margin")[-1] == "certain"


def test_class_matched_random_reproduces_the_entropy_composition(sel):
    rows = _pool(n=150, classes=(0, 0, 0, 0, 0, 0, 1, 2))
    labels = {uid: label for uid, _, label in rows}
    for budget in (8, 20, 45):
        entropy_pick = sel.select_uids(rows, budget, "entropy")
        matched = sel.select_uids(rows, budget, "class_matched_random", seed=5)
        assert sel.class_counts(labels[u] for u in matched) == sel.class_counts(labels[u] for u in entropy_pick)
        # The same composition, but not simply the entropy cases again.
        assert set(matched) != set(entropy_pick)
        # Within a class, the cases are the random arm's order filtered to that class.
        order = sel.random_order(labels, 5)
        for cls in set(labels.values()):
            chosen = [u for u in matched if labels[u] == cls]
            assert chosen == [u for u in order if labels[u] == cls][: len(chosen)]


def test_class_matched_random_counts_unlabelled_rows_as_their_own_stratum(sel):
    rows = [(f"u{i}", [0.34, 0.33, 0.33] if i < 3 else [0.9, 0.05, 0.05], None if i < 3 else 0) for i in range(10)]
    labels = {uid: label for uid, _, label in rows}
    picked = sel.select_uids(rows, 3, "class_matched_random", seed=1)
    assert sorted(labels[u] is None for u in picked) == [True, True, True]


def test_budget_edges(sel):
    rows = _pool(n=6)
    assert sel.select_uids(rows, 0, "entropy") == []
    assert sel.select_uids(rows, -3, "random") == []
    assert sorted(sel.select_uids(rows, 99, "random")) == sorted(r[0] for r in rows)
    assert sel.select_uids([], 5, "entropy") == []


def test_bad_input_is_rejected(sel):
    with pytest.raises(ValueError, match="unknown strategy"):
        sel.select_uids(_pool(n=3), 1, "leastconfident")
    with pytest.raises(ValueError, match="duplicate UID"):
        sel.select_uids([("x", [1, 0, 0], 0), ("x", [0, 1, 0], 1)], 1, "random")
    with pytest.raises(ValueError, match="non-empty string UID"):
        sel.select_uids([("", [1, 0, 0], 0)], 1, "random")
    with pytest.raises(ValueError, match="needs class probabilities"):
        sel.select_uids([("x", None, 0), ("y", None, 1)], 1, "entropy")
    # The random arm needs no scores (an include-list pool has none).
    assert sel.select_uids([("x", None, 0), ("y", None, 1)], 1, "random") in (["x"], ["y"])


def test_error_messages_never_name_a_uid(sel):
    with pytest.raises(ValueError) as excinfo:
        sel.select_uids([("RUMC_009_left", [1, 0, 0], 0), ("RUMC_009_left", [1, 0, 0], 0)], 1, "random")
    assert "RUMC_009_left" not in str(excinfo.value)


def test_uncertainty_measures(sel):
    import math

    assert sel.entropy([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(math.log(3))
    assert sel.entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert sel.entropy([0.0, 0.0, 0.0]) == pytest.approx(math.log(3))  # degenerate -> uniform
    assert sel.entropy([2.0, 2.0, 2.0]) == pytest.approx(math.log(3))  # unnormalised
    assert sel.margin_uncertainty([0.9, 0.08, 0.02]) < sel.margin_uncertainty([0.45, 0.44, 0.11])


def test_statistics_are_aggregates_and_the_digest_ignores_order(sel):
    rows = _pool(n=40)
    result = sel.select(rows, 12, "entropy", num_classes=3)
    stats = result.stats
    assert stats["pool_size"] == 40 and stats["n_selected"] == 12
    assert set(stats["class_counts"]) == {0, 1, 2}  # absent classes reported as 0
    assert sum(stats["class_counts"].values()) == 12
    assert sum(stats["pool_class_counts"].values()) == 40
    labels = {uid: label for uid, _, label in rows}
    assert stats["class_counts"] == sel.class_counts((labels[u] for u in result.uids), 3)
    expected_entropy = sum(sel.entropy(p) for u, p, _ in rows if u in set(result.uids)) / 12
    assert stats["mean_entropy"] == pytest.approx(expected_entropy)
    assert stats["digest"] == sel.selection_digest(reversed(result.uids))
    assert stats["digest_number"] == int(stats["digest"][:12], 16)
    assert stats["digest_number"] < 2**53  # survives a float64 metric transport exactly
    flat = repr(stats)
    assert not any(uid in flat for uid in result.uids)


def test_statistics_without_scores_have_no_mean_entropy(sel):
    rows = [(f"u{i}", None, i % 3) for i in range(9)]
    stats = sel.selection_statistics(rows, ["u0", "u4"], 3)
    assert stats["mean_entropy"] is None
    assert stats["class_counts"] == {0: 1, 1: 1, 2: 0}
    with pytest.raises(ValueError, match="not in the pool"):
        sel.selection_statistics(rows, ["nope"], 3)


# --------------------------------------------------------------------------------------------------
# The offline study runs the same implementation
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts" / "active_learning"))
    sys.modules.pop("selection", None)
    import selection

    yield selection
    sys.modules.pop("selection", None)


def test_offline_selection_uses_the_job_implementation(offline, sel):
    rows = _pool(n=50)
    csv_rows = [
        {"uid": uid, "ground_truth": str(label), **{f"prob_class_{k}": str(p) for k, p in enumerate(probs)}}
        for uid, probs, label in rows
    ]
    for strategy in ("entropy", "margin", "random", "class_matched_random"):
        offline_pick = [r["uid"] for r in offline.select(list(reversed(csv_rows)), 15, strategy, seed=4)]
        assert offline_pick == sel.select_uids(rows, 15, strategy, seed=4)
    assert offline.STRATEGIES == ("entropy", "margin", "random")
    assert offline.class_composition(csv_rows) == sel.class_counts(label for _, _, label in rows)
