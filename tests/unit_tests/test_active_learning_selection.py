"""Unit tests for the active-learning selection strategies (D3.2, #527).

These pin the properties the study depends on: that uncertainty is measured in a
consistent direction, that selection is reproducible across sites, and that the
class-composition check the analysis must not skip actually works.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "active_learning"))
import selection as sel  # noqa: E402


def _row(uid, gt, probs):
    r = {"uid": uid, "ground_truth": str(gt)}
    for i, p in enumerate(probs):
        r[f"prob_class_{i}"] = str(p)
    return r


# ---- uncertainty measures -------------------------------------------------

def test_entropy_is_maximal_for_a_uniform_distribution():
    import math
    assert sel.entropy([1/3, 1/3, 1/3]) == pytest.approx(math.log(3))
    assert sel.entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_entropy_orders_certain_below_uncertain():
    assert sel.entropy([0.9, 0.05, 0.05]) < sel.entropy([0.4, 0.35, 0.25])


def test_margin_points_the_same_way_as_entropy():
    """Both must mean 'higher = less certain', or one ranking routine cannot serve both."""
    confident, torn = [0.9, 0.08, 0.02], [0.45, 0.44, 0.11]
    assert sel.margin_uncertainty(confident) < sel.margin_uncertainty(torn)
    assert sel.entropy(confident) < sel.entropy(torn)


def test_degenerate_rows_do_not_crash():
    """Six model implementations write these CSVs; a bad row is a data problem."""
    assert sel.entropy([0.0, 0.0, 0.0]) == pytest.approx(sel.entropy([1/3, 1/3, 1/3]))
    assert sel.entropy([-1.0, 0.0, 0.0]) >= 0.0


def test_unnormalised_probabilities_are_normalised():
    assert sel.entropy([2.0, 2.0, 2.0]) == pytest.approx(sel.entropy([1/3, 1/3, 1/3]))


# ---- selection ------------------------------------------------------------

def test_entropy_selects_the_uncertain_cases():
    rows = [_row("certain", 0, [0.98, 0.01, 0.01]),
            _row("torn",    2, [0.34, 0.33, 0.33]),
            _row("clear",   0, [0.95, 0.03, 0.02])]
    picked = sel.select(rows, budget=1, strategy="entropy")
    assert [r["uid"] for r in picked] == ["torn"]


def test_selection_is_reproducible_regardless_of_input_order():
    """Sites produce rows in dataloader order; the same experiment must not
    depend on which order a given site happened to write."""
    rows = [_row(f"u{i}", i % 3, [0.5, 0.3, 0.2]) for i in range(10)]
    a = [r["uid"] for r in sel.select(rows, 4, "entropy")]
    b = [r["uid"] for r in sel.select(list(reversed(rows)), 4, "entropy")]
    assert a == b


def test_random_is_reproducible_for_a_seed_and_varies_across_seeds():
    rows = [_row(f"u{i}", 0, [0.5, 0.3, 0.2]) for i in range(50)]
    assert [r["uid"] for r in sel.select(rows, 5, "random", seed=1)] == \
           [r["uid"] for r in sel.select(rows, 5, "random", seed=1)]
    assert [r["uid"] for r in sel.select(rows, 5, "random", seed=1)] != \
           [r["uid"] for r in sel.select(rows, 5, "random", seed=2)]


def test_budget_bounds_and_edge_cases():
    rows = [_row(f"u{i}", 0, [0.4, 0.35, 0.25]) for i in range(5)]
    assert sel.select(rows, 0, "entropy") == []
    assert len(sel.select(rows, 3, "entropy")) == 3
    assert len(sel.select(rows, 99, "entropy")) == 5      # budget above pool size


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        sel.select([_row("u", 0, [1.0, 0, 0])], 1, "leastconfident")


# ---- the check the analysis must not skip ---------------------------------

def test_class_composition_exposes_rare_class_over_selection():
    """Uncertainty sampling over-selects rare classes; the study must report it."""
    rows = ([_row(f"n{i}", 0, [0.97, 0.02, 0.01]) for i in range(20)] +
            [_row(f"b{i}", 1, [0.36, 0.34, 0.30]) for i in range(3)])
    picked = sel.select(rows, budget=3, strategy="entropy")
    comp = sel.class_composition(picked)
    assert comp == {1: 3}, "entropy took every benign case out of a 20:3 pool"
    assert sel.class_composition(rows) == {0: 20, 1: 3}


def test_class_composition_ignores_rows_without_labels():
    assert sel.class_composition([{"uid": "x"}, _row("y", 2, [0.1, 0.2, 0.7])]) == {2: 1}
