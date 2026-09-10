"""Unit tests for the D3.4 Byzantine-robust aggregation rules (#529).

Deliberately dependency-free -- no torch, no nvflare -- so these run in CI
regardless of whether the training image is installed. A skipped test file
reports the same green as a passing one, which is how a broken metric path
survived two live eight-site runs (F10); this file must never be skippable.
"""

import importlib.util
import math
import os

import pytest

_RA = os.path.join(os.path.dirname(__file__), "..", "..", "application", "jobs",
                   "_shared", "custom", "robust_aggregation.py")
_spec = importlib.util.spec_from_file_location("robust_aggregation", _RA)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


# ---------------------------------------------------------------- primitives
class TestNormAndClip:
    def test_l2_norm(self):
        assert ra.l2_norm([3, 4]) == pytest.approx(5.0)
        assert ra.l2_norm([0, 0, 0]) == 0.0

    def test_clip_scales_down_to_exactly_max_norm(self):
        out = ra.clip_to_norm([30, 40], 5)
        assert ra.l2_norm(out) == pytest.approx(5.0)
        assert out == pytest.approx([3.0, 4.0])

    def test_clip_never_scales_up(self):
        """A vector already inside the ball must come back untouched."""
        assert ra.clip_to_norm([3, 4], 100) == pytest.approx([3.0, 4.0])

    def test_clip_preserves_direction(self):
        v = [1.0, -2.0, 3.0]
        out = ra.clip_to_norm(v, 0.5)
        scale = ra.l2_norm(out) / ra.l2_norm(v)
        assert out == pytest.approx([x * scale for x in v])

    def test_clip_handles_zero_vector(self):
        assert ra.clip_to_norm([0, 0], 1) == [0.0, 0.0]

    def test_clip_rejects_nonpositive_bound(self):
        with pytest.raises(ValueError, match="max_norm"):
            ra.clip_to_norm([1, 2], 0)


# ------------------------------------------------------------------ the rules
class TestWeightedMean:
    def test_matches_hand_computation(self):
        # (1*1 + 3*3) / 4 = 2.5
        assert ra.weighted_mean([[1, 1], [3, 3]], [1, 3]) == pytest.approx([2.5, 2.5])

    def test_defaults_to_unweighted(self):
        assert ra.weighted_mean([[0], [10]]) == pytest.approx([5.0])

    def test_rejects_ragged_input(self):
        with pytest.raises(ValueError, match="dimension"):
            ra.weighted_mean([[1, 2], [1]])

    def test_rejects_bad_weights(self):
        with pytest.raises(ValueError, match="weights"):
            ra.weighted_mean([[1], [2]], [1])
        with pytest.raises(ValueError, match="non-negative"):
            ra.weighted_mean([[1], [2]], [-1, 2])
        with pytest.raises(ValueError, match="sum to zero"):
            ra.weighted_mean([[1], [2]], [0, 0])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="no updates"):
            ra.weighted_mean([])


class TestTrimmedMean:
    def test_drops_the_extremes(self):
        # [1,2,3,4,100] -> drop 1 and 100 -> mean(2,3,4) = 3
        assert ra.trimmed_mean([[1], [2], [3], [4], [100]], trim=1) == pytest.approx([3.0])

    def test_trim_zero_is_the_plain_mean(self):
        assert ra.trimmed_mean([[1], [3]], trim=0) == pytest.approx([2.0])

    def test_trims_per_coordinate_independently(self):
        """The outlier is in a different coordinate for each update."""
        ups = [[100, 0], [0, 100], [1, 1], [1, 1], [1, 1]]
        assert ra.trimmed_mean(ups, trim=1) == pytest.approx([1.0, 1.0])

    def test_rejects_trim_that_would_empty_the_set(self):
        with pytest.raises(ValueError, match="discard all"):
            ra.trimmed_mean([[1], [2], [3]], trim=2)

    def test_rejects_negative_trim(self):
        with pytest.raises(ValueError, match="non-negative"):
            ra.trimmed_mean([[1], [2], [3]], trim=-1)


class TestCoordinateMedian:
    def test_odd_count(self):
        assert ra.coordinate_median([[1], [2], [100]]) == pytest.approx([2.0])

    def test_even_count_averages_the_middle_pair(self):
        assert ra.coordinate_median([[1], [2], [4], [100]]) == pytest.approx([3.0])


class TestKrum:
    def test_selects_the_cluster_and_rejects_the_outlier(self):
        ups = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [50, 50]]
        agg, chosen = ra.krum(ups, f=1, m=1)
        assert 4 not in chosen, "Krum must not select the outlier"
        assert ra.l2_norm(agg) < 1.0

    def test_multi_krum_averages_m_selections(self):
        # n=5, f=1 -> m may be at most n-f-2 = 2.
        ups = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [50, 50]]
        _, chosen = ra.krum(ups, f=1, m=2)
        assert len(chosen) == 2
        assert 4 not in chosen

    def test_requires_enough_participants(self):
        with pytest.raises(ValueError, match="n > 2f\\+2"):
            ra.krum([[1], [2], [3]], f=1)

    def test_rejects_out_of_range_m(self):
        ups = [[0], [0.1], [0.2], [0.3], [0.4]]
        with pytest.raises(ValueError, match="m must be in"):
            ra.krum(ups, f=1, m=99)


# ------------------------------------------------- the properties that matter
class TestBreakdownBehaviour:
    """The claims D3.4 rests on, as executable assertions."""

    HONEST = [[1.0, 1.0] for _ in range(7)]

    def test_weighted_mean_has_unbounded_influence(self):
        """One participant moves the plain weighted mean arbitrarily far."""
        prev = None
        for scale in (1e2, 1e4, 1e6):
            ups = self.HONEST + [[scale, scale]]
            got = ra.l2_norm(ra.weighted_mean(ups))
            assert prev is None or got > prev * 10, "influence must grow with the attack"
            prev = got
        assert prev > 1e5

    def test_norm_clipping_bounds_influence(self):
        """The same attack, bounded by a number the consortium chooses."""
        clip = 2.0
        results = [ra.l2_norm(ra.norm_clipped_mean(self.HONEST + [[s, s]], clip))
                   for s in (1e2, 1e4, 1e6)]
        assert max(results) <= clip, "no aggregate may exceed the clip"
        assert max(results) - min(results) < 1e-6, "attack scale must stop mattering"

    def test_clipping_bound_scales_with_adversary_weight(self):
        """The D3.4 caveat: clipping does not stop a majority contributor.

        Same clipped attack, once at 1/8 of the weight and once at 51.7 %.
        """
        clip = 2.0
        attack = [1e6, 1e6]
        small = ra.norm_clipped_mean(self.HONEST + [attack], clip, [1] * 7 + [1])
        large = ra.norm_clipped_mean(self.HONEST + [attack], clip, [1] * 7 + [7.5])
        honest_dir = ra.weighted_mean(self.HONEST)
        d_small = ra.l2_norm([a - b for a, b in zip(small, honest_dir)])
        d_large = ra.l2_norm([a - b for a, b in zip(large, honest_dir)])
        assert d_large > 2 * d_small, "a heavier adversary must move it further"

    def test_order_statistic_rules_ignore_weights(self):
        """Why they cost ODELIA so much: the weight argument cannot reach them."""
        ups = [[0.0], [0.0], [0.0], [10.0], [10.0]]
        # aggregate() accepts weights but the order-statistic rules must not use them
        heavy_on_the_tens = [1, 1, 1, 1000, 1000]
        assert ra.aggregate("trimmed_mean", ups, heavy_on_the_tens) == \
               ra.aggregate("trimmed_mean", ups, None)
        assert ra.aggregate("coordinate_median", ups, heavy_on_the_tens) == \
               ra.aggregate("coordinate_median", ups, None)
        # whereas the weighted mean does move
        assert ra.weighted_mean(ups, heavy_on_the_tens)[0] > \
               ra.weighted_mean(ups, None)[0]


class TestDispatch:
    def test_every_advertised_rule_dispatches(self):
        ups = [[float(i), float(i)] for i in range(6)]
        for rule in ra.RULES:
            kw = {"clip": 10.0} if rule == "norm_clipped_mean" else {}
            out = ra.aggregate(rule, ups, [1] * 6, **kw)
            assert len(out) == 2 and all(math.isfinite(x) for x in out), rule

    def test_norm_clipped_mean_requires_a_clip(self):
        with pytest.raises(ValueError, match="requires clip"):
            ra.aggregate("norm_clipped_mean", [[1], [2]])

    def test_unknown_rule_names_the_alternatives(self):
        with pytest.raises(ValueError, match="unknown rule"):
            ra.aggregate("fedavg_but_cooler", [[1], [2]])
