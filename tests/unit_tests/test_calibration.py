"""Unit tests for the D3.2 calibration analysis (#527, E4).

Dependency-free -- no torch, no nvflare, no sklearn -- so this cannot be
silently skipped in CI. A skipped test file reports the same green as a passing
one (F10, #565).
"""

import importlib.util
import math
import os
import random

import pytest

_P = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "calibration",
                  "analyse_calibration.py")
_spec = importlib.util.spec_from_file_location("analyse_calibration", _P)
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


def rows_from(pairs):
    """pairs: [(y, [p0,p1,p2]), ...]"""
    return [{"site": "S", "uid": f"S:{i}", "y": y, "p": p}
            for i, (y, p) in enumerate(pairs)]


class TestTemperatureScaling:
    def test_t_of_one_is_the_identity(self):
        p = [0.6, 0.3, 0.1]
        assert cal.softmax_t(p, 1.0) == pytest.approx(p, abs=1e-9)

    def test_output_is_a_distribution(self):
        for T in (0.2, 1.0, 5.0):
            out = cal.softmax_t([0.5, 0.4, 0.1], T)
            assert sum(out) == pytest.approx(1.0)
            assert all(0.0 <= x <= 1.0 for x in out)

    def test_high_temperature_flattens(self):
        out = cal.softmax_t([0.9, 0.07, 0.03], 100.0)
        assert out == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=0.02)

    def test_low_temperature_sharpens_towards_the_argmax(self):
        out = cal.softmax_t([0.5, 0.3, 0.2], 0.01)
        assert out[0] > 0.99

    def test_scaling_preserves_the_argmax(self):
        """The property the verdict depends on: temperature scaling is monotone
        in the logits, so it can never reorder classes within a case."""
        rng = random.Random(0)
        for _ in range(200):
            p = [rng.random() + 1e-6 for _ in range(3)]
            s = sum(p)
            p = [x / s for x in p]
            for T in (0.1, 0.5, 2.0, 9.0):
                assert cal.softmax_t(p, T).index(max(cal.softmax_t(p, T))) == p.index(max(p))

    def test_handles_a_zero_probability(self):
        out = cal.softmax_t([1.0, 0.0, 0.0], 1.0)
        assert sum(out) == pytest.approx(1.0)
        assert all(math.isfinite(x) for x in out)


class TestFitTemperature:
    def test_recovers_a_known_temperature(self):
        """Take confident predictions, flatten them by a known T, and check the
        fit recovers roughly that much sharpening."""
        rng = random.Random(1)
        base = []
        for _ in range(400):
            y = rng.randrange(3)
            p = [0.1, 0.1, 0.1]
            p[y] = 0.8
            base.append((y, cal.softmax_t(p, 3.0)))   # deliberately flattened
        T = cal.fit_temperature(rows_from(base))
        assert T < 1.0, "over-flattened predictions must be sharpened back"

    def test_perfectly_calibrated_input_stays_near_one(self):
        rng = random.Random(2)
        pairs = []
        for _ in range(600):
            p0 = rng.uniform(0.2, 0.8)
            rest = (1 - p0) / 2
            p = [p0, rest, rest]
            y = 0 if rng.random() < p0 else rng.choice([1, 2])
            pairs.append((y, p))
        T = cal.fit_temperature(rows_from(pairs))
        assert 0.6 < T < 1.7

    def test_fit_minimises_nll_within_the_search_interval(self):
        """The fit is a minimum over [lo, hi], not over all reals -- so only
        compare against points inside the interval."""
        rng = random.Random(7)
        pairs = []
        for _ in range(300):
            p0 = rng.uniform(0.3, 0.95)
            rest = (1 - p0) / 2
            y = 0 if rng.random() < p0 * 0.7 else rng.choice([1, 2])
            pairs.append((y, [p0, rest, rest]))
        rows = rows_from(pairs)
        lo, hi = 0.05, 20.0
        T = cal.fit_temperature(rows, lo, hi)
        best = cal.nll(rows, T)
        for other in (lo, 0.5, 1.0, 2.0, 5.0, hi):
            assert cal.nll(rows, other) >= best - 1e-9, f"T={T} beaten by {other}"

    def test_matches_brute_force(self):
        rng = random.Random(8)
        pairs = [(rng.randrange(3), [0.5, 0.3, 0.2]) for _ in range(150)]
        rows = rows_from(pairs)
        T = cal.fit_temperature(rows, 0.05, 20.0)
        grid = min((cal.nll(rows, 0.05 + i * (20 - 0.05) / 2000) for i in range(2001)))
        assert cal.nll(rows, T) <= grid + 1e-9

    def test_reports_when_the_fit_lands_on_a_bound(self):
        """A perfectly ordered ranking wants maximum sharpening, which pins T to
        the lower bound. That is a diagnosis, not a fitted temperature."""
        rng = random.Random(10)
        pairs = []
        for _ in range(400):
            y = rng.randrange(3)
            p = [0.1, 0.1, 0.1]
            p[y] = 0.8
            pairs.append((y, cal.softmax_t(p, 3.0)))   # argmax always correct
        T, at_bound = cal.fit_temperature_checked(rows_from(pairs), 0.05, 20.0)
        assert at_bound == "lo", f"expected a lower-bound hit, got {at_bound} at T={T}"

    def test_no_bound_flag_for_a_genuine_interior_fit(self):
        """Over-confident but imperfectly ordered -> a real interior optimum."""
        rng = random.Random(11)
        pairs = []
        for _ in range(600):
            a = rng.randrange(3)
            p = [0.05, 0.05, 0.05]
            p[a] = 0.90
            y = a if rng.random() < 0.60 else rng.choice([c for c in range(3) if c != a])
            pairs.append((y, p))
        T, at_bound = cal.fit_temperature_checked(rows_from(pairs), 0.05, 20.0)
        assert at_bound is None, f"unexpected bound hit at T={T}"
        assert T > 1.0, "over-confident predictions must be softened"


class TestMetrics:
    def test_perfect_confident_predictions_score_zero(self):
        rows = rows_from([(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]), (2, [0.0, 0.0, 1.0])])
        assert cal.brier(rows) == pytest.approx(0.0, abs=1e-6)
        assert cal.nll(rows) == pytest.approx(0.0, abs=1e-6)
        e, _ = cal.ece(rows)
        assert e == pytest.approx(0.0, abs=1e-6)

    def test_confidently_wrong_is_maximally_penalised(self):
        right = rows_from([(0, [1.0, 0.0, 0.0])])
        wrong = rows_from([(1, [1.0, 0.0, 0.0])])
        assert cal.brier(wrong) > cal.brier(right)
        assert cal.nll(wrong) > cal.nll(right)

    def test_ece_detects_systematic_overconfidence(self):
        """90 % confident, 50 % accurate -> ECE near 0.4."""
        rows = rows_from([(0 if i < 50 else 1, [0.9, 0.05, 0.05]) for i in range(100)])
        e, table = cal.ece(rows, bins=10)
        assert e == pytest.approx(0.4, abs=0.05)
        assert sum(r["n"] for r in table) == 100

    def test_ece_bins_partition_the_input(self):
        rng = random.Random(4)
        pairs = []
        for _ in range(300):
            p = [rng.random() + 1e-6 for _ in range(3)]
            s = sum(p)
            pairs.append((rng.randrange(3), [x / s for x in p]))
        _, table = cal.ece(rows_from(pairs), bins=10)
        assert sum(r["n"] for r in table) == 300

    def test_entropy_is_maximal_for_the_uniform_distribution(self):
        assert cal.entropy([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(math.log(3))
        assert cal.entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)

    def test_entropy_orders_distributions_by_spread(self):
        assert cal.entropy([0.9, 0.05, 0.05]) < cal.entropy([0.5, 0.3, 0.2])


class TestAcquisition:
    def test_malignant_yield_counts_class_two(self):
        rows = rows_from([(2, [0, 0, 1]), (0, [1, 0, 0]), (2, [0, 0, 1])])
        assert cal.malignant_yield(rows, rows, 3) == 2
        assert cal.malignant_yield(rows, rows, 1) == 1

    def test_acquisition_reports_all_three_arms(self):
        rng = random.Random(5)
        pairs = []
        for i in range(60):
            p = [rng.random() + 1e-6 for _ in range(3)]
            s = sum(p)
            pairs.append((2 if i % 4 == 0 else 0, [x / s for x in p]))
        out = cal.acquisition_test(rows_from(pairs), 1.2, [10, 20], seeds=30)
        assert set(out) == {"uncalibrated", "calibrated", "random"}
        for b in ("10", "20"):
            assert isinstance(out["uncalibrated"][b], int)
            assert "mean" in out["random"][b] and "sd" in out["random"][b]

    def test_random_arm_yield_grows_with_budget(self):
        rng = random.Random(6)
        pairs = [(2 if i % 3 == 0 else 0, [0.34, 0.33, 0.33]) for i in range(90)]
        out = cal.acquisition_test(rows_from(pairs), 1.0, [10, 50], seeds=50)
        assert out["random"]["50"]["mean"] > out["random"]["10"]["mean"]


class TestWrongArchitectureGuard:
    """E4: two published conclusions were computed on predictions from a
    checkpoint loaded with the wrong architecture. The guard is what stops
    that recurring, so it has to be tested, not trusted."""

    def _rows(self, malignant_separable):
        rng = random.Random(20)
        rows = []
        for i in range(120):
            y = 2 if i % 3 == 0 else 0
            if malignant_separable:
                p2 = rng.uniform(0.6, 0.95) if y == 2 else rng.uniform(0.02, 0.35)
            else:                                     # chance-level, the E2 signature
                p2 = rng.uniform(0.1, 0.9)
            rest = (1 - p2) / 2
            rows.append({"site": "S", "uid": f"S:{i}", "y": y, "p": [rest, rest, p2]})
        return rows

    def test_refuses_chance_level_predictions(self):
        with pytest.raises(SystemExit) as e:
            cal.guard_wrong_architecture(self._rows(False), "results/best")
        msg = str(e.value)
        assert "REFUSING" in msg
        assert "E2" in msg, "the guard must name the pitfall so it is searchable"
        assert "results/fixed" in msg, "the guard must point at the correct data"

    def test_allows_a_genuine_evaluation(self):
        cal.guard_wrong_architecture(self._rows(True), "results/fixed")   # must not raise

    def test_accepts_raw_csv_rows_too(self):
        """The two consumers hand it different row shapes; both must work."""
        raw = [{"ground_truth": "2" if i % 3 == 0 else "0",
                "prob_class_2": "0.9" if i % 3 == 0 else "0.1"} for i in range(60)]
        cal.guard_wrong_architecture(raw, "ok")
        bad = [{"ground_truth": "2" if i % 3 == 0 else "0",
                "prob_class_2": "0.5"} for i in range(60)]
        # all-tied scores give AUROC 0.5 -> must be refused
        with pytest.raises(SystemExit):
            cal.guard_wrong_architecture(bad, "tied")
