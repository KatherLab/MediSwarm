"""Tests for the calibrated-noise accountant (T3.3 Phase 2, #530).

A privacy accountant that is wrong in the optimistic direction is worse than
none at all -- it licenses a claim that is not true. These tests pin the
direction of every monotonicity as well as the values.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import SHARED_CUSTOM_DIR, import_module_from_path  # noqa: E402

MODULE = SHARED_CUSTOM_DIR / "privacy_accounting.py"


@pytest.fixture(scope="module")
def pa():
    return import_module_from_path("_privacy_accounting", MODULE)


def test_no_noise_is_no_guarantee(pa):
    """sigma = 0 must be infinite epsilon, never a finite comfortable number."""
    assert pa.compute_epsilon(0.0, 10) == float("inf")
    assert pa.compute_epsilon(-1.0, 10) == float("inf")


def test_zero_steps_costs_nothing(pa):
    assert pa.compute_epsilon(1.0, 0) == 0.0


def test_epsilon_grows_with_rounds(pa):
    """Composition must accumulate. A flat epsilon would hide the real cost."""
    e = [pa.compute_epsilon(1.0, r) for r in (1, 5, 20, 100)]
    assert e == sorted(e), e
    assert e[0] < e[-1]


def test_more_noise_buys_more_privacy(pa):
    """Epsilon must fall as sigma rises."""
    e = [pa.compute_epsilon(s, 20) for s in (0.5, 1.0, 2.0, 8.0)]
    assert e == sorted(e, reverse=True), e


def test_smaller_delta_costs_more_epsilon(pa):
    assert pa.compute_epsilon(1.0, 20, delta=1e-7) > pa.compute_epsilon(1.0, 20, delta=1e-3)


def test_matches_closed_form_at_a_fixed_order(pa):
    """Guard the arithmetic itself, not just its shape."""
    sigma, steps, alpha, delta = 2.0, 10, 8.0, 1e-5
    rdp = pa.rdp_gaussian(sigma, steps, alpha)
    assert rdp == pytest.approx(steps * alpha / (2 * sigma ** 2))
    expect = rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)
    assert pa.rdp_to_dp(rdp, alpha, delta) == pytest.approx(expect)
    # the searched minimum can only be <= any single order
    assert pa.compute_epsilon(sigma, steps, delta) <= expect + 1e-9


def test_accountant_accumulates_over_steps(pa):
    a = pa.PrivacyAccountant(noise_multiplier=1.0, delta=1e-5)
    first = a.step()
    for _ in range(9):
        a.step()
    assert a.steps == 10
    assert a.epsilon() > first


def test_budget_is_actually_enforced(pa):
    """A budget that never trips is not a budget."""
    a = pa.PrivacyAccountant(noise_multiplier=2.0, delta=1e-5, epsilon_target=5.0)
    assert not a.budget_exhausted()
    for _ in range(200):
        if a.budget_exhausted():
            break
        a.step()
    assert a.budget_exhausted(), f"eps={a.epsilon()} never reached target 5.0"
    assert a.epsilon() <= 5.0, "the accountant let a release past the target"


def test_budget_trips_before_the_over_budget_release(pa):
    """The pre-1.9.0 check fired one round late: at sigma 2.4, target 8 it tripped after release 15
    (epsilon 8.28), i.e. after the over-budget update had been sent. Now release 15 is refused up front."""
    a = pa.PrivacyAccountant(noise_multiplier=2.4, delta=1e-5, epsilon_target=8.0)
    for _ in range(13):
        a.check_release()
        a.step()
    assert not a.budget_exhausted() and not a.would_exceed()
    a.check_release()
    a.step()  # release 14: epsilon 7.94
    assert a.epsilon() == pytest.approx(7.94, abs=0.005)
    assert a.budget_exhausted() and a.would_exceed()
    with pytest.raises(pa.PrivacyBudgetExceeded, match="8.27"):
        a.check_release()
    assert a.steps == 14, "a refused release must not be counted"
    assert a.releases_remaining() == 0


def test_check_release_returns_the_epsilon_it_would_bring(pa):
    a = pa.PrivacyAccountant(noise_multiplier=2.4, delta=1e-5)
    assert a.check_release() == pytest.approx(pa.compute_epsilon(2.4, 1, 1e-5))
    assert a.check_release(3) == pytest.approx(pa.compute_epsilon(2.4, 3, 1e-5))
    assert a.steps == 0
    assert a.releases_remaining() is None  # no target, no budget


def test_target_is_never_read_from_the_environment(pa, monkeypatch):
    """The old accountant silently picked up PRIVACY_EPSILON_TARGET; nothing ever set it. The budget is
    now passed explicitly (dp_noise reads ODELIA_DP_TARGET_EPSILON), so no variable can set it implicitly."""
    monkeypatch.setenv("PRIVACY_EPSILON_TARGET", "3.5")
    assert pa.PrivacyAccountant(noise_multiplier=1.0).epsilon_target is None
    assert pa.PrivacyAccountant(noise_multiplier=1.0, epsilon_target=3.5).epsilon_target == 3.5


def test_report_states_the_guarantee_is_client_level(pa):
    """The report must not let a reader infer record-level DP."""
    r = pa.PrivacyAccountant(noise_multiplier=1.0).report()
    assert "client-level" in r["guarantee"]
    assert "NOT record-level" in r["guarantee"]


def test_reported_epsilon_is_conservative_vs_subsampling(pa):
    """q=1 is assumed; subsampling would only reduce epsilon, so we over-state."""
    full = pa.compute_epsilon(1.0, 50)
    assert full > 0 and math.isfinite(full)


# Verified trajectory at the planned DP run's settings (sigma 2.4, delta 1e-5), rounds 1..20; the
# T3.3 noise-cost table and docs/DIFFERENTIAL_PRIVACY.md quote these numbers.
SIGMA_2_4_EPSILONS = [1.77, 2.60, 3.26, 3.84, 4.36, 4.84, 5.29, 5.72, 6.12, 6.51,
                      6.89, 7.25, 7.60, 7.94, 8.28, 8.60, 8.92, 9.23, 9.54, 9.84]


def test_per_round_epsilon_trajectory_at_sigma_2_4(pa):
    got = [pa.compute_epsilon(2.4, r, 1e-5) for r in range(1, 21)]
    assert got == pytest.approx(SIGMA_2_4_EPSILONS, abs=0.005)
    assert pa.compute_epsilon(2.4, 1, 1e-5) == pytest.approx(1.7716, abs=1e-4)
    assert pa.compute_epsilon(2.4, 20, 1e-5) == pytest.approx(9.8416, abs=1e-4)


def test_replace_site_adjacency_doubles_the_sensitivity(pa):
    """Documented alternative: sensitivity 2C is the epsilon of sigma / 2 (23.61 after 20 rounds)."""
    assert pa.compute_epsilon(1.2, 20, 1e-5) == pytest.approx(23.61, abs=0.005)


@pytest.mark.parametrize("target,rounds", [(10.0, 20), (8.0, 14), (3.0, 2), (1.77, 0), (1.0, 0)])
def test_max_rounds_for_budget(pa, target, rounds):
    assert pa.max_rounds_for_budget(2.4, 1e-5, target) == rounds


@pytest.mark.parametrize("sigma,delta,target", [(0.8, 1e-5, 20.0), (2.4, 1e-6, 10.0), (6.7, 1e-5, 3.0),
                                                (18.1, 1e-5, 1.0), (50.0, 1e-5, 2.0)])
def test_max_rounds_for_budget_is_the_last_round_that_fits(pa, sigma, delta, target):
    r = pa.max_rounds_for_budget(sigma, delta, target)
    assert r >= 1
    assert pa.compute_epsilon(sigma, r, delta) <= target < pa.compute_epsilon(sigma, r + 1, delta)


def test_max_rounds_for_budget_without_noise_or_budget_is_zero(pa):
    assert pa.max_rounds_for_budget(0.0, 1e-5, 10.0) == 0
    assert pa.max_rounds_for_budget(2.4, 1e-5, 0.0) == 0
    assert pa.max_rounds_for_budget(2.4, 1e-5, float("nan")) == 0


def test_issue_530_table_still_holds(pa):
    """#530 promised: epsilon <= 10 / 3 / 1 over 20 rounds at sigma 2.4 / 6.7 / 18.1 (delta 1e-5)."""
    for sigma, target in ((2.4, 10.0), (6.7, 3.0), (18.1, 1.0)):
        assert pa.max_rounds_for_budget(sigma, 1e-5, target) >= 20


def test_accountant_module_needs_no_torch():
    """The job builder imports the accountant on the admin host, where torch may be missing."""
    import ast

    imported = set()
    for node in ast.walk(ast.parse(MODULE.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "math", "typing"}, imported
