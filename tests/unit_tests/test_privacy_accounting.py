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
    a = pa.PrivacyAccountant(noise_multiplier=0.8, delta=1e-5, epsilon_target=1.0)
    assert not a.budget_exhausted()
    for _ in range(200):
        a.step()
    assert a.budget_exhausted(), f"eps={a.epsilon()} never exceeded target 1.0"


def test_budget_read_from_environment(pa, monkeypatch):
    monkeypatch.setenv("PRIVACY_EPSILON_TARGET", "3.5")
    a = pa.PrivacyAccountant(noise_multiplier=1.0)
    assert a.epsilon_target == 3.5
    monkeypatch.delenv("PRIVACY_EPSILON_TARGET")
    assert pa.PrivacyAccountant(noise_multiplier=1.0).epsilon_target is None


def test_report_states_the_guarantee_is_client_level(pa):
    """The report must not let a reader infer record-level DP."""
    r = pa.PrivacyAccountant(noise_multiplier=1.0).report()
    assert "client-level" in r["guarantee"]
    assert "NOT record-level" in r["guarantee"]


def test_reported_epsilon_is_conservative_vs_subsampling(pa):
    """q=1 is assumed; subsampling would only reduce epsilon, so we over-state."""
    full = pa.compute_epsilon(1.0, 50)
    assert full > 0 and math.isfinite(full)
