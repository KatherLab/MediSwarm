#!/usr/bin/env python3
"""Self-contained RDP accountant for the client-level DP noise of dp_noise.py (T3.3, #530, #593).

The noise itself is added by the ``dp_noise`` round hook inside the training subprocess (see
dp_noise.py and docs/DIFFERENTIAL_PRIVACY.md). This module only does the arithmetic: it turns the
noise multiplier, delta and the number of released updates into an epsilon, decides before a release
whether it still fits a budget, and tells the job builder how many rounds a budget allows. It has no
torch or NVFlare dependency, so the builder can import it on the admin host.

What guarantee this provides, stated precisely
----------------------------------------------
This is **client-level** (participant-level) differential privacy by output
perturbation. The protected unit is one site's contribution to one round: an
observer of the updates a site sends (and of every global model aggregated
from them) cannot tell whether that site contributed, up to (epsilon, delta).

It is **not** record-level DP. It says nothing about whether one *patient* was
in a site's training set -- that requires DP-SGD during local training
(Phase 3). Do not describe output of this mechanism as "differentially private"
without the qualifier; the distinction is the whole difference between
protecting a hospital and protecting a patient.

The mechanism
-------------
Per round, per site: clip the update to L2 norm ``clip_norm`` (C), then add
Gaussian noise of standard deviation ``noise_multiplier * C``. Clipping is what
makes the sensitivity finite and known -- without it the noise is calibrated to
nothing and the guarantee is vacuous, which is precisely what was wrong with the
``PercentilePrivacy`` filter the job configs carried until 1.9.0 (per-coordinate
clipping, no noise; it also never ran in swarm mode).

Adjacency: add/remove-site. A site's clipped update is compared with no
contribution (a zero update), so the L2 sensitivity is C and the noise multiplier
is ``sigma``. Under replace-site adjacency (the site's data swapped for any other
data) the sensitivity is 2C, i.e. the epsilon of ``sigma / 2``: at sigma 2.4 and
delta 1e-5 that is 23.61 after 20 rounds instead of 9.84.

Accounting
----------
Renyi DP for the Gaussian mechanism, composed over rounds, converted to
(epsilon, delta). Opacus is not installed in the ODELIA image and adding it
would force an image rebuild and redistribution to every site, so the
accountant is implemented here. For the Gaussian mechanism at sampling rate
q = 1 -- which is our case, since a round uses the whole local dataset -- RDP
has a closed form and needs no numerical integration:

    eps_RDP(alpha) = alpha / (2 * sigma^2)          per round
    eps_RDP_total  = rounds * alpha / (2 * sigma^2)  (composition is additive)

converted with the standard bound, minimised over alpha:

    eps = eps_RDP_total + log((alpha - 1) / alpha) - (log(delta) + log(alpha)) / (alpha - 1)

The q = 1 assumption is stated rather than hidden: with subsampling the RDP is
strictly smaller, so treating q = 1 is conservative -- it over-states the
privacy cost, never under-states it. Over-stating is the safe direction for a
privacy claim.

Budget
------
A budget is checked BEFORE a release: ``would_exceed()`` / ``check_release()`` answer whether the
next release still fits, and ``budget_exhausted()`` is true as soon as it does not. (Until 1.9.0
``budget_exhausted()`` only became true after the over-budget release had been counted, i.e. one
round too late: at sigma 2.4, delta 1e-5 and target 8 it tripped after release 15, epsilon 8.28.)
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

# Orders to search when converting RDP to (eps, delta). The usual Opacus grid.
DEFAULT_ALPHAS: List[float] = (
    [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
)

# Upper end of the search in max_rounds_for_budget(); far beyond any swarm run.
MAX_ROUNDS_SEARCHED = 1_000_000


class PrivacyBudgetExceeded(RuntimeError):
    """The next release would take epsilon past the target. Raised before anything is released."""


def rdp_gaussian(noise_multiplier: float, steps: int, alpha: float) -> float:
    """RDP of `steps` compositions of the Gaussian mechanism at order `alpha`.

    Sampling rate is taken as 1: a swarm round consumes the whole local dataset.
    """
    if noise_multiplier <= 0:
        return float("inf")
    return steps * alpha / (2.0 * noise_multiplier ** 2)


def rdp_to_dp(rdp: float, alpha: float, delta: float) -> float:
    """Convert an RDP guarantee at one order into (eps, delta)-DP."""
    if alpha <= 1:
        return float("inf")
    return rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)


def compute_epsilon(noise_multiplier: float, steps: int, delta: float = 1e-5,
                    alphas: Optional[Iterable[float]] = None) -> float:
    """Tightest (eps, delta) over the order grid. inf if no noise is added."""
    if steps <= 0:
        return 0.0
    if noise_multiplier <= 0:
        return float("inf")
    best = float("inf")
    for a in (alphas or DEFAULT_ALPHAS):
        if a <= 1:
            continue
        eps = rdp_to_dp(rdp_gaussian(noise_multiplier, steps, a), a, delta)
        if eps < best:
            best = eps
    return best


def max_rounds_for_budget(sigma: float, delta: float, target: float) -> int:
    """Largest number of released rounds R with compute_epsilon(sigma, R, delta) <= target.

    For the job builder: a DP job must not be configured for more training rounds than this, because
    the dp_noise hook refuses the release that would exceed the target (and with it the job). Returns
    0 when not even one round fits (or sigma <= 0, i.e. no noise). epsilon grows strictly with the
    number of rounds, so a binary search is exact.
    """
    if not (target > 0) or sigma <= 0:
        return 0
    if compute_epsilon(sigma, 1, delta) > target:
        return 0
    if compute_epsilon(sigma, MAX_ROUNDS_SEARCHED, delta) <= target:
        return MAX_ROUNDS_SEARCHED
    lo, hi = 1, MAX_ROUNDS_SEARCHED  # invariant: eps(lo) <= target < eps(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if compute_epsilon(sigma, mid, delta) <= target:
            lo = mid
        else:
            hi = mid
    return lo


class PrivacyAccountant:
    """Track the cumulative privacy cost of one site's releases across rounds.

    ``epsilon_target``, when given, is a budget that is enforced before a release: call
    ``check_release()`` before sending an update and ``step()`` after it. A budget that is never
    enforced is not a budget. The target is passed explicitly (dp_noise reads it from
    ODELIA_DP_TARGET_EPSILON); the accountant does not read the environment, so no variable can set
    a budget behind the caller's back.
    """

    def __init__(self, noise_multiplier: float, delta: float = 1e-5,
                 epsilon_target: Optional[float] = None):
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.steps = 0
        self.epsilon_target = None if epsilon_target is None else float(epsilon_target)

    def step(self, n: int = 1) -> float:
        """Count ``n`` releases that have happened; returns the epsilon spent so far."""
        self.steps += n
        return self.epsilon()

    def epsilon(self) -> float:
        return compute_epsilon(self.noise_multiplier, self.steps, self.delta)

    def epsilon_after(self, n: int = 1) -> float:
        """Epsilon after ``n`` more releases."""
        return compute_epsilon(self.noise_multiplier, self.steps + n, self.delta)

    def would_exceed(self, n: int = 1) -> bool:
        """True if ``n`` more releases would take epsilon past the target."""
        return self.epsilon_target is not None and self.epsilon_after(n) > self.epsilon_target

    def budget_exhausted(self) -> bool:
        """True as soon as the next release no longer fits the target."""
        return self.would_exceed(1)

    def check_release(self, n: int = 1) -> float:
        """Raise PrivacyBudgetExceeded if ``n`` more releases would exceed the target.

        Returns the epsilon those releases would bring. Nothing is counted: call ``step()`` once the
        release has happened.
        """
        eps_after = self.epsilon_after(n)
        if self.epsilon_target is not None and eps_after > self.epsilon_target:
            raise PrivacyBudgetExceeded(
                f"privacy budget exhausted: release {self.steps + n} would bring epsilon to {eps_after:.4f} "
                f"> target {self.epsilon_target:g} (noise multiplier {self.noise_multiplier:g}, "
                f"delta {self.delta:g}); {self.steps} release(s) so far, epsilon {self.epsilon():.4f}"
            )
        return eps_after

    def releases_remaining(self) -> Optional[int]:
        """Releases that still fit the target (None without a target)."""
        if self.epsilon_target is None:
            return None
        return max(0, max_rounds_for_budget(self.noise_multiplier, self.delta, self.epsilon_target) - self.steps)

    def report(self) -> dict:
        return {
            "steps": self.steps,
            "noise_multiplier": self.noise_multiplier,
            "delta": self.delta,
            "epsilon": self.epsilon(),
            "epsilon_target": self.epsilon_target,
            "budget_exhausted": self.budget_exhausted(),
            "releases_remaining": self.releases_remaining(),
            "guarantee": "client-level (epsilon, delta)-DP by output perturbation, add/remove-site adjacency; "
                         "NOT record-level",
        }
