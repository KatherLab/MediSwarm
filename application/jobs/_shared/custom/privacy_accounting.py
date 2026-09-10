#!/usr/bin/env python3
"""Calibrated-noise filter and a self-contained RDP accountant (T3.3 Phase 2, #530).

What guarantee this provides, stated precisely
----------------------------------------------
This is **client-level** (participant-level) differential privacy by output
perturbation. The protected unit is one site's contribution to one round: an
observer of the aggregated model cannot tell whether any single site
participated in that round, up to (epsilon, delta).

It is **not** record-level DP. It says nothing about whether one *patient* was
in a site's training set -- that requires DP-SGD during local training
(Phase 3). Do not describe output of this filter as "differentially private"
without the qualifier; the distinction is the whole difference between
protecting a hospital and protecting a patient.

The mechanism
-------------
Per round, per site: clip the update to L2 norm ``clip_norm`` (C), then add
Gaussian noise of standard deviation ``noise_multiplier * C``. Clipping is what
makes the sensitivity finite and known -- without it the noise is calibrated to
nothing and the guarantee is vacuous, which is precisely the gap in the
``PercentilePrivacy`` filter currently wired into every job.

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
"""

from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional

# Orders to search when converting RDP to (eps, delta). The usual Opacus grid.
DEFAULT_ALPHAS: List[float] = (
    [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
)


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


class PrivacyAccountant:
    """Track cumulative privacy cost across rounds.

    `PRIVACY_EPSILON_TARGET`, when set, is a budget: once the spent epsilon
    would exceed it, `budget_exhausted()` is true and the run should stop
    rather than quietly continue spending. A budget that is never enforced is
    not a budget.
    """

    def __init__(self, noise_multiplier: float, delta: float = 1e-5,
                 epsilon_target: Optional[float] = None):
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.steps = 0
        if epsilon_target is None:
            env = os.environ.get("PRIVACY_EPSILON_TARGET", "").strip()
            epsilon_target = float(env) if env else None
        self.epsilon_target = epsilon_target

    def step(self, n: int = 1) -> float:
        self.steps += n
        return self.epsilon()

    def epsilon(self) -> float:
        return compute_epsilon(self.noise_multiplier, self.steps, self.delta)

    def budget_exhausted(self) -> bool:
        return self.epsilon_target is not None and self.epsilon() > self.epsilon_target

    def report(self) -> dict:
        return {
            "steps": self.steps,
            "noise_multiplier": self.noise_multiplier,
            "delta": self.delta,
            "epsilon": self.epsilon(),
            "epsilon_target": self.epsilon_target,
            "budget_exhausted": self.budget_exhausted(),
            "guarantee": "client-level (epsilon, delta)-DP by output perturbation; NOT record-level",
        }
