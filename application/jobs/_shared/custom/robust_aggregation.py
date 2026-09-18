#!/usr/bin/env python3
"""Byzantine-robust aggregation rules for swarm learning (D3.4, #529).

Why this exists
---------------
Every ODELIA and STAMP job aggregates with ``InTimeAccumulateWeightedAggregator``,
which is a plain size-weighted mean of client updates. A weighted mean has a
**breakdown point of zero**: one participant, by submitting a large enough
update, can move the aggregate arbitrarily far. In a swarm there is no central
party that holds the data and could sanity-check a contribution -- every
participant is a peer, and in CCWF one of them *is* the aggregator for the
round. That is what makes this different from the single-institution case, and
it is the attack surface D3.4 is asked to report on.

This module is self-contained and dependency-free, for the same reason
``privacy_accounting.py`` is: adding a library would force an image rebuild and
redistribution to all eight sites.

The ODELIA-specific catch, stated up front
------------------------------------------
Trimmed mean, coordinate-wise median and Krum all share one property that
matters more here than their breakdown points: **they cannot honour contribution
weights.** Their robustness comes from order statistics over contributions, and
re-weighting the survivors by dataset size would hand back exactly the influence
the trim just removed. So they effectively target the *unweighted* mean of site
optima.

For a consortium with equal sites that is a mild cost. For ODELIA it is not.
Training-set sizes span **94-fold**, and the class mix of the average site is
nothing like the class mix of the pooled data, because Aachen holds 83.9 % of
all benign cases:

===================  ==========  ========  ===========
Class distribution   no-lesion   benign    malignant
===================  ==========  ========  ===========
Pooled (34,462)      0.531       **0.409** 0.059
Mean of the 8 sites  0.660       **0.204** 0.136
===================  ==========  ========  ===========

An unweighted rule therefore aims at a distribution that under-represents benign
by half and over-represents malignant by more than double. Measured in
``scripts/adversarial/attack_simulation.py``: relative error against the pooled
optimum is **0.023** for the size-weighted mean and **0.401** for a plain
unweighted mean, before any trimming and with no attacker present. Trimming adds
only 0.065 on top of that -- so the cost is overwhelmingly *the loss of weights*,
not the discarding of outliers.

Two consequences for D3.4:

* Norm clipping is the only rule here that bounds an adversary's influence
  **while keeping weights**, which is why it is the recommendation.
* Norm clipping bounds influence *proportionally to weight*. It does not protect
  against a large-weight adversary: a poisoning Aachen still holds 51.7 % of the
  weight after clipping. Clipping is a bound, not a defence against the majority
  contributor, and D3.4 must say so.

What each rule guarantees
-------------------------
``n`` contributions of which at most ``f`` are adversarial:

=================  ==================  ==========  =================================
Rule               Tolerates           Weighted?   Cost
=================  ==================  ==========  =================================
weighted_mean      f = 0               yes         none (this is the status quo)
norm_clip + mean   bounded, any f      yes         bias <= 2 * p_adv * clip
trimmed_mean       f < n/2             **no**      discards 2f per coordinate
coord_median       f < n/2             **no**      discards all but the middle
krum               f < (n-2)/2         **no**      keeps exactly one contribution
=================  ==================  ==========  =================================

The "Weighted?" column is the one that decides this for ODELIA. See below.

"Tolerates" is about *unbounded* influence. Every rule here still lets an
adversary exert bounded influence, and none of them detect a colluding majority.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Vector = Sequence[float]

__all__ = [
    "l2_norm", "clip_to_norm", "weighted_mean", "norm_clipped_mean",
    "trimmed_mean", "coordinate_median", "krum", "aggregate", "RULES",
]


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def l2_norm(v: Vector) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def clip_to_norm(v: Vector, max_norm: float) -> List[float]:
    """Scale ``v`` down so its L2 norm is at most ``max_norm``. Never scales up.

    This is the operation that makes an adversary's influence *bounded* rather
    than unbounded, and it is the only rule here that does so without discarding
    anyone -- which is why it is the one that survives ODELIA's data imbalance.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    n = l2_norm(v)
    if n <= max_norm or n == 0.0:
        return [float(x) for x in v]
    scale = max_norm / n
    return [float(x) * scale for x in v]


def _check(updates: Sequence[Vector], weights: Optional[Sequence[float]]) -> List[float]:
    if not updates:
        raise ValueError("no updates to aggregate")
    dim = len(updates[0])
    for i, u in enumerate(updates):
        if len(u) != dim:
            raise ValueError(f"update {i} has dimension {len(u)}, expected {dim}")
    if weights is None:
        return [1.0] * len(updates)
    if len(weights) != len(updates):
        raise ValueError(f"{len(weights)} weights for {len(updates)} updates")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("weights sum to zero")
    return [float(w) for w in weights]


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------
def weighted_mean(updates: Sequence[Vector],
                  weights: Optional[Sequence[float]] = None) -> List[float]:
    """The status quo. Breakdown point zero: one participant can move this anywhere."""
    w = _check(updates, weights)
    total = sum(w)
    dim = len(updates[0])
    return [sum(w[i] * float(updates[i][j]) for i in range(len(updates))) / total
            for j in range(dim)]


def norm_clipped_mean(updates: Sequence[Vector], clip: float,
                      weights: Optional[Sequence[float]] = None) -> List[float]:
    """Clip every contribution to ``clip``, then take the weighted mean.

    Keeps every participant -- which for ODELIA is the point, since discarding
    outliers means discarding Aachen. An adversary holding weight fraction p can
    shift the aggregate by at most ``2 * p * clip``, so influence is bounded by a
    number the consortium chooses rather than by the adversary's honesty.
    """
    return weighted_mean([clip_to_norm(u, clip) for u in updates], weights)


def trimmed_mean(updates: Sequence[Vector], trim: int = 1) -> List[float]:
    """Coordinate-wise mean after dropping the ``trim`` highest and lowest values.

    Tolerates ``trim`` adversaries, and discards ``2 * trim`` contributions per
    coordinate. Unweighted by construction: the trim is what provides robustness,
    and re-weighting the survivors by dataset size would reintroduce the
    influence the trim just removed.
    """
    _check(updates, None)
    n = len(updates)
    if trim < 0:
        raise ValueError(f"trim must be non-negative, got {trim}")
    if 2 * trim >= n:
        raise ValueError(
            f"trim={trim} would discard all {n} contributions; need 2*trim < n")
    dim = len(updates[0])
    out = []
    for j in range(dim):
        col = sorted(float(u[j]) for u in updates)
        kept = col[trim:n - trim] if trim else col
        out.append(sum(kept) / len(kept))
    return out


def coordinate_median(updates: Sequence[Vector]) -> List[float]:
    """Coordinate-wise median. Tolerates f < n/2, discards all but the middle."""
    _check(updates, None)
    n = len(updates)
    dim = len(updates[0])
    mid = n // 2
    out = []
    for j in range(dim):
        col = sorted(float(u[j]) for u in updates)
        out.append(col[mid] if n % 2 else (col[mid - 1] + col[mid]) / 2.0)
    return out


def krum(updates: Sequence[Vector], f: int = 1,
         m: int = 1) -> Tuple[List[float], List[int]]:
    """(Multi-)Krum. Returns (aggregate, indices of the selected contributions).

    Scores each contribution by the summed squared distance to its ``n - f - 2``
    nearest neighbours and averages the ``m`` best. With ``m = 1`` this selects a
    single participant's update as the round's global model.

    Requires ``n > 2f + 2``. Read the ODELIA caveat in the module docstring
    before using this: selecting the most-agreeable contribution in a consortium
    where one site holds half the data selects *against* that site.
    """
    _check(updates, None)
    n = len(updates)
    if f < 0:
        raise ValueError(f"f must be non-negative, got {f}")
    if n <= 2 * f + 2:
        raise ValueError(f"Krum needs n > 2f+2; got n={n}, f={f}")
    if not 1 <= m <= n - f - 2:
        raise ValueError(f"m must be in [1, n-f-2] = [1, {n - f - 2}]; got {m}")

    d2 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = sum((float(a) - float(b)) ** 2 for a, b in zip(updates[i], updates[j]))
            d2[i][j] = d2[j][i] = s

    k = n - f - 2
    scores = [(sum(sorted(d2[i][:i] + d2[i][i + 1:])[:k]), i) for i in range(n)]
    scores.sort()
    chosen = sorted(i for _, i in scores[:m])
    dim = len(updates[0])
    agg = [sum(float(updates[i][j]) for i in chosen) / len(chosen) for j in range(dim)]
    return agg, chosen


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
RULES = ("weighted_mean", "norm_clipped_mean", "trimmed_mean",
         "coordinate_median", "krum")


def aggregate(rule: str, updates: Sequence[Vector],
              weights: Optional[Sequence[float]] = None,
              **kwargs) -> List[float]:
    """Dispatch to a rule by name. ``weights`` is ignored by the rules that
    cannot honour it without giving back the influence they just removed."""
    if rule == "weighted_mean":
        return weighted_mean(updates, weights)
    if rule == "norm_clipped_mean":
        if "clip" not in kwargs:
            raise ValueError("norm_clipped_mean requires clip=")
        return norm_clipped_mean(updates, kwargs["clip"], weights)
    if rule == "trimmed_mean":
        return trimmed_mean(updates, kwargs.get("trim", 1))
    if rule == "coordinate_median":
        return coordinate_median(updates)
    if rule == "krum":
        return krum(updates, kwargs.get("f", 1), kwargs.get("m", 1))[0]
    raise ValueError(f"unknown rule {rule!r}; available: {', '.join(RULES)}")
