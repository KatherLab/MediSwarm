#!/usr/bin/env python3
"""Sample-selection strategies for the active-learning study (D3.2 / T3.2, #527).

What this is for
----------------
D3.2 asks whether active learning is *useful*: whether selecting which cases to
label by model uncertainty reaches a given performance with fewer labels than
selecting at random. That is a property of the selection rule and the data, so
it is measured here on per-case class probabilities rather than by running a
swarm round per iteration -- 2.2 h per round measured on the 8-site run makes
the federated form of this experiment cost ~110 h across two arms, which does
not fit before M48.

Strategies
----------
``entropy``  predictive entropy over the class distribution. Highest when the
             model spreads mass evenly; picks cases it cannot commit on.
``margin``   top-1 probability minus top-2. Small margin means the model is
             torn between two specific classes -- often more useful than
             entropy when one class is rare, because entropy can be dominated
             by a long flat tail over classes the model never predicts.
``random``   the control. An active-learning result is only meaningful against
             this, at identical budget and seed.

A caution that shapes the analysis
----------------------------------
Uncertainty sampling naturally over-selects rare classes, and ODELIA's benign
class is rare at every site. A gain over random must therefore be shown to
survive when the class composition of the selected set is controlled for --
otherwise the experiment measures class imbalance and calls it active learning.
``class_composition`` is provided so that check is cheap and hard to skip.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

STRATEGIES = ("entropy", "margin", "random")


def _normalise(probs: Sequence[float]) -> List[float]:
    """Return a proper distribution, or a uniform one if the input is degenerate.

    Rows come from CSVs written by six different model implementations; a row
    that does not sum to 1 is a data problem, not a reason to crash a study.
    """
    vals = [max(0.0, float(p)) for p in probs]
    total = sum(vals)
    if total <= 0:
        n = len(vals) or 1
        return [1.0 / n] * n
    return [v / total for v in vals]


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in nats. Higher means less certain."""
    return -sum(p * math.log(p) for p in _normalise(probs) if p > 0)


def margin_uncertainty(probs: Sequence[float]) -> float:
    """1 - (top1 - top2), so that larger means less certain.

    Returned in the same direction as entropy so both can feed one ranking
    routine; a caller comparing raw margins directly would otherwise have to
    remember that small margin means high uncertainty.
    """
    ordered = sorted(_normalise(probs), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return 1.0 - (ordered[0] - ordered[1])


def score(row: Dict[str, str], strategy: str, rng: random.Random) -> float:
    """Uncertainty score for one prediction row. Higher is selected first."""
    if strategy == "random":
        return rng.random()
    probs = [float(row[k]) for k in sorted(row) if k.startswith("prob_class_")]
    if strategy == "entropy":
        return entropy(probs)
    if strategy == "margin":
        return margin_uncertainty(probs)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


def select(rows: Sequence[Dict[str, str]], budget: int, strategy: str,
           seed: int = 0) -> List[Dict[str, str]]:
    """Pick ``budget`` rows under ``strategy``.

    Ties are broken by uid, not by input order: the CSVs arrive in whatever
    order each site's dataloader produced, so an order-dependent tie-break
    would make the same experiment give different answers at different sites.
    """
    if budget <= 0:
        return []
    rng = random.Random(seed)
    scored = [(score(r, strategy, rng), str(r.get("uid", "")), r) for r in rows]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [r for _, _, r in scored[:budget]]


def class_composition(rows: Sequence[Dict[str, str]]) -> Dict[int, int]:
    """Ground-truth class counts of a selection.

    Required for reporting: a selection that beats random while quietly
    containing three times the malignant cases has not demonstrated that
    uncertainty sampling works.
    """
    counts: Dict[int, int] = {}
    for r in rows:
        gt = r.get("ground_truth")
        if gt is None or gt == "":
            continue
        k = int(float(gt))
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))
