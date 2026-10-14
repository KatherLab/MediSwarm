"""Budgeted case selection for active learning (D3.2, #527, #592): pure functions over scored rows.

Why this module exists
----------------------
D3.2 compares fine-tuning on k cases chosen by model uncertainty against k cases chosen at random. The
same rules run in two places: inside the swarm, where every site selects from its own pool with the
received global (al_subset.py), and offline at TUD on predict.py outputs
(scripts/active_learning/run_experiment.py, through scripts/active_learning/selection.py). One
implementation here keeps "entropy" and "random" meaning the same in both, and it travels with the job
(job code cannot import scripts/).

A row is ``(uid, probs, label)``: ``probs`` are class probabilities (non-negative numbers, normalised
here), ``label`` is the ground-truth class index or None. No torch and no numpy, so this module can be
tested and imported anywhere.

Strategies
----------
``entropy``               highest predictive entropy first.
``margin``                smallest top-1 minus top-2 probability first (the offline study reports it).
``random``                the control arm: seeded, independent of the scores.
``class_matched_random``  random within class, with exactly the class counts the entropy rule picks at the
                          same budget. Uncertainty sampling over-selects the rare classes by construction,
                          so a gain of entropy over plain random can be class composition alone; this arm
                          holds the composition fixed and changes only which cases of each class are used.

Reproducibility rules
---------------------
* Nothing depends on the order rows arrive in (each site's split.csv order, loader order). Uncertainty
  ties are broken by UID. The random arms sort the pool by UID and then shuffle it with
  ``random.Random(seed)``: the same pool and seed give the same cases in every run, at every site and in
  the offline study. (The previous offline rule drew one random number per row in input order, so
  reversing the rows changed 16 of 20 picks.)
* Budgets nest: on the same pool (and seed) the selection at budget b is a subset of the selection at any
  larger budget, for every rule, so a curve over budgets adds cases instead of redrawing them.
* UIDs must be unique within the pool. A duplicate is a data problem (the same case twice, or a bad input
  replaced by its neighbour) and raises instead of being resolved silently.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

STRATEGY_ENTROPY = "entropy"
STRATEGY_MARGIN = "margin"
STRATEGY_RANDOM = "random"
STRATEGY_CLASS_MATCHED_RANDOM = "class_matched_random"
# Not a rule over scores: the cases come from a UID list prepared elsewhere (ODELIA_TRAIN_INCLUDE_LIST).
STRATEGY_INCLUDE_LIST = "include_list"

STRATEGIES = (STRATEGY_ENTROPY, STRATEGY_MARGIN, STRATEGY_RANDOM, STRATEGY_CLASS_MATCHED_RANDOM)
# Rules whose result depends on the class probabilities (and therefore needs a scoring pass).
SCORED_STRATEGIES = (STRATEGY_ENTROPY, STRATEGY_MARGIN, STRATEGY_CLASS_MATCHED_RANDOM)

# al/selection_digest is streamed as a number. 12 hex digits = 48 bits, exactly representable as a float64
# (the per-site metric transport may turn it into one), and plenty to tell selections apart.
DIGEST_NUMBER_HEX_DIGITS = 12


class Row(NamedTuple):
    uid: str
    probs: Optional[Tuple[float, ...]]
    label: Optional[int]


@dataclass
class Selection:
    """Result of one selection: UIDs in rank order plus the statistics that may be reported.

    ``uids`` and the per-row data are site-local; everything in ``stats`` is an aggregate.
    """

    strategy: str
    budget: int
    seed: int
    uids: List[str]
    stats: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------
# Uncertainty measures (higher = less certain, for both)
# ---------------------------------------------------------------------------------------------------


def normalise(probs: Sequence[float]) -> List[float]:
    """Return a proper distribution, or a uniform one if the input is degenerate.

    Rows come from six different model implementations and from CSVs; a row that does not sum to 1 is a
    data problem, not a reason to stop a study.
    """
    values = [max(0.0, float(p)) for p in probs]
    total = sum(values)
    if total <= 0 or not math.isfinite(total):
        n = len(values) or 1
        return [1.0 / n] * n
    return [v / total for v in values]


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in nats. Higher means less certain."""
    return -sum(p * math.log(p) for p in normalise(probs) if p > 0)


def margin_uncertainty(probs: Sequence[float]) -> float:
    """1 - (top1 - top2), so that larger means less certain (same direction as entropy)."""
    ordered = sorted(normalise(probs), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return 1.0 - (ordered[0] - ordered[1])


_MEASURES = {STRATEGY_ENTROPY: entropy, STRATEGY_MARGIN: margin_uncertainty}


# ---------------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------------


def as_rows(rows: Iterable) -> List[Row]:
    """Validate ``(uid, probs, label)`` rows and return them as Row tuples.

    UIDs must be non-empty strings and unique. ``probs`` may be None (rows that were not scored, e.g. an
    include list); ``label`` may be None (unlabelled).
    """
    result: List[Row] = []
    seen = set()
    duplicates = 0
    for raw in rows:
        uid, probs, label = raw
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"every row needs a non-empty string UID, got {uid!r}")
        if uid in seen:
            duplicates += 1
            continue
        seen.add(uid)
        probs_t = None if probs is None else tuple(float(p) for p in probs)
        label_i = None if label is None else int(label)
        result.append(Row(uid, probs_t, label_i))
    if duplicates:
        # The UID itself is deliberately not named: messages end up in logs that leave the site.
        raise ValueError(
            f"{duplicates} duplicate UID(s) in the pool; selection by UID would be ambiguous. "
            f"Check the split for repeated cases or substituted (unreadable) inputs."
        )
    return result


def _require_probs(rows: Sequence[Row], strategy: str) -> None:
    missing = sum(1 for r in rows if r.probs is None or len(r.probs) == 0)
    if missing:
        raise ValueError(f"strategy {strategy!r} needs class probabilities, but {missing} row(s) have none")


# ---------------------------------------------------------------------------------------------------
# Orders and selection
# ---------------------------------------------------------------------------------------------------


def uncertainty_order(rows: Sequence[Row], measure) -> List[str]:
    """UIDs by descending uncertainty; ties broken by UID so input order never matters."""
    return [r.uid for r in sorted(rows, key=lambda r: (-measure(r.probs), r.uid))]


def random_order(uids: Iterable[str], seed: int) -> List[str]:
    """A seeded permutation of ``uids`` that does not depend on their input order: sort, then shuffle."""
    ordered = sorted(uids)
    random.Random(int(seed)).shuffle(ordered)
    return ordered


def _stratum(label: Optional[int]):
    # Unlabelled rows form their own stratum; sort key keeps None after the integer classes.
    return (1, 0) if label is None else (0, label)


def select_uids(rows: Iterable, budget: int, strategy: str, seed: int = 0) -> List[str]:
    """Pick ``budget`` UIDs from ``rows`` under ``strategy``, in rank order.

    ``budget <= 0`` selects nothing; a budget above the pool size selects the whole pool.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    pool = as_rows(rows)
    budget = int(budget)
    if budget <= 0 or not pool:
        return []

    if strategy in _MEASURES:
        _require_probs(pool, strategy)
        return uncertainty_order(pool, _MEASURES[strategy])[:budget]

    order = random_order((r.uid for r in pool), seed)
    if strategy == STRATEGY_RANDOM:
        return order[:budget]

    # class_matched_random: the class counts of the entropy selection at this budget, filled per class with
    # the first cases of that class in the random arm's order (so it nests across budgets like the others).
    _require_probs(pool, strategy)
    label_of = {r.uid: r.label for r in pool}
    target: Dict = {}
    for uid in uncertainty_order(pool, entropy)[:budget]:
        key = _stratum(label_of[uid])
        target[key] = target.get(key, 0) + 1
    picked: List[str] = []
    for key in sorted(target):
        in_class = [uid for uid in order if _stratum(label_of[uid]) == key]
        picked.extend(in_class[: target[key]])
    return picked


# ---------------------------------------------------------------------------------------------------
# Statistics and digests (the only things about a selection that may leave a site)
# ---------------------------------------------------------------------------------------------------


def class_counts(labels: Iterable[Optional[int]], num_classes: Optional[int] = None) -> Dict[int, int]:
    """Counts per class index; unlabelled entries are skipped. With ``num_classes``, absent classes are 0."""
    counts: Dict[int, int] = {k: 0 for k in range(num_classes or 0)}
    for label in labels:
        if label is None:
            continue
        counts[int(label)] = counts.get(int(label), 0) + 1
    return dict(sorted(counts.items()))


def selection_digest(uids: Iterable[str]) -> str:
    """SHA-256 of the sorted UID list joined by newlines: identical for the same set of cases, in any order.

    Can be recomputed from a UID list at the site (the local selection record keeps one) to show which
    selection a streamed digest belongs to.
    """
    return hashlib.sha256("\n".join(sorted(uids)).encode("utf-8")).hexdigest()


def digest_as_number(hexdigest: str, hex_digits: int = DIGEST_NUMBER_HEX_DIGITS) -> int:
    """Leading ``hex_digits`` of a hex digest as an int, for metric channels that only carry numbers."""
    return int(hexdigest[:hex_digits], 16)


def selection_statistics(rows: Iterable, selected: Sequence[str], num_classes: Optional[int] = None) -> Dict:
    """Aggregate description of a selection from ``rows`` (the pool): sizes, class counts, mean entropy, digest.

    ``mean_entropy`` is over the selected rows that carry probabilities, None if none do.
    """
    pool = as_rows(rows)
    by_uid = {r.uid: r for r in pool}
    missing = [uid for uid in selected if uid not in by_uid]
    if missing:
        raise ValueError(f"{len(missing)} selected UID(s) are not in the pool")
    chosen = [by_uid[uid] for uid in selected]
    entropies = [entropy(r.probs) for r in chosen if r.probs]
    digest = selection_digest(selected)
    return {
        "pool_size": len(pool),
        "pool_class_counts": class_counts((r.label for r in pool), num_classes),
        "n_selected": len(chosen),
        "class_counts": class_counts((r.label for r in chosen), num_classes),
        "mean_entropy": (sum(entropies) / len(entropies)) if entropies else None,
        "digest": digest,
        "digest_number": digest_as_number(digest),
    }


def select(rows: Iterable, budget: int, strategy: str, seed: int = 0, num_classes: Optional[int] = None) -> Selection:
    """Select ``budget`` cases under ``strategy`` and describe the result (see select_uids and selection_statistics)."""
    pool = as_rows(rows)
    uids = select_uids(pool, budget, strategy, seed)
    return Selection(
        strategy=strategy,
        budget=int(budget),
        seed=int(seed),
        uids=uids,
        stats=selection_statistics(pool, uids, num_classes),
    )
