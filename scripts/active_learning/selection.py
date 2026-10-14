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

One implementation
------------------
The rules live in ``application/jobs/_shared/custom/al_selection.py``, the copy
that travels with swarm jobs and selects on site (al_subset.py). This module
adapts it to the offline study's CSV rows (dicts with ``uid``,
``ground_truth`` and ``prob_class_<k>``), so "entropy" and "random" mean the
same thing offline and in the swarm.

Strategies
----------
``entropy``  predictive entropy over the class distribution. Highest when the
             model spreads mass evenly; picks cases it cannot commit on.
``margin``   top-1 probability minus top-2. Small margin means the model is
             torn between two specific classes -- often more useful than
             entropy when one class is rare, because entropy can be dominated
             by a long flat tail over classes the model never predicts.
``random``   the control. An active-learning result is only meaningful against
             this, at identical budget and seed. The pool is sorted by uid and
             then shuffled with the seed, so the same seed picks the same cases
             whatever order the rows arrive in (the earlier rule drew one random
             number per row in input order: reversing the rows changed 16 of 20
             picks).
``class_matched_random`` (``select`` only) random within class, with the class
             counts entropy picks at the same budget: the composition control.

A caution that shapes the analysis
----------------------------------
Uncertainty sampling naturally over-selects rare classes, and ODELIA's benign
class is rare at every site. A gain over random must therefore be shown to
survive when the class composition of the selected set is controlled for --
otherwise the experiment measures class imbalance and calls it active learning.
``class_composition`` is provided so that check is cheap and hard to skip.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Sequence

_CORE_PATH = (
    Path(__file__).resolve().parents[2] / "application" / "jobs" / "_shared" / "custom" / "al_selection.py"
)


def _load_core():
    """Import al_selection.py by path, so the job's custom/ directory never goes on sys.path here."""
    name = "odelia_al_selection_core"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(_CORE_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


core = _load_core()

# The offline study's table; class_matched_random is available through select() but needs many seeds,
# like random, which run_experiment.py gives only to "random".
STRATEGIES = ("entropy", "margin", "random")

entropy = core.entropy
margin_uncertainty = core.margin_uncertainty


def _probs(row: Dict[str, str]) -> List[float]:
    keys = sorted((k for k in row if k.startswith("prob_class_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    return [float(row[k]) for k in keys]


def _label(row: Dict[str, str]):
    gt = row.get("ground_truth")
    if gt is None or gt == "":
        return None
    return int(float(gt))


def select(rows: Sequence[Dict[str, str]], budget: int, strategy: str,
           seed: int = 0) -> List[Dict[str, str]]:
    """Pick ``budget`` rows under ``strategy``, in rank order (see al_selection.select_uids).

    Rows are keyed by ``uid``; a missing or repeated uid raises, because it would make the
    selection depend on row order.
    """
    by_uid = {}
    core_rows = []
    for row in rows:
        uid = str(row.get("uid", ""))
        by_uid.setdefault(uid, row)
        core_rows.append((uid, _probs(row), _label(row)))
    return [by_uid[uid] for uid in core.select_uids(core_rows, budget, strategy, seed)]


def class_composition(rows: Sequence[Dict[str, str]]) -> Dict[int, int]:
    """Ground-truth class counts of a selection.

    Required for reporting: a selection that beats random while quietly
    containing three times the malignant cases has not demonstrated that
    uncertainty sampling works.
    """
    return core.class_counts(_label(r) for r in rows)
