#!/usr/bin/env python3
"""Active-learning acquisition experiment (D3.2 / T3.2, #527).

What this measures, and what it does not
----------------------------------------
D3.2 asks whether active learning is *useful*. The complete answer is a
labelling-budget curve of **model performance**: retrain after each acquisition
round and show that uncertainty sampling reaches a given AUROC with fewer labels
than random. That requires a training run per round per arm. At the measured
2.2 h per round on the eight-site consortium, five iterations of five rounds
across two arms is ~110 h of consortium time, which does not fit before M48 and
would consume partner GPU for a question that can be partly answered first.

This experiment answers the part that needs no retraining: **acquisition
quality**. Given a fixed model's per-case probabilities, does an uncertainty
rule surface the cases worth labelling sooner than random?

Reported per budget:

* ``malignant_yield``  malignant cases in the selected set. The clinically
  decisive class, and the one a labelling budget is really being spent on.
* ``benign_yield``     benign cases. Rare everywhere (17 of 165 pooled), and the
  class whose scarcity has already produced a wrong conclusion about a centre.
* ``mean_entropy``     how uncertain the selected cases actually were.
* ``composition``      class counts of the selected set.

**What it cannot tell you:** whether a model trained on those cases is better.
An acquisition rule that surfaces rare cases faster is necessary for active
learning to help, not sufficient. Do not report this as "active learning works".

Why the composition column is not optional
------------------------------------------
Uncertainty sampling over-selects rare classes almost by construction, and this
dataset is 5.9 % malignant and 10.3 % benign. A rare-class yield above random is
therefore the *expected* behaviour of the rule, not evidence that the rule is
informative. The comparison that means something is yield at matched
composition, which is why composition is emitted alongside every point.

Usage
    python3 scripts/active_learning/run_experiment.py \
        --predictions workspace/eval_87c5bbee/results/fixed \
        --out workspace/active_learning
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import STRATEGIES, class_composition, entropy, select  # noqa: E402


def load_rows(base):
    """Per-case rows from every site under `base`, tagged with their site."""
    rows = []
    for site_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(site_dir):
            continue
        site = os.path.basename(site_dir)
        for f in sorted(glob.glob(os.path.join(site_dir, "predictions_*.csv"))):
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    r["_site"] = site
                    rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# E2 guard. The evaluation tree contains both a correct run and a
# wrong-architecture one (results/fixed vs results/best); "best" names the
# CHECKPOINT KIND, not the better result. This experiment was first run against
# the wrong one and produced a false null result. Refuse rather than repeat it.
# ---------------------------------------------------------------------------
def guard_wrong_architecture(rows, source):
    def _auroc(s, l):
        pos = sum(l); neg = len(l) - pos
        if not pos or not neg:
            return None
        o = sorted(range(len(s)), key=lambda i: s[i]); r = [0.0] * len(s); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and s[o[j + 1]] == s[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return (sum(x for x, y in zip(r, l) if y == 1) - pos * (pos + 1) / 2.0) / (pos * neg)

    gt = [int(float(r["ground_truth"])) for r in rows]
    p2 = [float(r["prob_class_2"]) for r in rows]
    a = _auroc(p2, [1 if g == 2 else 0 for g in gt])
    if a is not None and a < 0.60:
        raise SystemExit(
            f"\nREFUSING TO ANALYSE {source}: malignant AUROC {a:.3f} is at or below chance.\n"
            f"That is the signature of a checkpoint evaluated with the wrong architecture (E2).\n"
            f"Compare prediction_results.json 'model_name' against the checkpoint's train_conf.\n"
            f"The corrected ODELIA evaluation is workspace/eval_87c5bbee/results/fixed.\n")


def probs(row):
    return [float(row[f"prob_class_{c}"]) for c in (0, 1, 2)]


def _verdict(by, budgets, calib, total):
    """Derive the conclusion FROM THE NUMBERS.

    This used to be a hardcoded string. When the experiment was re-run against
    corrected predictions the numbers reversed and the string did not, so the
    summary asserted a null result on top of data showing the opposite. A
    conclusion that cannot change when the data changes is not a conclusion.
    """
    beats, loses = [], []
    for b in budgets:
        e = by[("entropy", b)]["malignant_yield"]
        r = by[("random", b)]
        if e > r["malignant_yield"] + r["malignant_sd"]:
            beats.append(b)
        elif e < r["malignant_yield"] - r["malignant_sd"]:
            loses.append(b)
    ent = calib["mean_entropy_by_class"]
    spread = max(ent.values()) - min(ent.values()) if ent else 0.0

    if beats and not loses:
        head = (f"Uncertainty sampling BEAT random at {len(beats)} of {len(budgets)} budgets "
                f"({', '.join(map(str, beats))}) and lost at none.")
    elif beats:
        head = (f"Uncertainty sampling beat random at {len(beats)} budgets "
                f"({', '.join(map(str, beats))}) and lost at {len(loses)} "
                f"({', '.join(map(str, loses))}).")
    elif loses:
        head = (f"Uncertainty sampling did NOT beat random -- it lost at "
                f"{len(loses)} of {len(budgets)} budgets.")
    else:
        head = "Uncertainty sampling was indistinguishable from random at every budget."

    tail = (f" Mean predictive entropy by true class {ent} (spread {spread:.3f}). "
            f"Pool composition {dict(total)}.")
    if beats:
        tail += (" Note the ceiling: at the largest budgets the selection covers most of "
                 "the pool, so every method converges on the same cases and the "
                 "advantage necessarily disappears. The informative region is the "
                 "small-budget end, which is also where a labelling budget actually "
                 "binds.")
    return head + tail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="workspace/eval_87c5bbee/results/fixed")
    ap.add_argument("--out", default="workspace/active_learning")
    ap.add_argument("--seeds", type=int, default=200,
                    help="repeats; random needs many to give a stable control")
    ap.add_argument("--budgets", default="10,20,30,40,50,60,80,100")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_rows(args.predictions)
    if not rows:
        raise SystemExit(f"no predictions under {args.predictions}")
    guard_wrong_architecture(rows, args.predictions)
    budgets = [int(b) for b in args.budgets.split(",") if int(b) <= len(rows)]

    total = class_composition(rows)
    n_mal, n_ben = total.get(2, 0), total.get(1, 0)
    print(f"  pool: {len(rows)} cases  ({total.get(0,0)} no-lesion, {n_ben} benign, {n_mal} malignant)")

    out = []
    for strategy in STRATEGIES:
        # A deterministic rule gives the same set every seed; only `random`
        # needs repeats. Running one seed for the others is not a shortcut,
        # it is what the rule is.
        seeds = args.seeds if strategy == "random" else 1
        for budget in budgets:
            mal, ben, ents = [], [], []
            for s in range(seeds):
                picked = select(rows, budget, strategy, seed=s)
                comp = class_composition(picked)
                mal.append(comp.get(2, 0))
                ben.append(comp.get(1, 0))
                ents.append(st.mean(entropy(probs(r)) for r in picked))
            out.append({
                "strategy": strategy, "budget": budget, "seeds": seeds,
                "malignant_yield": round(st.mean(mal), 2),
                "malignant_sd": round(st.pstdev(mal), 2) if seeds > 1 else 0.0,
                "benign_yield": round(st.mean(ben), 2),
                "mean_entropy": round(st.mean(ents), 4),
                "malignant_recall": round(st.mean(mal) / n_mal, 3) if n_mal else "",
                "frac_of_pool": round(budget / len(rows), 3),
            })

    csv_path = os.path.join(args.out, "acquisition_curve.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    by = {(r["strategy"], r["budget"]): r for r in out}
    print(f"\n  {'budget':>7} {'random':>18} {'entropy':>18} {'margin':>18}   (malignant found)")
    for b in budgets:
        cells = []
        for s in ("random", "entropy", "margin"):
            r = by[(s, b)]
            cells.append(f"{r['malignant_yield']:>6.1f}" + (f" ±{r['malignant_sd']:<4.1f}" if s == "random" else "      "))
        print(f"  {b:>7} " + " ".join(f"{c:>17}" for c in cells))

    # Diagnostic: is the model's uncertainty informative about the true class at
    # all? If entropy is flat across classes, an uncertainty rule cannot prefer
    # the rare one, and a null result below is a property of the model rather
    # than of active learning.
    by_class = {0: [], 1: [], 2: []}
    for r in rows:
        by_class[int(float(r["ground_truth"]))].append(entropy(probs(r)))
    ranked = sorted(((entropy(probs(r)), int(float(r["ground_truth"]))) for r in rows),
                    reverse=True)
    k = min(30, len(ranked))
    calib = {
        "mean_entropy_by_class": {str(c): round(st.mean(v), 3) for c, v in by_class.items() if v},
        f"top{k}_entropy_composition": {str(c): sum(1 for _, g in ranked[:k] if g == c) for c in (0, 1, 2)},
        f"bottom{k}_entropy_composition": {str(c): sum(1 for _, g in ranked[-k:] if g == c) for c in (0, 1, 2)},
    }
    print("\n  calibration diagnostic")
    print(f"    mean entropy by true class: {calib['mean_entropy_by_class']}")
    print(f"    highest-entropy {k}: {calib[f'top{k}_entropy_composition']}  (class 0/1/2)")
    print(f"    lowest-entropy  {k}: {calib[f'bottom{k}_entropy_composition']}")

    summary = {
        "calibration": calib,
        "pool_size": len(rows), "pool_composition": {str(k): v for k, v in total.items()},
        "budgets": budgets, "seeds_random": args.seeds,
        "source": args.predictions,
        "caveat": ("Measures acquisition quality only. A model trained on these "
                   "selections has not been evaluated; rare-class yield above "
                   "random is expected from uncertainty sampling on an imbalanced "
                   "pool and is not on its own evidence that active learning helps."),
        "result": _verdict(by, budgets, calib, total),
    }
    json.dump(summary, open(os.path.join(args.out, "experiment_summary.json"), "w"), indent=1)
    print(f"\n  wrote {csv_path}")


if __name__ == "__main__":
    main()
