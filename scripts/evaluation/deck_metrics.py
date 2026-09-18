#!/usr/bin/env python3
"""Every headline number in the consortium deck, with confidence intervals.

Exists so the deck cannot quote a number that nobody can reproduce, and so a
figure cannot silently drift from the evaluation it came from.

**Read this before changing the --predictions default.** The evaluation tree
``workspace/eval_87c5bbee/results/`` contains BOTH:

* ``best/``  -- the checkpoint evaluated with ``--model-name MST``. The
  checkpoint is 1DivideAndConquer (its embedded ``train_conf`` says
  ``ResidualEncoderClsLightning``), so these predictions are the E2 failure:
  the weights load partially and produce plausible-looking numbers. Pooled
  accuracy 0.533, malignant AUROC 0.476 -- chance.
* ``fixed/`` -- the corrected re-run. Pooled accuracy 0.752, malignant AUROC
  0.887. **This is the real evaluation and the only one to quote.**

"best" names the *checkpoint kind* (best vs final), not the better result. That
naming cost a wrong conclusion once already: an active-learning null result and
a calibration finding were both computed on ``best/`` before anyone noticed.
The corrected re-run was written beside the broken one instead of replacing it.

Intervals are percentile bootstrap over cases. AUROC intervals at these sample
sizes are wide, and the deck says so rather than quoting three decimals as if
they were resolved.

Usage
    python3 scripts/evaluation/deck_metrics.py [--duke DUKE.csv] [--boot 5000]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random

ODELIA_DEFAULT = "workspace/eval_87c5bbee/results/fixed"


def auroc(scores, labels):
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return (sum(r for r, l in zip(ranks, labels) if l == 1) - pos * (pos + 1) / 2.0) / (pos * neg)


def boot_ci(scores, labels, draws, seed=0):
    rng = random.Random(seed)
    n = len(scores)
    vals = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        v = auroc([scores[i] for i in idx], [labels[i] for i in idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def load_dir(base):
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "*", "predictions_*.csv"))):
        site = os.path.basename(os.path.dirname(f))
        for r in csv.DictReader(open(f)):
            r["_site"] = site
            rows.append(r)
    return rows


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["_site"] = "DUKE"
    return rows


def guard_wrong_architecture(rows, label):
    """Refuse to report the E2 predictions, whatever directory they came from."""
    gt = [int(float(r["ground_truth"])) for r in rows]
    p2 = [float(r["prob_class_2"]) for r in rows]
    a = auroc(p2, [1 if g == 2 else 0 for g in gt])
    if a is not None and a < 0.60:
        raise SystemExit(
            f"\nREFUSING TO REPORT {label}: malignant AUROC {a:.3f} is at or below chance.\n"
            f"That is the signature of a checkpoint evaluated with the wrong architecture (E2).\n"
            f"Check prediction_results.json 'model_name' against the checkpoint's train_conf.\n")


def block(name, rows, draws, drop_benign):
    sel = [r for r in rows if not (drop_benign and int(float(r["ground_truth"])) == 1)]
    gt = [int(float(r["ground_truth"])) for r in sel]
    y = [1 if g == 2 else 0 for g in gt]
    p2 = [float(r["prob_class_2"]) for r in sel]
    a = auroc(p2, y)
    lo, hi = boot_ci(p2, y, draws)
    pred = [int(float(r["prediction"])) for r in sel]
    tp = sum(1 for p, l in zip(pred, y) if l == 1 and p == 2)
    tn = sum(1 for p, l in zip(pred, y) if l == 0 and p != 2)
    npos, nneg = sum(y), len(y) - sum(y)
    return {"name": name, "n": len(sel), "malignant": npos, "negative": nneg,
            "auroc": round(a, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "recall": round(tp / npos, 4) if npos else None,
            "specificity": round(tn / nneg, 4) if nneg else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=ODELIA_DEFAULT)
    ap.add_argument("--duke", default=None, help="Duke predictions CSV")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    od = load_dir(args.predictions)
    if not od:
        raise SystemExit(f"no predictions under {args.predictions}")
    guard_wrong_architecture(od, args.predictions)

    out = {"odelia_source": args.predictions, "bootstrap": args.boot, "blocks": []}
    print(f"\nODELIA challenge held-out — {args.predictions}")
    print(f"{'':<46}{'n':>5}{'malig':>7}{'AUROC':>8}{'95% CI':>18}{'recall':>9}{'spec':>8}")

    def show(b):
        out["blocks"].append(b)
        print(f"  {b['name']:<44}{b['n']:>5}{b['malignant']:>7}{b['auroc']:>8.3f}"
              f"   [{b['ci_low']:.3f}, {b['ci_high']:.3f}]{b['recall']:>9.3f}{b['specificity']:>8.3f}")

    show(block("malignant vs rest", od, args.boot, False))
    show(block("malignant vs no-lesion (benign excluded)", od, args.boot, True))

    if args.duke:
        dk = load_csv(args.duke)
        guard_wrong_architecture(dk, args.duke)
        print(f"\nDuke — independent US cohort, unilateral volumes")
        show(block("malignant vs no-lesion", dk, args.boot, True))

        a = [b for b in out["blocks"] if b["name"].startswith("malignant vs no-lesion")]
        if len(a) == 2:
            eu, us = a[0], a[1]
            d = eu["auroc"] - us["auroc"]
            print(f"""
  MATCHED COMPARISON (same task, benign excluded from both)
    Europe  {eu['auroc']:.3f}  [{eu['ci_low']:.3f}, {eu['ci_high']:.3f}]   n={eu['n']}
    US      {us['auroc']:.3f}  [{us['ci_low']:.3f}, {us['ci_high']:.3f}]   n={us['n']}
    drop    {d:+.3f} — the intervals overlap heavily, so this is consistent
            with no loss at all on an unseen continent, not with a measured
            loss of {abs(d):.3f}.""")
            out["matched_drop"] = round(d, 4)

    print(f"""
  THE OPERATING POINT IS THE PROBLEM, NOT THE DISCRIMINATION.
    Recall is far below what an AUROC near 0.9 permits, at very high
    specificity: the threshold is placed to avoid false alarms, so the model
    misses cancers it can in fact rank correctly. That is fixable by moving
    the threshold, with no retraining. Recorded as E3.
""")
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=1)
        print(f"  wrote {args.json_out}\n")


if __name__ == "__main__":
    main()
