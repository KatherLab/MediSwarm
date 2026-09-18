#!/usr/bin/env python3
"""Compare a regionally fine-tuned model against the pan-European baseline (D2.5 / T2.4, #526).

Design notes, because the obvious analysis is the wrong one here
----------------------------------------------------------------
The regional test partitions carved out of the 8-site run are small, and their
class support is very uneven:

    Dutch (RUMC + UMCU)   n=49   no-lesion 38   benign  2   malignant 9
    Greek (MHA)           n=26   no-lesion 18   benign  1   malignant 7

A macro AUROC over three classes therefore averages in a benign one-vs-rest
figure computed from one or two positive cases. That is not a measurement, and
ranking by it is exactly what produced the wrong conclusion about UMCU --
recorded as E1 in ``docs/EVALUATION_PITFALLS.md``. So:

* the **primary endpoint is malignant-vs-rest AUROC**, the clinically decisive
  discrimination and the only one with usable support in both cohorts;
* ``lesion-vs-none`` is reported as a secondary;
* macro AUROC is reported only with per-class support attached, and never as the
  basis for a claim.

The comparison is **paired**. Both models score the *same* cases, so the
uncertainty that matters is on the difference, not on each arm separately.
Resampling cases (not sites) and recomputing the delta each draw gives a CI that
correctly cancels the shared case-difficulty variance; comparing two independent
per-arm CIs would be far more conservative and would hide a real effect at these
sample sizes. Cases are resampled jointly for both models, which is what makes it
paired.

Usage
-----
    compare_regional_finetune.py --baseline <dir> --finetuned <dir> \
        --sites RUMC_1 UMCU_1 --label "Dutch cohort" [--bootstrap 10000]

Each directory is a per-site tree of ``<SITE>/predictions_*.csv`` with columns
uid, ground_truth, prediction, prob_class_0..2 -- the layout
``scripts/evaluation/predict.py`` already writes.
"""

import argparse
import csv
import glob
import json
import os
import random
from collections import Counter

CLS = {0: "No lesion", 1: "Benign", 2: "Malignant"}


def auroc(scores, labels):
    """One-vs-rest AUROC via the rank statistic. None if either class is absent.

    Same implementation as workspace/eval_87c5bbee/analyze_best.py, kept
    dependency-free so this runs anywhere the predictions land.
    """
    pos = sum(1 for l in labels if l == 1)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):                        # average tied ranks
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rsum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rsum - pos * (pos + 1) / 2.0) / (pos * neg)


def load(base, sites):
    """Load per-case rows for the given sites, keyed by a stable case key.

    Two layouts are read, because the rows reach us two ways:

    * ``<site>/predictions_*.csv`` from ``scripts/evaluation/predict.py``, which
      has a ``uid`` column;
    * ``<site>.csv`` returned through the swarm by ``per_case_predictions``,
      which deliberately has **no** identifier and carries ``row_index``
      instead -- a position within that site's own validation split.

    Either way the key is namespaced by site, so two sites that both start
    their row indices at 0 cannot collide and silently pair a Radboud case
    with an Utrecht one.
    """
    rows = {}
    for site in sites:
        found = sorted(glob.glob(os.path.join(base, site, "predictions_*.csv")))
        found += sorted(glob.glob(os.path.join(base, f"{site}.csv")))
        for f in found:
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    key = r.get("uid") or r.get("row_index")
                    if key is None:
                        continue
                    r["_site"] = site
                    r["uid"] = f"{site}:{key}"
                    rows[r["uid"]] = r
    return rows


def _scores(rows, uids, kind):
    if kind == "malignant_vs_rest":
        return [float(rows[u]["prob_class_2"]) for u in uids]
    if kind == "lesion_vs_none":
        return [float(rows[u]["prob_class_1"]) + float(rows[u]["prob_class_2"]) for u in uids]
    raise ValueError(kind)


def _labels(rows, uids, kind):
    gt = [int(float(rows[u]["ground_truth"])) for u in uids]
    if kind == "malignant_vs_rest":
        return [1 if g == 2 else 0 for g in gt]
    return [1 if g in (1, 2) else 0 for g in gt]


def paired_bootstrap(base_rows, fine_rows, uids, kind, draws, seed=0):
    """Bootstrap the *difference* in AUROC over jointly resampled cases.

    Returns (delta_point, lo, hi, frac_favouring_finetuned). Draws in which
    either arm is undefined (a resample with no positives or no negatives) are
    skipped rather than counted as zero -- counting them would drag the interval
    toward no-effect for reasons that have nothing to do with the models.
    """
    rng = random.Random(seed)
    b_lab = _labels(base_rows, uids, kind)
    a0 = auroc(_scores(base_rows, uids, kind), b_lab)
    a1 = auroc(_scores(fine_rows, uids, kind), _labels(fine_rows, uids, kind))
    if a0 is None or a1 is None:
        return None, None, None, None
    point = a1 - a0

    deltas = []
    n = len(uids)
    for _ in range(draws):
        pick = [uids[rng.randrange(n)] for _ in range(n)]
        d0 = auroc(_scores(base_rows, pick, kind), _labels(base_rows, pick, kind))
        d1 = auroc(_scores(fine_rows, pick, kind), _labels(fine_rows, pick, kind))
        if d0 is None or d1 is None:
            continue
        deltas.append(d1 - d0)
    if not deltas:
        return point, None, None, None
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    favour = sum(1 for d in deltas if d > 0) / len(deltas)
    return point, lo, hi, favour


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="per-site predictions dir, pan-European model")
    ap.add_argument("--finetuned", required=True, help="per-site predictions dir, fine-tuned model")
    ap.add_argument("--sites", required=True, nargs="+", help="sites forming the regional cohort")
    ap.add_argument("--label", default="cohort", help="cohort name for the report")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    base_rows = load(args.baseline, args.sites)
    fine_rows = load(args.finetuned, args.sites)
    uids = sorted(set(base_rows) & set(fine_rows))
    if not uids:
        raise SystemExit("no overlapping cases between the two prediction sets")

    # A silently reduced overlap would compare the models on different cases.
    missing = (set(base_rows) | set(fine_rows)) - set(uids)
    gt = [int(float(base_rows[u]["ground_truth"])) for u in uids]
    support = {CLS[k]: v for k, v in sorted(Counter(gt).items())}

    print(f"\n{args.label} — {', '.join(args.sites)}")
    print(f"  paired cases: {len(uids)}"
          + (f"   (WARNING: {len(missing)} case(s) present in only one set, excluded)" if missing else ""))
    print(f"  support: " + ", ".join(f"{k} {v}" for k, v in support.items()))

    report = {"label": args.label, "sites": args.sites, "n_paired": len(uids),
              "n_excluded": len(missing), "support": support, "endpoints": {}}

    for kind, role in (("malignant_vs_rest", "PRIMARY"), ("lesion_vs_none", "secondary")):
        b = auroc(_scores(base_rows, uids, kind), _labels(base_rows, uids, kind))
        f = auroc(_scores(fine_rows, uids, kind), _labels(fine_rows, uids, kind))
        d, lo, hi, favour = paired_bootstrap(base_rows, fine_rows, uids, kind,
                                             args.bootstrap, args.seed)
        npos = sum(_labels(base_rows, uids, kind))
        print(f"\n  [{role}] {kind}   (positives {npos} / negatives {len(uids)-npos})")
        if b is None or f is None:
            print("    undefined: a class is absent from this cohort")
            report["endpoints"][kind] = {"undefined": True}
            continue
        print(f"    pan-European : {b:.3f}")
        print(f"    fine-tuned   : {f:.3f}")
        if lo is None:
            print(f"    delta        : {d:+.3f}  (CI unavailable)")
        else:
            print(f"    delta        : {d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]"
                  f"   favours fine-tuned in {favour:.0%} of draws")
            if lo < 0 < hi:
                print("    -> interval spans zero: no difference detectable at this sample size.")
        report["endpoints"][kind] = {"baseline": b, "finetuned": f, "delta": d,
                                     "ci_low": lo, "ci_high": hi, "favour_finetuned": favour,
                                     "n_pos": npos, "n_neg": len(uids) - npos}

    print("\n  Macro AUROC is deliberately not reported as a ranking metric for these")
    print("  cohorts: benign support is 1-2 cases. See docs/EVALUATION_PITFALLS.md (E1).\n")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  wrote {args.json_out}\n")


if __name__ == "__main__":
    main()
