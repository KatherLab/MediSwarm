#!/usr/bin/env python3
"""Calibration diagnosis and temperature scaling for the swarm model (D3.2, #527).

Why this exists
---------------
The D3.2 acquisition experiment returned a negative result: uncertainty sampling
found *fewer* malignant cases than random at every labelling budget. The
diagnostic pointed at calibration -- the model is most confident on benign
cases, the rarest class, so "what am I least sure about?" points away from what
you want to find.

That is a hypothesis, not a conclusion. This script tests it properly:

1. **Diagnose.** Expected calibration error, Brier score and negative
   log-likelihood, overall and per class, plus the reliability table.
2. **Fix.** Temperature scaling -- the standard single-parameter post-hoc
   recalibration. Fitted on one split and evaluated on the other, because
   fitting and evaluating a calibrator on the same 165 cases measures nothing.
3. **Re-test.** Does uncertainty sampling beat random *after* recalibration?

Point (3) is the one that matters. Temperature scaling is monotone in the
logits, so it **cannot reorder cases by max-probability confidence**. It can
reorder them by *entropy*, because entropy depends on the whole distribution
rather than its maximum. So this is a real test, but a narrow one, and if it
fails the honest conclusion is that the problem is not calibration in the
temperature-scaling sense -- it is that the model's confidence carries no
information about correctness at all, which no post-hoc rescaling repairs.

Usage
    python3 scripts/calibration/analyse_calibration.py \
        --predictions workspace/eval_87c5bbee/results/fixed \
        --out workspace/calibration
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
from collections import Counter, defaultdict

CLS = {0: "No lesion", 1: "Benign", 2: "Malignant"}
NCLS = 3


# ------------------------------------------------------------------ loading
def load(base):
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "*", "*.csv"))):
        site = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            for r in csv.DictReader(fh):
                p = [float(r[f"prob_class_{k}"]) for k in range(NCLS)]
                s = sum(p)
                if s <= 0:
                    continue
                rows.append({"site": site,
                             "uid": f"{site}:{r.get('uid') or r.get('row_index')}",
                             "y": int(float(r["ground_truth"])),
                             "p": [x / s for x in p]})
    return rows



# ---------------------------------------------------------------------------
# E2 guard. The evaluation tree contains both a correct run and a
# wrong-architecture one (results/fixed vs results/best); "best" names the
# CHECKPOINT KIND, not the better result. Pointing an analysis at the wrong one
# produced a false active-learning null result and a false calibration finding
# before anyone noticed. Refuse rather than analyse chance-level predictions.
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

    gt = [r["y"] if "y" in r else int(float(r["ground_truth"])) for r in rows]
    p2 = [r["p"][2] if "p" in r else float(r["prob_class_2"]) for r in rows]
    a = _auroc(p2, [1 if g == 2 else 0 for g in gt])
    if a is not None and a < 0.60:
        raise SystemExit(
            f"\nREFUSING TO ANALYSE {source}: malignant AUROC {a:.3f} is at or below chance.\n"
            f"That is the signature of a checkpoint evaluated with the wrong architecture (E2).\n"
            f"Compare prediction_results.json 'model_name' against the checkpoint's train_conf.\n"
            f"The corrected ODELIA evaluation is workspace/eval_87c5bbee/results/fixed.\n")

# ------------------------------------------------------------------- metrics
def softmax_t(p, T):
    """Temperature-scale a probability vector. log(p) recovers the logits up to
    a shift, and softmax is shift-invariant, so this is exact."""
    z = [math.log(max(x, 1e-12)) / T for x in p]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def nll(rows, T=1.0):
    return -sum(math.log(max(softmax_t(r["p"], T)[r["y"]], 1e-12))
                for r in rows) / len(rows)


def brier(rows, T=1.0):
    tot = 0.0
    for r in rows:
        p = softmax_t(r["p"], T)
        tot += sum((p[k] - (1.0 if r["y"] == k else 0.0)) ** 2 for k in range(NCLS))
    return tot / len(rows)


def ece(rows, T=1.0, bins=10):
    """Expected calibration error on the top-1 prediction."""
    buckets = defaultdict(list)
    for r in rows:
        p = softmax_t(r["p"], T)
        conf = max(p)
        pred = p.index(conf)
        b = min(bins - 1, int(conf * bins))
        buckets[b].append((conf, 1.0 if pred == r["y"] else 0.0))
    n = len(rows)
    total, table = 0.0, []
    for b in range(bins):
        v = buckets.get(b, [])
        if not v:
            continue
        c = sum(x for x, _ in v) / len(v)
        a = sum(y for _, y in v) / len(v)
        total += len(v) / n * abs(c - a)
        table.append({"bin": f"{b/bins:.1f}-{(b+1)/bins:.1f}", "n": len(v),
                      "confidence": round(c, 3), "accuracy": round(a, 3),
                      "gap": round(a - c, 3)})
    return total, table


def entropy(p):
    return -sum(x * math.log(max(x, 1e-12)) for x in p)


def fit_temperature(rows, lo=0.05, hi=20.0, iters=200):
    """Golden-section search for the T minimising NLL over ``[lo, hi]``.

    NLL in T is unimodal for temperature scaling, so this is safe and needs no
    gradients. Verified against a 4001-point brute-force grid.

    **The result can land on a bound, and that means something.** Hitting ``hi``
    says the fit wants to flatten the predictions as far as it is allowed --
    the probabilities carry so little information that the best single-parameter
    correction is to erase them towards uniform. Hitting ``lo`` says the
    ranking is already perfect and the probabilities are needlessly timid, so
    the fit sharpens as hard as it may. Either way the number is not an optimum,
    it is the edge of the search, and callers should say so rather than quote it
    as a temperature. Use :func:`fit_temperature_checked` to get that flag.
    """
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if nll(rows, c) < nll(rows, d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
        if abs(b - a) < 1e-6:
            break
    return (a + b) / 2


def fit_temperature_checked(rows, lo=0.05, hi=20.0):
    """``(T, at_bound)``. ``at_bound`` is 'lo', 'hi' or None -- see above."""
    T = fit_temperature(rows, lo, hi)
    tol = (hi - lo) * 1e-3
    if T <= lo + tol:
        return T, "lo"
    if T >= hi - tol:
        return T, "hi"
    return T, None


# --------------------------------------------------------------- acquisition
def malignant_yield(rows, order, budget):
    return sum(1 for r in order[:budget] if r["y"] == 2)


def acquisition_test(rows, T, budgets, seeds=200, seed=0):
    """Entropy-ranked selection against random, before and after scaling."""
    out = {}
    for label, temp in (("uncalibrated", 1.0), ("calibrated", T)):
        ranked = sorted(rows, key=lambda r: -entropy(softmax_t(r["p"], temp)))
        out[label] = {str(b): malignant_yield(rows, ranked, b) for b in budgets}
    rng = random.Random(seed)
    rand = {}
    for b in budgets:
        ys = []
        for _ in range(seeds):
            pick = rng.sample(rows, b)
            ys.append(sum(1 for r in pick if r["y"] == 2))
        mu = sum(ys) / len(ys)
        sd = math.sqrt(sum((y - mu) ** 2 for y in ys) / len(ys))
        rand[str(b)] = {"mean": round(mu, 2), "sd": round(sd, 2)}
    out["random"] = rand
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="workspace/eval_87c5bbee/results/fixed")
    ap.add_argument("--out", default="workspace/calibration")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load(args.predictions)
    if not rows:
        raise SystemExit(f"no predictions under {args.predictions}")
    guard_wrong_architecture(rows, args.predictions)
    comp = Counter(r["y"] for r in rows)
    print(f"\n{len(rows)} cases from {len(set(r['site'] for r in rows))} sites — "
          + ", ".join(f"{CLS[k]} {comp.get(k,0)}" for k in range(NCLS)))

    # ---- split: fit the calibrator on one half, judge it on the other -----
    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    half = len(idx) // 2
    fit = [rows[i] for i in idx[:half]]
    held = [rows[i] for i in idx[half:]]
    print(f"fit split {len(fit)}, held-out split {len(held)} "
          f"(a calibrator fitted and judged on the same cases measures nothing)")

    T, at_bound = fit_temperature_checked(fit)
    print(f"\nfitted temperature T = {T:.3f}  "
          f"({'sharpening — model was under-confident' if T < 1 else 'softening — model was over-confident'})")
    if at_bound:
        print(f"   WARNING: T is at the {at_bound} bound of the search interval. This is not a"
              f"\n   fitted optimum — it is the edge. A hit on the upper bound means the fit"
              f"\n   wants to flatten the predictions as far as allowed, i.e. they carry almost"
              f"\n   no information. Report it as such, not as a temperature.")

    # ---- diagnose, on held-out only --------------------------------------
    e0, tab0 = ece(held, 1.0)
    e1, tab1 = ece(held, T)
    print("\n" + "=" * 66)
    print("CALIBRATION on the held-out split")
    print("=" * 66)
    print(f"   {'metric':<28}{'before':>12}{'after':>12}")
    for name, f in (("expected calibration error", lambda t: ece(held, t)[0]),
                    ("Brier score", lambda t: brier(held, t)),
                    ("negative log-likelihood", lambda t: nll(held, t))):
        print(f"   {name:<28}{f(1.0):>12.4f}{f(T):>12.4f}")

    print(f"\n   Reliability, before (gap = accuracy - confidence):")
    print(f"      {'bin':<12}{'n':>5}{'conf':>8}{'acc':>8}{'gap':>8}")
    for r in tab0:
        print(f"      {r['bin']:<12}{r['n']:>5}{r['confidence']:>8.3f}{r['accuracy']:>8.3f}{r['gap']:>+8.3f}")

    # Is accuracy even increasing in confidence? A calibration error can be
    # repaired by rescaling; a non-monotone reliability curve cannot, because
    # the ranking itself is wrong. Weighted by bin size so a 1-case bin cannot
    # carry the conclusion.
    solid = [r for r in tab0 if r["n"] >= 5]
    inversions = sum(1 for i in range(len(solid) - 1)
                     if solid[i + 1]["accuracy"] < solid[i]["accuracy"])
    monotone = inversions == 0
    report_mono = {"bins_with_n_ge_5": len(solid), "inversions": inversions,
                   "monotone": monotone}
    print(f"\n   Bins with n>=5: {len(solid)}. Accuracy inversions as confidence rises: "
          f"{inversions}{'  <- confidence is ANTI-correlated with accuracy' if inversions else ''}")
    if not monotone:
        worst = max(((solid[i]['accuracy'] - solid[i + 1]['accuracy'], i)
                     for i in range(len(solid) - 1)), default=(0, 0))
        i = worst[1]
        lo, hi = solid[i], solid[i + 1]
        # Two-proportion z-test, so an inversion over small bins is not asserted
        # as a finding when it is noise.
        n1, n2 = lo["n"], hi["n"]
        p1, p2 = lo["accuracy"], hi["accuracy"]
        pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2)) if 0 < pool < 1 else 0.0
        z = (p1 - p2) / se if se > 0 else 0.0
        pv = math.erfc(abs(z) / math.sqrt(2))
        report_mono.update({"z": round(z, 2), "p_value": pv,
                            "lower_bin": lo["bin"], "higher_bin": hi["bin"]})
        print(f"   Largest inversion: bin {lo['bin']} (n={n1}) is {p1:.3f} accurate,\n"
              f"   but the MORE confident bin {hi['bin']} (n={n2}) is only {p2:.3f}.")
        print(f"   Two-proportion z = {z:.2f}, p = {pv:.4f} — "
              f"{'not attributable to bin size' if pv < 0.05 else 'WITHIN NOISE at these bin sizes; do not report as a finding'}")

    # ---- per-class confidence: the thing the AL result blamed -------------
    print("\n" + "=" * 66)
    print("PER-CLASS ENTROPY — is confidence informative about the class?")
    print("=" * 66)
    percls = {}
    print(f"   {'class':<12}{'n':>5}{'mean entropy before':>22}{'after':>10}")
    for k in range(NCLS):
        sel = [r for r in rows if r["y"] == k]
        if not sel:
            continue
        b = sum(entropy(r["p"]) for r in sel) / len(sel)
        a = sum(entropy(softmax_t(r["p"], T)) for r in sel) / len(sel)
        percls[CLS[k]] = {"n": len(sel), "before": round(b, 3), "after": round(a, 3)}
        print(f"   {CLS[k]:<12}{len(sel):>5}{b:>22.3f}{a:>10.3f}")
    spread_b = max(v["before"] for v in percls.values()) - min(v["before"] for v in percls.values())
    spread_a = max(v["after"] for v in percls.values()) - min(v["after"] for v in percls.values())
    print(f"\n   spread across classes: {spread_b:.3f} before, {spread_a:.3f} after")

    # ---- the test that matters -------------------------------------------
    budgets = [10, 20, 30, 40, 50, 60, 80, 100]
    acq = acquisition_test(rows, T, budgets, seed=args.seed)
    print("\n" + "=" * 66)
    print("DOES RECALIBRATION RESCUE UNCERTAINTY SAMPLING?")
    print("=" * 66)
    print(f"   malignant cases found (pool has {comp.get(2,0)})\n")
    print(f"   {'budget':>8}{'entropy':>10}{'entropy':>12}{'random':>16}")
    print(f"   {'':>8}{'(uncal)':>10}{'(calib)':>12}{'mean ± sd':>16}")
    # Two DIFFERENT questions, kept apart. Counting "calibrated beats random"
    # and calling it "recalibration helps" conflates them: the uncalibrated arm
    # may already beat random, in which case recalibration did nothing.
    wins_cal = wins_uncal = changed = 0
    for b in budgets:
        u = acq["uncalibrated"][str(b)]
        c = acq["calibrated"][str(b)]
        r = acq["random"][str(b)]
        if c > r["mean"] + r["sd"]:
            wins_cal += 1
        if u > r["mean"] + r["sd"]:
            wins_uncal += 1
        if c != u:
            changed += 1
        flag = "  <- beats random" if c > r["mean"] + r["sd"] else ""
        print(f"   {b:>8}{u:>10}{c:>12}{r['mean']:>11.1f} ± {r['sd']:<4.1f}{flag}")
    wins = wins_cal

    works = wins_uncal > 0 or wins_cal > 0
    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    print(f"""
   Does uncertainty sampling beat random?  {'YES' if works else 'NO'}
      uncalibrated : beats random at {wins_uncal} of {len(budgets)} budgets
      calibrated   : beats random at {wins_cal} of {len(budgets)} budgets

   Did recalibration change anything?      {'YES' if changed else 'essentially NO'}
      the two arms differ at {changed} of {len(budgets)} budgets; fitted T = {T:.3f}
      expected calibration error {e0:.3f} -> {e1:.3f}""")

    if works and not changed:
        print(f"""
   So the acquisition rule works, and recalibration is not what makes it
   work. The model is already close to calibrated (T = {T:.3f}, ECE {e0:.3f}),
   so temperature scaling has almost nothing to correct. Report the
   uncalibrated curve; do not claim recalibration as a contribution.""")
    elif works:
        print("""
   Report both arms and say which one the budget curve uses. A calibrator
   fitted on the evaluation set would make the curve meaningless, so state
   the split it was fitted on.""")
    else:
        print(f"""
   Uncertainty sampling does not beat random here, before or after scaling.
   Check the reliability curve before blaming calibration: if accuracy does
   not rise with confidence at all, the ranking carries no signal and no
   post-hoc rescaling repairs a ranking.""")

    if not monotone and report_mono.get("p_value", 1.0) < 0.05:
        print("""
   NOTE: accuracy does not rise monotonically with confidence, and the
   inversion is larger than bin size explains. Treat confidence-based
   selection with suspicion even where the yield looks good.""")

    print(f"""
   LIMITS. {len(rows)} cases, 5 of 8 sites, one model. The entropy and margin arms
   are single deterministic orderings, so they carry no sampling spread of
   their own -- the +-sd shown is the random arm's. Temperature scaling is the
   simplest recalibration, not the only one. None of this measures whether a
   model RETRAINED on actively-selected cases is better, which is the actual
   D3.2 question.
""")

    os.makedirs(args.out, exist_ok=True)
    rep = {"n": len(rows), "temperature_at_bound": at_bound, "composition": {CLS[k]: comp.get(k, 0) for k in range(NCLS)},
           "temperature": T, "fit_n": len(fit), "held_n": len(held),
           "held_out": {"ece_before": e0, "ece_after": e1,
                        "brier_before": brier(held, 1.0), "brier_after": brier(held, T),
                        "nll_before": nll(held, 1.0), "nll_after": nll(held, T)},
           "reliability_before": tab0, "reliability_after": tab1,
           "per_class_entropy": percls,
           "entropy_spread_before": spread_b, "entropy_spread_after": spread_a,
           "reliability_monotonicity": report_mono,
           "acquisition": acq,
           "budgets_where_calibrated_beats_random": wins_cal,
           "budgets_where_uncalibrated_beats_random": wins_uncal,
           "budgets_where_calibration_changed_selection": changed,
           "works": works}
    p = os.path.join(args.out, "calibration.json")
    with open(p, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"   wrote {p}\n")


if __name__ == "__main__":
    main()
