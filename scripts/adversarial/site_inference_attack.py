#!/usr/bin/env python3
"""White-hat probe: can the shared model's outputs reveal which hospital a case came from? (MS6, #531)

Why this exists, and what it is not
-----------------------------------
MS6 is a white-hat attack on the swarm-trained model. The proposal schedules it
**between partners** -- Cambridge and Radboud -- and that is still the right way
to run the full exercise, because the strongest attacks need access a single
partner does not have. But waiting for a scheduling round-trip is not a reason
to have measured nothing, so this runs the part TUD can run alone.

**Threat model.** An observer who sees the model's per-case outputs -- three
class probabilities, no identifier -- tries to infer *which hospital the case
came from*. That is a property-inference attack. If it succeeds, the model's
behaviour is site-identifiable, which matters because per-case predictions are
exactly what D2.5 and the M54 testing node propose to move between sites.

**What this is NOT.** It is not membership inference (was this patient in
training?) and not reconstruction (recover an image from weights). Those are the
attacks MS6 ultimately needs, they need training-set predictions or weight
access, and they remain scheduled with partners. Do not report this as "the
white-hat attack was done".

Method
------
Leave-one-out k-NN over the three output probabilities, against two baselines:

* **majority** -- always guess the largest site. The floor any attack must clear.
* **permutation null** -- reshuffle the site labels and repeat the whole LOO
  procedure. This is the honest baseline: with 5 imbalanced classes and 165
  cases, a k-NN can look better than majority purely by exploiting class
  imbalance, and only the permutation distribution shows whether the structure
  is real.

Reporting accuracy alone would overstate the attack. Balanced accuracy and the
permutation p-value are what the conclusion rests on.

Usage
    python3 scripts/adversarial/site_inference_attack.py [--k 5] [--perm 1000]
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

DEFAULT_PRED = "workspace/eval_87c5bbee/results/fixed"


def load(base):
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "*", "predictions_*.csv"))):
        site = os.path.basename(os.path.dirname(f))
        for r in csv.DictReader(open(f)):
            p = [float(r[f"prob_class_{c}"]) for c in range(3)]
            s = sum(p)
            if s <= 0:
                continue
            rows.append({"site": site, "p": [x / s for x in p],
                         "y": int(float(r["ground_truth"]))})
    return rows


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
    a = _auroc([r["p"][2] for r in rows], [1 if r["y"] == 2 else 0 for r in rows])
    if a is not None and a < 0.60:
        raise SystemExit(
            f"\nREFUSING TO ANALYSE {source}: malignant AUROC {a:.3f} is at or below chance.\n"
            f"That is the signature of a checkpoint evaluated with the wrong architecture (E2).\n"
            f"Use workspace/eval_87c5bbee/results/fixed.\n")


def loo_knn(feats, labels, k):
    """Leave-one-out k-NN accuracy and balanced accuracy."""
    n = len(feats)
    correct = 0
    per = defaultdict(lambda: [0, 0])          # label -> [correct, total]
    for i in range(n):
        d = []
        for j in range(n):
            if i == j:
                continue
            s = sum((a - b) ** 2 for a, b in zip(feats[i], feats[j]))
            d.append((s, labels[j]))
        d.sort()
        vote = Counter(l for _, l in d[:k]).most_common()
        top = max(c for _, c in vote)
        pred = sorted(l for l, c in vote if c == top)[0]      # deterministic tie-break
        per[labels[i]][1] += 1
        if pred == labels[i]:
            correct += 1
            per[labels[i]][0] += 1
    bal = sum(c / t for c, t in per.values()) / len(per)
    return correct / n, bal, {l: (c, t) for l, (c, t) in per.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=DEFAULT_PRED)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--min-per-site", type=int, default=5,
                    help="in the within-class control, ignore sites with fewer "
                         "cases than this; balanced accuracy is unstable below it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="workspace/adversarial/site_inference.json")
    args = ap.parse_args()

    rows = load(args.predictions)
    if not rows:
        raise SystemExit(f"no predictions under {args.predictions}")
    guard_wrong_architecture(rows, args.predictions)

    feats = [r["p"] for r in rows]
    labels = [r["site"] for r in rows]
    counts = Counter(labels)
    n = len(rows)
    majority = max(counts.values()) / n

    acc, bal, per = loo_knn(feats, labels, args.k)

    rng = random.Random(args.seed)
    null_acc, null_bal = [], []
    shuf = list(labels)
    for _ in range(args.perm):
        rng.shuffle(shuf)
        a, b, _ = loo_knn(feats, shuf, args.k)
        null_acc.append(a)
        null_bal.append(b)
    p_acc = (1 + sum(1 for v in null_acc if v >= acc)) / (args.perm + 1)
    p_bal = (1 + sum(1 for v in null_bal if v >= bal)) / (args.perm + 1)
    mu = sum(null_bal) / len(null_bal)
    sd = math.sqrt(sum((v - mu) ** 2 for v in null_bal) / len(null_bal))

    print(f"\nSite-inference attack on the shared model's outputs")
    print(f"  {n} cases, {len(counts)} sites: " + ", ".join(f"{s} {c}" for s, c in counts.most_common()))
    print(f"  attacker sees: three class probabilities per case. No identifier.\n")
    print(f"  {'':<26}{'accuracy':>10}{'balanced':>11}")
    print(f"  {'always guess largest site':<26}{majority:>10.3f}{1/len(counts):>11.3f}")
    print(f"  {f'k-NN attack (k={args.k})':<26}{acc:>10.3f}{bal:>11.3f}")
    print(f"  {'permutation null (mean)':<26}{sum(null_acc)/len(null_acc):>10.3f}{mu:>11.3f}")
    print(f"\n  permutation p-value:  accuracy p={p_acc:.4f}   balanced p={p_bal:.4f}   ({args.perm} shuffles)")
    print(f"  balanced accuracy is {(bal-mu)/sd:+.2f} sd from the null mean" if sd > 0 else "")

    print(f"\n  per-site recall (how often the attacker names the right hospital)")
    for s, (c, t) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"    {s:<8} {c:>3}/{t:<3}  {c/t:.3f}")

    # ---- control: is it the model, or just the case mix? ------------------
    # Sites differ enormously in class composition, so an attack on the whole
    # pool can succeed by inferring the CLASS and betting on the site that has
    # most of it. Re-running inside a single class removes that route: any
    # remaining signal is the model behaving differently per site on
    # like-for-like cases.
    control = {}
    print("\n  CONTROL — same attack restricted to one class at a time")
    print(f"  {'class':<12}{'n':>5}{'sites':>7}{'balanced':>11}{'null':>8}{'p':>9}")
    for cls, cname in ((0, "No lesion"), (1, "Benign"), (2, "Malignant")):
        idx = [i for i, r in enumerate(rows) if r["y"] == cls]
        sub_lab = [labels[i] for i in idx]
        if len(idx) < 25 or len(set(sub_lab)) < 3:
            print(f"  {cname:<12}{len(idx):>5}{len(set(sub_lab)):>7}     too few to test")
            continue
        # Balanced accuracy averages per-site recall, so a site with one or two
        # cases can swing it on a single lucky guess. Drop sites too small to
        # mean anything before testing, and report how many were dropped.
        cnt = Counter(sub_lab)
        keep = {s_ for s_, c in cnt.items() if c >= args.min_per_site}
        dropped = sorted(set(cnt) - keep)
        if len(keep) < 3:
            print(f"  {cname:<12}{len(idx):>5}{len(set(sub_lab)):>7}     "
                  f"only {len(keep)} site(s) with >={args.min_per_site} cases")
            continue
        idx = [i for i in idx if labels[i] in keep]
        sub_lab = [labels[i] for i in idx]
        sub_f = [feats[i] for i in idx]
        _, b2, _ = loo_knn(sub_f, sub_lab, args.k)
        r2 = random.Random(args.seed + 1)
        sh = list(sub_lab)
        nb = []
        for _ in range(args.perm):
            r2.shuffle(sh)
            _, bb, _ = loo_knn(sub_f, sh, args.k)
            nb.append(bb)
        pv = (1 + sum(1 for v in nb if v >= b2)) / (args.perm + 1)
        m2 = sum(nb) / len(nb)
        control[cname] = {"n": len(idx), "sites": sorted(keep), "dropped": dropped,
                          "balanced": round(b2, 4), "null_mean": round(m2, 4),
                          "p": round(pv, 4)}
        flag = "  <- signal" if pv < 0.05 else ""
        note = f"  (dropped {','.join(dropped)}: <{args.min_per_site} cases)" if dropped else ""
        print(f"  {cname:<12}{len(idx):>5}{len(keep):>7}{b2:>11.3f}{m2:>8.3f}{pv:>9.4f}{flag}{note}")

    within = [v for v in control.values() if v["p"] < 0.05]
    leaks = p_bal < 0.05
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    if leaks:
        print(f"""
   The model's outputs ARE site-identifiable (balanced p={p_bal:.4f}). An observer
   holding only three probabilities per case can name the originating hospital
   better than chance. That is a real, if narrow, leak, and it bears directly on
   the per-case return proposed for D2.5 and the M54 testing node: those rows
   carry no identifier, but they are not provenance-free.

   Sites differ enormously in case mix, so the obvious benign explanation is
   that the attack infers the CLASS and bets on the site holding most of it.
   The control tests exactly that by re-running inside a single class:""")
        if within:
            print(f"""   it still succeeds within {len(within)} class(es) taken alone, so the signal is
   NOT only case mix -- the model behaves measurably differently per site on
   like-for-like cases. That is the finding worth carrying into MS6.""")
        else:
            print("""   it does NOT survive within any single class. So the leak is most likely
   case mix, not per-site model behaviour: the attacker is inferring what
   kind of case it is and betting on the site that has most of them. Weaker
   than it first looks, and the honest way to report it.""")
    else:
        print(f"""
   No detectable site signal (balanced p={p_bal:.4f}). On this evidence an
   observer of the model's per-case outputs cannot say which hospital a case
   came from any better than shuffled labels would allow. That is a useful
   negative for the per-case return proposed in D2.5 and for the M54 testing
   node -- it does not make those rows provenance-free, but the obvious
   inference does not work.""")

    print(f"""
   LIMITS. {n} cases, 5 of 8 sites, one model, and the attacker is given only
   three numbers per case. A stronger attacker sees the weights. This is
   therefore a LOWER bound on leakage, not an upper one, and it is not
   membership inference and not reconstruction. The full MS6 exercise still
   wants partner involvement -- the proposal schedules it with Cambridge and
   Radboud -- and this does not replace it.
""")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump({"n": n, "sites": dict(counts), "k": args.k, "permutations": args.perm,
               "majority_baseline": majority, "accuracy": acc, "balanced_accuracy": bal,
               "null_mean_balanced": mu, "null_sd_balanced": sd,
               "p_accuracy": p_acc, "p_balanced": p_bal,
               "per_site_recall": {s: [c, t] for s, (c, t) in per.items()},
               "site_identifiable": leaks, "within_class_control": control}, open(args.json_out, "w"), indent=1)
    print(f"   wrote {args.json_out}\n")


if __name__ == "__main__":
    main()
