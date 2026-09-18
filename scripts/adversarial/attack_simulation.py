#!/usr/bin/env python3
"""Model-poisoning simulation against ODELIA's aggregation rules (D3.4, #529).

What this measures
------------------
Two things, and the second is the one that matters for this consortium.

1. **Attack resistance.** One participant submits a poisoned update. How far
   does each aggregation rule let it move the global model?

2. **The cost of that resistance, with no attacker present.** Robust rules work
   by discarding contributions that look unusual. ODELIA's contributions are
   wildly unequal -- Aachen holds 51.7 % of the training data and 83.9 % of all
   benign cases -- so the contribution that looks most unusual is the one
   carrying most of the evidence. This measures what that costs.

Reporting (1) without (2) would recommend a rule that halves the consortium's
effective dataset in order to defend against an attack nobody has mounted.

The model of a site's update
----------------------------
Each site's local optimum is driven by the class mix it actually trains on --
that is what non-IID means here, and ODELIA's class mixes differ enormously
(Nijmegen is 99.0 % no-lesion; Guildford is 63.0 % malignant). So site i's
update is modelled as::

    u_i = A @ p_i + noise_i,     noise_i ~ N(0, sigma^2 / n_i)

where ``p_i`` is site i's measured class-proportion vector, ``A`` is a fixed
random projection into ``dim`` dimensions, and the noise shrinks with the site's
sample count, as sampling error does.

The **target** is ``A @ p_pooled``, the optimum for the pooled consortium
distribution -- which is what training on all 34,462 volumes would approximate.
Because ``p_pooled`` is by definition the size-weighted mean of the ``p_i``, an
unattacked size-weighted mean recovers the target exactly up to noise. That is
not a rigged comparison in favour of the status quo; it is the reason the status
quo is used, and it is precisely what the robust rules give up.

Error is ``||aggregate - target||``, reported relative to ``||target||``.

Usage
    python3 scripts/adversarial/attack_simulation.py [--out DIR] [--trials 200]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RA = os.path.join(_HERE, "..", "..", "application", "jobs", "_shared",
                   "custom", "robust_aggregation.py")
_spec = importlib.util.spec_from_file_location("robust_aggregation", _RA)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

# Measured from each site's own trainer at dataset load. Classes sum to n.
SITES = [
    # code,   n,      no_lesion, benign, malignant
    ("UKA",   17834,  4747,  11836, 1251),
    ("UMCU",   6133,  5337,    740,   56),
    ("RUMC",   3608,  3571,      2,   35),
    ("USZ",    3448,  1911,   1244,  293),
    ("CAM",    1778,  1678,     42,   58),
    ("MHA",    1120,   904,    120,   96),
    ("RSH",     351,     4,    126,  221),
    ("VHIO",    190,   159,      0,   31),
]
DIM = 64
SIGMA = 3.0          # noise scale; divided by sqrt(n_i) per site


def _validate():
    for code, n, a, b, c in SITES:
        assert a + b + c == n, f"{code}: {a}+{b}+{c} != {n}"
    assert sum(s[1] for s in SITES) == 34462, "sites must sum to 34,462"


def proportions():
    return [[a / n, b / n, c / n] for _, n, a, b, c in SITES]


def pooled_proportions():
    tot = sum(s[1] for s in SITES)
    return [sum(s[2 + k] for s in SITES) / tot for k in range(3)]


def projection(rng):
    """Fixed random 3 -> DIM projection. Shared by every arm of the experiment."""
    return [[rng.gauss(0, 1) for _ in range(3)] for _ in range(DIM)]


def project(A, p):
    return [sum(A[d][k] * p[k] for k in range(3)) for d in range(DIM)]


def honest_updates(A, rng):
    out = []
    for (code, n, *_), p in zip(SITES, proportions()):
        base = project(A, p)
        s = SIGMA / math.sqrt(n)
        out.append([x + rng.gauss(0, s) for x in base])
    return out


def rel_err(v, target, tnorm):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v, target))) / tnorm


def run_rules(updates, weights, clip):
    """Every rule, on one set of updates. Returns {rule: aggregate}."""
    n = len(updates)
    res = {
        "weighted_mean":     ra.weighted_mean(updates, weights),
        "norm_clipped_mean": ra.norm_clipped_mean(updates, clip, weights),
        # Not a robust rule -- the diagnostic that isolates *why* the robust
        # rules cost what they do. They cannot use weights; this is what that
        # alone costs, with no trimming and no attacker.
        "unweighted_mean":   ra.weighted_mean(updates, None),
        "trimmed_mean":      ra.trimmed_mean(updates, trim=1),
        "coordinate_median": ra.coordinate_median(updates),
    }
    if n > 2 * 1 + 2:
        res["krum"] = ra.krum(updates, f=1, m=1)[0]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="workspace/adversarial")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _validate()
    rng = random.Random(args.seed)
    A = projection(rng)
    target = project(A, pooled_proportions())
    tnorm = math.sqrt(sum(x * x for x in target))
    weights = [s[1] for s in SITES]
    codes = [s[0] for s in SITES]

    # Clip budget: the median honest update norm. Chosen from honest behaviour,
    # not from the attack, so it is a number the consortium could actually set.
    probe = honest_updates(A, random.Random(args.seed + 999))
    clip = sorted(ra.l2_norm(u) for u in probe)[len(probe) // 2]

    report = {"dim": DIM, "sigma": SIGMA, "trials": args.trials,
              "clip": clip, "target_norm": tnorm,
              "pooled_proportions": pooled_proportions()}

    # ---- 1. no attacker: what does robustness cost? ----------------------
    acc = {}
    for t in range(args.trials):
        r = random.Random(args.seed * 1000 + t)
        ups = honest_updates(A, r)
        for rule, agg in run_rules(ups, weights, clip).items():
            acc.setdefault(rule, []).append(rel_err(agg, target, tnorm))
    baseline = {k: sum(v) / len(v) for k, v in acc.items()}
    report["baseline_error_no_attacker"] = baseline

    print("\n" + "=" * 74)
    print("1. NO ATTACKER — what each rule costs on ODELIA's data distribution")
    print("=" * 74)
    print(f"   {args.trials} trials, error = ||aggregate - pooled optimum|| / ||pooled optimum||\n")
    for rule, e in sorted(baseline.items(), key=lambda kv: kv[1]):
        bar = "#" * min(46, int(e * 220))
        print(f"   {rule:<19} {e:6.3f}  {bar}")
    gap = baseline["unweighted_mean"] - baseline["weighted_mean"]
    trim_extra = baseline["trimmed_mean"] - baseline["unweighted_mean"]
    pp = pooled_proportions()
    uw = [sum(p[k] for p in proportions()) / len(SITES) for k in range(3)]
    report["unweighted_proportions"] = uw
    report["cost_from_losing_weights"] = gap
    report["cost_from_trimming"] = trim_extra

    print(f"""
   WHY. It is not that the robust rules discard Aachen — a per-coordinate trim
   does not discard whole sites. It is that none of them can honour contribution
   weights, because re-weighting the survivors would hand back the influence the
   trim just removed. So they aim at the average *site* instead of the pooled
   *data*, and on ODELIA those are very different distributions:

        {'':<22}{'no-lesion':>11}{'benign':>10}{'malignant':>11}
        {'pooled (34,462)':<22}{pp[0]:>11.3f}{pp[1]:>10.3f}{pp[2]:>11.3f}
        {'mean of the 8 sites':<22}{uw[0]:>11.3f}{uw[1]:>10.3f}{uw[2]:>11.3f}

   Benign is {pp[1]:.1%} of the pooled data but only {uw[1]:.1%} of the average site,
   because Aachen holds 83.9 % of all benign cases. Splitting the cost:

        losing the weights   {gap:+.3f}
        the trimming itself  {trim_extra:+.3f}

   The weights are {abs(gap) / max(abs(gap) + abs(trim_extra), 1e-9):.0%} of it.""")

    r = random.Random(args.seed + 7)
    ups = honest_updates(A, r)
    _, chosen = ra.krum(ups, f=1, m=1)
    report["krum_selects"] = [codes[i] for i in chosen]
    print(f"\n   Krum, separately, selects {', '.join(codes[i] for i in chosen)} — one site's update"
          f"\n   becomes the global model for that round, and the other seven are dropped.")

    # ---- 2. one attacker, growing magnitude ------------------------------
    print("\n" + "=" * 74)
    print("2. ONE ATTACKER — sign-flip poisoning at growing magnitude")
    print("=" * 74)

    scales = [1, 5, 20, 100, 1000]
    attack = {}
    for who in ("VHIO", "UKA"):
        ai = codes.index(who)
        attack[who] = {}
        print(f"\n   Attacker = {who} "
              f"({SITES[ai][1]:,} volumes, {100*weights[ai]/sum(weights):.1f} % of consortium weight)")
        header = "   " + f"{'rule':<19}" + "".join(f"{'x' + str(s):>10}" for s in scales)
        print(header); print("   " + "-" * (len(header) - 3))
        rows = {}
        for scale in scales:
            per = {}
            for t in range(args.trials):
                r = random.Random(args.seed * 1000 + t)
                ups = honest_updates(A, r)
                ups[ai] = [-scale * x for x in target]      # sign-flip, scaled
                for rule, agg in run_rules(ups, weights, clip).items():
                    per.setdefault(rule, []).append(rel_err(agg, target, tnorm))
            for rule, v in per.items():
                rows.setdefault(rule, []).append(sum(v) / len(v))
        for rule in sorted(rows, key=lambda k: rows[k][-1]):
            print(f"   {rule:<19}" + "".join(f"{e:>10.2f}" for e in rows[rule]))
            attack[who][rule] = dict(zip((f"x{s}" for s in scales), rows[rule]))
    report["attack"] = attack
    report["attack_scales"] = scales

    # ---- verdict ---------------------------------------------------------
    wm   = attack["VHIO"]["weighted_mean"]["x1000"]
    nc   = attack["VHIO"]["norm_clipped_mean"]["x1000"]
    wm_u = attack["UKA"]["weighted_mean"]["x1000"]
    nc_u = attack["UKA"]["norm_clipped_mean"]["x1000"]
    report["verdict"] = {
        "small_adversary_weighted_mean_x1000": wm,
        "small_adversary_norm_clipped_x1000": nc,
        "large_adversary_weighted_mean_x1000": wm_u,
        "large_adversary_norm_clipped_x1000": nc_u,
    }
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"""
   1. Unbounded influence is real, and small sites are not harmless.
      At x1000 Barcelona — 190 volumes, 0.6 % of the consortium's data — moves
      the size-weighted mean {wm:.1f}x the target's own magnitude away from it.
      Weighted mean has a breakdown point of zero; the data imbalance does not
      protect it, it just changes who has to try harder.

   2. Norm clipping is the only rule that bounds influence while keeping weights.
      It holds that same attack to {nc:.2f}, and its no-attacker cost ({baseline['norm_clipped_mean']:.3f}) is close
      to the status quo ({baseline['weighted_mean']:.3f}). Every other rule here resists the attack by
      giving up weights, which costs {baseline['unweighted_mean'] - baseline['weighted_mean']:+.3f} before an adversary shows up.

   3. Clipping does NOT defend against the majority contributor. This is the
      limitation D3.4 has to state plainly. A poisoning Aachen still holds 51.7 %
      of the weight after its update is clipped, so clipping only takes the
      attack from {wm_u:.0f}x down to {nc_u:.2f} — bounded, but still a corrupted model.
      Influence is bounded proportionally to weight, and one participant here
      has half of it.

   RECOMMENDATION for D3.4:
     * Adopt norm-bounding with a published clip, plus per-round monitoring of
       contribution norms. It is cheap, keeps every participant, and turns
       unbounded influence into a number the consortium sets.
     * Do NOT adopt trimmed mean, median or Krum for ODELIA as it stands. They
       are designed for exchangeable contributions and ODELIA's are not.
     * The residual risk — a large-weight adversary — is not solvable by an
       aggregation rule. It needs a cap on any single site's weight share, which
       is a consortium governance decision with a real accuracy cost, and that
       trade-off belongs in the report rather than in a config file.

   NOT measured here: this is a simulation over modelled updates, not trained
   models. It establishes the ordering of the rules and the mechanism behind it.
   It does not give the accuracy a real ODELIA run would lose to a given clip;
   that needs one training run per clip value.
""")

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "attack_simulation.json")
    with open(p, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"   wrote {p}\n")


if __name__ == "__main__":
    main()
