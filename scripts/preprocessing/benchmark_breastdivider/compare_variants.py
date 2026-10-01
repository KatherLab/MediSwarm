#!/usr/bin/env python3
"""Compare a model's predictions on the same cases under different preprocessing variants.

Inputs: prediction CSVs written by scripts/evaluation/predict.py (columns uid, ground_truth,
prediction, prob_class_0, prob_class_1, prob_class_2), one per variant. Cases are aligned by
uid; only uids present in every variant are used, so every comparison is paired.

Reports per variant: n, accuracy, AUROC malignant vs rest (class 2), AUROC per class
(one vs rest), macro AUROC; and for every variant against the first one: the paired
bootstrap 95 % interval of the AUROC difference (2,000 resamples over cases). AUROC is the
Mann-Whitney statistic, no sklearn needed.

Usage: compare_variants.py --label old=path.csv --label BD=path.csv [--label existing=...]
"""
import argparse
import csv
import sys

import numpy as np


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based (ties count half)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": len(y), "acc": float((p.argmax(1) == y).mean())}
    for c in range(p.shape[1]):
        out[f"auc_c{c}"] = auroc((y == c).astype(int), p[:, c])
    out["auc_malignant"] = out["auc_c2"]
    out["auc_macro"] = float(np.nanmean([out[f"auc_c{c}"] for c in range(p.shape[1])]))
    return out


def load(path: str) -> dict:
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["uid"]] = (int(float(r["ground_truth"])),
                              np.array([float(r["prob_class_0"]), float(r["prob_class_1"]), float(r["prob_class_2"])]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", action="append", required=True, help="name=csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--split-csv", default=None, help="restrict to uids whose Split equals --split in this split.csv")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    data = {}
    for item in a.label:
        name, path = item.split("=", 1)
        data[name] = load(path)
    names = list(data)
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    if a.split_csv:
        keep = {r["UID"] for r in csv.DictReader(open(a.split_csv, encoding="utf-8-sig")) if r["Split"] == a.split}
        common = [u for u in common if u in keep]
        print(f"restricted to split '{a.split}' of {a.split_csv}: {len(common)} cases")
    if not common:
        sys.exit("no common uids")
    y = np.array([data[names[0]][u][0] for u in common])
    P = {n: np.array([data[n][u][1] for u in common]) for n in names}
    for n in names:
        if not all(data[n][u][0] == data[names[0]][u][0] for u in common):
            sys.exit(f"ground truth differs between {names[0]} and {n}")
    print(f"paired cases: {len(common)}  (labels: " + ", ".join(f"{c}: {(y == c).sum()}" for c in range(3)) + ")")
    print(f"{'variant':<10} {'n':>4} {'acc':>6} {'AUC mal':>8} {'AUC c0':>7} {'AUC c1':>7} {'AUC c2':>7} {'macro':>7}")
    base = names[0]
    for n in names:
        m = metrics(y, P[n])
        print(f"{n:<10} {m['n']:>4} {m['acc']:>6.3f} {m['auc_malignant']:>8.3f} {m['auc_c0']:>7.3f} {m['auc_c1']:>7.3f} {m['auc_c2']:>7.3f} {m['auc_macro']:>7.3f}")
    rng = np.random.default_rng(0)
    idx = np.arange(len(common))
    for n in names[1:]:
        d_mal, d_mac = [], []
        for _ in range(a.boot):
            b = rng.choice(idx, len(idx), replace=True)
            mb, mn = metrics(y[b], P[base][b]), metrics(y[b], P[n][b])
            d_mal.append(mn["auc_malignant"] - mb["auc_malignant"]); d_mac.append(mn["auc_macro"] - mb["auc_macro"])
        for label, d in (("malignant AUROC", d_mal), ("macro AUROC", d_mac)):
            d = np.array(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
            print(f"{n} minus {base}, {label}: {np.nanmean(d):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
                  + ("  (interval excludes 0)" if lo > 0 or hi < 0 else ""))


if __name__ == "__main__":
    main()
