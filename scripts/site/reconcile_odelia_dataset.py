#!/usr/bin/env python3
"""Count what the ODELIA swarm node will and will not use from a site's local data set.

Run it at the site, on the host, against the same directory you pass to
``./docker.sh --data_dir``. It reads only: annotation.csv, split.csv and the names of
the exam folders. It prints counts. With ``--list`` it also prints the UIDs behind each
count, which stay on your machine; send us the counts, not the UIDs.

What the node uses for a run (fold F) is the intersection of three lists:

  split.csv rows of fold F      x  annotation.csv rows  x  exam folders on disk
  (Split = train / val / test)     (UID -> Lesion)         (data_unilateral/<UID>/)

One row (one UID) is one breast of one exam, ``<exam>_left`` or ``<exam>_right``.
A bilateral exam therefore contributes two rows. Whatever is in only one or two of the
three lists is silently left out, and the node only says "detected, make sure this was
intended" without numbers, because UIDs never leave the site. This script gives the
numbers.

Usage
    python3 reconcile_odelia_dataset.py --data_dir /path/to/site/data [--fold 0] [--list]

``--data_dir`` may point at the folder that holds ``data_unilateral`` and
``metadata_unilateral`` directly, or one level above it (the layout the node sees).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path


def find_site_root(data_dir: Path) -> Path:
    if (data_dir / "data_unilateral").is_dir() and (data_dir / "metadata_unilateral").is_dir():
        return data_dir
    candidates = [p for p in sorted(data_dir.iterdir()) if p.is_dir()
                  and (p / "data_unilateral").is_dir() and (p / "metadata_unilateral").is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        sys.exit(f"no data_unilateral/ + metadata_unilateral/ under {data_dir} (or one level below)")
    sys.exit(f"several site folders under {data_dir}: {[c.name for c in candidates]}; point --data_dir at one of them")


def read_csv(path: Path) -> list[dict]:
    """Tolerate a UTF-8 byte-order mark, CRLF line endings and padded header names,
    all of which appear in files exported from spreadsheets."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def exam_folders(path: Path) -> tuple[set[str], list[str]]:
    found, dangling = set(), []
    with os.scandir(path) as entries:
        for e in entries:
            if e.is_dir():
                found.add(e.name)
            elif e.is_symlink() and not os.path.exists(e.path):
                dangling.append(e.name)
    return found, dangling


def side(uid: str) -> str:
    return "left" if uid.endswith("_left") else "right" if uid.endswith("_right") else "?"


def exam_of(uid: str) -> str:
    return uid[: -len("_left")] if uid.endswith("_left") else uid[: -len("_right")] if uid.endswith("_right") else uid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="the directory you pass to docker.sh --data_dir")
    ap.add_argument("--fold", type=int, default=0, help="fold the run uses (default 0)")
    ap.add_argument("--list", action="store_true", help="also print the UIDs behind each count (stays local)")
    a = ap.parse_args()

    root = find_site_root(Path(a.data_dir).expanduser().resolve())
    meta = root / "metadata_unilateral"
    ann = read_csv(meta / "annotation.csv")
    spl = read_csv(meta / "split.csv")
    images, dangling = exam_folders(root / "data_unilateral")

    ann_uids = [r["UID"] for r in ann]
    spl_uids = [r["UID"] for r in spl]
    ann_set, spl_set = set(ann_uids), set(spl_uids)
    lesion = {r["UID"]: r.get("Lesion", "") for r in ann}

    def show(title: str, uids) -> None:
        uids = sorted(uids)
        print(f"  {title}: {len(uids)}")
        if a.list and uids:
            for u in uids:
                print(f"      {u}")

    print(f"site folder      : {root}")
    print(f"annotation.csv   : {len(ann_uids)} rows, {len(ann_set)} distinct UIDs"
          + (f"  ({len(ann_uids) - len(ann_set)} duplicate rows)" if len(ann_uids) != len(ann_set) else ""))
    print(f"split.csv        : {len(spl_uids)} rows, {len(spl_set)} distinct UIDs; folds "
          f"{sorted({r['Fold'] for r in spl})}; per fold and split:")
    per = Counter((r["Fold"], r["Split"]) for r in spl)
    for (f, s), n in sorted(per.items()):
        print(f"                     fold {f} {s:5s}: {n}")
    print(f"exam folders     : {len(images)} under data_unilateral/"
          + (f"  ({len(dangling)} dangling symlinks skipped)" if dangling else ""))
    print(f"lesion labels    : {dict(Counter(lesion.values()))}")

    print("\nnot used by the node (in one or two of the three lists only):")
    show("in annotation.csv but not in split.csv", ann_set - spl_set)
    show("in split.csv but not in annotation.csv", spl_set - ann_set)
    show("in annotation.csv but no exam folder", ann_set - images)
    show("exam folder but not in annotation.csv", images - ann_set)
    show("exam folder but not in split.csv", images - spl_set)
    bad_suffix = [u for u in ann_set | spl_set | images if side(u) == "?"]
    show("UIDs not ending in _left or _right", bad_suffix)

    usable = ann_set & spl_set & images
    print(f"\nusable by the node (in all three lists): {len(usable)}")
    fold_rows = [r for r in spl if str(r["Fold"]) == str(a.fold)]
    if not fold_rows:
        print(f"  fold {a.fold} is not in split.csv; the run would abort naming the folds present")
    for s in ("train", "val", "test"):
        uids = {r["UID"] for r in fold_rows if r["Split"] == s}
        used = uids & usable
        cls = Counter(lesion.get(u, "") for u in used)
        print(f"  fold {a.fold} {s:5s}: {len(used)} used of {len(uids)} listed"
              + (f"; labels {dict(cls)}" if used else ""))
    exams = Counter(exam_of(u) for u in usable)
    sides = Counter(side(u) for u in usable)
    one_sided = [e for e, n in exams.items() if n == 1]
    print(f"  = {len(exams)} exams ({sides['left']} left, {sides['right']} right breasts); "
          f"{len(one_sided)} exams contribute one breast only")
    if a.list and one_sided:
        for e in sorted(one_sided):
            print(f"      {e}")


if __name__ == "__main__":
    main()
