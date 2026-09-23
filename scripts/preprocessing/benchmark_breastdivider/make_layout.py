#!/usr/bin/env python3
"""Build the bilateral input layout the two pipelines expect from the extracted challenge volumes.

Source (as extracted on dl0):   <src>/new_data_pre_post/<pid>_<k>.nii.gz   (HR series, k = 0..5)
                                <src>/new_data_pre_post_no_HR/<pid>_<k>.nii.gz (k = 0..7)
k = 0 is the pre-contrast phase, k >= 1 the post-contrast phases (checked on 23 Sep 2026:
phase 0 mean intensity 245, phases 1 to 5 about 400).

Target:  <dst>/data/<pid>/Pre.nii.gz, Post_1.nii.gz, Post_2.nii.gz, ...   (symlinks)

Only patients listed in --ids (one id per line) are linked, so the benchmark runs on the
labelled UMCU challenge cases only. A patient present in both source folders takes the HR copy.
Usage: make_layout.py <src> <dst> --ids ids.txt
"""
import argparse
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--ids", type=Path, required=True)
    a = ap.parse_args()
    wanted = {l.strip() for l in a.ids.read_text().splitlines() if l.strip()}
    files = defaultdict(dict)   # pid -> {k: path}
    origin = {}
    for folder in ("new_data_pre_post", "new_data_pre_post_no_HR"):
        for p in sorted((a.src / folder).glob("*.nii.gz")):
            m = re.match(r"^(\d+)_(\d+)\.nii\.gz$", p.name)
            if not m:
                continue
            pid, k = m.group(1), int(m.group(2))
            if pid not in wanted or (pid in origin and origin[pid] != folder):
                continue
            origin[pid] = folder
            files[pid][k] = p
    made, missing = 0, []
    for pid in sorted(wanted):
        ph = files.get(pid, {})
        if 0 not in ph or 1 not in ph:
            missing.append(pid)
            continue
        out = a.dst / "data" / pid
        out.mkdir(parents=True, exist_ok=True)
        # Hard links, not symlinks: the BreastDivider container only sees what is bind-mounted
        # under the root dir, and a symlink pointing outside it is invisible in there.
        def place(src: Path, dst: Path) -> None:
            dst.unlink(missing_ok=True)
            try:
                os.link(src.resolve(), dst)
            except OSError:
                shutil.copyfile(src.resolve(), dst)
        place(ph[0], out / "Pre.nii.gz")
        for k in sorted(k for k in ph if k >= 1):
            place(ph[k], out / f"Post_{k}.nii.gz")
        made += 1
    print(f"linked {made} patients into {a.dst / 'data'}; HR {sum(1 for p in origin if origin[p] == 'new_data_pre_post' and p in wanted)}, "
          f"noHR {sum(1 for p in origin if origin[p] != 'new_data_pre_post' and p in wanted)}; missing pre/post_1: {len(missing)} {missing[:5]}")


if __name__ == "__main__":
    main()
