#!/usr/bin/env python3
"""Reference (current) ODELIA unilateral preprocessing, made runnable on any root.

Reproduces scripts/preprocessing/step2_compute_sub.py + step3_unilateral.py from the
MediSwarm repository, unchanged in method, for the benchmark against BreastDivider:

  <root>/data/<case>/Pre.nii.gz, Post_1.nii.gz        (bilateral, full resolution)
    -> Sub_1 = Post_1 - Pre (int16, shifted to >= 0, uint16)          [step 2]
    -> ToCanonical, resample to (0.7, 0.7, 3) mm, crop-or-pad to 512x512x32,
       crop height to 256 by the 90 % intensity quantile, split at the midline
       -> <root>/<out>/<case>_left|_right/{Pre,Post_1,Sub_1}.nii.gz  [step 3]

Usage: old_unilateral.py <root> [--out data_unilateral_old] [--workers N] [--cases a,b]
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio


def compute_sub(case_dir: Path) -> None:
    pre = sitk.ReadImage(str(case_dir / "Pre.nii.gz"), sitk.sitkInt16)
    post = sitk.ReadImage(str(case_dir / "Post_1.nii.gz"), sitk.sitkInt16)
    sub = sitk.GetArrayFromImage(post) - sitk.GetArrayFromImage(pre)
    sub = (sub - sub.min()).astype(np.uint16)
    out = sitk.GetImageFromArray(sub)
    out.CopyInformation(pre)
    sitk.WriteImage(out, str(case_dir / "Sub_1.nii.gz"))


def crop_breast_height(image, margin_top=10):
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data > threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    top = min(max(512 - int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    bottom = 256 - top
    return tio.Crop((0, 0, bottom, top, 0, 0))


def preprocess(args):
    case_dir, out_root = args
    t0 = time.time()
    try:
        if not (case_dir / "Sub_1.nii.gz").exists():
            compute_sub(case_dir)
        ref_img = tio.ToCanonical()(tio.ScalarImage(case_dir / "Pre.nii.gz"))
        ref_img = tio.Resample((0.7, 0.7, 3))(ref_img)
        target_shape = (512, 512, 32)
        pad = ref_img.data.min().item()
        transform = tio.Compose([tio.Resample(ref_img), tio.CropOrPad(target_shape, padding_mode=pad)])
        crop_height = crop_breast_height(transform(ref_img))
        split_side = {"right": tio.Crop((256, 0, 0, 0, 0, 0)), "left": tio.Crop((0, 256, 0, 0, 0, 0))}
        for path_img in sorted(case_dir.glob("*.nii.gz")):
            if path_img.name not in ("Pre.nii.gz", "Post_1.nii.gz", "Sub_1.nii.gz"):
                continue
            img = tio.ScalarImage(path_img)
            pad = img.data.min().item()
            img = tio.Compose([tio.Resample(ref_img), tio.CropOrPad(target_shape, padding_mode=pad)])(img)
            img = crop_height(img)
            for side in ("left", "right"):
                out_dir = out_root / f"{case_dir.name}_{side}"
                out_dir.mkdir(parents=True, exist_ok=True)
                split_side[side](img).save(out_dir / path_img.name)
        return case_dir.name, "ok", round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001
        return case_dir.name, f"ERROR {type(e).__name__}: {e}", round(time.time() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", default="data_unilateral_old")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cases", default=None, help="comma-separated case ids (default all)")
    a = ap.parse_args()
    cases = sorted(p for p in (a.root / "data").iterdir() if p.is_dir() and (p / "Pre.nii.gz").exists())
    if a.cases:
        keep = set(a.cases.split(","))
        cases = [c for c in cases if c.name in keep]
    out_root = a.root / a.out
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases -> {out_root}", flush=True)
    ok = 0
    with Pool(a.workers) as pool:
        for name, status, secs in pool.imap_unordered(preprocess, [(c, out_root) for c in cases]):
            ok += status == "ok"
            print(f"{name}\t{status}\t{secs}s", flush=True)
    print(f"done: {ok}/{len(cases)} ok", flush=True)
    sys.exit(0 if ok == len(cases) else 1)


if __name__ == "__main__":
    main()
