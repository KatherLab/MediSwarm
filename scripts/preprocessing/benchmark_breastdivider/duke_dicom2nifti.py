#!/usr/bin/env python3
"""Duke DICOM -> bilateral NIfTI, the reference converter with its paths as arguments.

Same method as scripts/preprocessing/duke/step1_dicom2nifti.py (SimpleITK series reader
over the TCIA download, sequence names from the TCIA filepath mapping), restricted to the
sequences and patients asked for, so a benchmark subset converts in minutes instead of a day.

Output: <out>/data/<PatientID>/<Sequence>.nii.gz   (Pre.nii.gz, Post_1.nii.gz, ...)

Usage:
  duke_dicom2nifti.py --raw <.../Duke-Breast-Cancer-MRI> --mapping <mapping.csv> --out <root>
                      [--patients ids.txt] [--seqs Pre,Post_1] [--workers 8]
"""
import argparse
import csv
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
RAW = OUT = None


def convert(job):
    pid, seq, rel = job
    t0 = time.time()
    src = RAW / rel
    dst = OUT / "data" / pid / f"{seq}.nii.gz"
    if dst.exists():
        return pid, seq, "exists", 0.0
    if not src.is_dir():
        return pid, seq, f"missing dir {rel}", 0.0
    try:
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(str(src))
        if not names:
            return pid, seq, "no dicom files", 0.0
        reader.SetFileNames(names)
        img = reader.Execute()
        dst.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(dst))
        return pid, seq, f"ok {img.GetSize()} {tuple(round(s, 2) for s in img.GetSpacing())}", round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001
        return pid, seq, f"ERROR {type(e).__name__}: {str(e)[:120]}", round(time.time() - t0, 1)


def main() -> None:
    global RAW, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="directory that holds Breast_MRI_001/ ... (classic_path root)")
    ap.add_argument("--mapping", type=Path, required=True, help="Breast-Cancer-MRI-filepath_filename-mapping.csv (with PatientID, SequenceName, classic_path)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patients", type=Path, default=None, help="file with one Breast_MRI_xxx id per line (default: all in the mapping)")
    ap.add_argument("--seqs", default="Pre,Post_1")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    RAW, OUT = a.raw, a.out
    seqs = set(a.seqs.split(","))
    wanted = {l.strip() for l in a.patients.read_text().splitlines() if l.strip()} if a.patients else None
    jobs = []
    with open(a.mapping, newline="") as fh:
        for r in csv.DictReader(fh):
            seq = r["SequenceName"]
            rel = r["classic_path"]
            pid = rel.split("/")[0]
            if seq not in seqs or (wanted is not None and pid not in wanted):
                continue
            jobs.append((pid, seq, rel))
    jobs = sorted(set(jobs))
    logging.info("%d series to convert for %d patients", len(jobs), len({j[0] for j in jobs}))
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    ok = 0
    with Pool(a.workers) as pool:
        for pid, seq, status, secs in pool.imap_unordered(convert, jobs):
            ok += status.startswith("ok") or status == "exists"
            print(f"{pid}\t{seq}\t{status}\t{secs}s", flush=True)
    logging.info("done: %d/%d series ok", ok, len(jobs))


if __name__ == "__main__":
    main()
