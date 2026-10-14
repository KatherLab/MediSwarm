#!/usr/bin/env python3
"""Count what the ODELIA swarm node will and will not use from a site's local data set.

Run it at the site against the same directory you pass to ``./docker.sh --data_dir``: on
the host, or inside the image, which has the same version of the folder rules as the node:

    docker run --rm --network none -u "$(id -u):$(id -g)" -v "$DATADIR":/data:ro \\
        --entrypoint python3 jefftud/odelia:<version> \\
        /MediSwarm/scripts/site/reconcile_odelia_dataset.py --data_dir /data/<SITE> --config unilateral_BD

It reads only: annotation.csv, split.csv, the names of the exam folders, the processing.log
next to the crops and the headers of the image files. It prints counts. With ``--list`` it
also prints the UIDs behind each count, which stay on your machine; send us the counts,
not the UIDs.

What the node uses for a run (fold F) is the intersection of three lists:

  split.csv rows of fold F      x  annotation.csv rows  x  exam folders on disk
  (Split = train / val / test)     (UID -> Lesion)         (data_unilateral/<UID>/)

One row (one UID) is one breast of one exam, ``<exam>_left`` or ``<exam>_right``.
A bilateral exam therefore contributes two rows. Whatever is in only one or two of the
three lists is silently left out, and the node only says "detected, make sure this was
intended" without numbers, because UIDs never leave the site. This script gives the
numbers.

Which folders are read follows the run's CONFIG, exactly as on the node:

  --config unilateral (default)   data_unilateral/      + metadata_unilateral/
  --config unilateral_BD          data_unilateral_BD/   + metadata_unilateral_BD/
                                  (metadata_unilateral/ if metadata_unilateral_BD/ is missing)
  --config original               data/                 + metadata/

``--data_folder``, ``--metadata_folder`` and ``--image_file`` override them (the node reads
ODELIA_DATA_FOLDER, ODELIA_METADATA_FOLDER and ODELIA_IMAGE_FILE; so does this script). For
unilateral and unilateral_BD it also runs the node's crop-format check: BreastDivider's
processing.log (package version, untrusted-fallback count) and the voxel spacing of the
crops, which varies per case for BreastDivider and is a fixed 0.7 x 0.7 x 3 mm grid for the
standard crops.

Usage
    python3 reconcile_odelia_dataset.py --data_dir /path/to/site/data [--config unilateral_BD]
                                        [--fold 0] [--list] [--sample N]

``--data_dir`` may point at the site folder that holds the data and metadata folders
directly, or one level above it (the layout the node sees).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

# Copy of data_layout.CONFIG_FOLDERS for when this file is used on its own (a unit test keeps
# the two equal). From a MediSwarm checkout or inside the image the node's module is used.
STANDALONE_CONFIG_FOLDERS = {
    "original": ("data", "metadata", None),
    "unilateral": ("data_unilateral", "metadata_unilateral", None),
    "unilateral_BD": ("data_unilateral_BD", "metadata_unilateral_BD", "metadata_unilateral"),
}
DEFAULT_IMAGE_FILE = "Sub_1.nii.gz"


def load_layout_module():
    """The node decides which folders it reads in application/jobs/_shared/custom/data_layout.py.
    Use that very file when it is next to this script (checkout, or /MediSwarm in the image)."""
    here = Path(__file__).resolve()
    if len(here.parents) < 3:
        return None
    path = here.parents[2] / "application" / "jobs" / "_shared" / "custom" / "data_layout.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("odelia_data_layout", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses look their module up while it executes
    spec.loader.exec_module(module)
    return module


def layout_env(a) -> dict:
    """The command line in the variables the node reads (CLI first, then the environment)."""
    return {
        "CONFIG": a.config if a.config is not None else os.environ.get("CONFIG", ""),
        "ODELIA_DATA_FOLDER": a.data_folder if a.data_folder is not None else os.environ.get("ODELIA_DATA_FOLDER", ""),
        "ODELIA_METADATA_FOLDER": (a.metadata_folder if a.metadata_folder is not None
                                   else os.environ.get("ODELIA_METADATA_FOLDER", "")),
        "ODELIA_IMAGE_FILE": a.image_file if a.image_file is not None else os.environ.get("ODELIA_IMAGE_FILE", ""),
    }


def requested_folders(env: dict, layout_module) -> tuple[str, str, list[str], str]:
    """(config, images folder, metadata folders in order of preference, image file), before
    looking at the disk."""
    if layout_module is not None:
        try:
            layout = layout_module.requested_layout(env=env)
        except ValueError as exc:
            sys.exit(str(exc))
        fallback = layout_module.CONFIG_FOLDERS[layout.config][2]
        metadata = [layout.metadata_folder]
        if fallback and "ODELIA_METADATA_FOLDER" not in layout.overrides:
            metadata.append(fallback)
        return layout.config, layout.data_folder, metadata, layout.image_file
    raw = (env["CONFIG"] or "unilateral").strip()
    config = next((c for c in STANDALONE_CONFIG_FOLDERS if c.lower() == raw.lower()), None)
    if config is None:
        sys.exit(f"Unknown CONFIG={raw!r}. Valid values: {', '.join(STANDALONE_CONFIG_FOLDERS)}.")
    data, metadata, fallback = STANDALONE_CONFIG_FOLDERS[config]
    data = env["ODELIA_DATA_FOLDER"].strip() or data
    if env["ODELIA_METADATA_FOLDER"].strip():
        metadata_list = [env["ODELIA_METADATA_FOLDER"].strip()]
    else:
        metadata_list = [metadata] + ([fallback] if fallback else [])
    return config, data, metadata_list, env["ODELIA_IMAGE_FILE"].strip() or DEFAULT_IMAGE_FILE


def find_site_root(data_dir: Path, data_folder: str, metadata_folders: list[str]) -> Path:
    def is_site(p: Path) -> bool:
        return (p / data_folder).is_dir() and any((p / m).is_dir() for m in metadata_folders)

    if is_site(data_dir):
        return data_dir
    candidates = [p for p in sorted(data_dir.iterdir()) if p.is_dir() and is_site(p)] if data_dir.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    wanted = f"{data_folder}/ + {' or '.join(m + '/' for m in metadata_folders)}"
    if not candidates:
        sys.exit(f"no {wanted} under {data_dir} (or one level below)")
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
    ap.add_argument("--config", default=None,
                    help="CONFIG of the run: unilateral (default), unilateral_BD or original (default: $CONFIG)")
    ap.add_argument("--data_folder", default=None, help="images folder instead of the CONFIG default "
                                                        "(default: $ODELIA_DATA_FOLDER)")
    ap.add_argument("--metadata_folder", default=None, help="metadata folder instead of the CONFIG default "
                                                            "(default: $ODELIA_METADATA_FOLDER)")
    ap.add_argument("--image_file", default=None, help=f"image file in each exam folder (default: "
                                                       f"$ODELIA_IMAGE_FILE or {DEFAULT_IMAGE_FILE})")
    ap.add_argument("--sample", type=int, default=0,
                    help="read the headers of at most N crops for the format check (default: all)")
    a = ap.parse_args()

    layout_module = load_layout_module()
    env = layout_env(a)
    config, data_folder, metadata_folders, image_file = requested_folders(env, layout_module)
    root = find_site_root(Path(a.data_dir).expanduser().resolve(), data_folder, metadata_folders)
    metadata_folder = next(m for m in metadata_folders if (root / m).is_dir())
    meta = root / metadata_folder
    ann = read_csv(meta / "annotation.csv")
    spl = read_csv(meta / "split.csv")
    images_dir = root / data_folder
    images, dangling = exam_folders(images_dir)
    without_image = sorted(u for u in images if not (images_dir / u / image_file).is_file())

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
    print(f"config           : {config} (images {data_folder}/, metadata {metadata_folder}/, image file {image_file})")
    if metadata_folder != metadata_folders[0]:
        print(f"                   {metadata_folders[0]}/ not found: the node falls back to {metadata_folder}/ "
              f"and warns")
    print(f"annotation.csv   : {len(ann_uids)} rows, {len(ann_set)} distinct UIDs"
          + (f"  ({len(ann_uids) - len(ann_set)} duplicate rows)" if len(ann_uids) != len(ann_set) else ""))
    print(f"split.csv        : {len(spl_uids)} rows, {len(spl_set)} distinct UIDs; folds "
          f"{sorted({r['Fold'] for r in spl})}; per fold and split:")
    per = Counter((r["Fold"], r["Split"]) for r in spl)
    for (f, s), n in sorted(per.items()):
        print(f"                     fold {f} {s:5s}: {n}")
    print(f"exam folders     : {len(images)} under {data_folder}/"
          + (f"  ({len(dangling)} dangling symlinks skipped)" if dangling else ""))
    if without_image:
        show(f"exam folders without {image_file} (the node cannot load them)", without_image)
    print(f"lesion labels    : {dict(Counter(lesion.values()))}")

    if layout_module is None:
        print("crop format      : not checked (run this script from a MediSwarm checkout or inside the image)")
    elif config != "original":
        uids = sorted(images)
        sample = a.sample if a.sample > 0 else max(len(uids), 1)
        verdict = layout_module.check_crop_format(images_dir, config, uids, image_file, sample_size=sample)
        spacing = verdict.spacing
        print(f"crop format      : {verdict.summary()}")
        if spacing is not None and spacing.read:
            print(f"                   {spacing.distinct} distinct spacings among {spacing.read} crops, "
                  f"{spacing.on_fixed_grid} on the fixed 0.7 x 0.7 x 3 mm grid of the standard crops")
        for issue in verdict.issues:
            print(f"  WARNING: {issue}")

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
