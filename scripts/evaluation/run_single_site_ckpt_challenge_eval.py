#!/usr/bin/env python3
"""Evaluate single-site ODELIA checkpoints on the ODELIA challenge dataset.

This helper is intentionally orchestration-heavy and dependency-light.  It
collects local-training checkpoints from the sources used in the USZ/UKA/CAM/MHA
/RUMC comparison, stages them to dl0, runs the existing
``scripts/evaluation/predict.py`` inside the MediSwarm Docker image, summarizes
metrics, and regenerates the combined Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace" / "odelia_single_site_eval"
REPORT_PATH = ROOT / "docs" / "ODELIA_SINGLE_SITE_CKPT_CHALLENGE_EVAL_REPORT.md"
CONDENSED_REPORT_PATH = ROOT / "docs" / "ODELIA_SINGLE_SITE_CKPT_CHALLENGE_EVAL_CONDENSED_REPORT.md"
PARTNER_WORKBOOK_PATH = ROOT / "docs" / "supplementary" / "ODELIA_single_site_checkpoint_results_20260608.xlsx"
OLE_REPORT_PATH = ROOT / "docs" / "OLE_SWARM_EVALUATION_ARTIFACT_REPORT.md"
CHALLENGE_SWARM_REPORT_PATH = ROOT / "docs" / "CHALLENGE_SWARM_LOCAL_TEST_REPORT_20260513.md"
FIGURE_DIR = ROOT / "docs" / "figures" / "odelia_single_site_eval"

DL0_HOST = "swarm@dd-dl0"
DL0_STAGE = Path("/mnt/dlhd0/odelia_single_site_eval")
DL0_DATA = Path("/mnt/dlhd0/medswarmdata")
DOCKER_IMAGE = "jefftud/odelia:1.4.3-dev.260427.ab6397b"
EVAL_SITES = ("CAM", "MHA", "RSH", "RUMC", "UKA", "UMCU")
CLASSES = (0, 1, 2)

UKA_ZIP_DIR = Path("/mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/UKA")
UKA_ZIP_STAMP = "20260520T091501Z"
UKA_RUN_ID = "1DivideAndConquer_unilateral_2026_05_04_082228"
UMCU_ZIP_PATH = Path("/mnt/nvme0n1p1/scratch/jeff/Downloads/MST_unilateral_2026_06_01_205145-20260608T122612Z-3-001.zip")
UMCU_RUN_ID = "MST_unilateral_2026_06_01_205145"
RUMC_1DC_RUN_ID = "1DivideAndConquer_unilateral_2026_06_04_101650"
RUMC_1DC_ZIP_PATHS = (
    Path("/mnt/nvme0n1p1/scratch/jeff/Downloads/1DivideAndConquer_unilateral_2026_06_04_101650-20260619T085949Z-3-001.zip"),
    Path("/mnt/nvme0n1p1/scratch/jeff/Downloads/1DivideAndConquer_unilateral_2026_06_04_101650-20260619T085949Z-3-002.zip"),
)


@dataclass
class RunSource:
    run_key: str
    source_site: str
    model_name: str
    model_short: str
    run_id: str
    run_dir: str
    train_csv: str | None
    val_csv: str | None
    note: str = ""


@dataclass
class Snapshot:
    snapshot_id: str
    source_site: str
    model_name: str
    model_short: str
    run_key: str
    label: str
    path: str
    bytes: int | None = None
    sha256: str | None = None
    duplicate_of: str | None = None


def run(cmd: list[str], *, cwd: Path = ROOT, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def remote(command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    return run(["ssh", "-o", "ConnectTimeout=15", DL0_HOST, command], capture=capture, check=check)


def ensure_dirs() -> None:
    for path in [
        WORKSPACE,
        WORKSPACE / "raw" / "UKA",
        WORKSPACE / "manifests",
        WORKSPACE / "training_curves",
        WORKSPACE / "odelia_eval",
        WORKSPACE / "tables",
        FIGURE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def extract_uka() -> Path:
    """Extract one complete UKA zip triplet into the workspace."""
    dest = WORKSPACE / "raw" / "UKA"
    run_dir = dest / UKA_RUN_ID
    expected = [
        run_dir / "last.ckpt",
        run_dir / "last_global_model.ckpt",
        run_dir / "epoch=25-step=57980.ckpt",
        run_dir / "site_model_gt_and_classprob_train.csv",
        run_dir / "site_model_gt_and_classprob_validation.csv",
    ]
    if all(path.exists() for path in expected):
        return run_dir

    zips = sorted(UKA_ZIP_DIR.glob(f"*{UKA_ZIP_STAMP}*.zip"))
    if len(zips) != 3:
        raise FileNotFoundError(f"Expected 3 UKA zip chunks for {UKA_ZIP_STAMP}, found {len(zips)} in {UKA_ZIP_DIR}")
    for zip_path in zips:
        run(["unzip", "-n", str(zip_path), "-d", str(dest)])
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("UKA extraction did not produce expected files:\n" + "\n".join(map(str, missing)))
    return run_dir


def extract_umcu() -> Path:
    """Extract the UMCU MST local-training zip into the workspace."""
    dest = WORKSPACE / "raw" / "UMCU"
    run_dir = dest / UMCU_RUN_ID
    expected = [
        run_dir / "last.ckpt",
        run_dir / "last_global_model.ckpt",
        run_dir / "epoch=0-step=767.ckpt",
        run_dir / "site_model_gt_and_classprob_train.csv",
        run_dir / "site_model_gt_and_classprob_validation.csv",
    ]
    if all(path.exists() for path in expected):
        return run_dir
    if not UMCU_ZIP_PATH.exists():
        raise FileNotFoundError(f"UMCU zip not found: {UMCU_ZIP_PATH}")
    run(["unzip", "-n", str(UMCU_ZIP_PATH), "-d", str(dest)])
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("UMCU extraction did not produce expected files:\n" + "\n".join(map(str, missing)))
    return run_dir


def extract_rumc_1dc() -> Path:
    """Extract the RUMC 1DC local-training zip pair into the workspace."""
    dest = WORKSPACE / "raw" / "RUMC_1DC"
    run_dir = dest / RUMC_1DC_RUN_ID
    expected = [
        run_dir / "last.ckpt",
        run_dir / "epoch=0-step=451.ckpt",
        run_dir / "site_model_gt_and_classprob_train.csv",
        run_dir / "site_model_gt_and_classprob_validation.csv",
    ]
    if all(path.exists() for path in expected):
        return run_dir
    missing_zips = [path for path in RUMC_1DC_ZIP_PATHS if not path.exists()]
    if missing_zips:
        raise FileNotFoundError("RUMC 1DC zip(s) not found:\n" + "\n".join(map(str, missing_zips)))
    for zip_path in RUMC_1DC_ZIP_PATHS:
        run(["unzip", "-n", str(zip_path), "-d", str(dest)])
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("RUMC 1DC extraction did not produce expected files:\n" + "\n".join(map(str, missing)))
    return run_dir


def base_run_sources(uka_run_dir: Path, umcu_run_dir: Path, rumc_1dc_run_dir: Path) -> list[RunSource]:
    usz = ROOT / "workspace" / "usz_partner_eval"
    return [
        RunSource(
            "USZ_MST",
            "USZ",
            "MST",
            "MST",
            "MST_unilateral_2026_04_28_083041",
            str(usz / "checkpoints"),
            str(usz / "runs" / "MST_train.csv"),
            str(usz / "runs" / "MST_validation.csv"),
            "USZ local MST, 100 epochs; best checkpoint at epoch 33.",
        ),
        RunSource(
            "USZ_1DC",
            "USZ",
            "1DivideAndConquer",
            "1DC",
            "1DivideAndConquer_unilateral_2026_05_12_124440",
            str(usz / "checkpoints"),
            str(usz / "runs" / "1DC_train.csv"),
            str(usz / "runs" / "1DC_validation.csv"),
            "USZ local 1DivideAndConquer retry; available checkpoints are epoch 14 best and last.",
        ),
        RunSource(
            "UMCU_MST",
            "UMCU",
            "MST",
            "MST",
            UMCU_RUN_ID,
            str(umcu_run_dir),
            str(umcu_run_dir / "site_model_gt_and_classprob_train.csv"),
            str(umcu_run_dir / "site_model_gt_and_classprob_validation.csv"),
            "UMCU local MST artifacts supplied as a single zip on 2026-06-08. "
            "Lightning selected epoch 0 because `ModelCheckpoint` monitors `val/ACC`; in this run, validation accuracy is already maximized by the class-0 majority baseline. "
            "Later epochs improve AUROC/probability ranking but do not improve argmax accuracy or Class-2 recall.",
        ),
        RunSource(
            "UKA_1DC",
            "UKA",
            "1DivideAndConquer",
            "1DC",
            UKA_RUN_ID,
            str(uka_run_dir),
            str(uka_run_dir / "site_model_gt_and_classprob_train.csv"),
            str(uka_run_dir / "site_model_gt_and_classprob_validation.csv"),
            "UKA local 1DivideAndConquer artifacts extracted from local Google Drive zip chunks.",
        ),
        RunSource(
            "CAM_1DC",
            "CAM",
            "1DivideAndConquer",
            "1DC",
            "1DivideAndConquer_unilateral_2026_04_28_161733",
            "/srv/mediswarm/live/CAM_1/local/1DivideAndConquer_unilateral_2026_04_28_161733/run_dir",
            "/srv/mediswarm/live/CAM_1/local/1DivideAndConquer_unilateral_2026_04_28_161733/run_dir/site_model_gt_and_classprob_train.csv",
            "/srv/mediswarm/live/CAM_1/local/1DivideAndConquer_unilateral_2026_04_28_161733/run_dir/site_model_gt_and_classprob_validation.csv",
        ),
        RunSource(
            "MHA_1DC",
            "MHA",
            "1DivideAndConquer",
            "1DC",
            "1DivideAndConquer_unilateral_2026_04_22_154631",
            "/srv/mediswarm/live/MHA_1/local/1DivideAndConquer_unilateral_2026_04_22_154631/run_dir",
            "/srv/mediswarm/live/MHA_1/local/1DivideAndConquer_unilateral_2026_04_22_154631/run_dir/site_model_gt_and_classprob_train.csv",
            "/srv/mediswarm/live/MHA_1/local/1DivideAndConquer_unilateral_2026_04_22_154631/run_dir/site_model_gt_and_classprob_validation.csv",
        ),
        RunSource(
            "RSH_1DC",
            "RSH",
            "1DivideAndConquer",
            "1DC",
            "1DivideAndConquer_unilateral_2026_05_28_090751",
            str(WORKSPACE / "raw" / "RSH_1DC" / "run_dir"),
            str(WORKSPACE / "raw" / "RSH_1DC" / "run_dir" / "site_model_gt_and_classprob_train.csv"),
            str(WORKSPACE / "raw" / "RSH_1DC" / "run_dir" / "site_model_gt_and_classprob_validation.csv"),
            "RSH local 1DivideAndConquer artifacts supplied as local tar/zip chunks on 2026-06-02.",
        ),
        RunSource(
            "RSH_5Pimed",
            "RSH",
            "5Pimed",
            "5Pimed",
            "challenge_5pimed_unilateral_2026_04_03_182744",
            "/srv/mediswarm/live/RSH_1/local/challenge_5pimed_unilateral_2026_04_03_182744/run_dir",
            "/srv/mediswarm/live/RSH_1/local/challenge_5pimed_unilateral_2026_04_03_182744/run_dir/site_model_gt_and_classprob_train.csv",
            "/srv/mediswarm/live/RSH_1/local/challenge_5pimed_unilateral_2026_04_03_182744/run_dir/site_model_gt_and_classprob_validation.csv",
            "RSH local 5Pimed run from the Cosmos dashboard mirror.",
        ),
        RunSource(
            "RUMC_1DC",
            "RUMC",
            "1DivideAndConquer",
            "1DC",
            RUMC_1DC_RUN_ID,
            str(rumc_1dc_run_dir),
            str(rumc_1dc_run_dir / "site_model_gt_and_classprob_train.csv"),
            str(rumc_1dc_run_dir / "site_model_gt_and_classprob_validation.csv"),
            "RUMC local 1DivideAndConquer artifacts supplied as two zip chunks on 2026-06-19. "
            "The validation cohort is almost entirely class 0, so `val/ACC` selects epoch 0 and should not be read as a useful malignant-detection endpoint.",
        ),
        RunSource(
            "RUMC_MST_20260413",
            "RUMC",
            "MST",
            "MST",
            "MST_unilateral_2026_04_13_162111",
            "/srv/mediswarm/live/RUMC_1/local/MST_unilateral_2026_04_13_162111/run_dir",
            "/srv/mediswarm/live/RUMC_1/local/MST_unilateral_2026_04_13_162111/run_dir/site_model_gt_and_classprob_train.csv",
            "/srv/mediswarm/live/RUMC_1/local/MST_unilateral_2026_04_13_162111/run_dir/site_model_gt_and_classprob_validation.csv",
            "Short April RUMC MST run.",
        ),
        RunSource(
            "RUMC_MST_20260218",
            "RUMC",
            "MST",
            "MST",
            "MST_unilateral_2026_02_18_120355",
            "/srv/mediswarm/live/RUMC_1/local/MST_unilateral_2026_02_18_120355/run_dir",
            "/srv/mediswarm/live/RUMC_1/local/MST_unilateral_2026_02_18_120355/run_dir/site_model_gt_and_classprob_train.csv",
            "/srv/mediswarm/live/RUMC_1/local/MST_unilateral_2026_02_18_120355/run_dir/site_model_gt_and_classprob_validation.csv",
            "Earlier February RUMC MST run.",
        ),
    ]


def materialize_source_runs(run_sources: list[RunSource]) -> list[RunSource]:
    """Copy dashboard mirror run directories into the workspace if needed.

    The Cosmos dashboard mirror stores some uploaded checkpoint blobs with
    ``0600 mediswarm-upload`` permissions.  Listing works, but reading those
    blobs as the current user does not.  We keep the mirror untouched and copy
    the relevant run_dir into the workspace with readable ownership.
    """
    materialized: list[RunSource] = []
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    for source in run_sources:
        run_dir = Path(source.run_dir)
        if not str(run_dir).startswith("/srv/mediswarm/live/"):
            materialized.append(source)
            continue
        dest = WORKSPACE / "raw" / source.run_key / "run_dir"
        expected = [
            dest / "site_model_gt_and_classprob_train.csv",
            dest / "site_model_gt_and_classprob_validation.csv",
        ]
        expected.extend(dest.glob("*.ckpt"))
        if not dest.exists() or not expected or not all(path.exists() and os.access(path, os.R_OK) for path in expected):
            dest.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "sudo",
                    "rsync",
                    "-aL",
                    "--delete",
                    "--chmod=Du+rwx,Dgo+rx,Fu+rw,Fgo+r",
                    str(run_dir) + "/",
                    str(dest) + "/",
                ]
            )
            run(["sudo", "chown", "-R", uid_gid, str(dest)])
        copied = RunSource(
            run_key=source.run_key,
            source_site=source.source_site,
            model_name=source.model_name,
            model_short=source.model_short,
            run_id=source.run_id,
            run_dir=str(dest),
            train_csv=str(dest / "site_model_gt_and_classprob_train.csv"),
            val_csv=str(dest / "site_model_gt_and_classprob_validation.csv"),
            note=source.note,
        )
        materialized.append(copied)
    return materialized


def label_from_ckpt(path: Path) -> str:
    name = path.name
    if name == "last.ckpt":
        return "last"
    if name == "last_global_model.ckpt":
        return "last_global"
    match = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt", name)
    if match:
        return f"epoch{match.group(1)}_step{match.group(2)}"
    return path.stem.replace("=", "").replace("-", "_")


def known_snapshots(run_sources: list[RunSource]) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    for source in run_sources:
        run_dir = Path(source.run_dir)
        if source.run_key == "USZ_MST":
            ckpts = [
                (run_dir / "USZ_MST_best.ckpt", "epoch33_best"),
                (run_dir / "USZ_MST_last.ckpt", "last"),
            ]
        elif source.run_key == "USZ_1DC":
            ckpts = [
                (run_dir / "USZ_1DC_epoch14_best.ckpt", "epoch14_best"),
                (run_dir / "USZ_1DC_last.ckpt", "last"),
            ]
        else:
            ckpts = [(path, label_from_ckpt(path)) for path in sorted(run_dir.glob("*.ckpt"))]
        for ckpt_path, label in ckpts:
            if not ckpt_path.exists():
                print(f"WARNING: missing checkpoint: {ckpt_path}", file=sys.stderr)
                continue
            label_safe = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")
            snapshot_id = f"{source.source_site}_{source.model_short}_{label_safe}"
            if source.run_key.startswith("RUMC_MST"):
                snapshot_id = f"{source.run_key}_{label_safe}"
            snapshots.append(
                Snapshot(
                    snapshot_id=snapshot_id,
                    source_site=source.source_site,
                    model_name=source.model_name,
                    model_short=source.model_short,
                    run_key=source.run_key,
                    label=label,
                    path=str(ckpt_path),
                    bytes=ckpt_path.stat().st_size,
                )
            )
    return snapshots


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024 * 8), b""):
            digest.update(block)
    return digest.hexdigest()


def annotate_checksums(snapshots: list[Snapshot]) -> list[Snapshot]:
    seen: dict[tuple[str, str], str] = {}
    for snapshot in snapshots:
        path = Path(snapshot.path)
        snapshot.bytes = path.stat().st_size
        snapshot.sha256 = sha256_file(path)
        key = (snapshot.model_name, snapshot.sha256)
        snapshot.duplicate_of = seen.get(key)
        if snapshot.duplicate_of is None:
            seen[key] = snapshot.snapshot_id
    return snapshots


def write_manifest(run_sources: list[RunSource], snapshots: list[Snapshot]) -> None:
    manifest = {
        "workspace": str(WORKSPACE.relative_to(ROOT)),
        "dl0_stage": str(DL0_STAGE),
        "docker_image": DOCKER_IMAGE,
        "eval_sites": list(EVAL_SITES),
        "run_sources": [asdict(item) for item in run_sources],
        "snapshots": [asdict(item) for item in snapshots],
    }
    with (WORKSPACE / "manifests" / "artifact_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    with (WORKSPACE / "tables" / "checkpoint_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snapshot_id",
                "source_site",
                "model_short",
                "run_key",
                "label",
                "bytes",
                "sha256",
                "duplicate_of",
                "path",
            ],
        )
        writer.writeheader()
        for snapshot in snapshots:
            row = asdict(snapshot)
            row.pop("model_name")
            writer.writerow(row)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def binary_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def macro_auroc(labels: list[int], probabilities: list[list[float]]) -> tuple[float | None, dict[int, float | None]]:
    per_class: dict[int, float | None] = {}
    for cls in CLASSES:
        binary_labels = [1 if label == cls else 0 for label in labels]
        scores = [row[cls] for row in probabilities]
        per_class[cls] = binary_auc(binary_labels, scores)
    valid = [value for value in per_class.values() if value is not None]
    return (sum(valid) / len(valid), per_class) if valid else (None, per_class)


def f1_scores(labels: list[int], predictions: list[int]) -> tuple[float, float, dict[int, float], dict[int, float]]:
    counts = Counter(labels)
    per_class_f1: dict[int, float] = {}
    per_class_recall: dict[int, float] = {}
    for cls in CLASSES:
        tp = sum(1 for label, pred in zip(labels, predictions) if label == cls and pred == cls)
        fp = sum(1 for label, pred in zip(labels, predictions) if label != cls and pred == cls)
        fn = sum(1 for label, pred in zip(labels, predictions) if label == cls and pred != cls)
        denom = 2 * tp + fp + fn
        per_class_f1[cls] = 0.0 if denom == 0 else (2 * tp) / denom
        per_class_recall[cls] = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    macro = sum(per_class_f1.values()) / len(CLASSES)
    total = len(labels)
    weighted = sum(per_class_f1[cls] * counts[cls] for cls in CLASSES) / total if total else 0.0
    return macro, weighted, per_class_f1, per_class_recall


def load_epoch_prediction_csv(path: Path) -> dict[int, tuple[list[int], list[list[float]]]]:
    by_epoch: dict[int, tuple[list[int], list[list[float]]]] = defaultdict(lambda: ([], []))
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or len(row) < 5:
                continue
            try:
                epoch = int(row[0])
                label = int(row[1])
                probabilities = [float(value) for value in row[2:5]]
            except ValueError:
                continue
            by_epoch[epoch][0].append(label)
            by_epoch[epoch][1].append(probabilities)
    return dict(by_epoch)


def epoch_metrics(path: Path) -> list[dict]:
    metrics = []
    for epoch, (labels, probabilities) in sorted(load_epoch_prediction_csv(path).items()):
        predictions = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
        accuracy = sum(1 for label, pred in zip(labels, predictions) if label == pred) / len(labels)
        macro_auc, per_class_auc = macro_auroc(labels, probabilities)
        macro_f1, weighted_f1, per_class_f1, per_class_recall = f1_scores(labels, predictions)
        metrics.append(
            {
                "epoch": epoch,
                "samples": len(labels),
                "accuracy": accuracy,
                "macro_auroc": macro_auc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "class2_auroc": per_class_auc[2],
                "class2_f1": per_class_f1[2],
                "class2_recall": per_class_recall[2],
            }
        )
    return metrics


def write_epoch_metrics(run_sources: list[RunSource]) -> dict[str, dict[str, list[dict]]]:
    all_metrics: dict[str, dict[str, list[dict]]] = {}
    for source in run_sources:
        all_metrics[source.run_key] = {}
        for split, path_str in [("train", source.train_csv), ("validation", source.val_csv)]:
            if not path_str:
                continue
            path = Path(path_str)
            if not path.exists():
                print(f"WARNING: missing {split} CSV for {source.run_key}: {path}", file=sys.stderr)
                continue
            metrics = epoch_metrics(path)
            all_metrics[source.run_key][split] = metrics
            out = WORKSPACE / "training_curves" / f"{source.run_key}_{split}_epoch_metrics.csv"
            with out.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()) if metrics else ["epoch"])
                writer.writeheader()
                writer.writerows(metrics)
        if all_metrics[source.run_key]:
            write_training_svg(source, all_metrics[source.run_key])
    return all_metrics


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "NA"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    value = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{value}" />'


def write_training_svg(source: RunSource, split_metrics: dict[str, list[dict]]) -> None:
    width, height = 920, 440
    left, right, top, bottom = 70, 30, 42, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    rows = []
    for split in ("train", "validation"):
        for row in split_metrics.get(split, []):
            rows.append(row)
    if not rows:
        return
    max_epoch = max(row["epoch"] for row in rows) or 1
    values = [row[key] for row in rows for key in ("accuracy", "class2_auroc") if row.get(key) is not None]
    y_min = max(0.0, min(values) - 0.05) if values else 0.0
    y_max = min(1.0, max(values) + 0.05) if values else 1.0
    if y_max - y_min < 0.1:
        y_max = min(1.0, y_min + 0.1)

    def xy(epoch: int, value: float | None) -> tuple[float, float] | None:
        if value is None or math.isnan(value):
            return None
        x = left + (epoch / max_epoch) * plot_w
        y = top + (1 - (value - y_min) / (y_max - y_min)) * plot_h
        return x, y

    series = [
        ("train", "accuracy", "#2864c9", "train accuracy"),
        ("validation", "accuracy", "#d55e00", "val accuracy"),
        ("train", "class2_auroc", "#00876c", "train class-2 AUROC"),
        ("validation", "class2_auroc", "#7b3294", "val class-2 AUROC"),
    ]
    polylines = []
    legend = []
    for idx, (split, key, color, label) in enumerate(series):
        points = [xy(row["epoch"], row.get(key)) for row in split_metrics.get(split, [])]
        points = [point for point in points if point is not None]
        polylines.append(svg_polyline(points, color))
        lx = left + (idx % 2) * 280
        ly = height - 44 + (idx // 2) * 22
        legend.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{color}" stroke-width="3" />')
        legend.append(f'<text x="{lx+36}" y="{ly+4}" font-size="13">{html.escape(label)}</text>')

    y_ticks = []
    for tick in range(0, 6):
        value = y_min + (y_max - y_min) * tick / 5
        y = top + (1 - (value - y_min) / (y_max - y_min)) * plot_h
        y_ticks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e7e7e7" />')
        y_ticks.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.2f}</text>')
    x_ticks = []
    for tick in range(0, 6):
        epoch = round(max_epoch * tick / 5)
        x = left + (epoch / max_epoch) * plot_w
        x_ticks.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#f2f2f2" />')
        x_ticks.append(f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{epoch}</text>')

    title = f"{source.source_site} {source.model_short} local training: {source.run_id}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{left}" y="24" font-size="18" font-weight="700">{html.escape(title)}</text>
  {''.join(y_ticks)}
  {''.join(x_ticks)}
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>
  {''.join(polylines)}
  <text x="{left + plot_w / 2}" y="{height-18}" text-anchor="middle" font-size="13">epoch</text>
  <text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" text-anchor="middle" font-size="13">metric</text>
  {''.join(legend)}
</svg>
"""
    (FIGURE_DIR / f"{source.run_key}_training_curves.svg").write_text(svg)


def stage_checkpoints(snapshots: list[Snapshot]) -> None:
    remote(f"mkdir -p {DL0_STAGE}/checkpoints {DL0_STAGE}/odelia_eval")
    for snapshot in snapshots:
        local_path = Path(snapshot.path)
        remote_path = f"{DL0_STAGE}/checkpoints/{snapshot.snapshot_id}.ckpt"
        if snapshot.duplicate_of:
            print(f"Skipping stage for duplicate {snapshot.snapshot_id} (duplicate of {snapshot.duplicate_of})")
            continue
        exists = remote(f"test -s {remote_path} && echo yes || echo no", capture=True).stdout.strip()
        if exists == "yes":
            continue
        run(["rsync", "-aL", "--progress", str(local_path), f"{DL0_HOST}:{remote_path}"])


def stage_prediction_helper() -> None:
    remote(f"mkdir -p {DL0_STAGE}/tools")
    run(["rsync", "-aL", str(ROOT / "scripts" / "evaluation" / "predict.py"), f"{DL0_HOST}:{DL0_STAGE}/tools/predict.py"])


def eval_on_dl0(snapshots: list[Snapshot], *, force: bool = False) -> None:
    stage_prediction_helper()
    unique = [snapshot for snapshot in snapshots if not snapshot.duplicate_of]
    by_model: dict[str, list[Snapshot]] = defaultdict(list)
    for snapshot in unique:
        by_model[snapshot.model_name].append(snapshot)

    for eval_site in EVAL_SITES:
        for model_name, model_snapshots in sorted(by_model.items()):
            model_short = "1DC" if model_name == "1DivideAndConquer" else model_name
            out_dir = f"{DL0_STAGE}/odelia_eval/{model_short}/{eval_site}"
            required_snapshot_checks = " && ".join(
                f"grep -q {snapshot.snapshot_id!r} {out_dir}/prediction_results.json"
                for snapshot in model_snapshots
            )
            done = remote(
                f"test -s {out_dir}/prediction_results.json && "
                f"grep -q '\"metrics\"' {out_dir}/prediction_results.json && "
                f"{required_snapshot_checks} && echo yes || echo no",
                capture=True,
            ).stdout.strip()
            if done == "yes" and not force:
                print(f"Skipping existing eval: {model_short} on {eval_site}")
                continue
            checkpoint_args = " ".join(f"/workspace/{snapshot.snapshot_id}.ckpt" for snapshot in model_snapshots)
            command = f"""
set -euo pipefail
mkdir -p {out_dir}
docker run --rm \\
  --gpus=device=0 \\
  --ulimit nofile=65536:65536 \\
  --net=host --ipc=host \\
  -v {DL0_DATA}:/data:ro \\
  -v {DL0_STAGE}/checkpoints:/workspace:ro \\
  -v {DL0_STAGE}/tools/predict.py:/MediSwarm/scripts/evaluation/predict.py:ro \\
  -v {out_dir}:/output \\
  --env SITE_NAME={eval_site} \\
  --env DATA_DIR=/data \\
  --env SCRATCH_DIR=/output \\
  --env MODEL_NAME={model_name} \\
  --env TORCH_HOME=/torch_home \\
  --env CONFIG=unilateral \\
  --env TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_system \\
  {DOCKER_IMAGE} \\
  python3 /MediSwarm/scripts/evaluation/predict.py \\
    --checkpoint {checkpoint_args} \\
    --checkpoint-type lightning \\
    --model-name {model_name} \\
    --output-dir /output \\
    --split test > {out_dir}/predict_stdout.log 2>&1
"""
            remote(command)


def pull_dl0_results() -> None:
    run(["rsync", "-a", f"{DL0_HOST}:{DL0_STAGE}/odelia_eval/", str(WORKSPACE / "odelia_eval") + "/"])


def summarize_prediction_results(snapshots: list[Snapshot]) -> list[dict]:
    by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
    rows: list[dict] = []
    for result_path in sorted((WORKSPACE / "odelia_eval").glob("*/*/prediction_results.json")):
        model_short = result_path.parents[1].name
        eval_site = result_path.parent.name
        with result_path.open() as handle:
            results = json.load(handle)
        for item in results:
            if "metrics" not in item:
                continue
            snapshot_id = Path(item["checkpoint"]).stem
            snapshot = by_id.get(snapshot_id)
            metrics = item["metrics"]
            per_class_auroc = metrics.get("auroc_per_class", {})
            per_class_f1 = metrics.get("per_class_f1", {})
            per_class_recall = metrics.get("per_class_recall", {})
            row = {
                "snapshot_id": snapshot_id,
                "source_site": snapshot.source_site if snapshot else "unknown",
                "model_short": snapshot.model_short if snapshot else model_short,
                "run_key": snapshot.run_key if snapshot else "unknown",
                "label": snapshot.label if snapshot else item.get("kind", ""),
                "eval_site": eval_site,
                "samples": metrics.get("num_samples"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "macro_auroc": metrics.get("auroc_macro"),
                "class0_auroc": per_class_auroc.get("0"),
                "class1_auroc": per_class_auroc.get("1"),
                "class2_auroc": per_class_auroc.get("2"),
                "class2_f1": per_class_f1.get("2"),
                "class2_recall": per_class_recall.get("2"),
            }
            rows.append(row)

    out = WORKSPACE / "tables" / "challenge_summary_metrics.csv"
    fieldnames = [
        "snapshot_id",
        "source_site",
        "model_short",
        "run_key",
        "label",
        "eval_site",
        "samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auroc",
        "class0_auroc",
        "class1_auroc",
        "class2_auroc",
        "class2_f1",
        "class2_recall",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_aggregate_metrics(rows)
    write_challenge_svgs(rows)
    return rows


def weighted_mean(rows: Iterable[dict], key: str) -> float | None:
    total = 0.0
    weight = 0
    for row in rows:
        value = row.get(key)
        if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
            continue
        total += float(value) * int(row["samples"])
        weight += int(row["samples"])
    return total / weight if weight else None


def write_aggregate_metrics(rows: list[dict]) -> list[dict]:
    by_snapshot: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_snapshot[row["snapshot_id"]].append(row)
    aggregates = []
    for snapshot_id, group in sorted(by_snapshot.items()):
        first = group[0]
        item = {
            "snapshot_id": snapshot_id,
            "source_site": first["source_site"],
            "model_short": first["model_short"],
            "run_key": first["run_key"],
            "label": first["label"],
            "samples": sum(int(row["samples"]) for row in group),
        }
        for metric in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_auroc",
            "class2_auroc",
            "class2_f1",
            "class2_recall",
        ]:
            item[metric] = weighted_mean(group, metric)
        aggregates.append(item)

    out = WORKSPACE / "tables" / "challenge_aggregate_metrics.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0].keys()) if aggregates else ["snapshot_id"])
        writer.writeheader()
        writer.writerows(aggregates)
    return aggregates


def write_bar_svg(items: list[dict], metric: str, path: Path, title: str, limit: int = 20) -> None:
    items = [item for item in items if item.get(metric) is not None]
    items = sorted(items, key=lambda item: item[metric], reverse=True)[:limit]
    width = 1060
    bar_h = 24
    gap = 8
    left = 270
    right = 70
    top = 54
    height = top + len(items) * (bar_h + gap) + 40
    plot_w = width - left - right
    max_value = max((item[metric] for item in items), default=1.0) or 1.0
    rows = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    rows.append('<rect width="100%" height="100%" fill="white"/>')
    rows.append(f'<text x="20" y="30" font-size="18" font-weight="700">{html.escape(title)}</text>')
    for idx, item in enumerate(items):
        y = top + idx * (bar_h + gap)
        value = item[metric]
        w = (value / max_value) * plot_w
        label = f"{item['snapshot_id']} ({item['model_short']})"
        rows.append(f'<text x="{left-10}" y="{y+17}" text-anchor="end" font-size="12">{html.escape(label)}</text>')
        rows.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="#386cb0"/>')
        rows.append(f'<text x="{left+w+6:.1f}" y="{y+17}" font-size="12">{value:.3f}</text>')
    rows.append("</svg>\n")
    path.write_text("\n".join(rows))


def write_challenge_svgs(rows: list[dict]) -> None:
    aggregate_path = WORKSPACE / "tables" / "challenge_aggregate_metrics.csv"
    if not aggregate_path.exists():
        return
    with aggregate_path.open(newline="") as handle:
        aggregates = list(csv.DictReader(handle))
    for item in aggregates:
        for key, value in list(item.items()):
            if key in {"snapshot_id", "source_site", "model_short", "run_key", "label"}:
                continue
            item[key] = float(value) if value not in {"", "None"} else None
    write_bar_svg(
        aggregates,
        "class2_auroc",
        FIGURE_DIR / "challenge_aggregate_class2_auroc.svg",
        "ODELIA challenge weighted mean Class-2 AUROC by checkpoint",
    )
    write_bar_svg(
        aggregates,
        "macro_auroc",
        FIGURE_DIR / "challenge_aggregate_macro_auroc.svg",
        "ODELIA challenge weighted mean macro AUROC by checkpoint",
    )


def markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def class_count_text(counts: dict[int, int] | Counter) -> str:
    if not counts:
        return "NA"
    total = sum(int(counts.get(cls, 0)) for cls in CLASSES)
    return f"n={total}; " + ", ".join(f"{cls}={int(counts.get(cls, 0))}" for cls in CLASSES)


def epoch_label_counts(path_str: str | None) -> tuple[Counter, int | None]:
    if not path_str:
        return Counter(), None
    path = Path(path_str)
    if not path.exists():
        return Counter(), None
    by_epoch = load_epoch_prediction_csv(path)
    if not by_epoch:
        return Counter(), None
    epoch = 0 if 0 in by_epoch else min(by_epoch)
    labels, _ = by_epoch[epoch]
    return Counter(labels), epoch


def internal_class_distribution_rows(run_sources: list[RunSource]) -> list[dict]:
    rows: list[dict] = []
    for source in run_sources:
        for split, path_str in [("train", source.train_csv), ("validation", source.val_csv)]:
            counts, epoch = epoch_label_counts(path_str)
            if not counts:
                continue
            rows.append(
                {
                    "source_site": source.source_site,
                    "model_short": source.model_short,
                    "run_key": source.run_key,
                    "run_id": source.run_id,
                    "split": split,
                    "epoch_used": epoch,
                    "samples": sum(counts.values()),
                    "class0": counts.get(0, 0),
                    "class1": counts.get(1, 0),
                    "class2": counts.get(2, 0),
                    "distribution": class_count_text(counts),
                }
            )
        if source.run_key == "RUMC_1DC":
            # The partner-provided console log includes the held-out local test
            # split; only train/validation prediction CSVs are included in the
            # Google Drive zip pair.
            test_counts = Counter({0: 1110, 1: 1, 2: 11})
            rows.append(
                {
                    "source_site": source.source_site,
                    "model_short": source.model_short,
                    "run_key": source.run_key,
                    "run_id": source.run_id,
                    "split": "test",
                    "epoch_used": "",
                    "samples": sum(test_counts.values()),
                    "class0": test_counts.get(0, 0),
                    "class1": test_counts.get(1, 0),
                    "class2": test_counts.get(2, 0),
                    "distribution": class_count_text(test_counts),
                }
            )
    return rows


def external_challenge_class_distribution_rows() -> list[dict]:
    rows: list[dict] = []
    base = WORKSPACE / "odelia_eval"
    for site in EVAL_SITES:
        candidates = sorted(base.glob(f"*/{site}/predictions_*.csv"))
        if not candidates:
            continue
        counts: Counter = Counter()
        with candidates[0].open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    counts[int(row["ground_truth"])] += 1
                except (KeyError, TypeError, ValueError):
                    continue
        rows.append(
            {
                "eval_site": site,
                "samples": sum(counts.values()),
                "class0": counts.get(0, 0),
                "class1": counts.get(1, 0),
                "class2": counts.get(2, 0),
                "distribution": class_count_text(counts),
                "source_file": str(candidates[0].relative_to(ROOT)),
            }
        )
    return rows


def write_class_distribution_tables(run_sources: list[RunSource]) -> tuple[list[dict], list[dict]]:
    internal_rows = internal_class_distribution_rows(run_sources)
    external_rows = external_challenge_class_distribution_rows()

    internal_path = WORKSPACE / "tables" / "internal_class_distribution.csv"
    with internal_path.open("w", newline="") as handle:
        fieldnames = [
            "source_site",
            "model_short",
            "run_key",
            "run_id",
            "split",
            "epoch_used",
            "samples",
            "class0",
            "class1",
            "class2",
            "distribution",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in internal_rows])

    external_path = WORKSPACE / "tables" / "external_challenge_class_distribution.csv"
    with external_path.open("w", newline="") as handle:
        fieldnames = ["eval_site", "samples", "class0", "class1", "class2", "distribution", "source_file"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in external_rows])

    return internal_rows, external_rows


def distribution_lookup(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["run_key"], row["split"]): row for row in rows}


def class_distribution_section(run_sources: list[RunSource]) -> str:
    internal_rows, external_rows = write_class_distribution_tables(run_sources)
    sections = ["\n## Cohort Class Distributions\n"]
    sections.append(
        "Class labels are shown as `0=No lesion`, `1=Benign`, `2=Malignant`. "
        "Internal rows are the first epoch of each local training CSV; the label set is static across epochs, so this is the run's train/validation cohort size. "
        "External rows are read from the ODELIA challenge prediction CSVs and are independent of checkpoint choice.\n"
    )

    if external_rows:
        sections.append("\n### External ODELIA Challenge Test Cohorts\n")
        sections.append(
            markdown_table(
                [[row["eval_site"], row["distribution"]] for row in external_rows],
                ["Challenge site", "Cases by class"],
            )
        )
        total = Counter()
        for row in external_rows:
            for cls in CLASSES:
                total[cls] += int(row[f"class{cls}"])
        sections.append(f"\n\nExternal challenge total: **{class_count_text(total)}**.\n")

    if internal_rows:
        sections.append("\n### Internal Local-Training Cohorts\n")
        sections.append(
            markdown_table(
                [
                    [row["source_site"], row["model_short"], row["run_id"], row["split"], row["distribution"]]
                    for row in internal_rows
                ],
                ["Source", "Model", "Run ID", "Split", "Cases by class"],
            )
        )
        sections.append(
            "\n\nThese internal distributions make the validation AUROC numbers interpretable: several sites have strongly imbalanced class-2 prevalence, so argmax sensitivity can be low even when Class-2 AUROC is useful.\n"
        )

    return "".join(sections)


def latest_training_summary(metrics_by_run: dict[str, dict[str, list[dict]]], run_sources: list[RunSource]) -> list[list[object]]:
    rows = []
    by_key = {source.run_key: source for source in run_sources}
    for run_key, split_metrics in metrics_by_run.items():
        val = split_metrics.get("validation", [])
        train = split_metrics.get("train", [])
        if not val:
            continue
        best_acc = max(val, key=lambda row: row["accuracy"])
        best_c2 = max((row for row in val if row["class2_auroc"] is not None), key=lambda row: row["class2_auroc"], default=None)
        last = val[-1]
        source = by_key[run_key]
        rows.append(
            [
                source.source_site,
                source.model_short,
                source.run_id,
                len(train) if train else "NA",
                f"e{best_acc['epoch']} / {fmt(best_acc['accuracy'])}",
                f"e{best_c2['epoch']} / {fmt(best_c2['class2_auroc'])}" if best_c2 else "NA",
                f"e{last['epoch']} / {fmt(last['accuracy'])} / {fmt(last['class2_auroc'])}",
            ]
        )
    return rows


def metric_by_epoch(metrics_by_run: dict[str, dict[str, list[dict]]], run_key: str, split: str, epoch: int) -> dict | None:
    for row in metrics_by_run.get(run_key, {}).get(split, []):
        if int(row["epoch"]) == epoch:
            return row
    return None


def validation_metric_for_label(metrics_by_run: dict[str, dict[str, list[dict]]], run_key: str, label: str) -> dict | None:
    val = metrics_by_run.get(run_key, {}).get("validation", [])
    if not val:
        return None
    if label in {"last", "last_global"}:
        return val[-1]
    match = re.search(r"epoch(\d+)", label)
    if not match:
        return None
    return metric_by_epoch(metrics_by_run, run_key, "validation", int(match.group(1)))


def source_model_selection_rows(aggregate_rows: list[dict], metrics_by_run: dict[str, dict[str, list[dict]]]) -> list[list[object]]:
    selected: dict[tuple[str, str], dict] = {}
    for row in aggregate_rows:
        key = (row["source_site"], row["model_short"])
        current = selected.get(key)
        if current is None or (row.get("class2_auroc") or 0.0) > (current.get("class2_auroc") or 0.0):
            selected[key] = row

    rows = []
    for row in sorted(selected.values(), key=lambda item: item.get("class2_auroc") or 0.0, reverse=True):
        internal = validation_metric_for_label(metrics_by_run, row["run_key"], row["label"])
        rows.append(
            [
                row["source_site"],
                row["model_short"],
                row["snapshot_id"],
                fmt(internal["class2_auroc"]) if internal else "NA",
                fmt(internal["accuracy"]) if internal else "NA",
                fmt(row["class2_auroc"]),
                fmt(row["macro_auroc"]),
                fmt(row["accuracy"]),
                fmt(row["class2_recall"]),
            ]
        )
    return rows


def model_family_counts(run_sources: list[RunSource]) -> str:
    counts = Counter(source.model_short for source in run_sources)
    return ", ".join(f"{model}={counts[model]}" for model in sorted(counts))


def run_training_status(epoch_count: int) -> str:
    if epoch_count >= 90:
        return f"complete ({epoch_count} epochs)"
    if epoch_count > 1:
        return f"short run ({epoch_count} epochs)"
    return "single-epoch artifact"


def external_result_readout(row: dict | None) -> str:
    if not row or row.get("class2_auroc") is None:
        return "not evaluated externally"
    class2 = float(row["class2_auroc"])
    recall = float(row.get("class2_recall") or 0.0)
    if class2 >= 0.80:
        strength = "strong external transfer"
    elif class2 >= 0.70:
        strength = "useful external transfer"
    elif class2 >= 0.60:
        strength = "modest external transfer"
    else:
        strength = "weak external transfer"
    if recall == 0.0:
        return f"{strength}; ranks better than argmax behavior (Class-2 recall 0)"
    if recall < 0.20:
        return f"{strength}; low Class-2 recall at default argmax"
    return strength


def local_training_review_rows(
    run_sources: list[RunSource],
    snapshots: list[Snapshot],
    metrics_by_run: dict[str, dict[str, list[dict]]],
    aggregate_rows: list[dict],
) -> list[list[object]]:
    aggregate_by_run: dict[str, list[dict]] = defaultdict(list)
    for row in aggregate_rows:
        aggregate_by_run[row["run_key"]].append(row)
    unique_snapshot_counts = Counter(snapshot.run_key for snapshot in snapshots if not snapshot.duplicate_of)
    alias_counts = Counter(snapshot.run_key for snapshot in snapshots if snapshot.duplicate_of)

    rows: list[list[object]] = []
    for source in run_sources:
        val = metrics_by_run.get(source.run_key, {}).get("validation", [])
        if not val:
            continue
        train = metrics_by_run.get(source.run_key, {}).get("train", [])
        best_acc = max(val, key=lambda row: row["accuracy"])
        best_c2 = max((row for row in val if row["class2_auroc"] is not None), key=lambda row: row["class2_auroc"], default=None)
        external_rows = sorted(
            aggregate_by_run.get(source.run_key, []),
            key=lambda row: row.get("class2_auroc") or 0.0,
            reverse=True,
        )
        selected = external_rows[0] if external_rows else None
        ckpt_text = f"{unique_snapshot_counts[source.run_key]} unique"
        if alias_counts[source.run_key]:
            ckpt_text += f" (+{alias_counts[source.run_key]} duplicate alias)"
        rows.append(
            [
                source.source_site,
                source.model_short,
                source.run_id,
                run_training_status(len(val)),
                ckpt_text,
                f"best ACC e{best_acc['epoch']}={fmt(best_acc['accuracy'])}; "
                + (f"best C2 e{best_c2['epoch']}={fmt(best_c2['class2_auroc'])}" if best_c2 else "best C2 NA"),
                f"{selected['snapshot_id']} / C2 {fmt(selected['class2_auroc'])} / macro {fmt(selected['macro_auroc'])} / recall {fmt(selected['class2_recall'])}"
                if selected
                else "NA",
                external_result_readout(selected),
            ]
        )
    return rows


def local_training_review_section(
    run_sources: list[RunSource],
    snapshots: list[Snapshot],
    metrics_by_run: dict[str, dict[str, list[dict]]],
    aggregate_rows: list[dict],
) -> str:
    sites = sorted({source.source_site for source in run_sources})
    unique_count = sum(1 for snapshot in snapshots if not snapshot.duplicate_of)
    duplicate_aliases = sum(1 for snapshot in snapshots if snapshot.duplicate_of)
    sections = ["\n## Local Training Completion Review\n"]
    sections.append(
        f"Available local-training artifacts cover **{len(sites)} source sites** "
        f"({', '.join(sites)}), **{len(run_sources)} run artifacts**, and **{model_family_counts(run_sources)}**. "
        f"They contribute **{unique_count} unique checkpoints** to external ODELIA challenge evaluation"
        f"{f' plus {duplicate_aliases} exact duplicate aliases' if duplicate_aliases else ''}. "
        + (
            "If the target list is the full 8-site swarm participant set, **VHIO local-training results are still missing from this report**. "
            if "VHIO" not in sites
            else ""
        )
        + "All rows below have local train/validation prediction CSVs and at least one retained checkpoint; `short run` means the artifact is usable but is not a full 100-epoch local-training run.\n"
    )
    sections.append(
        markdown_table(
            local_training_review_rows(run_sources, snapshots, metrics_by_run, aggregate_rows),
            [
                "Source",
                "Model",
                "Run ID",
                "Training status",
                "Retained ckpts",
                "Internal validation result",
                "Best external ODELIA result",
                "Readout",
            ],
        )
    )
    sections.append(
        "\n\nMain pattern: 1DC transfers best externally (UKA, USZ, MHA are the strongest rows), while MST runs trained on very Class-2-sparse sites often show usable AUROC/ranking but poor default argmax Class-2 recall. "
        "Internal validation is useful for overfitting and checkpoint-selection diagnosis, but it does not reliably rank external challenge transfer.\n"
    )
    return "".join(sections)


def per_source_training_sections(
    run_sources: list[RunSource],
    metrics_by_run: dict[str, dict[str, list[dict]]],
    aggregate_rows: list[dict],
) -> str:
    internal_rows, _ = write_class_distribution_tables(run_sources)
    dist_by_run = distribution_lookup(internal_rows)
    aggregate_by_run: dict[str, list[dict]] = defaultdict(list)
    for row in aggregate_rows:
        aggregate_by_run[row["run_key"]].append(row)

    sections = ["\n## Per-Source Local Training and External Transfer\n"]
    sections.append(
        "Each source run below uses the same fields: internal train/validation class counts, internal validation checkpoint behavior, "
        "the externally strongest retained checkpoint on the ODELIA challenge cohort, the full retained-checkpoint external table, and the local training curve. "
        "This is the comparable section; site-specific deployment notes are kept separately.\n"
    )

    for source in run_sources:
        val = metrics_by_run.get(source.run_key, {}).get("validation", [])
        train = metrics_by_run.get(source.run_key, {}).get("train", [])
        if not val:
            continue
        train_dist = dist_by_run.get((source.run_key, "train"), {})
        val_dist = dist_by_run.get((source.run_key, "validation"), {})
        best_acc = max(val, key=lambda row: row["accuracy"])
        best_c2 = max((row for row in val if row["class2_auroc"] is not None), key=lambda row: row["class2_auroc"], default=None)
        last_val = val[-1]
        last_train = train[-1] if train else None
        external_rows = sorted(
            aggregate_by_run.get(source.run_key, []),
            key=lambda row: row.get("class2_auroc") or 0.0,
            reverse=True,
        )
        selected = external_rows[0] if external_rows else None

        sections.append(f"\n### {source.source_site} {source.model_short} — `{source.run_id}`\n")
        if source.note:
            sections.append(f"{source.note}\n\n")
        summary_rows = [
            ["Internal train cohort", train_dist.get("distribution", "NA")],
            ["Internal validation cohort", val_dist.get("distribution", "NA")],
            ["Internal best val ACC", f"epoch {best_acc['epoch']} / ACC {fmt(best_acc['accuracy'])} / C2 AUROC {fmt(best_acc['class2_auroc'])}"],
            [
                "Internal best val Class-2 AUROC",
                f"epoch {best_c2['epoch']} / C2 AUROC {fmt(best_c2['class2_auroc'])} / ACC {fmt(best_c2['accuracy'])}" if best_c2 else "NA",
            ],
            ["Internal last validation", f"epoch {last_val['epoch']} / ACC {fmt(last_val['accuracy'])} / C2 AUROC {fmt(last_val['class2_auroc'])}"],
        ]
        if last_train:
            summary_rows.append(
                [
                    "Internal last train",
                    f"epoch {last_train['epoch']} / ACC {fmt(last_train['accuracy'])} / C2 AUROC {fmt(last_train['class2_auroc'])}",
                ]
            )
        if selected:
            internal = validation_metric_for_label(metrics_by_run, selected["run_key"], selected["label"])
            summary_rows.append(
                [
                    "Externally strongest retained checkpoint",
                    f"`{selected['snapshot_id']}` / external C2 AUROC {fmt(selected['class2_auroc'])} / "
                    f"external macro AUROC {fmt(selected['macro_auroc'])} / internal C2 AUROC {fmt(internal['class2_auroc']) if internal else 'NA'}",
                ]
            )
        sections.append(markdown_table(summary_rows, ["Field", "Value"]))

        svg = FIGURE_DIR / f"{source.run_key}_training_curves.svg"
        if svg.exists():
            sections.append(
                f"\n\n![{source.run_key} local training curves]({svg.relative_to(REPORT_PATH.parent)})\n"
            )
        lightning_svg = FIGURE_DIR / f"{source.run_key}_lightning_training_curves.svg"
        if lightning_svg.exists():
            sections.append(
                f"\n\n![{source.run_key} Lightning-log training curves]({lightning_svg.relative_to(REPORT_PATH.parent)})\n"
            )

        if external_rows:
            sections.append("\nRetained checkpoints on external ODELIA challenge:\n")
            sections.append(
                markdown_table(
                    [
                        [
                            row["snapshot_id"],
                            row["label"],
                            fmt(validation_metric_for_label(metrics_by_run, row["run_key"], row["label"])["class2_auroc"])
                            if validation_metric_for_label(metrics_by_run, row["run_key"], row["label"])
                            else "NA",
                            fmt(validation_metric_for_label(metrics_by_run, row["run_key"], row["label"])["accuracy"])
                            if validation_metric_for_label(metrics_by_run, row["run_key"], row["label"])
                            else "NA",
                            fmt(row["class2_auroc"]),
                            fmt(row["macro_auroc"]),
                            fmt(row["accuracy"]),
                            fmt(row["class2_recall"]),
                        ]
                        for row in external_rows
                    ],
                    [
                        "Snapshot",
                        "Label",
                        "Internal val C2 AUROC",
                        "Internal val ACC",
                        "External C2 AUROC",
                        "External macro AUROC",
                        "External ACC",
                        "External C2 recall",
                    ],
                )
            )
            sections.append("\n")
        else:
            sections.append("\nExternal challenge evaluation rows were not found for this run.\n")

    return "".join(sections)


def validation_framing_section() -> str:
    return (
        "\n## Validation Framing\n"
        "This report intentionally separates two different validation regimes:\n\n"
        "- **Internal validation** means metrics computed during the source training run on that run's local validation split. "
        "These rows answer whether a checkpoint learned its own site's distribution and are reported as validation ACC, macro AUROC, and Class-2 AUROC from `site_model_gt_and_classprob_validation.csv`.\n"
        "- **External validation** means checkpoint inference on held-out institutions that were not used to train that checkpoint. "
        "The main external endpoint here is the ODELIA challenge test data on `dd-dl0:/mnt/dlhd0/medswarmdata` across CAM/MHA/RSH/RUMC/UKA/UMCU.\n"
        "- **Reference swarm packages** (`CHALLENGE_SWARM_LOCAL_TEST_REPORT_20260513.md` and `OLE_SWARM_EVALUATION_ARTIFACT_REPORT.md`) are artifact/validation-audit context. "
        "They are useful for understanding available swarm models and internal validation behavior, but they are not the same endpoint as the single-site checkpoint external challenge evaluation unless explicitly stated.\n"
    )


def reference_swarm_context_section() -> str:
    return (
        "\n## Reference Swarm/Artifact Context\n"
        "Two source reports are appended verbatim at the end of this document so the single-site checkpoint findings can be read against the existing swarm-artifact evidence.\n\n"
        "### Challenge Swarm/Local Package, 2026-05-13\n"
        "The challenge-swarm package is a six-model swarm/local artifact audit with complete final artifacts across CAM_1, MHA_1, RSH_1, RUMC_1, UKA_1, and UMCU_1. "
        "Its metrics are **internal validation streams**, not the external checkpoint-on-challenge-test endpoint used for the single-site comparison above.\n\n"
        + markdown_table(
            [
                ["MST", "0.775 @ 104", "0.801 @ 159", "complete artifacts"],
                ["1DivideAndConquer", "0.811 @ 128", "0.870 @ 120", "complete artifacts"],
                ["2BCN_AIM", "0.760 @ 144", "0.775 @ 109", "complete artifacts"],
                ["3agaldran", "0.830 @ 152", "0.902 @ 180", "complete artifacts"],
                ["4LME_ABMIL", "0.900 @ 40", "0.910 @ 140", "complete artifacts"],
                ["5Pimed", "0.789 @ 180", "0.866 @ 145", "complete artifacts after retry"],
            ],
            ["Model", "Best aggregated val AUROC", "Best site val AUROC", "Status"],
        )
        + "\n\n### OLE/Duke Swarm Package\n"
        "The OLE/Duke package is a three-node Duke swarm artifact audit. It is retained as source context because it documents the older Duke swarm result, including the important caveats around train/validation/test UID overlap warnings and failed global-best selection.\n"
    )


def usz_detailed_section(metrics_by_run: dict[str, dict[str, list[dict]]], aggregate_rows: list[dict]) -> str:
    parts = ["\n## USZ Partner Supplement\n"]
    parts.append(
        "Comparable USZ training, class-distribution, curve, and external ODELIA challenge fields are included in the per-source section above. "
        "This supplement records USZ-specific deployment/data-hygiene notes and the supplemental Duke cross-evaluation.\n"
    )
    data_summary_path = ROOT / "workspace" / "usz_partner_eval" / "usz_data_summary.json"
    if data_summary_path.exists():
        data_summary = json.loads(data_summary_path.read_text())
        lesion_counts = data_summary.get("annotation_lesion_distribution", {})
        parts.append(
            "USZ `Data_all/USZ_1` contains "
            f"**{data_summary.get('annotation_unique_uids', 'NA')} annotated unilateral UIDs** and "
            f"**{data_summary.get('split_unique_uids', 'NA')} split UIDs**. "
            f"Class distribution is 0={lesion_counts.get('0', 'NA')}, "
            f"1={lesion_counts.get('1', 'NA')}, 2={lesion_counts.get('2', 'NA')} "
            "(Class 2/Malignant is about 9%). "
            f"All split UIDs have image data (`split_uids_missing_image={data_summary.get('split_uids_missing_image', 'NA')}`); "
            f"{data_summary.get('in_annotation_but_not_split', 'NA')} annotated UIDs are excluded by the split and "
            f"{data_summary.get('data_dir_dirs_unrelated_to_split', 'NA')} image directories are unused by this fold.\n"
        )
    parts.append(
        "The trainer plausibility audit reported no duplicate UID or split-overlap errors. "
        "The remaining warnings were non-blocking: 4 byte-identical image-data groups, annotation/split/image set drift, "
        "and unused image directories from older preprocessing output.\n"
    )

    duke_path = ROOT / "workspace" / "usz_partner_eval" / "cross_eval_duke" / "prediction_results.json"
    if duke_path.exists():
        duke_rows = []
        for result in json.loads(duke_path.read_text()):
            metrics = result.get("metrics", {})
            c2_auc = (metrics.get("auroc_per_class") or {}).get("2")
            c2_recall = (metrics.get("per_class_recall") or {}).get("2")
            c2_f1 = (metrics.get("per_class_f1") or {}).get("2")
            duke_rows.append(
                [
                    result.get("site", "NA"),
                    result.get("kind", "NA"),
                    metrics.get("num_samples", "NA"),
                    fmt(metrics.get("accuracy")),
                    fmt(c2_auc),
                    fmt(c2_recall),
                    fmt(c2_f1),
                ]
            )
        if duke_rows:
            parts.append("\n### Supplemental USZ MST -> Duke Held-Out Test\n")
            parts.append(
                markdown_table(
                    duke_rows,
                    ["Snapshot", "Kind", "Samples", "ACC", "Class-2 AUROC", "Class-2 Recall", "Class-2 F1"],
                )
            )
            parts.append(
                "\n\nThese Duke numbers are supplemental binary 0-vs-2 cross-evaluation results for the USZ-trained MST checkpoints; Duke has no true class-1 labels in this slice.\n"
            )

    parts.append("\n### USZ Data and Output Footprint\n")
    parts.append(
        "- `Data_all/USZ_1` is about **114 GB** total; the unilateral training directory is about **17 GB** / 11,657 `Sub_1.nii.gz` files.\n"
        "- Each fold-0 100-epoch local-training run reads about **6.4 GB unique** training+validation data and roughly **640 GB logical epoch I/O**, mostly served from OS page cache after the first pass.\n"
        "- MST writes about **840 MB** of run output; 1DC writes about **3.1 GB** because each checkpoint is about 1.1 GB.\n"
        "- USZ artifacts are under `workspace/usz_partner_eval/`, and the unified challenge-eval outputs are under `workspace/odelia_single_site_eval/`.\n"
    )
    return "".join(parts)


def generate_report(run_sources: list[RunSource], snapshots: list[Snapshot], metrics_by_run: dict[str, dict[str, list[dict]]]) -> None:
    challenge_rows = load_csv_dicts(WORKSPACE / "tables" / "challenge_summary_metrics.csv")
    aggregate_rows = load_csv_dicts(WORKSPACE / "tables" / "challenge_aggregate_metrics.csv")
    inventory_rows = load_csv_dicts(WORKSPACE / "tables" / "checkpoint_inventory.csv")
    ole = OLE_REPORT_PATH.read_text() if OLE_REPORT_PATH.exists() else "_OLE source report not found._\n"
    challenge_swarm = CHALLENGE_SWARM_REPORT_PATH.read_text() if CHALLENGE_SWARM_REPORT_PATH.exists() else "_Challenge swarm/local source report not found._\n"

    for row in aggregate_rows:
        for key in ["samples", "accuracy", "macro_f1", "weighted_f1", "macro_auroc", "class2_auroc", "class2_f1", "class2_recall"]:
            if key in row and row[key] != "":
                row[key] = float(row[key])

    top_class2 = sorted(
        [row for row in aggregate_rows if row.get("class2_auroc") is not None],
        key=lambda row: row["class2_auroc"],
        reverse=True,
    )[:12]
    top_macro = sorted(
        [row for row in aggregate_rows if row.get("macro_auroc") is not None],
        key=lambda row: row["macro_auroc"],
        reverse=True,
    )[:12]

    duplicate_rows = [row for row in inventory_rows if row.get("duplicate_of")]
    unique_snapshots = [snapshot for snapshot in snapshots if not snapshot.duplicate_of]
    eval_count = len(challenge_rows)

    sections = []
    sections.append("# ODELIA Single-Site Checkpoint Challenge Evaluation Report\n")
    sections.append(
        "> Status: generated combined report. The challenge swarm/local and OLE swarm artifact reports are appended verbatim at the end as source context.\n"
    )
    sections.append("## Executive Summary\n")
    if aggregate_rows:
        best = top_class2[0]
        source_names = ", ".join(sorted({source.source_site for source in run_sources}))
        sections.append(
            f"- Evaluated **{len(unique_snapshots)} unique checkpoints** from **{len(run_sources)} local-training runs** across {source_names} on the six-institution ODELIA challenge set.\n"
        )
        sections.append(
            f"- Best weighted mean Class-2 (Malignant) AUROC so far: **{fmt(best['class2_auroc'])}** from `{best['snapshot_id']}`.\n"
        )
        selected_rows = source_model_selection_rows(aggregate_rows, metrics_by_run)
        if selected_rows:
            sections.append(
                "- For presentation-level comparison, this report also selects **one checkpoint per training-source/model family** by external weighted Class-2 AUROC; the condensed report uses that collapsed view.\n"
            )
        sections.append(
            "- Results are weighted by ODELIA challenge site sample count when aggregating across CAM/MHA/RSH/RUMC/UKA/UMCU.\n"
        )
        if PARTNER_WORKBOOK_PATH.exists():
            sections.append(
                f"- Partner-shareable workbook for Google Sheets import: [{PARTNER_WORKBOOK_PATH.relative_to(ROOT)}]({PARTNER_WORKBOOK_PATH.relative_to(REPORT_PATH.parent)}).\n"
            )
    else:
        sections.append("- Checkpoint inventory and local training curves are prepared; ODELIA challenge evaluations have not completed yet.\n")
    sections.append(
        "- UKA artifacts were supplied as six zip files; one timestamp triplet (`20260520T091501Z`) was extracted, while the second triplet appears to be a duplicate download and remains untouched.\n"
    )
    if duplicate_rows:
        sections.append(
            f"- Exact checkpoint duplicates were detected and not re-evaluated separately: **{len(duplicate_rows)} duplicate snapshot aliases**.\n"
        )
    sections.append("- Internal validation AUROC and external challenge AUROC are reported separately throughout; they answer different questions and should not be read as the same endpoint.\n")

    sections.append(validation_framing_section())
    sections.append(class_distribution_section(run_sources))
    sections.append(local_training_review_section(run_sources, snapshots, metrics_by_run, aggregate_rows))

    if aggregate_rows:
        sections.append("\n## Condensed Selection: One Checkpoint per Source Model\n")
        sections.append(
            markdown_table(
                source_model_selection_rows(aggregate_rows, metrics_by_run),
                [
                    "Source",
                    "Model",
                    "Selected checkpoint",
                    "Internal val C2 AUROC",
                    "Internal val ACC",
                    "External C2 AUROC",
                    "External macro AUROC",
                    "External ACC",
                    "External C2 recall",
                ],
            )
        )
        sections.append(
            "\n\nSelection is by external ODELIA challenge weighted Class-2 AUROC within each `(training source, model family)` group. "
            "The internal validation columns show the same persisted checkpoint's validation metrics when the checkpoint epoch can be mapped back to the local training CSV.\n"
        )

    sections.append("\n## Artifact Inventory\n")
    sections.append(
        markdown_table(
            [
                [
                    row["snapshot_id"],
                    row["source_site"],
                    row["model_short"],
                    row["run_key"],
                    row["label"],
                    fmt_size(int(row["bytes"])) if row.get("bytes") else "NA",
                    row["duplicate_of"] or "",
                ]
                for row in inventory_rows
            ],
            ["Snapshot", "Source", "Model", "Run", "Label", "Size", "Duplicate of"],
        )
    )
    sections.append("\n\n## Local Training Curves\n")
    sections.append(
        markdown_table(
            latest_training_summary(metrics_by_run, run_sources),
            ["Source", "Model", "Run ID", "Train epochs", "Best val ACC", "Best val Class-2 AUROC", "Last val ACC / C2 AUROC"],
        )
    )
    sections.append("\n\nTraining curve SVGs are generated under `docs/figures/odelia_single_site_eval/`:\n")
    for source in run_sources:
        svg = FIGURE_DIR / f"{source.run_key}_training_curves.svg"
        if svg.exists():
            sections.append(f"- [{source.run_key} training curves]({svg.relative_to(REPORT_PATH.parent)})\n")
        lightning_svg = FIGURE_DIR / f"{source.run_key}_lightning_training_curves.svg"
        if lightning_svg.exists():
            sections.append(f"- [{source.run_key} Lightning-log training curves]({lightning_svg.relative_to(REPORT_PATH.parent)})\n")

    sections.append(per_source_training_sections(run_sources, metrics_by_run, aggregate_rows))

    sections.append("\n## ODELIA Challenge Evaluation\n")
    if aggregate_rows:
        sections.append(
            f"Completed per-site checkpoint evaluations: **{eval_count}** rows in `workspace/odelia_single_site_eval/tables/challenge_summary_metrics.csv`.\n\n"
        )
        sections.append("### Top Checkpoints by Weighted Mean Class-2 AUROC\n")
        sections.append(
            markdown_table(
                [
                    [
                        row["snapshot_id"],
                        row["source_site"],
                        row["model_short"],
                        fmt(row["class2_auroc"]),
                        fmt(row["macro_auroc"]),
                        fmt(row["accuracy"]),
                        fmt(row["class2_recall"]),
                    ]
                    for row in top_class2
                ],
                ["Snapshot", "Source", "Model", "Class-2 AUROC", "Macro AUROC", "Accuracy", "Class-2 Recall"],
            )
        )
        sections.append("\n\n### Top Checkpoints by Weighted Mean Macro AUROC\n")
        sections.append(
            markdown_table(
                [
                    [
                        row["snapshot_id"],
                        row["source_site"],
                        row["model_short"],
                        fmt(row["macro_auroc"]),
                        fmt(row["class2_auroc"]),
                        fmt(row["accuracy"]),
                    ]
                    for row in top_macro
                ],
                ["Snapshot", "Source", "Model", "Macro AUROC", "Class-2 AUROC", "Accuracy"],
            )
        )
        sections.append(
            "\n\n![Challenge Class-2 AUROC](figures/odelia_single_site_eval/challenge_aggregate_class2_auroc.svg)\n\n"
        )
        sections.append(
            "![Challenge Macro AUROC](figures/odelia_single_site_eval/challenge_aggregate_macro_auroc.svg)\n"
        )
    else:
        sections.append("Challenge evaluation has not completed yet.\n")

    sections.append(usz_detailed_section(metrics_by_run, aggregate_rows))

    sections.append(reference_swarm_context_section())

    sections.append("\n## Methods\n")
    sections.append(
        "- Checkpoints are loaded as PyTorch Lightning `.ckpt` files through `scripts/evaluation/predict.py --checkpoint-type lightning`.\n"
    )
    sections.append(
        f"- Inference runs on `dd-dl0` with Docker image `{DOCKER_IMAGE}` and challenge data root `{DL0_DATA}`.\n"
    )
    sections.append(
        "- Evaluated ODELIA challenge target sites: `CAM`, `MHA`, `RSH`, `RUMC`, `UKA`, `UMCU`.\n"
    )
    sections.append(
        "- Training curves are computed from `site_model_gt_and_classprob_{train,validation}.csv`, whose rows are `epoch, ground_truth, prob_class_0, prob_class_1, prob_class_2`.\n"
    )
    sections.append(
        "- Class-distribution tables are written to `workspace/odelia_single_site_eval/tables/internal_class_distribution.csv` and `workspace/odelia_single_site_eval/tables/external_challenge_class_distribution.csv`.\n"
    )
    if PARTNER_WORKBOOK_PATH.exists():
        sections.append(
            f"- Partner workbook for Google Sheets import: [{PARTNER_WORKBOOK_PATH.relative_to(ROOT)}]({PARTNER_WORKBOOK_PATH.relative_to(REPORT_PATH.parent)}).\n"
        )
    sections.append("\n## Open Items\n")
    sections.append("- Confirm whether any unavailable intermediate checkpoints (for example an epoch-36 USZ 1DC checkpoint) were retained elsewhere; the USZ run currently exposes epoch-14 best and last only.\n")
    sections.append("- Decide whether exact `last.ckpt` / `last_global_model.ckpt` duplicates should be kept as aliases in the final table or collapsed entirely.\n")

    sections.append("\n---\n\n# Appended Source Context: ODELIA Challenge Swarm/Local Artifact Report\n\n")
    sections.append(challenge_swarm)
    sections.append("\n\n---\n\n# Appended Source Context: ODELIA `Ole_swarm` Evaluation Artifact Report\n\n")
    sections.append(ole)
    REPORT_PATH.write_text("".join(sections))


def generate_condensed_report(run_sources: list[RunSource], snapshots: list[Snapshot], metrics_by_run: dict[str, dict[str, list[dict]]]) -> None:
    challenge_rows = load_csv_dicts(WORKSPACE / "tables" / "challenge_summary_metrics.csv")
    aggregate_rows = load_csv_dicts(WORKSPACE / "tables" / "challenge_aggregate_metrics.csv")
    for row in aggregate_rows:
        for key in ["samples", "accuracy", "macro_f1", "weighted_f1", "macro_auroc", "class2_auroc", "class2_f1", "class2_recall"]:
            if key in row and row[key] != "":
                row[key] = float(row[key])

    top_class2 = sorted(
        [row for row in aggregate_rows if row.get("class2_auroc") is not None],
        key=lambda row: row["class2_auroc"],
        reverse=True,
    )
    selected_rows = source_model_selection_rows(aggregate_rows, metrics_by_run)

    sections = []
    sections.append("# ODELIA Checkpoint Evaluation: Condensed Summary\n")
    sections.append(
        "> Presentation-oriented summary. It collapses multiple checkpoints for the same training-source/model family to one externally strongest checkpoint and keeps detailed provenance in the full report.\n"
    )
    sections.append("\n## Storyline\n")
    if top_class2:
        best = top_class2[0]
        sections.append(
            f"- The strongest single-site checkpoint on external ODELIA challenge Class-2/Malignant AUROC is **{best['snapshot_id']}** with weighted Class-2 AUROC **{fmt(best['class2_auroc'])}**.\n"
        )
    sections.append(
        "- **Internal validation** is the source site's local validation split during training; it is useful for checkpoint selection and overfitting diagnosis.\n"
        "- **External validation** is held-out ODELIA challenge inference across CAM/MHA/RSH/RUMC/UKA/UMCU; this is the main transfer/generalisation endpoint.\n"
        "- Internal validation and external challenge performance are related but not interchangeable: some high internal-val checkpoints transfer poorly, and some later checkpoints trade AUROC for recall/specificity differently.\n"
        "- The 2026-05-13 challenge swarm/local package is included as reference internal-validation context for six swarm-trained models; it is not the same endpoint as the single-site external challenge evaluation below.\n"
    )

    internal_dist_rows, external_dist_rows = write_class_distribution_tables(run_sources)
    if external_dist_rows or internal_dist_rows:
        sections.append("\n## Cohort Distributions\n")
        if external_dist_rows:
            sections.append("External ODELIA challenge cohorts (cases by class):\n")
            sections.append(
                markdown_table(
                    [[row["eval_site"], row["distribution"]] for row in external_dist_rows],
                    ["Challenge site", "Cases"],
                )
            )
            sections.append("\n\n")
        validation_dist_rows = [row for row in internal_dist_rows if row["split"] == "validation"]
        if validation_dist_rows:
            sections.append("Internal validation cohorts used for checkpoint selection:\n")
            sections.append(
                markdown_table(
                    [
                        [row["source_site"], row["model_short"], row["run_id"], row["distribution"]]
                        for row in validation_dist_rows
                    ],
                    ["Source", "Model", "Run ID", "Validation cases"],
                )
            )
            sections.append("\n")

    sites = sorted({source.source_site for source in run_sources})
    sections.append("\n## Local Training Coverage\n")
    sections.append(
        f"Completed/available local-training artifacts cover **{len(sites)} source sites** "
        f"({', '.join(sites)}), **{len(run_sources)} run artifacts**, and **{model_family_counts(run_sources)}**. "
        + (
            "If the target list is the full 8-site swarm participant set, **VHIO local-training results are still missing from this report**. "
            if "VHIO" not in sites
            else ""
        )
        + "The full report has the per-run completion table; the selected-checkpoint table below keeps one externally strongest checkpoint per source/model family.\n"
    )

    sections.append("\n## Selected Single-Site Checkpoints\n")
    sections.append(
        markdown_table(
            selected_rows,
            [
                "Source",
                "Model",
                "Selected checkpoint",
                "Internal val C2 AUROC",
                "Internal val ACC",
                "External C2 AUROC",
                "External macro AUROC",
                "External ACC",
                "External C2 recall",
            ],
        )
    )
    sections.append(
        "\n\nSelection rule: one checkpoint per `(training source, model family)`, choosing the highest external ODELIA challenge weighted Class-2 AUROC. "
        "This removes repeated checkpoint variants from the presentation view while preserving the main comparison.\n"
    )

    if top_class2:
        sections.append("\n## Top External Checkpoints Overall\n")
        sections.append(
            markdown_table(
                [
                    [
                        row["snapshot_id"],
                        row["source_site"],
                        row["model_short"],
                        fmt(row["class2_auroc"]),
                        fmt(row["macro_auroc"]),
                        fmt(row["accuracy"]),
                        fmt(row["class2_recall"]),
                    ]
                    for row in top_class2[:8]
                ],
                ["Checkpoint", "Source", "Model", "External C2 AUROC", "External macro AUROC", "External ACC", "External C2 recall"],
            )
        )
        sections.append("\n")

    sections.append("\n## Internal Validation Summary\n")
    sections.append(
        markdown_table(
            latest_training_summary(metrics_by_run, run_sources),
            ["Source", "Model", "Run ID", "Train epochs", "Best val ACC", "Best val Class-2 AUROC", "Last val ACC / C2 AUROC"],
        )
    )
    sections.append(
        "\n\nInternal validation curves are available as SVGs under `docs/figures/odelia_single_site_eval/`. "
        "The most relevant plots for presentation are the per-run training curves plus the aggregate Class-2 AUROC bar chart.\n\n"
        "![External Class-2 AUROC](figures/odelia_single_site_eval/challenge_aggregate_class2_auroc.svg)\n"
    )

    sections.append("\n## Reference Swarm Context\n")
    sections.append(
        "The 2026-05-13 challenge swarm/local artifact report has complete final artifacts for six models across six sites. "
        "Its best **internal** aggregated validation AUROCs were: 4LME_ABMIL 0.900, 3agaldran 0.830, 1DivideAndConquer 0.811, 5Pimed 0.789, MST 0.775, and 2BCN_AIM 0.760. "
        "Those numbers are useful for model-family context, but the external single-site checkpoint endpoint in this report is the ODELIA challenge test inference summarized above.\n"
    )

    sections.append("\n## Files\n")
    sections.append(
        f"- Full detailed report: [{REPORT_PATH.relative_to(ROOT)}]({REPORT_PATH.relative_to(CONDENSED_REPORT_PATH.parent)})\n"
        f"- Google Sheets / Excel workbook: [{PARTNER_WORKBOOK_PATH.relative_to(ROOT)}]({PARTNER_WORKBOOK_PATH.relative_to(CONDENSED_REPORT_PATH.parent)})\n"
        "- External per-site metrics: `workspace/odelia_single_site_eval/tables/challenge_summary_metrics.csv`\n"
        "- External aggregate metrics: `workspace/odelia_single_site_eval/tables/challenge_aggregate_metrics.csv`\n"
        "- Class distributions: `workspace/odelia_single_site_eval/tables/internal_class_distribution.csv` and `workspace/odelia_single_site_eval/tables/external_challenge_class_distribution.csv`\n"
        "- Appended source reports: `docs/CHALLENGE_SWARM_LOCAL_TEST_REPORT_20260513.md` and `docs/OLE_SWARM_EVALUATION_ARTIFACT_REPORT.md`\n"
    )
    CONDENSED_REPORT_PATH.write_text("".join(sections))


def prepare() -> tuple[list[RunSource], list[Snapshot], dict[str, dict[str, list[dict]]]]:
    ensure_dirs()
    uka_run_dir = extract_uka()
    umcu_run_dir = extract_umcu()
    rumc_1dc_run_dir = extract_rumc_1dc()
    run_sources = materialize_source_runs(base_run_sources(uka_run_dir, umcu_run_dir, rumc_1dc_run_dir))
    snapshots = annotate_checksums(known_snapshots(run_sources))
    write_manifest(run_sources, snapshots)
    metrics_by_run = write_epoch_metrics(run_sources)
    return run_sources, snapshots, metrics_by_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Extract/inventory/plot only; do not run dl0 inference.")
    parser.add_argument("--skip-stage", action="store_true", help="Do not rsync checkpoints to dl0.")
    parser.add_argument("--skip-eval", action="store_true", help="Do not run dl0 inference.")
    parser.add_argument("--force-eval", action="store_true", help="Re-run inference even when prediction_results.json already exists.")
    parser.add_argument("--summary-only", action="store_true", help="Only summarize existing results and regenerate report.")
    args = parser.parse_args()

    if args.summary_only:
        ensure_dirs()
        uka_run_dir = WORKSPACE / "raw" / "UKA" / UKA_RUN_ID
        umcu_run_dir = extract_umcu()
        rumc_1dc_run_dir = extract_rumc_1dc()
        run_sources = materialize_source_runs(base_run_sources(uka_run_dir, umcu_run_dir, rumc_1dc_run_dir))
        snapshots = annotate_checksums(known_snapshots(run_sources))
        write_manifest(run_sources, snapshots)
        metrics_by_run = write_epoch_metrics(run_sources)
        summarize_prediction_results(snapshots)
        generate_report(run_sources, snapshots, metrics_by_run)
        generate_condensed_report(run_sources, snapshots, metrics_by_run)
        return

    run_sources, snapshots, metrics_by_run = prepare()
    if args.prepare_only:
        summarize_prediction_results(snapshots)
        generate_report(run_sources, snapshots, metrics_by_run)
        generate_condensed_report(run_sources, snapshots, metrics_by_run)
        return

    if not args.skip_stage:
        stage_checkpoints(snapshots)
    if not args.skip_eval:
        eval_on_dl0(snapshots, force=args.force_eval)
        pull_dl0_results()
    else:
        pull_dl0_results()

    summarize_prediction_results(snapshots)
    generate_report(run_sources, snapshots, metrics_by_run)
    generate_condensed_report(run_sources, snapshots, metrics_by_run)


if __name__ == "__main__":
    main()
