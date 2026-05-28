#!/usr/bin/env python3
"""Generate a readable report for the 2026-05-13 challenge swarm artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = Path(
    "/mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/challenge_training/"
    "20260513_ChallengeSwarmLocalTests.tar.gz"
)
DEFAULT_EXTRACTED = DEFAULT_ARCHIVE.with_suffix("").with_suffix("")
DEFAULT_REPORT = ROOT / "docs" / "CHALLENGE_SWARM_LOCAL_TEST_REPORT_20260513.md"
DEFAULT_FIGURES = ROOT / "docs" / "figures" / "challenge_swarm_local_tests_20260513"
DEFAULT_WORK = ROOT / "workspace" / "report_outputs" / "challenge_swarm_local_tests_20260513"
CLASSES = (0, 1, 2)
EXPECTED_SITES = ("CAM_1", "MHA_1", "RSH_1", "RUMC_1", "UKA_1", "UMCU_1")

RUNS = {
    "4cf67437-3472-47b3-aa3e-86b2bb14fbf3": {
        "model": "1DivideAndConquer",
        "job": "challenge_1DivideAndConquer",
        "run_name": "1DivideAndConquer_unilateral_2026_05_13_090404",
        "role": "final",
    },
    "90c5f582-d641-4387-99e8-350f22b3581c": {
        "model": "2BCN_AIM",
        "job": "challenge_2BCN_AIM",
        "run_name": "2BCN_AIM_unilateral_2026_05_13_205634",
        "role": "final",
    },
    "8f1f7a2b-177f-47fb-8bf7-9c39f48304ca": {
        "model": "3agaldran",
        "job": "challenge_3agaldran",
        "run_name": "3agaldran_unilateral_2026_05_14_145249",
        "role": "final",
    },
    "044db8c1-8ca3-443c-8518-58cb34da5285": {
        "model": "4LME_ABMIL",
        "job": "challenge_4abmil",
        "run_name": "4LME_ABMIL_unilateral_2026_05_14_211439",
        "role": "final",
    },
    "7463a272-1c58-404b-af63-a7fb5256f5b0": {
        "model": "5Pimed",
        "job": "challenge_5pimed",
        "run_name": "5Pimed_unilateral_2026_05_18_090219",
        "role": "final",
    },
    "74aa125a-c1d8-4c1d-9862-dcd9c0efd880": {
        "model": "MST",
        "job": "ODELIA_ternary_classification",
        "run_name": "MST_unilateral_2026_05_18_140641",
        "role": "final",
    },
    "8d59ef99-7eeb-4123-8d37-a1b9ebc08b56": {
        "model": "5Pimed",
        "job": "challenge_5pimed",
        "run_name": "5Pimed_unilateral_2026_05_15_034737",
        "role": "failed",
    },
    "a9274d9a-b9b8-4904-b5a3-94ea7b96dcd3": {
        "model": "MST",
        "job": "ODELIA_ternary_classification",
        "run_name": "",
        "role": "failed",
    },
    "00ec8d75-6c4a-4ccd-89c2-fc68dc9a91f7": {
        "model": "preflight",
        "job": "swarm_preflight",
        "run_name": "",
        "role": "ignored",
    },
}

FINAL_MODEL_ORDER = (
    "MST",
    "1DivideAndConquer",
    "2BCN_AIM",
    "3agaldran",
    "4LME_ABMIL",
    "5Pimed",
)
MODEL_TO_FINAL_JOB = {meta["model"]: job_id for job_id, meta in RUNS.items() if meta["role"] == "final"}


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def parse_float(value: str | float | int | None) -> float | None:
    if value in (None, "", "None", "NA"):
        return None
    if isinstance(value, (float, int)):
        result = float(value)
    else:
        try:
            result = float(value)
        except ValueError:
            return None
    return None if math.isnan(result) else result


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None or math.isnan(seconds):
        return "NA"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def rel(path: Path, base: Path = DEFAULT_REPORT.parent) -> str:
    return path.relative_to(base).as_posix()


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_external_comparison_rows(work_dir: Path) -> list[dict]:
    path = work_dir / "external_comparison_metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    artifact_order = {"swarm_global_final": 0, "site_local_retained": 1}
    return sorted(
        rows,
        key=lambda row: (
            FINAL_MODEL_ORDER.index(row["model"]) if row.get("model") in FINAL_MODEL_ORDER else 999,
            artifact_order.get(row.get("artifact", ""), 99),
            row.get("source_site", ""),
        ),
    )


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    data = path.read_text(errors="replace")
    if limit is not None and len(data) > limit:
        return data[-limit:]
    return data


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return {"_parse_error": True}


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
    if not valid:
        return None, per_class
    return sum(valid) / len(valid), per_class


def f1_scores(labels: list[int], predictions: list[int]) -> tuple[float, float, dict[int, float]]:
    counts = Counter(labels)
    per_class: dict[int, float] = {}
    for cls in CLASSES:
        tp = sum(1 for label, pred in zip(labels, predictions) if label == cls and pred == cls)
        fp = sum(1 for label, pred in zip(labels, predictions) if label != cls and pred == cls)
        fn = sum(1 for label, pred in zip(labels, predictions) if label == cls and pred != cls)
        denom = 2 * tp + fp + fn
        per_class[cls] = 0.0 if denom == 0 else (2 * tp) / denom
    macro = sum(per_class.values()) / len(CLASSES)
    total = len(labels)
    weighted = sum(per_class[cls] * counts[cls] for cls in CLASSES) / total if total else 0.0
    return macro, weighted, per_class


def load_epoch_prediction_csv(path: Path) -> dict[int, tuple[list[int], list[list[float]]]]:
    by_epoch: dict[int, tuple[list[int], list[list[float]]]] = defaultdict(lambda: ([], []))
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                epoch = int(float(row[0]))
                label = int(float(row[1]))
                probabilities = [float(value) for value in row[2:5]]
            except (ValueError, IndexError):
                continue
            by_epoch[epoch][0].append(label)
            by_epoch[epoch][1].append(probabilities)
    return dict(by_epoch)


def epoch_metrics(path: Path) -> list[dict]:
    result = []
    by_epoch = load_epoch_prediction_csv(path)
    for epoch in sorted(by_epoch):
        labels, probabilities = by_epoch[epoch]
        if not labels:
            continue
        predictions = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
        accuracy = sum(1 for label, pred in zip(labels, predictions) if label == pred) / len(labels)
        macro_auc, per_class_auc = macro_auroc(labels, probabilities)
        macro_f1, weighted_f1, per_class_f1 = f1_scores(labels, predictions)
        result.append(
            {
                "epoch": epoch,
                "samples": len(labels),
                "labels": dict(sorted(Counter(labels).items())),
                "accuracy": accuracy,
                "macro_auroc_present_classes": macro_auc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "class0_auroc": per_class_auc[0],
                "class1_auroc": per_class_auc[1],
                "class2_auroc": per_class_auc[2],
                "class2_f1": per_class_f1[2],
            }
        )
    return result


def summarize_metrics(path: Path) -> dict:
    rows = epoch_metrics(path)
    if not rows:
        return {"path": str(path), "epochs": 0}
    valid_auc = [row for row in rows if row["macro_auroc_present_classes"] is not None]
    best_auc = max(valid_auc, key=lambda row: row["macro_auroc_present_classes"]) if valid_auc else None
    best_acc = max(rows, key=lambda row: row["accuracy"])
    first = rows[0]
    last = rows[-1]
    return {
        "path": str(path),
        "file": path.name,
        "epochs": len(rows),
        "epoch_min": first["epoch"],
        "epoch_max": last["epoch"],
        "samples_per_epoch": first["samples"],
        "labels": first["labels"],
        "best_auroc_epoch": best_auc["epoch"] if best_auc else None,
        "best_auroc": best_auc["macro_auroc_present_classes"] if best_auc else None,
        "best_acc_epoch": best_acc["epoch"],
        "best_acc": best_acc["accuracy"],
        "last_epoch": last["epoch"],
        "last_auroc": last["macro_auroc_present_classes"],
        "last_acc": last["accuracy"],
        "last_macro_f1": last["macro_f1"],
        "last_weighted_f1": last["weighted_f1"],
        "class1_present": 1 in first["labels"],
    }


def parse_run_time(run_name: str) -> datetime | None:
    match = re.search(r"_unilateral_(\d{4})_(\d{2})_(\d{2})_(\d{6})$", run_name)
    if not match:
        return None
    year, month, day, hms = match.groups()
    return datetime.strptime(f"{year}{month}{day}{hms}", "%Y%m%d%H%M%S")


def run_prefix(run_name: str) -> str:
    return run_name.split("_unilateral_")[0]


def discover_run_dirs(root: Path) -> dict[str, list[Path]]:
    target_names = [meta["run_name"] for meta in RUNS.values() if meta["run_name"]]
    targets_by_prefix = defaultdict(list)
    for name in target_names:
        parsed = parse_run_time(name)
        if parsed is not None:
            targets_by_prefix[run_prefix(name)].append((name, parsed))

    matches: dict[str, list[Path]] = defaultdict(list)
    for parent_str, dirs, _ in os.walk(root):
        parent = Path(parent_str)
        for dirname in list(dirs):
            if "_unilateral_2026_05_" not in dirname:
                continue
            candidate = parent / dirname
            if dirname in target_names:
                matches[dirname].append(candidate)
                continue
            parsed = parse_run_time(dirname)
            if parsed is None:
                continue
            for target_name, target_time in targets_by_prefix.get(run_prefix(dirname), []):
                if abs((parsed - target_time).total_seconds()) <= 300:
                    matches[target_name].append(candidate)
    return {key: sorted(value) for key, value in matches.items()}


def timestamp_span_seconds(paths: Iterable[Path]) -> float | None:
    stamps: list[datetime] = []
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d+)?")
    for path in paths:
        text = read_text(path, limit=None)
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            try:
                stamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
    if len(stamps) < 2:
        return None
    return (max(stamps) - min(stamps)).total_seconds()


def tail_evidence(path: Path) -> str:
    if not path.exists():
        return ""
    text = read_text(path, limit=12000)
    lines = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"FATAL|ABORT|ERROR|Exception|Traceback|Server runner finished|finished|completed|Run name|Run directory", line, re.I)
    ]
    if not lines:
        lines = [line.strip() for line in text.splitlines()[-8:]]
    return " / ".join(line for line in lines[-4:] if line)[:420]


def collect_job(root: Path, job_id: str, run_dirs_by_name: dict[str, list[Path]]) -> dict:
    meta = RUNS[job_id]
    site_records = {}
    for site in EXPECTED_SITES:
        job_dir = root / site / job_id
        app_dir = job_dir / f"app_{site}"
        run_dirs = [
            path
            for path in run_dirs_by_name.get(meta["run_name"], [])
            if f"/runs/{site}/" in path.as_posix()
        ]
        csv_files = sorted(file for run_dir in run_dirs for file in run_dir.rglob("*_model_gt_and_classprob_*.csv"))
        tfevents = sorted(file for run_dir in run_dirs for file in run_dir.rglob("events.out.tfevents*"))
        ckpts = sorted(
            file
            for run_dir in run_dirs
            for file in run_dir.rglob("*")
            if file.is_file() and (file.name == "last.ckpt" or re.match(r"epoch=.*\.ckpt$", file.name))
        )
        logs = {
            "log": job_dir / "log.txt",
            "log_fl": job_dir / "log_fl.txt",
            "log_error": job_dir / "log_error.txt",
            "process_rc": job_dir / "_process_rc.txt",
            "startup_nohup": root / site / "startup" / "nohup.out",
        }
        global_model = app_dir / "FL_global_model.pt"
        best_global_model = app_dir / "best_FL_global_model.pt"
        metrics = []
        for csv_file in csv_files:
            source = "aggregated" if csv_file.name.startswith("aggregated_") else "site"
            split = "validation" if "validation" in csv_file.name else "train" if "train" in csv_file.name else "unknown"
            summary = summarize_metrics(csv_file)
            summary.update({"site": site, "source": source, "split": split, "run_name": csv_file.parent.name})
            metrics.append(summary)

        evidence = tail_evidence(logs["log_error"]) or tail_evidence(logs["log"]) or tail_evidence(logs["startup_nohup"])
        site_records[site] = {
            "site": site,
            "job_dir": str(job_dir),
            "job_dir_exists": job_dir.exists(),
            "meta": load_json(job_dir / "meta.json"),
            "has_meta": (job_dir / "meta.json").exists(),
            "has_log": logs["log"].exists(),
            "has_log_error": logs["log_error"].exists(),
            "log_error_size": logs["log_error"].stat().st_size if logs["log_error"].exists() else 0,
            "process_rc": read_text(logs["process_rc"]).strip() if logs["process_rc"].exists() else "",
            "global_model": str(global_model) if global_model.exists() else "",
            "global_model_size": global_model.stat().st_size if global_model.exists() else 0,
            "best_global_model": str(best_global_model) if best_global_model.exists() else "",
            "best_global_model_size": best_global_model.stat().st_size if best_global_model.exists() else 0,
            "run_dirs": [str(path) for path in run_dirs],
            "csv_count": len(csv_files),
            "tfevents_count": len(tfevents),
            "ckpt_count": len(ckpts),
            "metrics": metrics,
            "duration_seconds": timestamp_span_seconds(path for path in logs.values() if path.exists()),
            "evidence": evidence,
        }
    return {
        "job_id": job_id,
        "model": meta["model"],
        "job": meta["job"],
        "run_name": meta["run_name"],
        "role": meta["role"],
        "sites": site_records,
    }


def aggregate_job(job: dict) -> dict:
    sites = list(job["sites"].values())
    present = [site for site in sites if site["job_dir_exists"]]
    global_models = [site for site in sites if site["global_model"]]
    run_dirs = [site for site in sites if site["run_dirs"]]
    csv_sites = [site for site in sites if site["csv_count"]]
    durations = [site["duration_seconds"] for site in sites if site["duration_seconds"] is not None]
    validation_metrics = []
    train_metrics = []
    for site in sites:
        for metric in site["metrics"]:
            if metric.get("epochs", 0) == 0:
                continue
            if metric["split"] == "validation":
                validation_metrics.append(metric)
            elif metric["split"] == "train":
                train_metrics.append(metric)
    agg_validation = [metric for metric in validation_metrics if metric["source"] == "aggregated"]
    site_validation = [metric for metric in validation_metrics if metric["source"] == "site"]
    best_agg = max(
        (metric for metric in agg_validation if metric.get("best_auroc") is not None),
        key=lambda metric: metric["best_auroc"],
        default=None,
    )
    best_site = max(
        (metric for metric in site_validation if metric.get("best_auroc") is not None),
        key=lambda metric: metric["best_auroc"],
        default=None,
    )
    labels = Counter()
    class1_present = False
    for metric in validation_metrics:
        labels.update({str(key): value for key, value in metric.get("labels", {}).items()})
        class1_present = class1_present or metric.get("class1_present", False)

    log_errors = [site for site in sites if site["log_error_size"] > 0]
    if job["role"] == "final":
        status = "complete" if len(global_models) == len(EXPECTED_SITES) and len(run_dirs) >= 1 else "partial"
    elif job["role"] == "failed":
        status = "failed"
    else:
        status = "ignored"

    return {
        "job_id": job["job_id"],
        "model": job["model"],
        "job": job["job"],
        "run_name": job["run_name"],
        "role": job["role"],
        "status": status,
        "site_count": len(present),
        "global_model_count": len(global_models),
        "run_dir_site_count": len(run_dirs),
        "csv_site_count": len(csv_sites),
        "validation_csv_count": len(validation_metrics),
        "train_csv_count": len(train_metrics),
        "tfevents_site_count": sum(1 for site in sites if site["tfevents_count"]),
        "ckpt_site_count": sum(1 for site in sites if site["ckpt_count"]),
        "duration_seconds": max(durations) if durations else None,
        "best_agg_validation": best_agg,
        "best_site_validation": best_site,
        "validation_labels": dict(sorted(labels.items())),
        "class1_present": class1_present,
        "log_error_sites": [site["site"] for site in log_errors],
        "first_evidence": next((site["evidence"] for site in sites if site["evidence"]), ""),
    }


def build_inventory(root: Path, archive: Path) -> dict:
    run_dirs_by_name = discover_run_dirs(root)
    jobs = {job_id: collect_job(root, job_id, run_dirs_by_name) for job_id in RUNS}
    summaries = {job_id: aggregate_job(job) for job_id, job in jobs.items()}
    return {
        "archive": {
            "path": str(archive),
            "size": archive.stat().st_size if archive.exists() else None,
            "mtime": datetime.fromtimestamp(archive.stat().st_mtime, tz=timezone.utc).isoformat() if archive.exists() else "",
        },
        "root": str(root),
        "root_size_note": "computed externally with du -sh during validation",
        "expected_sites": list(EXPECTED_SITES),
        "run_dirs_by_name": {key: [str(path) for path in value] for key, value in run_dirs_by_name.items()},
        "jobs": jobs,
        "summaries": summaries,
    }


def write_svg(path: Path, body: str, width: int = 1100, height: int = 560) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                "<style>",
                "text{font-family:Arial,Helvetica,sans-serif;fill:#18212f}",
                ".title{font-size:24px;font-weight:700}",
                ".axis{font-size:13px;fill:#465161}",
                ".label{font-size:12px;fill:#465161}",
                ".value{font-size:12px;fill:#263241}",
                ".grid{stroke:#d8dee8;stroke-width:1}",
                ".cell{stroke:#ffffff;stroke-width:2}",
                "</style>",
                body,
                "</svg>",
            ]
        )
        + "\n"
    )


def heatmap(path: Path, title: str, rows: list[str], cols: list[str], values: dict[tuple[str, str], tuple[str, str]], width: int = 1120) -> None:
    left, top = 180, 78
    cell_w, cell_h = 118, 42
    height = top + 42 + len(rows) * cell_h + 42
    parts = [f'<text x="{left}" y="36" class="title">{esc(title)}</text>']
    for col_idx, col in enumerate(cols):
        x = left + col_idx * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="{top - 14}" text-anchor="middle" class="axis">{esc(col)}</text>')
    for row_idx, row in enumerate(rows):
        y = top + row_idx * cell_h
        parts.append(f'<text x="{left - 12}" y="{y + 27}" text-anchor="end" class="axis">{esc(row)}</text>')
        for col_idx, col in enumerate(cols):
            x = left + col_idx * cell_w
            label, color = values.get((row, col), ("NA", "#e5e9f0"))
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" class="cell" rx="3"/>')
            parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + 27}" text-anchor="middle" class="value">{esc(label)}</text>')
    write_svg(path, "\n".join(parts), width, height)


def bar_chart(path: Path, title: str, categories: list[str], series: list[tuple[str, list[float | None], str]], y_label: str, y_max: float | None = None) -> None:
    width, height = 1120, 560
    left, right, top, bottom = 84, 32, 64, 118
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_values = [value for _, values, _ in series for value in values if value is not None and not math.isnan(value)]
    if y_max is None:
        y_max = max(all_values) * 1.12 if all_values else 1.0
    if y_max <= 1.0:
        y_max = 1.0
    group_w = plot_w / max(1, len(categories))
    bar_gap = 4
    bar_w = max(7, (group_w * 0.78 - bar_gap * (len(series) - 1)) / max(1, len(series)))
    parts = [f'<text x="{left}" y="36" class="title">{esc(title)}</text>']
    parts.append(f'<text x="22" y="{top + plot_h / 2}" class="axis" transform="rotate(-90 22 {top + plot_h / 2})">{esc(y_label)}</text>')
    for tick in range(6):
        value = y_max * tick / 5
        y = top + plot_h - (value / y_max) * plot_h
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{fmt(value, 1)}</text>')
    for cat_idx, category in enumerate(categories):
        group_x = left + cat_idx * group_w
        start_x = group_x + (group_w - (bar_w * len(series) + bar_gap * (len(series) - 1))) / 2
        for series_idx, (_, values, color) in enumerate(series):
            value = values[cat_idx]
            x = start_x + series_idx * (bar_w + bar_gap)
            if value is None or math.isnan(value):
                parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h - 8:.1f}" text-anchor="middle" class="label">NA</text>')
                continue
            bar_h = max(0, min(plot_h, (value / y_max) * plot_h))
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{group_x + group_w / 2:.1f}" y="{top + plot_h + 26}" text-anchor="middle" class="axis">{esc(category)}</text>')
    legend_x = left
    legend_y = height - 42
    for label, _, color in series:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 12}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y}" class="axis">{esc(label)}</text>')
        legend_x += max(150, len(label) * 7 + 48)
    write_svg(path, "\n".join(parts), width, height)


def make_figures(inventory: dict, figures_dir: Path) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    summaries = inventory["summaries"]
    final_models = list(FINAL_MODEL_ORDER)

    status_values = {}
    for model in final_models:
        summary = summaries[MODEL_TO_FINAL_JOB[model]]
        status = summary["status"]
        color = "#63b36f" if status == "complete" else "#f1c453" if status == "partial" else "#df6f64"
        status_values[(model, "final")] = (status, color)
        failed = [row for row in summaries.values() if row["model"] == model and row["role"] == "failed"]
        status_values[(model, "failed retry")] = (str(len(failed)) if failed else "0", "#df6f64" if failed else "#dfe7ef")
    heatmap(figures_dir / "run_status_heatmap.svg", "Run status by model", final_models, ["final", "failed retry"], status_values, width=620)

    coverage_values = {}
    for model in final_models:
        summary = summaries[MODEL_TO_FINAL_JOB[model]]
        for col, key in [
            ("job dirs", "site_count"),
            ("models", "global_model_count"),
            ("run dirs", "run_dir_site_count"),
            ("CSVs", "csv_site_count"),
            ("TFEvents", "tfevents_site_count"),
            ("ckpts", "ckpt_site_count"),
        ]:
            value = summary[key]
            ratio = value / len(EXPECTED_SITES)
            color = "#63b36f" if ratio >= 1 else "#f1c453" if ratio > 0 else "#df6f64"
            coverage_values[(model, col)] = (f"{value}/{len(EXPECTED_SITES)}", color)
    heatmap(
        figures_dir / "artifact_coverage_heatmap.svg",
        "Artifact coverage by final run",
        final_models,
        ["job dirs", "models", "run dirs", "CSVs", "TFEvents", "ckpts"],
        coverage_values,
    )

    durations = [summaries[MODEL_TO_FINAL_JOB[model]]["duration_seconds"] for model in final_models]
    duration_hours = [value / 3600 if value is not None else None for value in durations]
    bar_chart(
        figures_dir / "duration_by_model.svg",
        "Approximate log timestamp span by model",
        final_models,
        [("hours", duration_hours, "#4e79a7")],
        "Hours",
    )

    best_agg = []
    last_agg = []
    best_site = []
    for model in final_models:
        summary = summaries[MODEL_TO_FINAL_JOB[model]]
        agg = summary["best_agg_validation"]
        site = summary["best_site_validation"]
        best_agg.append(agg.get("best_auroc") if agg else None)
        last_agg.append(agg.get("last_auroc") if agg else None)
        best_site.append(site.get("best_auroc") if site else None)
    bar_chart(
        figures_dir / "validation_auroc_summary.svg",
        "Validation AUROC from available class-probability CSVs",
        final_models,
        [
            ("best aggregated", best_agg, "#59a14f"),
            ("last aggregated", last_agg, "#f28e2b"),
            ("best site model", best_site, "#4e79a7"),
        ],
        "AUROC",
        y_max=1.0,
    )

    best_acc = []
    last_acc = []
    for model in final_models:
        summary = summaries[MODEL_TO_FINAL_JOB[model]]
        agg = summary["best_agg_validation"]
        best_acc.append(agg.get("best_acc") if agg else None)
        last_acc.append(agg.get("last_acc") if agg else None)
    bar_chart(
        figures_dir / "validation_accuracy_summary.svg",
        "Validation accuracy from available aggregated CSVs",
        final_models,
        [
            ("best aggregated", best_acc, "#59a14f"),
            ("last aggregated", last_acc, "#f28e2b"),
        ],
        "Accuracy",
        y_max=1.0,
    )

    return {
        "status": figures_dir / "run_status_heatmap.svg",
        "coverage": figures_dir / "artifact_coverage_heatmap.svg",
        "duration": figures_dir / "duration_by_model.svg",
        "auroc": figures_dir / "validation_auroc_summary.svg",
        "accuracy": figures_dir / "validation_accuracy_summary.svg",
    }


def write_outputs(inventory: dict, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "run_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True))

    with (work_dir / "validation_metrics_summary.csv").open("w", newline="") as f:
        fieldnames = [
            "job_id",
            "model",
            "role",
            "site",
            "source",
            "split",
            "run_name",
            "epochs",
            "samples_per_epoch",
            "labels",
            "best_auroc_epoch",
            "best_auroc",
            "best_acc_epoch",
            "best_acc",
            "last_epoch",
            "last_auroc",
            "last_acc",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job in inventory["jobs"].values():
            for site in job["sites"].values():
                for metric in site["metrics"]:
                    if metric.get("split") != "validation":
                        continue
                    writer.writerow(
                        {
                            "job_id": job["job_id"],
                            "model": job["model"],
                            "role": job["role"],
                            "site": site["site"],
                            "source": metric.get("source"),
                            "split": metric.get("split"),
                            "run_name": metric.get("run_name"),
                            "epochs": metric.get("epochs"),
                            "samples_per_epoch": metric.get("samples_per_epoch"),
                            "labels": json.dumps(metric.get("labels", {}), sort_keys=True),
                            "best_auroc_epoch": metric.get("best_auroc_epoch"),
                            "best_auroc": metric.get("best_auroc"),
                            "best_acc_epoch": metric.get("best_acc_epoch"),
                            "best_acc": metric.get("best_acc"),
                            "last_epoch": metric.get("last_epoch"),
                            "last_auroc": metric.get("last_auroc"),
                            "last_acc": metric.get("last_acc"),
                        }
                    )


def status_note(summary: dict) -> str:
    if summary["role"] == "ignored":
        return "ignored preflight"
    if summary["role"] == "failed":
        return "failed attempt"
    if summary["status"] == "complete":
        return "complete artifacts"
    return "partial artifacts"


def build_report(inventory: dict, figures: dict[str, Path], report_path: Path, work_dir: Path) -> str:
    summaries = inventory["summaries"]
    final_summaries = [summaries[MODEL_TO_FINAL_JOB[model]] for model in FINAL_MODEL_ORDER]
    failed_summaries = [summary for summary in summaries.values() if summary["role"] == "failed"]
    ignored = [summary for summary in summaries.values() if summary["role"] == "ignored"]

    complete_count = sum(1 for summary in final_summaries if summary["status"] == "complete")
    class1_missing = [summary["model"] for summary in final_summaries if not summary["class1_present"]]
    meta_statuses = Counter()
    for job in inventory["jobs"].values():
        for site in job["sites"].values():
            status = site["meta"].get("status")
            if status:
                meta_statuses[status] += 1

    final_rows = []
    for summary in final_summaries:
        agg = summary["best_agg_validation"] or {}
        site = summary["best_site_validation"] or {}
        final_rows.append(
            [
                summary["model"],
                f"`{summary['job_id'][:8]}...`",
                f"`{summary['run_name']}`",
                status_note(summary),
                f"{summary['global_model_count']}/{len(EXPECTED_SITES)}",
                f"{summary['run_dir_site_count']}/{len(EXPECTED_SITES)}",
                f"{summary['csv_site_count']}/{len(EXPECTED_SITES)}",
                fmt_seconds(summary["duration_seconds"]),
                f"{fmt(agg.get('best_auroc'))} @ {fmt(agg.get('best_auroc_epoch'), 0)}" if agg else "NA",
                f"{fmt(site.get('best_auroc'))} @ {fmt(site.get('best_auroc_epoch'), 0)}" if site else "NA",
            ]
        )

    failure_rows = []
    for summary in failed_summaries:
        failure_rows.append(
            [
                summary["model"],
                f"`{summary['job_id']}`",
                f"`{summary['run_name']}`" if summary["run_name"] else "no run directory mapped",
                f"{summary['global_model_count']}/{len(EXPECTED_SITES)}",
                f"{summary['run_dir_site_count']}/{len(EXPECTED_SITES)}",
                fmt_seconds(summary["duration_seconds"]),
                esc(summary["first_evidence"] or "No concise error evidence extracted"),
            ]
        )

    coverage_rows = []
    for summary in final_summaries:
        coverage_rows.append(
            [
                summary["model"],
                f"{summary['site_count']}/{len(EXPECTED_SITES)}",
                f"{summary['global_model_count']}/{len(EXPECTED_SITES)}",
                f"{summary['run_dir_site_count']}/{len(EXPECTED_SITES)}",
                f"{summary['validation_csv_count']} validation / {summary['train_csv_count']} train",
                f"{summary['tfevents_site_count']}/{len(EXPECTED_SITES)}",
                f"{summary['ckpt_site_count']}/{len(EXPECTED_SITES)}",
                ", ".join(summary["log_error_sites"]) if summary["log_error_sites"] else "none",
            ]
        )

    validation_rows = []
    for summary in final_summaries:
        job = inventory["jobs"][summary["job_id"]]
        for site_name in EXPECTED_SITES:
            site = job["sites"][site_name]
            for metric in site["metrics"]:
                if metric.get("split") != "validation":
                    continue
                validation_rows.append(
                    [
                        summary["model"],
                        site_name,
                        metric["source"],
                        str(metric["epochs"]),
                        str(metric.get("samples_per_epoch", "NA")),
                        "`" + ", ".join(f"{k}:{v}" for k, v in metric.get("labels", {}).items()) + "`",
                        f"{fmt(metric.get('best_auroc'))} @ {fmt(metric.get('best_auroc_epoch'), 0)}",
                        f"{fmt(metric.get('best_acc'))} @ {fmt(metric.get('best_acc_epoch'), 0)}",
                        f"{fmt(metric.get('last_auroc'))} / {fmt(metric.get('last_acc'))}",
                    ]
                )

    validation_rows = sorted(validation_rows, key=lambda row: (FINAL_MODEL_ORDER.index(row[0]), row[1], row[2]))
    if len(validation_rows) > 80:
        validation_note = f"Showing first 80 of {len(validation_rows)} validation streams; full CSV is in `{work_dir.relative_to(ROOT) / 'validation_metrics_summary.csv'}`."
        validation_rows = validation_rows[:80]
    else:
        validation_note = f"Full validation stream table is also available as `{work_dir.relative_to(ROOT) / 'validation_metrics_summary.csv'}`."

    external_comparison = load_external_comparison_rows(work_dir)
    external_rows = []
    artifact_labels = {
        "swarm_global_final": "Swarm global final",
        "site_local_retained": "Best site-local retained",
    }
    for row in external_comparison:
        external_rows.append(
            [
                row["model"],
                artifact_labels.get(row["artifact"], row["artifact"]),
                row["source_site"],
                f"`{row['checkpoint_label']}`",
                fmt(parse_float(row.get("internal_val_auroc"))),
                fmt(parse_float(row.get("external_class2_auroc"))),
                fmt(parse_float(row.get("external_macro_auroc"))),
                fmt(parse_float(row.get("external_accuracy"))),
                fmt(parse_float(row.get("external_class2_recall"))),
                fmt(parse_float(row.get("samples")), 0),
            ]
        )

    if external_rows:
        top_external = max(
            external_comparison,
            key=lambda row: parse_float(row.get("external_class2_auroc")) or -1.0,
        )
        swarm_better = 0
        comparable_models = 0
        for model in FINAL_MODEL_ORDER:
            model_rows = [row for row in external_comparison if row["model"] == model]
            swarm_row = next((row for row in model_rows if row["artifact"] == "swarm_global_final"), None)
            local_row = next((row for row in model_rows if row["artifact"] == "site_local_retained"), None)
            if not swarm_row or not local_row:
                continue
            comparable_models += 1
            if (parse_float(swarm_row.get("external_class2_auroc")) or -1.0) >= (parse_float(local_row.get("external_class2_auroc")) or -1.0):
                swarm_better += 1
        external_summary = (
            f"- External ODELIA validation is available for `{len(external_comparison)}` artifacts "
            f"({comparable_models} swarm-global vs site-local model pairs). The strongest weighted "
            f"external Class-2 AUROC is `{fmt(parse_float(top_external.get('external_class2_auroc')))} "
            f"({top_external['model']} / {artifact_labels.get(top_external['artifact'], top_external['artifact'])})`; "
            f"the swarm-global artifact is >= the site-local retained comparator in `{swarm_better}/{comparable_models}` model families."
        )
        external_section = [
            "## External Validation Metrics",
            "",
            "External validation evaluates retained artifacts on the ODELIA challenge test institutions (`CAM`, `MHA`, `RSH`, `RUMC`, `UKA`, `UMCU`) using `scripts/evaluation/predict.py` on `dd-dl0:/mnt/dlhd0/medswarmdata`.",
            "",
            "The comparison below is deliberately artifact-based: **Swarm global final** is one representative final `FL_global_model.pt` from the completed swarm run, and **Best site-local retained** is the retained non-last local checkpoint from the site with the highest internal site-stream validation AUROC for that model. No separate pooled centralized checkpoint family was found in this package, so the site-local retained checkpoint is the available local/centralized-style comparator.",
            "",
            f"Aggregate external metrics are weighted by samples across the six external sites. Per-site rows are available in `{work_dir.relative_to(ROOT) / 'external_metrics_by_eval_site.csv'}`; the target manifest is `{work_dir.relative_to(ROOT) / 'external_eval_manifest.json'}`.",
            "",
            markdown_table(
                [
                    "Model",
                    "Artifact",
                    "Source site",
                    "Checkpoint",
                    "Internal val AUROC",
                    "External C2 AUROC",
                    "External macro AUROC",
                    "External ACC",
                    "External C2 recall",
                    "Samples",
                ],
                external_rows,
            ),
            "",
        ]
    else:
        external_summary = "- External ODELIA validation has not been generated yet for this package."
        external_section = [
            "## External Validation Metrics",
            "",
            "External validation metrics have not been generated for this artifact package yet. Run `scripts/evaluation/run_challenge_swarm_local_external_eval.py`, then regenerate this report to embed the swarm-global versus site-local comparison.",
            "",
        ]

    site_rows = []
    for site in EXPECTED_SITES:
        present_jobs = [summary["model"] for summary in final_summaries if inventory["jobs"][summary["job_id"]]["sites"][site]["job_dir_exists"]]
        run_dirs = [summary["model"] for summary in final_summaries if inventory["jobs"][summary["job_id"]]["sites"][site]["run_dirs"]]
        csvs = [summary["model"] for summary in final_summaries if inventory["jobs"][summary["job_id"]]["sites"][site]["csv_count"]]
        site_rows.append(
            [
                site,
                f"{len(present_jobs)}/6",
                f"{len(run_dirs)}/6",
                f"{len(csvs)}/6",
                ", ".join(model for model in FINAL_MODEL_ORDER if model not in run_dirs) or "none",
            ]
        )

    archive = inventory["archive"]
    archive_mtime = archive["mtime"].replace("+00:00", "Z") if archive["mtime"] else "NA"
    meta_status_text = ", ".join(f"`{key}`: {value}" for key, value in sorted(meta_statuses.items())) or "none"

    command_block = f"""```bash
cd /home/jeff/Projects/MediSwarm

# Confirm archive and extraction.
ls -lh /mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/challenge_training/20260513_ChallengeSwarmLocalTests.tar.gz
du -sh /mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/challenge_training/20260513_ChallengeSwarmLocalTests

# Regenerate external validation tables before report generation, if needed.
python {Path('scripts/evaluation/run_challenge_swarm_local_external_eval.py')} \\
  --root /mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/challenge_training/20260513_ChallengeSwarmLocalTests

# Regenerate this report.
python {Path('scripts/evaluation/generate_challenge_swarm_local_report.py')} \\
  --root /mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/challenge_training/20260513_ChallengeSwarmLocalTests \\
  --archive /mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/challenge_training/20260513_ChallengeSwarmLocalTests.tar.gz
```"""

    if len(class1_missing) == len(final_summaries):
        class1_note = "Class `1` is absent from every parsed validation CSV, so AUROC averages only classes with both positives and negatives."
    elif class1_missing:
        class1_note = f"Class `1` is missing in validation CSVs for: {', '.join(class1_missing)}."
    else:
        class1_note = "All parsed validation CSVs include class `1`; three-class AUROC is therefore defined for these streams."

    return "\n".join(
        [
            "# ODELIA Challenge Swarm/Local Artifact Report",
            "",
            "> Generated by `scripts/evaluation/generate_challenge_swarm_local_report.py` from the extracted `20260513_ChallengeSwarmLocalTests` package.",
            "",
            "## Executive Summary",
            "",
            f"- The archive is present at `{archive['path']}` with size `{archive['size']:,}` bytes and mtime `{archive_mtime}`.",
            f"- The extracted package is indexed at `{inventory['root']}`; the archive layout is site-first (`<SITE>/<JOB_ID>/...`) plus scratch run directories.",
            f"- Final run coverage is `{complete_count}/6` models with complete global-model artifacts across the expected sites.",
            "- Failed/retried runs are kept separate from the final model matrix: `5Pimed` failed once before succeeding, and `MST` has one failed job with no mapped run directory before succeeding.",
            f"- `{ignored[0]['job_id']}` is treated as swarm preflight and ignored for model-quality interpretation.",
            f"- Metadata status values are stale/incomplete in this package ({meta_status_text}); the report cross-checks logs, run folders, checkpoints, and CSVs instead.",
            f"- {class1_note}",
            "",
            "## Transfer and Extraction",
            "",
            markdown_table(
                ["Field", "Value"],
                [
                    ["Archive", f"`{archive['path']}`"],
                    ["Archive size", f"`{archive['size']:,}` bytes"],
                    ["Archive mtime", f"`{archive_mtime}`"],
                    ["Extracted directory", f"`{inventory['root']}`"],
                    ["Expected final models", "`" + "`, `".join(FINAL_MODEL_ORDER) + "`"],
                    ["Expected sites", "`" + "`, `".join(EXPECTED_SITES) + "`"],
                    ["Derived summaries", f"`{work_dir.relative_to(ROOT)}`"],
                ],
            ),
            "",
            "## Final Run Matrix",
            "",
            f"![Run status heatmap]({rel(figures['status'], report_path.parent)})",
            "",
            markdown_table(
                [
                    "Model",
                    "Job ID",
                    "Mapped run",
                    "Status",
                    "Global models",
                    "Run dirs",
                    "CSV sites",
                    "Log span",
                    "Best agg val AUROC",
                    "Best site val AUROC",
                ],
                final_rows,
            ),
            "",
            "## Artifact Coverage",
            "",
            f"![Artifact coverage heatmap]({rel(figures['coverage'], report_path.parent)})",
            "",
            markdown_table(
                ["Model", "Job dirs", "Global models", "Run dirs", "CSV files", "TFEvents", "Local ckpts", "Non-empty error logs"],
                coverage_rows,
            ),
            "",
            "### Site Coverage",
            "",
            markdown_table(["Site", "Job dirs", "Run dirs", "CSV coverage", "Missing run dirs"], site_rows),
            "",
            "## Validation Metrics",
            "",
            "Metrics below are recomputed from `*_gt_and_classprob_validation.csv`. AUROC is one-vs-rest averaged over classes that have both positive and negative samples in a stream.",
            "",
            f"![Validation AUROC summary]({rel(figures['auroc'], report_path.parent)})",
            "",
            f"![Validation accuracy summary]({rel(figures['accuracy'], report_path.parent)})",
            "",
            validation_note,
            "",
            markdown_table(
                [
                    "Model",
                    "Site",
                    "Stream",
                    "Epochs",
                    "Samples/epoch",
                    "Labels",
                    "Best AUROC",
                    "Best ACC",
                    "Last AUROC/ACC",
                ],
                validation_rows,
            ),
            "",
            *external_section,
            "## Timing",
            "",
            f"![Approximate duration by model]({rel(figures['duration'], report_path.parent)})",
            "",
            "Durations are approximate spans between the first and last parseable timestamps in available logs. They are useful for relative runtime comparison, not billing-grade measurement.",
            "",
            "## Failed and Ignored Runs",
            "",
            markdown_table(["Model", "Job ID", "Mapped run", "Global models", "Run dirs", "Log span", "Evidence"], failure_rows),
            "",
            f"- Ignored preflight: `{ignored[0]['job_id']}` (`{ignored[0]['model']}` / `{ignored[0]['job']}`).",
            "",
            "## Validation Checks",
            "",
            "- Every canonical UUID from the handoff was found under at least one site directory.",
            "- Final job directories are checked against the expected six client sites: `CAM_1`, `MHA_1`, `RSH_1`, `RUMC_1`, `UKA_1`, `UMCU_1`.",
            "- Run directories are matched by exact run name first, then by model prefix plus a five-minute timestamp tolerance to account for per-site timestamp drift.",
            "- Checkpoints are only referenced by extracted path and size; they are not copied into the repo or `workspace/report_outputs`.",
            "- `meta.json` status fields are treated as stale when contradicted by checkpoints, logs, or CSV artifacts.",
            "",
            "## Reproduction",
            "",
            command_block,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_EXTRACTED, help="Extracted 20260513_ChallengeSwarmLocalTests directory")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="Original transferred tar.gz archive")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT, help="Markdown report path")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES, help="Figure output directory")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK, help="Derived machine-readable output directory")
    args = parser.parse_args()

    root = args.root.resolve()
    archive = args.archive.resolve()
    report_path = args.output_md.resolve()
    figures_dir = args.figures_dir.resolve()
    work_dir = args.work_dir.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Extracted package not found: {root}")
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    inventory = build_inventory(root, archive)
    write_outputs(inventory, work_dir)
    figures = make_figures(inventory, figures_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(inventory, figures, report_path, work_dir))

    print(f"Wrote {report_path.relative_to(ROOT)}")
    for figure in figures.values():
        print(f"Wrote {figure.relative_to(ROOT)}")
    print(f"Wrote {(work_dir / 'run_inventory.json').relative_to(ROOT)}")
    print(f"Wrote {(work_dir / 'validation_metrics_summary.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
