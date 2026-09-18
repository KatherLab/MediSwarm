#!/usr/bin/env python3
"""Generate a readable report for the extracted ODELIA Ole_swarm artifacts.

The script intentionally uses only the Python standard library so it can run on
the lightweight project environment used for deployment validation.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "workspace" / "usz_partner_eval"
DOCS_DIR = ROOT / "docs"
FIGURE_DIR = DOCS_DIR / "figures" / "usz_partner_eval"
OLE_FIGURE_DIR = DOCS_DIR / "figures" / "ole_swarm"
OLE_EVALUATION_PNG = OLE_FIGURE_DIR / "evaluation.png"
REPORT_PATH = DOCS_DIR / "OLE_SWARM_EVALUATION_ARTIFACT_REPORT.md"
CLASSES = (0, 1, 2)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value * 100:.{digits}f}%"


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "NA"
    return f"{int(value):,}"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def ensure_inputs() -> None:
    required = [
        SOURCE_DIR / "usz_data_summary.json",
        SOURCE_DIR / "odelia_challenge_eval" / "summary_metrics.csv",
        SOURCE_DIR / "cross_eval_duke" / "prediction_results.json",
        SOURCE_DIR / "runs" / "1DC_train.csv",
        SOURCE_DIR / "runs" / "1DC_validation.csv",
        SOURCE_DIR / "runs" / "MST_train.csv",
        SOURCE_DIR / "runs" / "MST_validation.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required report input(s):\n{joined}")


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_challenge_metrics(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["samples"] = int(row["samples"])
        for key, value in list(row.items()):
            if key in {"run", "site", "samples"}:
                continue
            row[key] = float(value) if value != "" else math.nan
    return rows


def weighted_mean(rows: Iterable[dict], key: str) -> float | None:
    weighted_total = 0.0
    weight_sum = 0
    for row in rows:
        value = row[key]
        if isinstance(value, float) and math.isnan(value):
            continue
        weighted_total += value * row["samples"]
        weight_sum += row["samples"]
    if weight_sum == 0:
        return None
    return weighted_total / weight_sum


def aggregate_challenge(rows: list[dict]) -> list[dict]:
    metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auroc",
        "class2_auroc",
        "class2_f1",
        "class2_recall",
    ]
    aggregates = []
    for run in sorted({row["run"] for row in rows}):
        run_rows = [row for row in rows if row["run"] == run]
        item = {
            "run": run,
            "samples": sum(row["samples"] for row in run_rows),
        }
        item.update({metric: weighted_mean(run_rows, metric) for metric in metrics})
        aggregates.append(item)
    return aggregates


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
    weighted = sum(per_class[cls] * counts[cls] for cls in CLASSES) / total
    return macro, weighted, per_class


def load_epoch_prediction_csv(path: Path) -> dict[int, tuple[list[int], list[list[float]]]]:
    by_epoch: dict[int, tuple[list[int], list[list[float]]]] = defaultdict(lambda: ([], []))
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            epoch = int(row[0])
            label = int(row[1])
            probabilities = [float(value) for value in row[2:5]]
            by_epoch[epoch][0].append(label)
            by_epoch[epoch][1].append(probabilities)
    return dict(by_epoch)


def epoch_metrics(path: Path) -> list[dict]:
    result = []
    by_epoch = load_epoch_prediction_csv(path)
    for epoch in sorted(by_epoch):
        labels, probabilities = by_epoch[epoch]
        predictions = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
        accuracy = sum(1 for label, pred in zip(labels, predictions) if label == pred) / len(labels)
        macro_auc, per_class_auc = macro_auroc(labels, probabilities)
        macro_f1, weighted_f1, per_class_f1 = f1_scores(labels, predictions)
        result.append(
            {
                "epoch": epoch,
                "samples": len(labels),
                "accuracy": accuracy,
                "macro_auroc": macro_auc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "class2_auroc": per_class_auc[2],
                "class2_f1": per_class_f1[2],
            }
        )
    return result


def parse_log_summary(path: Path) -> dict[str, str | int | None]:
    if not path.exists():
        return {"path": str(path), "best_checkpoint": None, "last_checkpoint": None, "exit_code": None, "exit_at": None}
    text = path.read_text(errors="replace")
    best_match = re.findall(r"Best model checkpoint: (.+)", text)
    last_match = re.findall(r"Last model saved to: (.+)", text)
    exit_match = re.findall(r"=== exit=(\d+) at (.+?) ===", text)
    return {
        "path": str(path.relative_to(ROOT)),
        "best_checkpoint": best_match[-1] if best_match else None,
        "last_checkpoint": last_match[-1] if last_match else None,
        "exit_code": int(exit_match[-1][0]) if exit_match else None,
        "exit_at": exit_match[-1][1] if exit_match else None,
    }


def file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


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
                ".legend{font-size:13px;fill:#263241}",
                ".value{font-size:12px;fill:#263241}",
                ".grid{stroke:#d8dee8;stroke-width:1}",
                ".axisline{stroke:#2e3a48;stroke-width:1.2}",
                "</style>",
                body,
                "</svg>",
            ]
        )
        + "\n"
    )


def bar_chart(
    path: Path,
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float | None], str]],
    y_label: str,
    y_max: float = 1.0,
    width: int = 1100,
    height: int = 560,
) -> None:
    left, right, top, bottom = 82, 28, 62, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(1, len(categories))
    bar_gap = 3
    bar_w = max(6, (group_w * 0.76 - bar_gap * (len(series) - 1)) / max(1, len(series)))

    parts = [f'<text x="{left}" y="34" class="title">{esc(title)}</text>']
    parts.append(f'<text x="20" y="{top + plot_h / 2}" class="axis" transform="rotate(-90 20 {top + plot_h / 2})">{esc(y_label)}</text>')

    for tick in range(0, 6):
        value = y_max * tick / 5
        y = top + plot_h - (value / y_max) * plot_h
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{fmt(value, 1)}</text>')
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" class="axisline"/>')
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" class="axisline"/>')

    for category_idx, category in enumerate(categories):
        group_x = left + category_idx * group_w
        start_x = group_x + (group_w - (bar_w * len(series) + bar_gap * (len(series) - 1))) / 2
        for series_idx, (_, values, color) in enumerate(series):
            value = values[category_idx]
            x = start_x + series_idx * (bar_w + bar_gap)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h - 8:.1f}" text-anchor="middle" class="label">NA</text>')
                continue
            bar_h = max(0, min(plot_h, (value / y_max) * plot_h))
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"/>')
        label_x = group_x + group_w / 2
        parts.append(f'<text x="{label_x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="axis">{esc(category)}</text>')

    legend_x = left
    legend_y = height - 42
    for label, _, color in series:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 12}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y}" class="legend">{esc(label)}</text>')
        legend_x += max(140, len(label) * 7 + 40)

    write_svg(path, "\n".join(parts), width, height)


def line_chart(
    path: Path,
    title: str,
    lines: list[tuple[str, list[tuple[int, float | None]], str]],
    y_label: str,
    y_min: float = 0.0,
    y_max: float = 1.0,
    width: int = 1100,
    height: int = 560,
) -> None:
    left, right, top, bottom = 76, 28, 62, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_epochs = [epoch for _, points, _ in lines for epoch, value in points if value is not None]
    min_epoch = min(all_epochs)
    max_epoch = max(all_epochs)

    def sx(epoch: int) -> float:
        if max_epoch == min_epoch:
            return left
        return left + (epoch - min_epoch) / (max_epoch - min_epoch) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    parts = [f'<text x="{left}" y="34" class="title">{esc(title)}</text>']
    parts.append(f'<text x="20" y="{top + plot_h / 2}" class="axis" transform="rotate(-90 20 {top + plot_h / 2})">{esc(y_label)}</text>')
    for tick in range(0, 6):
        value = y_min + (y_max - y_min) * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{fmt(value, 1)}</text>')
    for epoch in [min_epoch, 20, 40, 60, 80, max_epoch]:
        if epoch < min_epoch or epoch > max_epoch:
            continue
        x = sx(epoch)
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="axis">{epoch}</text>')
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" class="axisline"/>')
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" class="axisline"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" class="axis">Epoch</text>')

    for label, points, color in lines:
        filtered = [(epoch, value) for epoch, value in points if value is not None]
        polyline = " ".join(f"{sx(epoch):.1f},{sy(value):.1f}" for epoch, value in filtered)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        if filtered:
            last_epoch, last_value = filtered[-1]
            parts.append(f'<circle cx="{sx(last_epoch):.1f}" cy="{sy(last_value):.1f}" r="3.5" fill="{color}"/>')

    legend_x = left
    legend_y = height - 48
    for label, _, color in lines:
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 18}" y1="{legend_y - 5}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{legend_y}" class="legend">{esc(label)}</text>')
        legend_x += max(130, len(label) * 7 + 48)

    write_svg(path, "\n".join(parts), width, height)


def split_chart(path: Path, summary: dict) -> None:
    categories = ["Train", "Validation", "Test"]
    values = [
        summary["split_distribution"]["train"],
        summary["split_distribution"]["val"],
        summary["split_distribution"]["test"],
    ]
    colors = ["#2c7fb8", "#f28e2b", "#59a14f"]
    width, height = 960, 420
    left, top = 170, 86
    bar_w, bar_h = 620, 38
    max_value = max(values)
    parts = [f'<text x="{left}" y="38" class="title">USZ split sizes and annotation classes</text>']
    for idx, (category, value, color) in enumerate(zip(categories, values, colors)):
        y = top + idx * 58
        w = value / max_value * bar_w
        parts.append(f'<text x="{left - 18}" y="{y + 25}" text-anchor="end" class="axis">{esc(category)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{left + w + 10:.1f}" y="{y + 25}" class="value">{fmt_int(value)} UIDs</text>')

    class_y = top + 205
    total_ann = sum(summary["annotation_lesion_distribution"].values())
    parts.append(f'<text x="{left}" y="{class_y - 20}" class="axis">Annotation lesion distribution ({fmt_int(total_ann)} UIDs)</text>')
    x = left
    class_colors = {"0": "#4e79a7", "1": "#e15759", "2": "#76b7b2"}
    for cls in ["0", "1", "2"]:
        value = summary["annotation_lesion_distribution"][cls]
        w = value / total_ann * bar_w
        parts.append(f'<rect x="{x:.1f}" y="{class_y}" width="{w:.1f}" height="42" fill="{class_colors[cls]}"/>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{class_y + 27}" text-anchor="middle" class="value">class {cls}: {fmt_pct(value / total_ann)}</text>')
        x += w
    write_svg(path, "\n".join(parts), width, height)


def confusion_chart(path: Path, duke_results: list[dict]) -> None:
    width, height = 980, 500
    parts = [f'<text x="64" y="38" class="title">Duke cross-evaluation confusion matrices</text>']
    max_cell = max(max(max(row) for row in item["metrics"]["confusion_matrix"]) for item in duke_results)
    colors = ("#f8fbff", "#0f5e9c")

    def cell_color(value: int) -> str:
        ratio = 0 if max_cell == 0 else value / max_cell
        start = tuple(int(colors[0][i : i + 2], 16) for i in (1, 3, 5))
        end = tuple(int(colors[1][i : i + 2], 16) for i in (1, 3, 5))
        rgb = tuple(round(start[idx] + (end[idx] - start[idx]) * ratio) for idx in range(3))
        return "#" + "".join(f"{component:02x}" for component in rgb)

    for block_idx, item in enumerate(duke_results):
        x0 = 88 + block_idx * 445
        y0 = 88
        cell = 78
        title = item["site"].replace("USZ_", "").replace("_", " ")
        parts.append(f'<text x="{x0}" y="{y0 - 28}" class="legend">{esc(title)}</text>')
        parts.append(f'<text x="{x0 + cell * 1.5}" y="{y0 - 8}" text-anchor="middle" class="axis">Predicted</text>')
        parts.append(f'<text x="{x0 - 48}" y="{y0 + cell * 1.5}" class="axis" transform="rotate(-90 {x0 - 48} {y0 + cell * 1.5})">True</text>')
        for idx, cls in enumerate(CLASSES):
            parts.append(f'<text x="{x0 + idx * cell + cell / 2}" y="{y0 - 10}" text-anchor="middle" class="axis">{cls}</text>')
            parts.append(f'<text x="{x0 - 14}" y="{y0 + idx * cell + cell / 2 + 5}" text-anchor="end" class="axis">{cls}</text>')
        for row_idx, row in enumerate(item["metrics"]["confusion_matrix"]):
            for col_idx, value in enumerate(row):
                x = x0 + col_idx * cell
                y = y0 + row_idx * cell
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{cell_color(value)}" stroke="#d8dee8"/>')
                text_color = "#ffffff" if value / max_cell > 0.55 else "#18212f"
                parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" style="font-family:Arial,Helvetica,sans-serif;font-size:20px;font-weight:700;fill:{text_color}">{value}</text>')
        y_metrics = y0 + cell * 3 + 32
        parts.append(
            f'<text x="{x0}" y="{y_metrics}" class="value">Acc {fmt(item["metrics"]["accuracy"])} | weighted F1 {fmt(item["metrics"]["weighted_f1"])} | class 2 AUROC {fmt(item["metrics"]["auroc_per_class"]["2"])}</text>'
        )
    write_svg(path, "\n".join(parts), width, height)


def make_figures(summary: dict, challenge_rows: list[dict], challenge_agg: list[dict], curves: dict, duke_results: list[dict]) -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OLE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "ole_evaluation_png": OLE_EVALUATION_PNG,
        "ole_validation_auroc": OLE_FIGURE_DIR / "validation_auroc_summary.svg",
        "ole_validation_accuracy": OLE_FIGURE_DIR / "validation_accuracy_summary.svg",
        "split": FIGURE_DIR / "usz_split_distribution.svg",
        "challenge_aggregate": FIGURE_DIR / "challenge_aggregate_metrics.svg",
        "challenge_site_auroc": FIGURE_DIR / "challenge_site_macro_auroc.svg",
        "training": FIGURE_DIR / "local_training_curves.svg",
        "duke_confusion": FIGURE_DIR / "duke_cross_eval_confusion.svg",
    }
    split_chart(figures["split"], summary)

    nodes = ["node_A", "node_B", "node_C"]
    bar_chart(
        figures["ole_validation_auroc"],
        "Extracted Duke swarm validation: AUROC for present classes 0 and 2",
        nodes,
        [
            ("Site model best", [0.9447, 0.8681, 0.9704], "#4e79a7"),
            ("Swarm agg best", [0.9482, 0.8720, 0.9645], "#59a14f"),
            ("Swarm agg last", [0.9482, 0.8622, 0.9556], "#f28e2b"),
        ],
        "AUROC",
    )
    bar_chart(
        figures["ole_validation_accuracy"],
        "Extracted Duke swarm validation: accuracy",
        nodes,
        [
            ("Site model best", [0.8846, 0.8077, 0.9231], "#4e79a7"),
            ("Swarm agg best", [0.8750, 0.8077, 0.8462], "#59a14f"),
            ("Swarm agg last", [0.8462, 0.7308, 0.7692], "#f28e2b"),
        ],
        "Accuracy",
    )

    run_order = ["1DC_epoch14_best", "1DC_last", "MST_best", "MST_last"]
    metric_labels = ["Accuracy", "Macro F1", "Macro AUROC", "Class 2 AUROC", "Class 2 recall"]
    metric_keys = ["accuracy", "macro_f1", "macro_auroc", "class2_auroc", "class2_recall"]
    series = []
    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    by_run = {row["run"]: row for row in challenge_agg}
    for run, color in zip(run_order, palette):
        series.append((run.replace("_", " "), [by_run[run][key] for key in metric_keys], color))
    bar_chart(figures["challenge_aggregate"], "Challenge-style evaluation: sample-weighted aggregate metrics", metric_labels, series, "Score")

    sites = sorted({row["site"] for row in challenge_rows})
    by_run_site = {(row["run"], row["site"]): row for row in challenge_rows}
    site_series = []
    for run, color in zip(run_order, palette):
        site_series.append((run.replace("_", " "), [by_run_site[(run, site)]["macro_auroc"] for site in sites], color))
    bar_chart(figures["challenge_site_auroc"], "Challenge-style evaluation: macro AUROC by site", sites, site_series, "Macro AUROC")

    lines = []
    line_specs = [
        ("1DC val AUROC", curves["1DC"]["validation"], "macro_auroc", "#4e79a7"),
        ("1DC train AUROC", curves["1DC"]["train"], "macro_auroc", "#9ecae1"),
        ("MST val AUROC", curves["MST"]["validation"], "macro_auroc", "#f28e2b"),
        ("MST train AUROC", curves["MST"]["train"], "macro_auroc", "#ffbe7d"),
    ]
    for label, data, key, color in line_specs:
        lines.append((label, [(row["epoch"], row[key]) for row in data], color))
    line_chart(figures["training"], "Local training curves on USZ split", lines, "Macro AUROC")

    confusion_chart(figures["duke_confusion"], duke_results)
    return figures


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def rel(path: Path) -> str:
    return path.relative_to(REPORT_PATH.parent).as_posix()


def build_report(
    summary: dict,
    challenge_rows: list[dict],
    challenge_agg: list[dict],
    curves: dict,
    duke_results: list[dict],
    figures: dict[str, Path],
) -> str:
    split_total = sum(summary["split_distribution"].values())
    split_checks = [
        ["Split UIDs", fmt_int(split_total), "train + val + test"],
        ["Split UIDs with image", fmt_int(summary["split_uids_with_image"]), "matches split UIDs"],
        ["Split UIDs missing image", fmt_int(summary["split_uids_missing_image"]), "expected 0"],
        [
            "Image dirs outside split",
            fmt_int(summary["data_dir_dirs_unrelated_to_split"]),
            f"{fmt_int(summary['data_unilateral_dir_count'])} dirs total - {fmt_int(summary['split_uids_with_image'])} split dirs",
        ],
        ["Annotation UIDs not in split", fmt_int(summary["in_annotation_but_not_split"]), "not evaluated in these artifacts"],
    ]

    split_rows = [
        ["Train", fmt_int(summary["split_distribution"]["train"]), fmt_pct(summary["split_distribution"]["train"] / split_total)],
        ["Validation", fmt_int(summary["split_distribution"]["val"]), fmt_pct(summary["split_distribution"]["val"] / split_total)],
        ["Test", fmt_int(summary["split_distribution"]["test"]), fmt_pct(summary["split_distribution"]["test"] / split_total)],
    ]
    lesion_total = sum(summary["annotation_lesion_distribution"].values())
    lesion_rows = [
        [f"Class {cls}", fmt_int(count), fmt_pct(count / lesion_total)]
        for cls, count in sorted(summary["annotation_lesion_distribution"].items())
    ]

    local_logs = {
        "MST": parse_log_summary(SOURCE_DIR / "logs" / "local_MST.log"),
        "1DC": parse_log_summary(SOURCE_DIR / "logs" / "local_1DC_retry.log"),
    }
    crashed_1dc = parse_log_summary(SOURCE_DIR / "logs" / "local_1DC.log")

    local_rows = []
    for model in ["1DC", "MST"]:
        validation = curves[model]["validation"]
        train = curves[model]["train"]
        best_val = max(validation, key=lambda row: row["macro_auroc"] if row["macro_auroc"] is not None else -1)
        last_val = validation[-1]
        last_train = train[-1]
        ckpt_name = "USZ_1DC_epoch14_best.ckpt" if model == "1DC" else "USZ_MST_best.ckpt"
        ckpt_path = SOURCE_DIR / "checkpoints" / ckpt_name
        local_rows.append(
            [
                "1DivideAndConquer" if model == "1DC" else "MST",
                "epoch 14" if model == "1DC" else "epoch 33",
                str(best_val["epoch"]),
                fmt(best_val["macro_auroc"]),
                fmt(best_val["accuracy"]),
                fmt(best_val["macro_f1"]),
                fmt(last_val["macro_auroc"]),
                fmt(last_val["accuracy"]),
                fmt(last_train["macro_auroc"]),
                fmt(file_size_mb(ckpt_path), 1) + " MB",
            ]
        )

    challenge_rows_md = []
    for item in sorted(challenge_agg, key=lambda row: row["run"]):
        challenge_rows_md.append(
            [
                item["run"],
                fmt_int(item["samples"]),
                fmt(item["accuracy"]),
                fmt(item["macro_f1"]),
                fmt(item["weighted_f1"]),
                fmt(item["macro_auroc"]),
                fmt(item["class2_auroc"]),
                fmt(item["class2_f1"]),
                fmt(item["class2_recall"]),
            ]
        )

    best_site_rows = []
    for run in sorted({row["run"] for row in challenge_rows}):
        run_rows = [row for row in challenge_rows if row["run"] == run and not math.isnan(row["macro_auroc"])]
        if not run_rows:
            continue
        best = max(run_rows, key=lambda row: row["macro_auroc"])
        worst = min(run_rows, key=lambda row: row["macro_auroc"])
        best_site_rows.append([run, best["site"], fmt(best["macro_auroc"]), worst["site"], fmt(worst["macro_auroc"])])

    duke_rows = []
    for item in duke_results:
        metrics = item["metrics"]
        duke_rows.append(
            [
                item["site"],
                item["kind"],
                fmt_int(metrics["num_samples"]),
                fmt(metrics["accuracy"]),
                fmt(metrics["weighted_f1"]),
                fmt(metrics["macro_f1"]),
                fmt(metrics["auroc_per_class"]["0"]),
                fmt(metrics["auroc_per_class"]["2"]),
            ]
        )

    ole_inventory_rows = [
        ["Extracted directory", "`20260424_DUKE_Experiment/`", "`2.6G` after extraction"],
        ["Archive size", "`2,431,153,159 bytes`", "`2.3G` from `ls -lh`"],
        ["Top-level content", "`admin@test.odelia`, `localhost`, `node_A`, `node_B`, `node_C`, `scratch`", "full three-client swarm package"],
        ["Job ID", "`56580928-9848-41bd-b713-61c8818908a3`", "same ID under all node job directories"],
        ["Application", "`ODELIA_ternary_classification`", "from `meta.json`"],
        ["Clients", "`node_A`, `node_B`, `node_C`", "from `meta.json`"],
        ["Submitted", "`2026-04-24T14:13:09.390659+00:00`", "from `meta.json`"],
        ["Global model files", "`app_node_A/B/C/FL_global_model.pt`", "`94,193,524` bytes each"],
        ["Included plot", "`evaluation.png`", "`1492 x 2790`, copied to `docs/figures/ole_swarm/evaluation.png`"],
    ]

    ole_runtime_rows = [
        ["Swarm completion", "Completed", "`round 19` finished on all nodes; logs state 20 rounds completed"],
        ["Metadata status", "Stale/incomplete", "`meta.json` still says `SUBMITTED` even though logs show completion"],
        ["Process return code", "Not numeric", "`_process_rc.txt` contains `None` on each node"],
        ["Job-run error logs", "Empty", "`node_*/565.../log_error.txt` files are empty"],
        ["Top-level node errors", "Important", "all three nodes report UID overlap across train/validation/test"],
        ["Best-model selection", "Broken", "config looked for `accuracy`, but metrics used `val/ACC`; aggregator logged `No global best result!`"],
        ["Final model behavior", "Last result", "node_A broadcast the last result, not a selected global-best result"],
        ["Dataset statistics", "Not included", "`stats_pool_summary.json` files are NVFlare timing histograms, not data distribution reports"],
    ]

    ole_run_rows = [
        ["node_A", "`MST_unilateral_2026_04_24_140845`", "preflight", "1", "104", "`0:49`, `2:55`", "exclude from model-quality interpretation"],
        ["node_A", "`MST_unilateral_2026_04_24_141333`", "real run", "120", "104", "`0:49`, `2:55`", "site plus aggregated prediction CSVs"],
        ["node_B", "`MST_unilateral_2026_04_24_141346`", "real run", "160", "78", "`0:38`, `2:40`", "site plus aggregated prediction CSVs"],
        ["node_C", "`MST_unilateral_2026_04_24_141332`", "real run", "200", "26", "`0:13`, `2:13`", "site plus aggregated prediction CSVs"],
    ]

    ole_validation_rows = [
        ["node_A", "preflight site", "104", "0", "0.561", "0 / 0.519", "0.561 / 0.519"],
        ["node_A", "real site", "104", "0-119", "0.945 @ 66", "50 / 0.885", "0.918 / 0.808"],
        ["node_A", "swarm aggregated", "104", "20 rounds", "0.948 @ 114", "54 / 0.875", "0.948 / 0.846"],
        ["node_B", "real site", "78", "0-159", "0.868 @ 113", "96 / 0.808", "0.823 / 0.654"],
        ["node_B", "swarm aggregated", "78", "20 rounds", "0.872 @ 112", "88 / 0.808", "0.862 / 0.731"],
        ["node_C", "real site", "26", "0-199", "0.970 @ 159", "159 / 0.923", "0.953 / 0.846"],
        ["node_C", "swarm aggregated", "26", "20 rounds", "0.965 @ 180", "160 / 0.846", "0.956 / 0.769"],
    ]

    ole_training_rows = [
        ["node_A real site train", "416", "0-119", "0.949 @ 115", "83 / 0.894", "0.947 / 0.875"],
        ["node_B real site train", "312", "0-159", "0.943 @ 159", "126 / 0.865", "0.943 / 0.782"],
        ["node_C real site train", "104", "0-199", "0.969 @ 199", "159 / 0.913", "0.969 / 0.885"],
    ]

    command_block = """```bash
cd /mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/Ole_swarm

# Confirm the received archive.
ls -lh 20260424_DUKE_Experiment.tar.bz2
sha256sum 20260424_DUKE_Experiment.tar.bz2

# Inspect the archive without extracting it.
tar -tjf 20260424_DUKE_Experiment.tar.bz2 | sed -n '1,120p'

# The current extracted directory should already be present.
du -sh 20260424_DUKE_Experiment
```"""

    return "\n".join(
        [
            "# ODELIA `Ole_swarm` Evaluation Artifact Report",
            "",
            "> Generated by `scripts/evaluation/generate_ole_swarm_report.py`. The main sections summarize the extracted Cosmos package; the final section keeps earlier `workspace/usz_partner_eval` material as supplemental context only.",
            "",
            "## Executive Summary",
            "",
            "- The archive has been extracted on `Cosmos` under `/mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/Ole_swarm/20260424_DUKE_Experiment` and occupies about `2.6G` after extraction.",
            "- The FL job completed 20 swarm rounds across `node_A`, `node_B`, and `node_C`; the `meta.json` status field is stale because it still reads `SUBMITTED`.",
            "- `node_A` has two scratch runs. `MST_unilateral_2026_04_24_140845` is a one-epoch data-access preflight check; `MST_unilateral_2026_04_24_141333` is the actual run to use for model-quality interpretation.",
            "- All extracted prediction CSVs contain only labels `0` and `2` although the model was configured with `num_classes: 3`. Class `1` AUROC is therefore undefined; AUROC values below average only the present classes `0` and `2`.",
            "- The run has two serious caveats: all top-level node error logs report UID overlap across train/validation/test, and global-best selection failed because the aggregator looked for `accuracy` while the clients reported `val/ACC`.",
            "- The included evaluation plot is present but mostly empty/redundant for the absent-class comparisons. The tables and summary charts below are the clearer source of evidence.",
            "",
            "## Transfer and Extraction",
            "",
            markdown_table(
                ["Field", "Value"],
                [
                    ["Remote host", "`jeff@Cosmos`"],
                    ["Remote directory", "`/mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/Ole_swarm`"],
                    ["Archive", "`20260424_DUKE_Experiment.tar.bz2`"],
                    ["Archive size", "`2,431,153,159 bytes` (`2.3G` from `ls -lh`)"],
                    ["Extracted directory", "`20260424_DUKE_Experiment/` (`2.6G`)"],
                    ["Modified", "`2026-05-20 11:43:22 +0200`"],
                    ["SHA-256", "`2f4ee34c3403d3b1851f7a1739018f52ee799ea3f9eb54d521b2f09b9b5734a2`"],
                ],
            ),
            "",
            "## Extracted Package Inventory",
            "",
            markdown_table(["Item", "Value", "Note"], ole_inventory_rows),
            "",
            "## Runtime and Integrity Checks",
            "",
            markdown_table(["Check", "Finding", "Evidence"], ole_runtime_rows),
            "",
            "## Run Inventory",
            "",
            "The sample counts below are per validation epoch and come from the extracted prediction CSVs, not from a separate dataset-statistics file.",
            "",
            markdown_table(["Node", "Run folder", "Role", "Epochs", "Val samples/epoch", "Val labels", "Use"], ole_run_rows),
            "",
            "## Extracted Validation Results",
            "",
            "Metrics were recomputed from `*_gt_and_classprob_validation.csv`. `Best AUROC` is the average of one-vs-rest AUROC for classes `0` and `2`; class `1` is absent and is not included.",
            "",
            f"![Extracted Duke swarm validation AUROC]({rel(figures['ole_validation_auroc'])})",
            "",
            f"![Extracted Duke swarm validation accuracy]({rel(figures['ole_validation_accuracy'])})",
            "",
            markdown_table(
                [
                    "Node",
                    "Stream",
                    "Samples",
                    "Epochs",
                    "Best AUROC",
                    "Best ACC epoch/value",
                    "Last AUROC/ACC",
                ],
                ole_validation_rows,
            ),
            "",
            "### Training CSV Check",
            "",
            "Training CSVs are useful as a sanity check only. They should not override the validation caveats above.",
            "",
            markdown_table(
                ["Stream", "Samples/epoch", "Epochs", "Best AUROC", "Best ACC epoch/value", "Last AUROC/ACC"],
                ole_training_rows,
            ),
            "",
            "## Included Evaluation Plot",
            "",
            f"![Included evaluation plot]({rel(figures['ole_evaluation_png'])})",
            "",
            "The plot confirms the class-coverage issue visually. Rows for `macro`, `none vs benign (0v1)`, and `benign vs malignant (1v2)` are empty because class `1` is not represented. The rows involving class `2` carry the real signal and appear repeated across equivalent binary views when only classes `0` and `2` exist.",
            "",
            "## Interpretation",
            "",
            "- This is a completed three-client swarm run, but it should be described as a last-model artifact rather than a valid selected-global-best artifact.",
            "- The strongest extracted validation AUROC is on `node_C`, but that node has only 26 validation samples per epoch; `node_A` has more stable support with 104 validation samples and a final aggregated AUROC of `0.948`.",
            "- `node_B` is the weakest validation site by both AUROC and final aggregated accuracy, so site-level variability is material.",
            "- Do not present class-1 performance or three-class macro AUROC from this package. The extracted labels only support binary class-0/class-2 interpretation.",
            "- The UID-overlap errors are the main blocker for treating the metrics as final evidence. They need split auditing before the numbers are used externally.",
            "",
            "## Supplemental Workspace Context",
            "",
            "Everything below comes from `workspace/usz_partner_eval`, not from the extracted `Ole_swarm` transfer package. Keep it as internal validation context only.",
            "",
            f"![Challenge aggregate metrics]({rel(figures['challenge_aggregate'])})",
            "",
            f"![Challenge site macro AUROC]({rel(figures['challenge_site_auroc'])})",
            "",
            f"![Duke confusion matrices]({rel(figures['duke_confusion'])})",
            "",
            "### Challenge-Style External Site Evaluation",
            "",
            "The table below is sample-weighted over sites. `macro_auroc` excludes a site only when that site has no valid value for that metric.",
            "",
            markdown_table(
                [
                    "Run",
                    "Samples",
                    "Accuracy",
                    "Macro F1",
                    "Weighted F1",
                    "Macro AUROC",
                    "Class 2 AUROC",
                    "Class 2 F1",
                    "Class 2 Recall",
                ],
                challenge_rows_md,
            ),
            "",
            "### Challenge Site Spread",
            "",
            markdown_table(["Run", "Best site", "Best macro AUROC", "Lowest site", "Lowest macro AUROC"], best_site_rows),
            "",
            "### Supplemental Duke Cross-Evaluation",
            "",
            "These supplemental results use the USZ-trained MST checkpoints on the Duke binary test split. True class `1` is absent in this Duke slice, so class-1 recall/F1 are not clinically interpretable and macro AUROC is unavailable.",
            "",
            markdown_table(
                ["Checkpoint", "Kind", "Samples", "Accuracy", "Weighted F1", "Macro F1", "Class 0 AUROC", "Class 2 AUROC"],
                duke_rows,
            ),
            "",
            f"![Supplemental USZ split distribution]({rel(figures['split'])})",
            "",
            f"![Supplemental prediction-history curves]({rel(figures['training'])})",
            "",
            "### Supplemental Data Checks",
            "",
            markdown_table(["Check", "Value", "Interpretation"], split_checks),
            "",
            "### Supplemental Split Sizes",
            "",
            markdown_table(["Split", "UIDs", "Share"], split_rows),
            "",
            "### Supplemental Annotation Labels",
            "",
            markdown_table(["Label", "UIDs", "Share"], lesion_rows),
            "",
            "### Supplemental Prediction-History Analysis",
            "",
            "The table recomputes metrics from per-epoch prediction CSVs already present in the workspace. It is not evidence that local training was run as part of the `Ole_swarm` package transfer.",
            "",
            markdown_table(
                [
                    "Model",
                    "Saved Best Ckpt",
                    "Peak Val AUROC Epoch",
                    "Peak Val AUROC",
                    "Peak Val Acc",
                    "Peak Val Macro F1",
                    "Last Val AUROC",
                    "Last Val Acc",
                    "Last Train AUROC",
                    "Best Ckpt Size",
                ],
                local_rows,
            ),
            "",
            "### Supplemental Local Run Logs",
            "",
            markdown_table(
                ["Run", "Log", "Exit", "Best checkpoint", "Last checkpoint"],
                [
                    [
                        "1DC retry",
                        f"`{local_logs['1DC']['path']}`",
                        str(local_logs["1DC"]["exit_code"]),
                        f"`{local_logs['1DC']['best_checkpoint']}`",
                        f"`{local_logs['1DC']['last_checkpoint']}`",
                    ],
                    [
                        "MST",
                        f"`{local_logs['MST']['path']}`",
                        str(local_logs["MST"]["exit_code"]),
                        f"`{local_logs['MST']['best_checkpoint']}`",
                        f"`{local_logs['MST']['last_checkpoint']}`",
                    ],
                    [
                        "1DC first attempt",
                        f"`{crashed_1dc['path']}`",
                        str(crashed_1dc["exit_code"]),
                        "NA",
                        "NA",
                    ],
                ],
            ),
            "",
            "## Transfer Note",
            "",
            "The extracted artifact is on the server at:",
            "",
            "`jeff@Cosmos:/mnt/nvme0n1p1/Jeff_projects/Odelia_local_training/Ole_swarm/20260424_DUKE_Experiment`",
            "",
            "Useful archive commands on the server:",
            "",
            command_block,
            "",
        ]
    )


def main() -> None:
    ensure_inputs()
    summary = load_json(SOURCE_DIR / "usz_data_summary.json")
    challenge_rows = load_challenge_metrics(SOURCE_DIR / "odelia_challenge_eval" / "summary_metrics.csv")
    challenge_agg = aggregate_challenge(challenge_rows)
    duke_results = load_json(SOURCE_DIR / "cross_eval_duke" / "prediction_results.json")
    curves = {
        "1DC": {
            "train": epoch_metrics(SOURCE_DIR / "runs" / "1DC_train.csv"),
            "validation": epoch_metrics(SOURCE_DIR / "runs" / "1DC_validation.csv"),
        },
        "MST": {
            "train": epoch_metrics(SOURCE_DIR / "runs" / "MST_train.csv"),
            "validation": epoch_metrics(SOURCE_DIR / "runs" / "MST_validation.csv"),
        },
    }
    figures = make_figures(summary, challenge_rows, challenge_agg, curves, duke_results)
    REPORT_PATH.write_text(build_report(summary, challenge_rows, challenge_agg, curves, duke_results, figures))
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    for figure in figures.values():
        print(f"Wrote {figure.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
