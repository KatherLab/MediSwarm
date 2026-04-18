#!/usr/bin/env python3
"""
Evaluate Duke split predictions as a binary malignant-vs-no-lesion task.

This helper is intentionally lightweight and depends only on the Python
standard library plus scikit-learn so it can run from the repo venv on Cosmos.

Expected input is a predictions CSV produced by `scripts/evaluation/predict.py`
with columns like:
    uid,ground_truth,prediction,prob_class_0,prob_class_1,prob_class_2

Compatibility is also provided for the older CSV shape:
    UID,GT,NN,NN_prob
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Duke predictions as a binary task.",
    )
    parser.add_argument(
        "--predictions-csv",
        required=True,
        help="Path to predictions CSV produced by scripts/evaluation/predict.py",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for metrics JSON output.",
    )
    return parser.parse_args()


def _pick_key(row: Dict[str, str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in row and row[candidate] != "":
            return candidate
    raise KeyError(f"None of the expected columns are present: {candidates}")


def _parse_scalar(value: str) -> float:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            if len(parsed) != 1:
                raise ValueError(f"Expected single-value list, got: {value}")
            return float(parsed[0])
        return float(parsed)
    return float(value)


def load_predictions(path: Path) -> Dict[str, List[float]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")

    for row in rows:
        gt_key = _pick_key(row, "ground_truth", "GT")
        pred_key = _pick_key(row, "prediction", "NN")
        score_key = _pick_key(row, "prob_class_2", "NN_prob")

        raw_gt = int(round(_parse_scalar(row[gt_key])))
        raw_pred = int(round(_parse_scalar(row[pred_key])))
        raw_score = float(_parse_scalar(row[score_key]))

        if raw_gt not in (0, 2):
            raise ValueError(
                f"Unexpected ground-truth label {raw_gt} in {path}. "
                "This Duke split evaluator expects labels 0 and 2 only."
            )

        y_true.append(1 if raw_gt == 2 else 0)
        y_pred.append(1 if raw_pred == 2 else 0)
        y_score.append(raw_score)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def evaluate_binary(data: Dict[str, List[float]]) -> Dict[str, object]:
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_score = data["y_score"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp = cm[0]
    fn, tp = cm[1]

    metrics = {
        "num_samples": len(y_true),
        "num_negative": int(sum(1 for value in y_true if value == 0)),
        "num_positive": int(sum(1 for value in y_true if value == 1)),
        "binary_auroc": float(roc_auc_score(y_true, y_score)),
        "binary_accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "binary_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "binary_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": cm,
        "confusion_matrix_labels": [0, 1],
        "pred_positive_rule": "prediction == 2",
        "score_positive_rule": "prob_class_2",
    }
    return metrics


def main() -> None:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv).resolve()
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    data = load_predictions(predictions_csv)
    metrics = evaluate_binary(data)
    metrics["predictions_csv"] = str(predictions_csv)

    print(
        "DUKE binary metrics:",
        f"samples={metrics['num_samples']}",
        f"auroc={metrics['binary_auroc']:.4f}",
        f"accuracy={metrics['binary_accuracy']:.4f}",
        f"f1={metrics['binary_f1']:.4f}",
        f"recall={metrics['binary_recall']:.4f}",
        f"specificity={metrics['specificity']:.4f}",
    )

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Metrics JSON written to: {output_json}")


if __name__ == "__main__":
    main()
