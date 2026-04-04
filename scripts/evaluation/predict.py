#!/usr/bin/env python3
"""
Run prediction on external test datasets using trained MediSwarm models.

Loads one or more global model checkpoint(s) produced by swarm training
(FL_global_model.pt, best_FL_global_model.pt) and evaluates them on a
test dataset.  Outputs per-sample predictions as CSV and aggregate
metrics (accuracy, AUC-ROC, per-class F1, confusion matrix) to both
the console and a JSON file.

Supports:
  - Single model evaluation (one checkpoint)
  - Multi-model evaluation (all checkpoints across sites)
  - Ensemble prediction (average probabilities from multiple models)

Typical usage after a swarm training run:

    # Evaluate best global model from one site:
    python predict.py \\
        --checkpoint /path/to/best_FL_global_model.pt \\
        --model-name MST

    # Evaluate all checkpoints from a swarm workspace:
    python predict.py \\
        --workspace /path/to/workspace/prod_00/ \\
        --model-name MST

    # Ensemble all site models and also report individual results:
    python predict.py \\
        --workspace /path/to/workspace/prod_00/ \\
        --model-name MST \\
        --ensemble

Environment variables (same as normal MediSwarm training):
    DATA_DIR          Path to the ODELIA dataset (or external test data)
    SCRATCH_DIR       Scratch/output directory for results
    SITE_NAME         Site identifier (for dataset loading)
    MEDISWARM_VERSION (optional) Version string
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Path setup -- make the shared custom directory importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # scripts/evaluation -> project root
_CUSTOM_DIR = _PROJECT_ROOT / "application" / "jobs" / "_shared" / "custom"

sys.path.insert(0, str(_CUSTOM_DIR))
sys.path.insert(0, str(_CUSTOM_DIR / "models"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("predict")

# ---------------------------------------------------------------------------
# Standard NVFlare checkpoint filenames produced by PTFileModelPersistor
# ---------------------------------------------------------------------------
CHECKPOINT_NAMES = ["best_FL_global_model.pt", "FL_global_model.pt"]


# ========================================================================== #
#  Argument parsing                                                          #
# ========================================================================== #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict on external test data using trained MediSwarm models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint", type=str, nargs="+",
        help="Path(s) to one or more model checkpoint files (.pt / .ckpt).",
    )
    group.add_argument(
        "--workspace", type=str,
        help=(
            "Path to a swarm workspace directory (e.g. workspace/prod_00/). "
            "The script will discover all FL_global_model.pt and "
            "best_FL_global_model.pt files across sites."
        ),
    )

    parser.add_argument(
        "--model-name", type=str, default=None,
        help=(
            "Model architecture name (e.g. MST, ResNet101, Swin3D, "
            "challenge_1DivideAndConquer).  If not given, reads MODEL_NAME "
            "from environment."
        ),
    )
    parser.add_argument(
        "--checkpoint-type", type=str, default="state_dict",
        choices=["state_dict", "lightning"],
        help=(
            "Checkpoint format. 'state_dict' (default) for NVFlare "
            "FL_global_model.pt files (bare state dicts). 'lightning' "
            "for PyTorch Lightning .ckpt files."
        ),
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help=(
            "When evaluating multiple checkpoints, also produce an ensemble "
            "prediction by averaging class probabilities."
        ),
    )
    parser.add_argument(
        "--best-only", action="store_true",
        help=(
            "When using --workspace, only evaluate best_FL_global_model.pt "
            "files (skip FL_global_model.pt)."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output files. Default: <SCRATCH_DIR>/predictions/",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Dataset split to predict on (default: test).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for prediction (default: 1).",
    )
    parser.add_argument(
        "--num-classes", type=int, default=3,
        help="Number of output classes (default: 3).",
    )
    return parser.parse_args()


# ========================================================================== #
#  Checkpoint discovery                                                      #
# ========================================================================== #

def discover_checkpoints(workspace: str, best_only: bool = False) -> List[Dict]:
    """
    Walk a workspace directory and find all NVFlare global model checkpoints.

    Returns a list of dicts with keys:
        - path: absolute path to the checkpoint
        - site: site name (extracted from app_<site> directory)
        - kind: 'best' or 'latest'
    """
    workspace = Path(workspace)
    checkpoints = []

    # NVFlare workspace structure:
    # workspace/prod_XX/<server_fqdn>/<job_id>/app_<SITE>/FL_global_model.pt
    # Also search flat structure: workspace/app_<SITE>/FL_global_model.pt
    patterns = [
        # Deep structure: prod_XX/server/job_id/app_SITE/
        "**/app_*/best_FL_global_model.pt",
        "**/app_*/FL_global_model.pt",
    ]

    if best_only:
        patterns = [p for p in patterns if "best_" in p]

    for pattern in patterns:
        for ckpt_path in workspace.glob(pattern):
            # Extract site name from parent directory app_<SITE>
            app_dir = ckpt_path.parent.name  # e.g. "app_UKA"
            if app_dir.startswith("app_"):
                site = app_dir[4:]  # strip "app_" prefix
            else:
                site = app_dir

            kind = "best" if "best_" in ckpt_path.name else "latest"

            checkpoints.append({
                "path": str(ckpt_path.resolve()),
                "site": site,
                "kind": kind,
            })

    # Sort: best before latest, then by site name
    checkpoints.sort(key=lambda c: (c["kind"] != "best", c["site"]))

    return checkpoints


# ========================================================================== #
#  Model loading                                                             #
# ========================================================================== #

def load_model(
    model_name: str,
    checkpoint_path: str,
    checkpoint_type: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Create a model and load a checkpoint into it.

    Args:
        model_name: Architecture name for the model factory.
        checkpoint_path: Path to the checkpoint file.
        checkpoint_type: 'state_dict' or 'lightning'.
        num_classes: Number of output classes.
        device: Target device.

    Returns:
        Model in eval mode on the specified device.
    """
    from models_config import create_model

    logger.info(f"Creating model: {model_name}")
    model = create_model(
        logger=logger,
        model_name=model_name,
        num_classes=num_classes,
    )

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if checkpoint_type == "lightning":
        # PyTorch Lightning checkpoint: state_dict is nested
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        # NVFlare PTFileModelPersistor: bare state_dict or nested
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        elif isinstance(ckpt, dict):
            # Try loading as a direct state dict
            model.load_state_dict(ckpt, strict=False)
        else:
            raise ValueError(
                f"Unexpected checkpoint format at {checkpoint_path}. "
                f"Got type: {type(ckpt)}"
            )

    model = model.to(device)
    model.eval()
    return model


# ========================================================================== #
#  Prediction                                                                #
# ========================================================================== #

def predict_with_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run prediction with a single model.

    Returns:
        probs: (N, C) array of class probabilities
        preds: (N,) array of predicted class indices
        targets: (N,) array of ground-truth class indices
        uids: list of sample UIDs
    """
    all_probs = []
    all_preds = []
    all_targets = []
    all_uids = []

    with torch.no_grad():
        for batch in dataloader:
            source = batch["source"].to(device)
            target = batch["target"]

            logits = model(source)
            probs = model.logits2probabilities(logits)
            preds = model.logits2labels(logits)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy().squeeze())
            all_targets.append(target.numpy().squeeze())

            # Collect UIDs if present
            if "uid" in batch:
                if isinstance(batch["uid"], (list, tuple)):
                    all_uids.extend(batch["uid"])
                else:
                    all_uids.extend([str(u) for u in batch["uid"]])

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    return probs, preds, targets, all_uids


# ========================================================================== #
#  Metrics computation                                                       #
# ========================================================================== #

def compute_metrics(
    probs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> Dict:
    """Compute classification metrics and return as a dict."""
    # Classification report (per-class precision, recall, F1)
    report = classification_report(
        targets, preds, output_dict=True, zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(targets, preds).tolist()

    # AUC-ROC (macro) -- requires probabilities
    try:
        if num_classes == 2:
            auroc = roc_auc_score(targets, probs[:, 1])
        else:
            auroc = roc_auc_score(
                targets, probs, multi_class="ovr", average="macro",
            )
    except ValueError as e:
        logger.warning(f"Could not compute AUC-ROC: {e}")
        auroc = None

    # Per-class AUC-ROC
    per_class_auroc = {}
    for cls_idx in range(num_classes):
        try:
            binary_targets = (targets == cls_idx).astype(int)
            per_class_auroc[str(cls_idx)] = round(
                roc_auc_score(binary_targets, probs[:, cls_idx]), 4
            )
        except ValueError:
            per_class_auroc[str(cls_idx)] = None

    metrics = {
        "accuracy": round(report["accuracy"], 4),
        "auroc_macro": round(auroc, 4) if auroc is not None else None,
        "auroc_per_class": per_class_auroc,
        "per_class_f1": {
            str(k): round(v["f1-score"], 4)
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "per_class_precision": {
            str(k): round(v["precision"], 4)
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "per_class_recall": {
            str(k): round(v["recall"], 4)
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "confusion_matrix": cm,
        "num_samples": int(len(targets)),
    }
    return metrics


# ========================================================================== #
#  Output                                                                    #
# ========================================================================== #

def write_predictions_csv(
    output_path: Path,
    uids: List[str],
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
) -> None:
    """Write per-sample predictions to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["uid", "ground_truth", "prediction"] + [
            f"prob_class_{i}" for i in range(num_classes)
        ]
        writer.writerow(header)
        for i in range(len(targets)):
            uid = uids[i] if i < len(uids) else f"sample_{i}"
            row = [uid, int(targets[i]), int(preds[i])]
            row.extend([round(float(p), 6) for p in probs[i]])
            writer.writerow(row)
    logger.info(f"Predictions CSV written to: {output_path}")


def print_metrics(label: str, metrics: Dict) -> None:
    """Print metrics to console in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    if metrics["auroc_macro"] is not None:
        print(f"  AUC-ROC:      {metrics['auroc_macro']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"  Samples:      {metrics['num_samples']}")

    # Per-class
    print(f"\n  Per-class F1:")
    for cls, f1 in metrics["per_class_f1"].items():
        auroc_str = ""
        if metrics["auroc_per_class"].get(cls) is not None:
            auroc_str = f"  AUC-ROC={metrics['auroc_per_class'][cls]:.4f}"
        print(f"    Class {cls}: F1={f1:.4f}{auroc_str}")

    # Confusion matrix
    print(f"\n  Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print("    " + "  ".join(f"{v:>5}" for v in row))
    print()


# ========================================================================== #
#  Data loading                                                              #
# ========================================================================== #

def setup_dataloader(
    split: str,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """
    Set up a DataLoader for the specified split using ODELIA_Dataset3D.

    Requires environment variables: DATA_DIR, SITE_NAME.
    """
    import torch.multiprocessing as mp
    from data.datasets import ODELIA_Dataset3D

    institution = os.environ.get("INSTITUTION", os.environ.get("SITE_NAME"))
    config = os.environ.get("CONFIG", "unilateral")

    if not institution:
        raise ValueError(
            "SITE_NAME (or INSTITUTION) environment variable must be set "
            "for dataset loading."
        )

    logger.info(f"Loading {split} split for institution={institution}, config={config}")

    ds = ODELIA_Dataset3D(
        institutions=institution,
        split=split,
        config=config,
    )

    logger.info(f"Dataset size: {len(ds)} samples")

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )

    return dataloader


# ========================================================================== #
#  Main                                                                      #
# ========================================================================== #

def main():
    args = _parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required for prediction.")
        sys.exit(1)

    device = torch.device("cuda")

    # Resolve model name
    model_name = args.model_name or os.environ.get("MODEL_NAME", "MST")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        scratch = os.environ.get("SCRATCH_DIR", "/tmp")
        output_dir = Path(scratch) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Collect checkpoints to evaluate                                    #
    # ------------------------------------------------------------------ #
    checkpoint_infos: List[Dict] = []

    if args.checkpoint:
        for ckpt_path in args.checkpoint:
            checkpoint_infos.append({
                "path": str(Path(ckpt_path).resolve()),
                "site": Path(ckpt_path).stem,
                "kind": "best" if "best" in Path(ckpt_path).name else "single",
            })
    elif args.workspace:
        checkpoint_infos = discover_checkpoints(
            args.workspace, best_only=args.best_only,
        )
        if not checkpoint_infos:
            logger.error(
                f"No checkpoints found in workspace: {args.workspace}\n"
                f"Expected files named: {', '.join(CHECKPOINT_NAMES)}"
            )
            sys.exit(1)

    logger.info(f"Found {len(checkpoint_infos)} checkpoint(s) to evaluate")
    for info in checkpoint_infos:
        logger.info(f"  [{info['kind']}] {info['site']}: {info['path']}")

    # ------------------------------------------------------------------ #
    #  Set up data                                                        #
    # ------------------------------------------------------------------ #
    dataloader = setup_dataloader(split=args.split, batch_size=args.batch_size)

    # ------------------------------------------------------------------ #
    #  Evaluate each checkpoint                                           #
    # ------------------------------------------------------------------ #
    all_results = []
    ensemble_probs = []
    reference_targets = None
    reference_uids = None

    for info in checkpoint_infos:
        label = f"{info['site']} ({info['kind']})"
        logger.info(f"\nEvaluating: {label}")

        try:
            model = load_model(
                model_name=model_name,
                checkpoint_path=info["path"],
                checkpoint_type=args.checkpoint_type,
                num_classes=args.num_classes,
                device=device,
            )

            probs, preds, targets, uids = predict_with_model(
                model=model,
                dataloader=dataloader,
                device=device,
                num_classes=args.num_classes,
            )

            # Store reference targets/UIDs for ensemble
            if reference_targets is None:
                reference_targets = targets
                reference_uids = uids

            metrics = compute_metrics(probs, preds, targets, args.num_classes)
            print_metrics(label, metrics)

            # Save per-sample CSV
            safe_name = f"{info['site']}_{info['kind']}"
            csv_path = output_dir / f"predictions_{safe_name}.csv"
            write_predictions_csv(
                csv_path, uids, targets, preds, probs, args.num_classes,
            )

            result = {
                "checkpoint": info["path"],
                "site": info["site"],
                "kind": info["kind"],
                "model_name": model_name,
                "split": args.split,
                "metrics": metrics,
                "predictions_csv": str(csv_path),
            }
            all_results.append(result)

            # Accumulate for ensemble
            if args.ensemble:
                ensemble_probs.append(probs)

            # Free GPU memory between models
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to evaluate {label}: {e}")
            all_results.append({
                "checkpoint": info["path"],
                "site": info["site"],
                "kind": info["kind"],
                "model_name": model_name,
                "split": args.split,
                "status": "error",
                "error": str(e),
            })

    # ------------------------------------------------------------------ #
    #  Ensemble prediction (if requested and multiple models)             #
    # ------------------------------------------------------------------ #
    if args.ensemble and len(ensemble_probs) > 1 and reference_targets is not None:
        logger.info("\nComputing ensemble prediction (probability averaging)...")

        avg_probs = np.mean(ensemble_probs, axis=0)
        avg_preds = np.argmax(avg_probs, axis=1)

        ensemble_metrics = compute_metrics(
            avg_probs, avg_preds, reference_targets, args.num_classes,
        )
        print_metrics(f"ENSEMBLE ({len(ensemble_probs)} models)", ensemble_metrics)

        csv_path = output_dir / "predictions_ensemble.csv"
        write_predictions_csv(
            csv_path,
            reference_uids,
            reference_targets,
            avg_preds,
            avg_probs,
            args.num_classes,
        )

        all_results.append({
            "checkpoint": "ensemble",
            "site": "ensemble",
            "kind": f"avg_{len(ensemble_probs)}_models",
            "model_name": model_name,
            "split": args.split,
            "metrics": ensemble_metrics,
            "predictions_csv": str(csv_path),
            "source_checkpoints": [info["path"] for info in checkpoint_infos],
        })

    # ------------------------------------------------------------------ #
    #  Summary table                                                      #
    # ------------------------------------------------------------------ #
    successful = [r for r in all_results if "metrics" in r]
    if successful:
        print(f"\n{'=' * 70}")
        print("PREDICTION SUMMARY")
        print(f"{'=' * 70}")
        header = f"{'Source':<30} {'ACC':>8} {'AUC-ROC':>8} {'F1-macro':>8} {'Samples':>8}"
        print(header)
        print("-" * len(header))
        for r in successful:
            m = r["metrics"]
            source = f"{r['site']} ({r['kind']})"
            auroc = f"{m['auroc_macro']:.4f}" if m["auroc_macro"] is not None else "N/A"
            print(
                f"{source:<30} "
                f"{m['accuracy']:>8.4f} "
                f"{auroc:>8} "
                f"{m['macro_f1']:>8.4f} "
                f"{m['num_samples']:>8}"
            )
        print()

    # ------------------------------------------------------------------ #
    #  Save JSON results                                                  #
    # ------------------------------------------------------------------ #
    results_file = output_dir / "prediction_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
