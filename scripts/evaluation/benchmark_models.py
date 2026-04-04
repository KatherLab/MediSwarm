#!/usr/bin/env python3
"""
Benchmark all available MediSwarm models on a consistent train/val/test split.

Outputs a comparison table with:
  - Accuracy, AUC-ROC (macro), per-class F1
  - Confusion matrix
  - Training time per epoch

Usage:
    # Benchmark built-in models only (ResNet variants, MST, Swin3D):
    python benchmark_models.py

    # Benchmark specific models:
    python benchmark_models.py --models MST ResNet18 Swin3D

    # Include challenge models:
    python benchmark_models.py --include-challenge

    # Dry run — list models without training:
    python benchmark_models.py --dry-run

Environment variables (same as normal MediSwarm training):
    DATA_DIR          Path to the ODELIA dataset
    SCRATCH_DIR       Scratch/output directory
    SITE_NAME         Site identifier
    TRAINING_MODE     Should be "local_training" or "preflight_check"
    MEDISWARM_VERSION Version string (default: "benchmark")
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Path setup — make the shared custom directory importable
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
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Built-in model catalogue
# ---------------------------------------------------------------------------
BUILTIN_MODELS = [
    "ResNet10", "ResNet18", "ResNet34", "ResNet50",
    "ResNet101", "ResNet152", "MST", "Swin3D",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MediSwarm models on a consistent split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model names to benchmark.  Default: all built-in models.",
    )
    parser.add_argument(
        "--include-challenge", action="store_true",
        help="Also benchmark registered challenge models.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=5,
        help="Max training epochs per model (default: 5).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for JSON results file (default: <scratch>/benchmark_results.json).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List models that would be benchmarked and exit.",
    )
    return parser.parse_args()


def _get_model_list(args: argparse.Namespace) -> List[str]:
    """Determine which models to benchmark."""
    if args.models:
        return args.models

    models = list(BUILTIN_MODELS)

    if args.include_challenge:
        try:
            from models_config import get_all_model_names
            for name in get_all_model_names():
                models.append(f"challenge_{name}")
        except ImportError:
            logger.warning("Could not import models_config — skipping challenge models.")

    return models


def _setup_data() -> tuple:
    """Set up the data module using the shared MediSwarm pipeline."""
    from threedcnn_ptl import set_up_data_module
    dm, path_run_dir, run_name, num_classes, loss_kwargs = set_up_data_module(logger)
    return dm, path_run_dir, num_classes, loss_kwargs


def _create_model(model_name: str, num_classes: int, loss_kwargs: dict):
    """Create a model instance via the centralized factory."""
    from models_config import create_model
    return create_model(
        logger=logger,
        model_name=model_name,
        num_classes=num_classes,
        loss_kwargs=loss_kwargs.copy(),
    )


def _benchmark_single(
    model_name: str,
    dm,
    num_classes: int,
    loss_kwargs: dict,
    max_epochs: int,
    output_dir: Path,
) -> Dict:
    """Train and evaluate a single model; return metrics dict."""
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from torchmetrics import AUROC, Accuracy
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    result: Dict = {"model": model_name, "status": "error"}

    try:
        logger.info(f"{'=' * 60}")
        logger.info(f"Benchmarking: {model_name}")
        logger.info(f"{'=' * 60}")

        model = _create_model(model_name, num_classes, loss_kwargs)

        run_dir = output_dir / model_name
        run_dir.mkdir(parents=True, exist_ok=True)

        checkpointing = ModelCheckpoint(
            dirpath=str(run_dir),
            monitor="val/ACC",
            save_last=True,
            save_top_k=1,
            mode="max",
        )

        trainer = Trainer(
            accelerator="gpu",
            accumulate_grad_batches=8,
            gradient_clip_val=1.0,
            precision="16-mixed",
            default_root_dir=str(run_dir),
            callbacks=[checkpointing],
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=50,
            max_epochs=max_epochs,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=run_dir),
        )

        # --- Train ---
        t0 = time.time()
        trainer.fit(model, datamodule=dm)
        train_time = time.time() - t0

        # --- Evaluate on test set ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in dm.test_dataloader():
                source = batch["source"].to(device)
                target = batch["target"].to(device)

                logits = model(source)
                probs = model.logits2probabilities(logits)
                preds = model.logits2labels(logits)

                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu().squeeze())
                all_targets.append(target.cpu().squeeze())

        all_probs = torch.cat(all_probs, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        auroc = AUROC(task="multiclass", num_classes=num_classes)
        accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        auroc_val = auroc(all_probs, all_targets).item()
        acc_val = accuracy(all_preds, all_targets).item()

        # Per-class report via sklearn
        y_true = all_targets.numpy()
        y_pred = all_preds.numpy()
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred).tolist()

        result = {
            "model": model_name,
            "status": "success",
            "accuracy": round(acc_val, 4),
            "auroc_macro": round(auroc_val, 4),
            "per_class_f1": {
                str(k): round(v["f1-score"], 4)
                for k, v in report.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            },
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            "confusion_matrix": cm,
            "train_time_seconds": round(train_time, 1),
            "train_time_per_epoch": round(train_time / max(max_epochs, 1), 1),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "best_checkpoint": checkpointing.best_model_path,
        }

        logger.info(
            f"  ACC={acc_val:.4f}  AUC-ROC={auroc_val:.4f}  "
            f"macro-F1={result['macro_f1']:.4f}  "
            f"time={train_time:.0f}s"
        )

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        result["error"] = str(e)

    return result


def _print_summary_table(results: List[Dict]) -> None:
    """Print a formatted summary table to stdout."""
    header = f"{'Model':<25} {'ACC':>8} {'AUC-ROC':>8} {'F1-macro':>8} {'Params':>12} {'Time(s)':>8}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("BENCHMARK RESULTS")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        if r["status"] == "success":
            print(
                f"{r['model']:<25} "
                f"{r['accuracy']:>8.4f} "
                f"{r['auroc_macro']:>8.4f} "
                f"{r['macro_f1']:>8.4f} "
                f"{r['num_parameters']:>12,} "
                f"{r['train_time_seconds']:>8.0f}"
            )
        else:
            print(f"{r['model']:<25} {'FAILED':>8}   {r.get('error', 'unknown')[:40]}")
    print(sep)

    # Per-class F1 detail
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        classes = sorted(successful[0]["per_class_f1"].keys())
        class_header = f"{'Model':<25}" + "".join(f" {'F1-c' + c:>8}" for c in classes)
        print(f"\nPer-class F1 scores:")
        print(class_header)
        for r in successful:
            line = f"{r['model']:<25}"
            for c in classes:
                line += f" {r['per_class_f1'].get(c, 0):>8.4f}"
            print(line)

    # Confusion matrices
    for r in successful:
        cm = r["confusion_matrix"]
        print(f"\nConfusion Matrix — {r['model']}:")
        for row in cm:
            print("  " + "  ".join(f"{v:>5}" for v in row))


def main():
    args = _parse_args()
    model_list = _get_model_list(args)

    if args.dry_run:
        print("Models to benchmark:")
        for m in model_list:
            print(f"  - {m}")
        print(f"\nTotal: {len(model_list)} models")
        return

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required for benchmarking.")
        sys.exit(1)

    logger.info(f"Benchmarking {len(model_list)} models: {model_list}")

    # Set up data once — all models use the same split
    dm, path_run_dir, num_classes, loss_kwargs = _setup_data()

    output_dir = Path(args.output).parent if args.output else path_run_dir / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output) if args.output else output_dir / "benchmark_results.json"

    results: List[Dict] = []
    for model_name in model_list:
        r = _benchmark_single(
            model_name=model_name,
            dm=dm,
            num_classes=num_classes,
            loss_kwargs=loss_kwargs,
            max_epochs=args.max_epochs,
            output_dir=output_dir,
        )
        results.append(r)

        # Save incrementally so partial results survive crashes
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    _print_summary_table(results)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
