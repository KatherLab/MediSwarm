#!/usr/bin/env python3
"""
STAMP post-training evaluation / prediction (issue #270).

The STAMP equivalent of ``scripts/evaluation/predict.py``: load a federated
global checkpoint produced by the swarm run (``FL_global_model.pt`` /
``best_FL_global_model.pt``, written by NVFlare's ``PTFileModelPersistor``),
run inference on a held-out split of the evaluation site's data, and record
task-appropriate metrics to a JSON file.

The deploy test (``scripts/deploy/run_stamp_deploy_test.sh``) collects the
per-client checkpoints into a staging workspace laid out as
``<workspace>/app_<SITE>/FL_global_model.pt`` and runs this script inside the
STAMP Docker image with the evaluation site's STAMP_* environment.

Only ``classification`` metrics (AUROC + accuracy) are implemented here.
``survival`` / ``regression`` metrics are left for #271, which extends the
deploy test to those tasks; for those tasks this script records the sample
count and a ``metric_not_implemented`` note rather than failing.

Usage (inside the STAMP image):
    python3 stamp_predict.py --workspace /workspace --output-dir /output
    python3 stamp_predict.py --checkpoint /workspace/app_RUMC_1/best_FL_global_model.pt \\
        --output-dir /output

Configuration is read from the same STAMP_* environment variables the training
uses (STAMP_CLINI_TABLE, STAMP_FEATURE_DIR, STAMP_MODEL_NAME, STAMP_TASK, ...).
The held-out split reproduces training's stratified val partition (STAMP_SEED).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stamp_predict")

# NVFlare PTFileModelPersistor filenames, best first.
CHECKPOINT_NAMES = ["best_FL_global_model.pt", "FL_global_model.pt"]

# Make the STAMP custom dir (this file's dir) and _shared importable when run
# as a plain script inside the image, so ``import stamp_training`` works.
_THIS_DIR = Path(__file__).resolve().parent
for _p in (str(_THIS_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
#  Pure helpers (no torch / STAMP needed — unit-testable standalone)          #
# --------------------------------------------------------------------------- #

def discover_checkpoints(workspace: str, best_only: bool = False) -> List[Dict[str, str]]:
    """Find swarm global checkpoints under a staging workspace.

    Matches ``<workspace>/**/app_<SITE>/{best_,}FL_global_model.pt`` (the layout
    produced by ``collect_checkpoints()`` in the deploy test, and by NVFlare job
    workspaces). De-duplicates on (site, filename), preferring ``best_`` first.

    Returns a list of ``{"site", "name", "path"}`` dicts.
    """
    patterns = ["**/app_*/best_FL_global_model.pt"]
    if not best_only:
        patterns.append("**/app_*/FL_global_model.pt")

    found: List[Dict[str, str]] = []
    seen = set()
    root = Path(workspace)
    for pattern in patterns:
        for ckpt_path in sorted(root.glob(pattern)):
            app_dir = ckpt_path.parent.name  # e.g. "app_RUMC_1"
            site = app_dir[4:] if app_dir.startswith("app_") else app_dir
            key = (site, ckpt_path.name)
            if key in seen:
                continue
            seen.add(key)
            found.append({"site": site, "name": ckpt_path.name, "path": str(ckpt_path)})
    return found


def extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    """Pull the model state dict out of a loaded checkpoint object.

    Handles the three shapes PTFileModelPersistor / Lightning produce:
    ``{"model": sd}``, ``{"state_dict": sd}``, or a bare state dict.
    """
    if isinstance(ckpt, dict):
        model_val = ckpt.get("model")
        if isinstance(model_val, dict):
            return model_val
        sd_val = ckpt.get("state_dict")
        if isinstance(sd_val, dict):
            return sd_val
        return ckpt
    raise ValueError(f"Unexpected checkpoint format (type={type(ckpt).__name__})")


def compute_classification_metrics(
    y_true: List[int], y_prob: List[List[float]]
) -> Dict[str, Any]:
    """Compute accuracy + (macro OVR) AUROC from ground truth and probabilities.

    ``y_prob[i]`` is the per-class probability vector for sample ``i``. AUROC is
    ``None`` when it is undefined (fewer than two classes present in ``y_true``).
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score

    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if y_prob_arr.ndim != 2:
        raise ValueError(f"y_prob must be 2-D (n_samples, n_classes); got shape {y_prob_arr.shape}")

    n_classes = int(y_prob_arr.shape[1])
    y_pred = y_prob_arr.argmax(axis=1)
    accuracy = float(accuracy_score(y_true_arr, y_pred))

    auroc: Optional[float] = None
    classes_present = sorted(set(int(v) for v in y_true_arr.tolist()))
    if len(classes_present) < 2:
        logger.warning(
            "AUROC undefined — only %d class(es) present in the eval labels", len(classes_present)
        )
    else:
        try:
            if n_classes == 2:
                auroc = float(roc_auc_score(y_true_arr, y_prob_arr[:, 1]))
            else:
                auroc = float(
                    roc_auc_score(
                        y_true_arr,
                        y_prob_arr,
                        multi_class="ovr",
                        average="macro",
                        labels=list(range(n_classes)),
                    )
                )
        except Exception as exc:  # noqa: BLE001 — metric is best-effort
            logger.warning("Could not compute AUROC: %s", exc)

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "num_samples": int(y_true_arr.shape[0]),
        "num_classes": n_classes,
    }


def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
    """Compute MSE, MAE, and (with >=2 samples) R^2 for a regression task (#271)."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = int(y_true_arr.shape[0])
    return {
        "mse": float(mean_squared_error(y_true_arr, y_pred_arr)) if n else None,
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)) if n else None,
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if n >= 2 else None,
        "num_samples": n,
    }


def compute_survival_metrics(
    times: List[float], events: List[int], risks: List[float]
) -> Dict[str, Any]:
    """Harrell's concordance index for survival (#271).

    A higher predicted ``risk`` should correspond to a shorter observed
    ``time``. A pair (i, j) is comparable when patient i had an event and
    ``time[i] < time[j]``; concordant when ``risk[i] > risk[j]`` (ties = 0.5).
    ``c_index`` is None when there are no comparable pairs.
    """
    import numpy as np

    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=int)
    risks_arr = np.asarray(risks, dtype=float)
    n = int(times_arr.shape[0])

    concordant = 0.0
    comparable = 0.0
    for i in range(n):
        if events_arr[i] != 1:
            continue
        for j in range(n):
            if times_arr[i] < times_arr[j]:
                comparable += 1.0
                if risks_arr[i] > risks_arr[j]:
                    concordant += 1.0
                elif risks_arr[i] == risks_arr[j]:
                    concordant += 0.5

    return {
        "c_index": float(concordant / comparable) if comparable > 0 else None,
        "num_comparable_pairs": int(comparable),
        "num_samples": n,
    }


# --------------------------------------------------------------------------- #
#  STAMP-dependent runtime (torch + STAMP required)                           #
# --------------------------------------------------------------------------- #

def load_eval_dataloader(env: Dict[str, Any]):
    """Build the held-out validation dataloader for the eval site's data.

    Reuses the training pipeline (``load_stamp_data`` + ``create_stamp_training_model``)
    so the split matches training's seeded stratified val partition. The freshly
    built model is discarded — only its ``valid_dl`` is used, with the *global*
    checkpoint loaded into a separate model for inference.
    """
    from stamp_training import create_stamp_training_model, load_stamp_data

    patient_to_data, feature_type = load_stamp_data(env)
    _model, _train_dl, valid_dl = create_stamp_training_model(env, patient_to_data, feature_type)
    return valid_dl


def build_and_load_model(checkpoint_path: str, device):
    """Instantiate the STAMP model (env-driven) and load a global checkpoint."""
    import torch

    from stamp_model_wrapper import create_stamp_model

    model = create_stamp_model()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(ckpt)
    result = model.load_state_dict(state_dict, strict=False)
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing:
        logger.warning("load_state_dict: %d missing key(s), e.g. %s", len(missing), missing[:3])
    if unexpected:
        logger.warning("load_state_dict: %d unexpected key(s), e.g. %s", len(unexpected), unexpected[:3])
    return model.to(device).eval()


def _unpack_batch(batch):
    """Return (bags, targets) from a STAMP dataloader batch (tuple or dict)."""
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    if isinstance(batch, dict):
        bags = batch.get("bags", batch.get("features"))
        targets = batch.get("targets", batch.get("labels"))
        return bags, targets
    raise ValueError(f"Unexpected batch type: {type(batch).__name__}")


def run_inference(model, dataloader, device):
    """Run the model over a dataloader; return (targets, outputs) as CPU tensors.

    Task-agnostic collection (interpreted per task by evaluate_checkpoint):
    forward pass mirrors ``STAMPPredictionCallback._write_predictions``.
    """
    import torch

    targets_all: List[Any] = []
    outputs_all: List[Any] = []
    with torch.no_grad():
        for batch in dataloader:
            bags, targets = _unpack_batch(batch)
            if bags is None or targets is None:
                continue
            bags = bags.to(device)
            try:
                out = model(bags)
            except Exception:  # noqa: BLE001 — some STAMP models use kwargs
                out = model(features=bags)
            out = out.detach().cpu()
            tgt = targets.detach().cpu() if hasattr(targets, "detach") else targets
            for i in range(out.shape[0]):
                outputs_all.append(out[i])
                targets_all.append(tgt[i])
    return targets_all, outputs_all


def _to_scalar(x) -> float:
    """Squeeze a tensor/number to a single float (first element if a vector)."""
    if hasattr(x, "numel"):
        return float(x.item() if x.numel() == 1 else x.reshape(-1)[0].item())
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


def _survival_target(t):
    """Extract (time, event) from a survival target; None if the shape is unrecognized."""
    try:
        if hasattr(t, "numel") and t.numel() >= 2:
            flat = t.reshape(-1)
            return float(flat[0].item()), int(round(float(flat[1].item())))
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            return float(t[0]), int(round(float(t[1])))
    except Exception:  # noqa: BLE001
        return None
    return None


def evaluate_checkpoint(checkpoint: Dict[str, str], dataloader, env: Dict[str, Any], device) -> Dict[str, Any]:
    """Evaluate one checkpoint on the eval dataloader; dispatch metrics by task."""
    import torch

    task = env.get("task", "classification")
    record: Dict[str, Any] = {
        "site": checkpoint["site"],
        "checkpoint": checkpoint["name"],
        "path": checkpoint["path"],
        "task": task,
    }
    model = build_and_load_model(checkpoint["path"], device)
    targets, outputs = run_inference(model, dataloader, device)
    record["num_samples"] = len(targets)
    if not targets:
        record["error"] = "no_eval_samples"
        return record

    if task == "classification":
        y_true: List[int] = []
        y_prob: List[List[float]] = []
        for tgt, out in zip(targets, outputs):
            if getattr(tgt, "dim", lambda: 1)() != 0:
                continue  # survival/regression target, not a scalar class index
            y_true.append(int(tgt.item()))
            y_prob.append([float(x) for x in torch.softmax(out, dim=-1).tolist()])
        if not y_true:
            record["error"] = "no_scalar_class_targets"
            return record
        record.update(compute_classification_metrics(y_true, y_prob))
    elif task == "regression":
        y_true_r = [_to_scalar(t) for t in targets]
        y_pred_r = [_to_scalar(o) for o in outputs]
        record.update(compute_regression_metrics(y_true_r, y_pred_r))
    elif task == "survival":
        times: List[float] = []
        events: List[int] = []
        risks: List[float] = []
        for tgt, out in zip(targets, outputs):
            te = _survival_target(tgt)
            if te is None:
                record["error"] = "survival_target_shape_unrecognized"
                return record
            times.append(te[0])
            events.append(te[1])
            risks.append(_to_scalar(out))
        record.update(compute_survival_metrics(times, events, risks))
    else:
        record["note"] = f"metric_not_implemented_for_task:{task}"
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--workspace", type=str, help="Staging dir containing app_<SITE>/FL_global_model.pt files.")
    group.add_argument("--checkpoint", type=str, help="Path to a single checkpoint file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for the results JSON.")
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="With --workspace, only evaluate best_FL_global_model.pt.",
    )
    parser.add_argument("--results-name", type=str, default="stamp_eval_results.json", help="Results JSON filename.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import torch

    from stamp_training import load_stamp_environment

    env = load_stamp_environment()
    logger.info("STAMP eval — task=%s model=%s site=%s", env.get("task"), env.get("model_name"), env.get("site_name"))

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        app_dir = ckpt_path.parent.name
        site = app_dir[4:] if app_dir.startswith("app_") else env.get("site_name", "unknown")
        checkpoints = [{"site": site, "name": ckpt_path.name, "path": str(ckpt_path)}]
    else:
        checkpoints = discover_checkpoints(args.workspace, best_only=args.best_only)

    if not checkpoints:
        logger.error("No checkpoints found to evaluate")
        return 1

    logger.info("Found %d checkpoint(s): %s", len(checkpoints), [c["path"] for c in checkpoints])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = load_eval_dataloader(env)

    results: List[Dict[str, Any]] = []
    for ckpt in checkpoints:
        logger.info("Evaluating %s (%s)", ckpt["name"], ckpt["site"])
        try:
            results.append(evaluate_checkpoint(ckpt, dataloader, env, device))
        except Exception as exc:  # noqa: BLE001 — record per-checkpoint failure, keep going
            logger.exception("Evaluation failed for %s", ckpt["path"])
            results.append({**ckpt, "error": str(exc)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.results_name
    payload = {
        "task": env.get("task"),
        "model_name": env.get("model_name"),
        "eval_site": env.get("site_name"),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote results to %s", out_path)

    # Success if at least one checkpoint produced a usable, task-appropriate metric.
    metric_keys = ("accuracy", "auroc", "mse", "mae", "c_index")

    def _scored(r: Dict[str, Any]) -> bool:
        return any(r.get(k) is not None for k in metric_keys)

    scored = [r for r in results if _scored(r)]
    if not scored:
        logger.error("No checkpoint produced a usable metric")
        return 1
    for r in scored:
        summary = {k: r.get(k) for k in metric_keys if r.get(k) is not None}
        logger.info("  %s/%s: %s (n=%s)", r["site"], r["checkpoint"], summary, r.get("num_samples"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
