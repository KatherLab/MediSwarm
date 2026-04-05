"""
STAMP training pipeline adapted for MediSwarm/NVFlare swarm learning.

This module bridges STAMP's data loading and model creation with NVFlare's
federated training loop. Only the training section of STAMP is integrated
here — preprocessing, deployment, and statistics remain standalone workflows.

Key differences from standalone STAMP training:
1. Data loading uses STAMP 2.4.0's pipeline (H5 features + clinical tables)
2. Model creation uses STAMP 2.4.0's setup_model_for_training() which
   handles dataloaders, class weights, and model instantiation together
3. Training loop is controlled by NVFlare via flare.patch(trainer)
4. Train/val split is done by STAMP internally per site
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.multiprocessing as mp
# STAMP 2.4.0 models inherit from ``lightning.LightningModule`` (the unified
# ``lightning`` package), which is a **different class** from
# ``pytorch_lightning.LightningModule`` in lightning >= 2.0.  Using
# ``pytorch_lightning.Trainer`` to fit a ``lightning.LightningModule`` raises
# ``TypeError: model must be a LightningModule``.  Import from ``lightning``
# to match STAMP's class hierarchy.
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

def load_stamp_environment():
    """Load STAMP-specific environment variables for MediSwarm.

    Returns a dict with all configuration needed for STAMP training.
    Environment variables use the STAMP_ prefix to avoid collision with
    ODELIA-specific variables.
    """
    env = {
        # Core paths
        "site_name": os.environ["SITE_NAME"],
        "scratch_dir": os.environ["SCRATCH_DIR"],
        "mediswarm_version": os.environ.get("MEDISWARM_VERSION", "unset"),

        # STAMP data paths
        "clini_table": os.environ["STAMP_CLINI_TABLE"],
        "feature_dir": os.environ["STAMP_FEATURE_DIR"],
        "slide_table": os.environ.get("STAMP_SLIDE_TABLE", ""),
        "output_dir": os.environ.get("STAMP_OUTPUT_DIR", ""),

        # STAMP task configuration
        "task": os.environ.get("STAMP_TASK", "classification"),
        "ground_truth_label": os.environ.get("STAMP_GROUND_TRUTH_LABEL", ""),
        "patient_label": os.environ.get("STAMP_PATIENT_LABEL", "PATIENT"),
        "filename_label": os.environ.get("STAMP_FILENAME_LABEL", "FILENAME"),
        "time_label": os.environ.get("STAMP_TIME_LABEL", ""),
        "status_label": os.environ.get("STAMP_STATUS_LABEL", ""),

        # STAMP model configuration
        "model_name": os.environ.get("STAMP_MODEL_NAME", "vit"),
        "feature_type": os.environ.get("STAMP_FEATURE_TYPE", ""),  # auto-detect if empty
        "dim_input": int(os.environ.get("STAMP_DIM_INPUT", "1024")),
        "num_classes": int(os.environ.get("STAMP_NUM_CLASSES", "3")),

        # Training hyperparameters
        "bag_size": int(os.environ.get("STAMP_BAG_SIZE", "512")),
        "batch_size": int(os.environ.get("STAMP_BATCH_SIZE", "64")),
        "max_epochs": int(os.environ.get("STAMP_MAX_EPOCHS", "32")),
        "patience": int(os.environ.get("STAMP_PATIENCE", "16")),
        "max_lr": float(os.environ.get("STAMP_MAX_LR", "1e-4")),
        "div_factor": float(os.environ.get("STAMP_DIV_FACTOR", "25.0")),
        "num_workers": int(os.environ.get("STAMP_NUM_WORKERS", str(min(mp.cpu_count(), 8)))),
        "seed": int(os.environ.get("STAMP_SEED", "42")),
    }

    # Derive output_dir if not explicitly set
    if not env["output_dir"]:
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        env["output_dir"] = str(
            Path(env["scratch_dir"]) / "runs" / env["site_name"]
            / f"STAMP_{env['model_name']}_{current_time}"
        )

    return env


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stamp_data(env: dict) -> Tuple[Dict[str, Any], str]:
    """Load STAMP patient data from clinical table + H5 feature files.

    Uses STAMP 2.4.0's ``load_patient_level_data`` to build a
    patient_id → PatientData mapping.  Feature-type detection is done
    separately via ``detect_feature_type``.

    Returns:
        patient_to_data: Mapping of patient_id → PatientData
        feature_type: Detected or overridden feature type string
    """
    from stamp.modeling.data import load_patient_level_data, detect_feature_type

    clini_table = Path(env["clini_table"])
    feature_dir = Path(env["feature_dir"])
    task = env["task"]
    ground_truth_label = env["ground_truth_label"] if env["ground_truth_label"] else None
    patient_label = env["patient_label"]
    time_label = env["time_label"] if env["time_label"] else None
    status_label = env["status_label"] if env["status_label"] else None

    # Load patient data (STAMP 2.4.0 API)
    patient_to_data = load_patient_level_data(
        task=task,
        clini_table=clini_table,
        feature_dir=feature_dir,
        patient_label=patient_label,
        ground_truth_label=ground_truth_label,
        time_label=time_label,
        status_label=status_label,
    )

    # Detect feature type from H5 files, or use override
    if env["feature_type"]:
        feature_type = env["feature_type"]
    else:
        feature_type = detect_feature_type(feature_dir)

    logger.info(f"Loaded {len(patient_to_data)} patients, feature_type={feature_type}")
    logger.info(f"Task: {task}, model: {env['model_name']}")

    return patient_to_data, feature_type


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_stamp_training_model(
    env: dict,
    patient_to_data: Dict[str, Any],
    feature_type: str,
    max_epochs_per_round: int = 0,
    total_rounds: int = 1,
) -> Tuple[Any, DataLoader, DataLoader]:
    """Create a STAMP model configured for training, along with dataloaders.

    Uses STAMP 2.4.0's ``setup_model_for_training()`` which:
    1. Creates train/val dataloaders with stratified split
    2. Computes class weights from training data
    3. Selects correct Lightning wrapper + backbone via registry
    4. Calculates OneCycleLR scheduler steps from data size

    Args:
        max_epochs_per_round: Actual per-round epoch count (after weighted
            epoch computation, if applicable).  When > 0, this value is
            used instead of ``env["max_epochs"]`` for scheduler sizing.
        total_rounds: In swarm mode, the number of federated rounds.  STAMP's
            OneCycleLR scheduler is configured once with ``total_steps =
            max_epochs × steps_per_epoch``.  When NVFlare calls
            ``trainer.fit()`` multiple times (once per round), the scheduler
            must have enough total steps for **all** rounds.  Passing
            ``total_rounds > 1`` multiplies the epoch count used to compute
            ``total_steps`` so the scheduler doesn't overflow.

    Returns:
        model, train_dl, valid_dl
    """
    from stamp.modeling.config import AdvancedConfig, ModelParams
    from stamp.modeling.train import setup_model_for_training
    from stamp.modeling.transforms import VaryPrecisionTransform
    from stamp.modeling.registry import ModelName

    # Build AdvancedConfig from environment.
    # For swarm mode, inflate max_epochs by total_rounds so that STAMP's
    # OneCycleLR scheduler has enough total_steps for the entire training
    # run.  The per-round epoch budget is controlled by the Trainer, not
    # by AdvancedConfig.
    epochs_for_scheduler = max_epochs_per_round if max_epochs_per_round > 0 else env["max_epochs"]
    scheduler_epochs = epochs_for_scheduler * total_rounds
    advanced = AdvancedConfig(
        seed=env["seed"],
        max_epochs=scheduler_epochs,
        patience=env["patience"],
        batch_size=env["batch_size"],
        bag_size=env["bag_size"],
        max_lr=env["max_lr"],
        div_factor=env["div_factor"],
        model_name=ModelName(env["model_name"]),
        num_workers=env["num_workers"],
        model_params=ModelParams(),
    )

    clini_table = Path(env["clini_table"])
    feature_dir = Path(env["feature_dir"])
    slide_table = Path(env["slide_table"]) if env["slide_table"] else None
    ground_truth_label = env["ground_truth_label"] if env["ground_truth_label"] else None
    time_label = env["time_label"] if env["time_label"] else None
    status_label = env["status_label"] if env["status_label"] else None

    model, train_dl, valid_dl = setup_model_for_training(
        patient_to_data=patient_to_data,
        task=env["task"],
        categories=None,  # auto-infer from data
        train_transform=VaryPrecisionTransform(min_fraction_bits=1),
        feature_type=feature_type,
        advanced=advanced,
        ground_truth_label=ground_truth_label,
        time_label=time_label,
        status_label=status_label,
        clini_table=clini_table,
        slide_table=slide_table,
        feature_dir=feature_dir,
    )

    logger.info(f"Created STAMP model: {env['model_name']} with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, train_dl, valid_dl


# ---------------------------------------------------------------------------
# Training preparation
# ---------------------------------------------------------------------------

class ValidationMetricCallback(Callback):
    """Callback to log validation metrics in a format NVFlare can consume."""

    def __init__(self):
        super().__init__()
        self.last_val_loss = None
        self.last_val_auroc = None

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.last_val_loss = metrics.get("validation_loss")
        if self.last_val_loss is not None:
            self.last_val_loss = self.last_val_loss.item()

        # Try to get AUROC if available (classification only)
        for key in ["val_auroc", "val_MulticlassAUROC"]:
            val = metrics.get(key)
            if val is not None:
                self.last_val_auroc = val.item()
                break


def compute_weighted_epochs(num_train_samples: int, site_name: str = "") -> int:
    """Compute per-round epoch count weighted by local dataset size.

    Sites with fewer training samples get more local epochs per round so
    that every site contributes roughly the same number of gradient updates
    to each aggregation round.  The formula is::

        epochs = base_epochs * (reference_size / num_train_samples)

    clamped to [1, max_cap].

    Environment variables (use STAMP_ prefix to avoid collision with ODELIA):
        STAMP_EPOCHS_PER_ROUND              Base epoch count (default 5).
        STAMP_EPOCHS_REFERENCE_DATASET_SIZE Reference dataset size — a site
                                            with exactly this many patients
                                            trains for base_epochs (default 200).
        STAMP_EPOCHS_MAX_CAP                Upper bound (default 20).
    """
    base_epochs = int(os.environ.get("STAMP_EPOCHS_PER_ROUND", "5"))
    reference_size = int(os.environ.get("STAMP_EPOCHS_REFERENCE_DATASET_SIZE", "200"))
    max_cap = int(os.environ.get("STAMP_EPOCHS_MAX_CAP", "20"))

    if num_train_samples <= 0:
        logger.warning(f"num_train_samples={num_train_samples}; falling back to base_epochs={base_epochs}")
        return base_epochs

    raw = base_epochs * (reference_size / num_train_samples)
    epochs = max(1, min(int(round(raw)), max_cap))

    logger.info(
        f"Weighted epochs — site={site_name}, train_samples={num_train_samples}, "
        f"reference_size={reference_size}, base={base_epochs}, "
        f"raw={raw:.1f}, clamped={epochs}"
    )
    return epochs


def prepare_training(
    env: dict,
    max_epochs: int,
    weighted_epochs: bool = False,
    total_rounds: int = 1,
):
    """Set up everything needed for STAMP training.

    Args:
        env: Environment configuration dict from load_stamp_environment().
        max_epochs: Maximum training epochs per round.
        weighted_epochs: If True, compute per-round epoch count from training
                         data size via compute_weighted_epochs().
        total_rounds: Number of federated rounds (swarm mode).  Used to
                      size the OneCycleLR scheduler correctly — see
                      :func:`create_stamp_training_model` for details.
                      For local / preflight training, leave at 1.

    Returns:
        train_dl, valid_dl, model, checkpointing, trainer, output_dir, metric_callback
    """
    torch.set_float32_matmul_precision("high")

    output_dir = Path(env["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"MediSwarm version: {env['mediswarm_version']}")
    logger.info(f"Output directory: {output_dir}")

    # Load patient data
    patient_to_data, feature_type = load_stamp_data(env)

    # Compute weighted epochs from training data size
    if weighted_epochs:
        max_epochs = compute_weighted_epochs(
            len(patient_to_data), env.get("site_name", "")
        )

    # Create model and dataloaders (STAMP 2.4.0 creates both together)
    # Pass the actual per-round epoch count so the scheduler is sized
    # correctly.  In swarm mode with weighted epochs, max_epochs is the
    # weighted value (possibly much larger than env["max_epochs"]).
    model, train_dl, valid_dl = create_stamp_training_model(
        env, patient_to_data, feature_type,
        max_epochs_per_round=max_epochs,
        total_rounds=total_rounds,
    )

    # Determine monitor metric based on task
    task = env["task"]
    if task == "survival":
        monitor_metric, mode = "val_cindex", "max"
    elif task == "classification":
        monitor_metric, mode = "validation_loss", "min"
    else:
        monitor_metric, mode = "validation_loss", "min"

    # Set up callbacks
    checkpointing = ModelCheckpoint(
        dirpath=str(output_dir),
        monitor=monitor_metric,
        save_last=True,
        save_top_k=1,
        mode=mode,
    )

    metric_callback = ValidationMetricCallback()

    # TensorBoard logger is optional — gracefully degrade if tensorboard
    # is not installed (e.g. minimal Docker image, CI environments).
    try:
        tb_logger = TensorBoardLogger(save_dir=output_dir)
    except (ModuleNotFoundError, ImportError):
        logger.warning("tensorboard not available — training will proceed without TensorBoard logging")
        tb_logger = False  # Lightning accepts False to disable logging

    # STAMP models train on pre-extracted features, so training is fast.
    # No gradient accumulation needed (batch_size is already 64).
    # Use mixed precision on GPU for speed; fall back to full precision on
    # CPU because bf16/fp16 backward is not supported on all CPU platforms
    # (e.g. DNNL with avx2_vnni_2 raises RuntimeError).
    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        precision="16-mixed" if use_gpu else "32-true",
        default_root_dir=str(output_dir),
        callbacks=[checkpointing, metric_callback],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=max(len(train_dl), 1),
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        logger=tb_logger,
        devices=1,
    )

    return train_dl, valid_dl, model, checkpointing, trainer, output_dir, metric_callback


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------

def validate_and_train(
    train_dl: DataLoader,
    valid_dl: DataLoader,
    model,
    trainer: Trainer,
):
    """Run one round of validation + training (called each swarm round)."""
    logger.info("--- Validate global model ---")
    trainer.validate(model, dataloaders=valid_dl)

    logger.info("--- Train new model ---")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def finalize_training(model, checkpointing, trainer, output_dir: Path):
    """Save best and latest checkpoints after training completes."""
    # Save best checkpoint (selected by monitor metric)
    best_path = checkpointing.best_model_path
    if best_path:
        final_best = output_dir / "best_model.ckpt"
        shutil.copy(best_path, final_best)
        logger.info(f"Best model saved to: {final_best}")
    else:
        logger.warning("No best checkpoint found")

    # Save latest (last) checkpoint — useful for resuming training or when
    # the best checkpoint was from an early round and the final aggregated
    # model is preferred for deployment.
    last_path = checkpointing.last_model_path
    if last_path:
        final_last = output_dir / "last_model.ckpt"
        shutil.copy(last_path, final_last)
        logger.info(f"Last model saved to: {final_last}")
    else:
        logger.warning("No last checkpoint found")

    logger.info("STAMP training completed successfully.")
