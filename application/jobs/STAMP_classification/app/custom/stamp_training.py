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
import re
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
    from stamp.modeling.data import (
        detect_feature_type,
        filter_complete_patient_data_,
        load_patient_level_data,
        patient_to_ground_truth_from_clini_table_,
        patient_to_survival_from_clini_table_,
        slide_to_patient_from_slide_table_,
    )

    clini_table = Path(env["clini_table"])
    feature_dir = Path(env["feature_dir"])
    task = env["task"]
    ground_truth_label = env["ground_truth_label"] if env["ground_truth_label"] else None
    patient_label = env["patient_label"]
    time_label = env["time_label"] if env["time_label"] else None
    status_label = env["status_label"] if env["status_label"] else None
    slide_table = Path(env["slide_table"]) if env["slide_table"] else None
    filename_label = env["filename_label"]

    if slide_table is not None:
        # A site whose patients have several slides (or whose H5 files are named
        # per slide rather than per patient). ``load_patient_level_data`` cannot
        # be used here: it hardcodes ``feature_dir / f"{patient_id}.h5"``, so every
        # patient would look "missing" and we'd silently train on 0 patients.
        # Compose STAMP's slide-table path instead.
        if not slide_table.exists():
            raise FileNotFoundError(
                f"STAMP_SLIDE_TABLE is set but the file does not exist: {slide_table}. "
                "Note this must be a path *inside* the container (under /data/)."
            )

        if task == "survival":
            if not (time_label and status_label):
                raise ValueError(
                    "task=survival requires STAMP_TIME_LABEL and STAMP_STATUS_LABEL"
                )
            patient_to_ground_truth = patient_to_survival_from_clini_table_(
                clini_table_path=clini_table,
                patient_label=patient_label,
                time_label=time_label,
                status_label=status_label,
            )
        else:
            if not ground_truth_label:
                raise ValueError(
                    f"task={task} requires STAMP_GROUND_TRUTH_LABEL"
                )
            patient_to_ground_truth = patient_to_ground_truth_from_clini_table_(
                clini_table_path=clini_table,
                patient_label=patient_label,
                ground_truth_label=ground_truth_label,
            )

        slide_to_patient = slide_to_patient_from_slide_table_(
            slide_table_path=slide_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            filename_label=filename_label,
        )

        patient_to_data = filter_complete_patient_data_(
            patient_to_ground_truth=patient_to_ground_truth,
            slide_to_patient=slide_to_patient,
            drop_patients_with_missing_ground_truth=True,
        )

        logger.info(
            f"Slide table: {slide_table} "
            f"({len(slide_to_patient)} slides -> {len(patient_to_data)} patients, "
            f"patient_label={patient_label!r}, filename_label={filename_label!r})"
        )
    else:
        # One feature file per patient, named <patient_id>.h5.
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

    if not patient_to_data:
        # Without this, training dies much later inside sklearn with the opaque
        # "With n_samples=0, test_size=0.25 ... the resulting train set will be empty".
        hint = (
            "Every patient in the clinical table lacks a matching feature file.\n"
            f"  clini table : {clini_table} (patient column: {patient_label!r})\n"
            f"  feature dir : {feature_dir}\n"
        )
        if slide_table is not None:
            hint += (
                f"  slide table : {slide_table} "
                f"(patient column: {patient_label!r}, filename column: {filename_label!r})\n"
                "  Check that STAMP_PATIENT_LABEL / STAMP_FILENAME_LABEL name real columns in the\n"
                "  slide table, and that its filenames match the files in the feature directory.\n"
            )
        else:
            hint += (
                "  No STAMP_SLIDE_TABLE is set, so each patient's features must be a single file\n"
                f"  named <patient_id>.h5 in the feature directory. If your H5 files are named per\n"
                "  slide, or a patient has several slides, provide a slide table via\n"
                "  STAMP_SLIDE_TABLE (+ STAMP_FILENAME_LABEL).\n"
            )
        raise ValueError("No patients could be loaded — nothing to train on.\n" + hint)

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
        # "validation_auroc" is what STAMP actually logs (confirmed from a real run).
        for key in ["validation_auroc", "val_auroc", "val_MulticlassAUROC"]:
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


# ---------------------------------------------------------------------------
# Scheduler horizon (#503)
# ---------------------------------------------------------------------------
# STAMP sizes its OneCycleLR scheduler once, for max_epochs x total_rounds steps.
# The rounds actually executed come from num_rounds in the job's
# config_fed_server.conf. These were two independent sources of truth: the client
# read STAMP_NUM_ROUNDS from its environment, nothing checked the two agreed, and
# when the server outran the scheduler training died mid-run with
#
#     ValueError: Tried to step 9 times. The specified number of total steps is 8
#
# reported to the server as EXECUTION_EXCEPTION -> FATAL_SYSTEM_ERROR. The message
# names neither variable, and because oversizing is harmless the mismatch stays
# invisible until someone RAISES num_rounds -- so it surfaces late, on a long run.
#
# The job's app (including config/) is deployed to every client, so the client can
# read the authoritative value straight off disk instead of being told separately.

ENV_NUM_ROUNDS = "STAMP_NUM_ROUNDS"
DEFAULT_TOTAL_ROUNDS = 20


def parse_num_rounds(text):
    """Extract ``num_rounds`` from a config_fed_server.conf body, or None."""
    if not text:
        return None
    match = re.search(r"^\s*num_rounds\s*=\s*(\d+)", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def find_server_config(explicit_path=None):
    """Locate the deployed ``config_fed_server.conf``, or None if not found.

    main.py runs as ``app_<SITE>/custom/main.py``, so the job's config sits at
    ``app_<SITE>/config/``. Also tries the working directory, since the launcher's
    cwd has differed between NVFlare versions.
    """
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    here = Path(__file__).resolve().parent
    candidates += [
        here.parent / "config" / "config_fed_server.conf",
        Path.cwd() / "config" / "config_fed_server.conf",
        Path.cwd().parent / "config" / "config_fed_server.conf",
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def resolve_total_rounds(env_value=None, server_config=None, default=DEFAULT_TOTAL_ROUNDS):
    """Return the round count the LR scheduler must be sized for.

    Precedence: the job's ``num_rounds`` (authoritative -- it is what the server
    actually runs) > ``STAMP_NUM_ROUNDS`` > ``default``. When both are present and
    disagree, the job wins and the discrepancy is logged loudly, because following
    the environment instead is what kills the run partway through.
    """
    if env_value is None:
        env_value = os.environ.get(ENV_NUM_ROUNDS)
    env_rounds = None
    if env_value not in (None, ""):
        try:
            env_rounds = int(env_value)
        except (TypeError, ValueError):
            logger.warning("%s=%r is not an integer — ignoring it", ENV_NUM_ROUNDS, env_value)

    config_path = server_config if server_config is not None else find_server_config()
    job_rounds = None
    if config_path is not None:
        try:
            job_rounds = parse_num_rounds(Path(config_path).read_text())
        except OSError as exc:
            logger.warning("Could not read %s: %s", config_path, exc)

    if job_rounds is not None:
        if env_rounds is not None and env_rounds != job_rounds:
            logger.warning(
                "%s=%d disagrees with the job's num_rounds=%d. Using the job's value: "
                "it is what the server runs, and sizing the LR scheduler for fewer "
                "rounds makes training fail partway through (#503).",
                ENV_NUM_ROUNDS, env_rounds, job_rounds,
            )
        logger.info("Scheduler horizon: %d rounds (from the job's config)", job_rounds)
        return job_rounds

    if env_rounds is not None:
        logger.info(
            "Scheduler horizon: %d rounds (from %s; the job's config_fed_server.conf "
            "was not found, so this value is unverified)", env_rounds, ENV_NUM_ROUNDS,
        )
        return env_rounds

    logger.warning(
        "Scheduler horizon: falling back to %d rounds — neither the job's config nor "
        "%s was readable. If the job runs more rounds than this, training will fail "
        "partway through (#503).", default, ENV_NUM_ROUNDS,
    )
    return default


# ---------------------------------------------------------------------------
# Class-order verification
# ---------------------------------------------------------------------------
# Swarm aggregation averages the classifier head position by position. If one
# site infers the class order ["no", "yes"] and another ["yes", "no"], every
# round averages weights that mean opposite things: the run completes, the loss
# looks plausible, and the result is silently meaningless. Nothing in NVFlare or
# STAMP detects this — the shapes match, only the semantics differ.
#
# STAMP infers the order from each site's own clinical table, so it depends on
# that site's label spelling and on which labels are present. Two real examples:
# a site that leaves a third label ("not provided", "FAP") in its table gets a
# 3-class head, and sites that disagree on capitalisation ("Lynch syndrome" vs
# "Lynch Syndrome") produce different orders. The second cost us a DECADE run,
# which aborted mid-flight with EXECUTION_EXCEPTION.
#
# Opt in per run by setting STAMP_EXPECTED_CATEGORIES to the agreed order, e.g.
#     STAMP_EXPECTED_CATEGORIES="Lynch syndrome,Sporadic"
#     STAMP_EXPECTED_CATEGORIES="no,yes"
# Comparison is case-insensitive and whitespace-insensitive but ORDER-SENSITIVE,
# because the order is exactly what must agree. When unset, the inferred order is
# logged and not enforced, so existing runs are unaffected.

ENV_EXPECTED_CATEGORIES = "STAMP_EXPECTED_CATEGORIES"


def parse_expected_categories(raw):
    """Parse a comma-separated class order into a list, or None if unset/blank."""
    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    parts = [part for part in parts if part]
    return parts or None


def categories_match(observed, expected):
    """Order-sensitive, case- and whitespace-insensitive class-order comparison."""
    if observed is None or expected is None:
        return False
    normalize = lambda values: [str(v).strip().casefold() for v in values]  # noqa: E731
    return normalize(observed) == normalize(expected)


def extract_categories(model):
    """Best-effort read of the class order STAMP inferred, or None if unavailable.

    STAMP exposes it as ``model.categories``; some versions only keep it in
    ``model.hparams``. Returns stripped strings so comparison is not thrown off by
    stray whitespace in a clinical table.
    """
    categories = getattr(model, "categories", None)
    if categories is None:
        hparams = getattr(model, "hparams", None)
        if hparams is not None:
            categories = (
                hparams.get("categories")
                if hasattr(hparams, "get")
                else getattr(hparams, "categories", None)
            )
    if categories is None:
        return None
    try:
        return [str(value).strip() for value in list(categories)]
    except TypeError:
        return None


def verify_class_order(env: dict, model, expected_categories=None):
    """Fail before round 0 if this site's class order is not the agreed one.

    Raises ValueError on a mismatch — deliberately before any training happens,
    since the alternative is a run that completes and cannot be trusted. Returns
    the observed class order (or None if it could not be read).
    """
    observed = extract_categories(model)
    if observed is None:
        logger.warning(
            "Could not read the inferred class order from the model — skipping "
            "class-order verification. A cross-site order mismatch would NOT be "
            "detected for this run."
        )
        return None

    # Always worth checking: a stray third label (e.g. 'not provided', 'FAP')
    # silently widens the head and guarantees a mismatch with the other sites.
    num_classes = env.get("num_classes")
    if num_classes is not None and len(observed) != int(num_classes):
        raise ValueError(
            f"Class-order check: this site inferred {len(observed)} classes "
            f"{observed!r} but STAMP_NUM_CLASSES is {num_classes}. Remove any extra "
            "label (e.g. 'not provided') from the clinical table, or correct "
            "STAMP_NUM_CLASSES, before starting the run."
        )

    if expected_categories is None:
        expected_categories = parse_expected_categories(
            os.environ.get(ENV_EXPECTED_CATEGORIES)
        )

    if expected_categories is None:
        logger.info(
            "Inferred class order %r (not verified — set %s to the agreed order "
            "to enforce it across sites)", observed, ENV_EXPECTED_CATEGORIES,
        )
        return observed

    if not categories_match(observed, expected_categories):
        raise ValueError(
            f"Class-order mismatch: this site inferred {observed!r} but the run "
            f"expects {list(expected_categories)!r} (compared case-insensitively, "
            "order matters). Swarm aggregation averages the classifier head by "
            "position, so continuing would silently combine classes that mean "
            "different things. Align the labels in the clinical table, or correct "
            f"{ENV_EXPECTED_CATEGORIES}."
        )

    logger.info("Class order verified: %r", observed)
    return observed


def _select_precision(task: str, use_gpu: bool) -> str:
    """Pick a Lightning precision that the task's loss can actually backprop.

    ``STAMP_PRECISION`` overrides this if set.

    Mixed fp16 is a big speed win and is fine for classification/regression. It is
    **not** usable for ``survival``: STAMP's Cox loss calls ``torch.logcumsumexp``
    (``stamp/modeling/models/cox.py``), whose backward has no half kernel, so
    training dies with::

        RuntimeError: "logcumsumexp_backward" not implemented for 'Half'

    Measured on a client GPU: fp16 backward FAILS, bf16 OK, fp32 OK. So for
    survival prefer bf16 where the GPU supports it and fall back to full precision.
    On CPU always use full precision (fp16/bf16 backward is unsupported on some
    CPU platforms, e.g. DNNL with avx2_vnni_2).
    """
    override = os.environ.get("STAMP_PRECISION", "").strip()
    if override:
        return override

    if not use_gpu:
        return "32-true"

    if task == "survival":
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:  # noqa: BLE001 — older torch / odd driver
            bf16_ok = False
        return "bf16-mixed" if bf16_ok else "32-true"

    return "16-mixed"


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
    # Emit the run directory in the phrasing live-sync's nohup parser recognises,
    # so swarm-mode syncing can locate this run's CSVs even before the directory
    # listing fallback kicks in (#492).
    logger.info(f"Run directory: {output_dir}")

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

    # Per-epoch metrics summary CSV + per-patient prediction CSVs.
    # The prediction callback computes the site model's validation AUROC and
    # publishes it to ``metric_callback`` (shared holder); it must run BEFORE the
    # summary callback so each summary row carries this epoch's AUROC (#492), and
    # it logs ``val_auroc`` so NVFlare's IntimeModelSelector can pick the best
    # round rather than the last (#493).
    from stamp_metrics_callback import STAMPMetricsSummaryCallback, STAMPPredictionCallback
    prediction_callback = STAMPPredictionCallback(
        train_dl, valid_dl, output_dir, metric_holder=metric_callback,
    )
    summary_callback = STAMPMetricsSummaryCallback(output_dir, metric_holder=metric_callback)
    callbacks = [prediction_callback, summary_callback, metric_callback, checkpointing]
    logger.info(f"Metrics CSV output enabled in {output_dir}")

    # FedProx proximal term: penalise local model deviation from global model.
    # Enabled via STAMP_FEDPROX_MU env var (default 0 = disabled).
    fedprox_mu = float(os.environ.get("STAMP_FEDPROX_MU", "0"))
    if fedprox_mu > 0:
        from fedprox_callback import FedProxCallback
        callbacks.append(FedProxCallback(mu=fedprox_mu))
        logger.info(f"FedProx enabled with mu={fedprox_mu}")

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
    precision = _select_precision(env["task"], use_gpu)
    logger.info(f"Trainer precision: {precision} (task={env['task']}, gpu={use_gpu})")
    trainer = Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        precision=precision,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
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

def output_aggregated_predictions(
    model,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    epoch: int,
    output_dir: Path,
):
    """Write per-patient predictions for the aggregated (global) model.

    Called before local training each swarm round to record predictions of
    the globally aggregated model.  This mirrors ODELIA's
    ``output_GT_and_classprobs_csv()`` for the aggregated model.
    """
    from stamp_metrics_callback import (
        STAMPPredictionCallback,
        FILENAME_GT_PREDPROB_AGGREGATED_TRAIN,
        FILENAME_GT_PREDPROB_AGGREGATED_VALIDATION,
    )

    try:
        cb = STAMPPredictionCallback.__new__(STAMPPredictionCallback)
        cb.train_dl = train_dl
        cb.valid_dl = valid_dl
        cb.csv_train = output_dir / FILENAME_GT_PREDPROB_AGGREGATED_TRAIN
        cb.csv_valid = output_dir / FILENAME_GT_PREDPROB_AGGREGATED_VALIDATION

        cb._write_predictions(model, train_dl, epoch, cb.csv_train)
        cb._write_predictions(model, valid_dl, epoch, cb.csv_valid)
    except Exception as e:
        logger.warning(f"Could not write aggregated model predictions: {e}")


def validate_and_train(
    train_dl: DataLoader,
    valid_dl: DataLoader,
    model,
    trainer: Trainer,
    output_dir: Path = None,
    current_round: int = 0,
):
    """Run one round of validation + training (called each swarm round)."""
    logger.info("--- Validate global model ---")
    trainer.validate(model, dataloaders=valid_dl)

    # Write aggregated model predictions (before local training)
    if output_dir is not None:
        output_aggregated_predictions(
            model, train_dl, valid_dl, current_round, output_dir,
        )

    # trainer.validate() (and the prediction export above) leave the module in
    # eval mode, and Lightning does not restore it before fit(). It warns
    # "Found N module(s) in eval mode at the start of training" — which alarmed
    # sites — and, more importantly, any module left in eval would train with
    # dropout disabled. Put it back explicitly.
    model.train()

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
