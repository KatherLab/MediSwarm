import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha224 as hash_function
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as torch_mp
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from data.datasets import ODELIA_Dataset3D
from data.datamodules import DataModule
from env_config import (
    build_odelia_manifests,
    generate_run_directory,
    load_environment_variables,
    prepare_odelia_dataset,
    prepare_odelia_dataset_without_augmentation,
)
from models.models_config import create_model

FILENAME_GT_PREDPROB_AGGREGATED_MODEL_TRAIN = 'aggregated_model_gt_and_classprob_train.csv'
FILENAME_GT_PREDPROB_SITE_MODEL_TRAIN = 'site_model_gt_and_classprob_train.csv'

FILENAME_GT_PREDPROB_AGGREGATED_MODEL_VALIDATION = 'aggregated_model_gt_and_classprob_validation.csv'
FILENAME_GT_PREDPROB_SITE_MODEL_VALIDATION = 'site_model_gt_and_classprob_validation.csv'

OMP_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# Default matches the `key_metric` of every job that uses this training script
# (application/jobs/*/app/config/config_fed_client.conf). Those jobs forward their
# selector metric as a KEY_METRIC=... prefix on the launcher command, which the
# SubprocessLauncher strips into the environment.
DEFAULT_KEY_METRIC = "val/AUC_ROC"


def resolve_key_metric():
    """Return (metric, mode) for the local best-model checkpoint.

    Keeps Lightning's ModelCheckpoint in step with NVFlare's IntimeModelSelector.
    The training script runs in a separate subprocess and cannot read the NVFlare
    component config, so the job forwards `key_metric` through the environment (#409).
    """
    metric = os.environ.get("KEY_METRIC", "").strip() or DEFAULT_KEY_METRIC
    mode = os.environ.get("KEY_METRIC_MODE", "").strip().lower()
    if mode not in ("min", "max"):
        mode = "min" if "loss" in metric.lower() else "max"
    return metric, mode


@dataclass
class UIDwithHash:
    uid: str
    hash: str


def _hexdigest(data) -> str:
    return hash_function(data).hexdigest()


def _hexdigest_string(data: str) -> str:
    return _hexdigest(data.encode('utf-8'))


def configure_odelia_runtime(env_vars: dict, logger) -> None:
    thread_count = max(1, env_vars['odelia_threads_per_worker'])
    interop_threads = max(1, env_vars['odelia_interop_threads'])
    sharing_strategy = os.environ.get("TORCH_MULTIPROCESSING_SHARING_STRATEGY", "file_system")

    for env_name in OMP_THREAD_ENV_VARS:
        os.environ[env_name] = str(thread_count)

    if sharing_strategy:
        try:
            torch_mp.set_sharing_strategy(sharing_strategy)
        except RuntimeError as exc:
            logger.warning("Could not set torch multiprocessing sharing strategy to %s: %s", sharing_strategy, exc)

    torch.set_num_threads(thread_count)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            logger.debug("torch interop threads were already configured; keeping existing value")

    logger.info(
        "ODELIA runtime controls — loader_workers=%s, hash_workers=%s, "
        "threads_per_worker=%s, interop_threads=%s, torch_sharing_strategy=%s",
        env_vars['odelia_num_workers'],
        env_vars['odelia_hash_num_workers'],
        thread_count,
        interop_threads,
        torch_mp.get_sharing_strategy(),
    )


@contextmanager
def timed_stage(logger, stage_name: str, stage_timings: Dict[str, float] | None = None):
    logger.info(f"[STAGE] Start: {stage_name}")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error(f"[STAGE] Failed: {stage_name} after {elapsed:.2f}s")
        raise
    else:
        elapsed = time.perf_counter() - start
        if stage_timings is not None:
            stage_timings[stage_name] = elapsed
        logger.info(f"[STAGE] Done: {stage_name} in {elapsed:.2f}s")


def log_stage_timing_summary(logger, stage_timings: Dict[str, float], heading: str) -> None:
    if not stage_timings:
        return
    logger.info(heading)
    for stage_name, elapsed in stage_timings.items():
        logger.info(f"  {stage_name}: {elapsed:.2f}s")


def _load_json_cache(cache_path: Path) -> Dict[str, str]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return cache if isinstance(cache, dict) else {}


def _save_json_cache(cache_path: Path, cache: Dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, sort_keys=True)
    os.replace(tmp_path, cache_path)


def _hash_cache_signature(path_img: Path) -> str:
    stat = path_img.stat()
    return f"{path_img.resolve()}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"


def compute_weighted_epochs(num_train_samples: int, site_name: str = "") -> int:
    """Compute per-round epoch count weighted by local dataset size.

    Sites with fewer training samples get more local epochs per round so
    that every site contributes roughly the same number of gradient updates
    to each aggregation round.  The formula is::

        epochs = base_epochs * (reference_size / num_train_samples)

    clamped to [1, max_cap].

    Environment variables:
        EPOCHS_PER_ROUND              Base epoch count (default 5).
        EPOCHS_REFERENCE_DATASET_SIZE Reference dataset size — a site with
                                      exactly this many samples trains for
                                      base_epochs (default 500).
        EPOCHS_MAX_CAP                Upper bound on computed epochs to
                                      prevent very small sites from running
                                      excessively many epochs (default 10).
    """
    logger = logging.getLogger(__name__)
    base_epochs = int(os.environ.get("EPOCHS_PER_ROUND", "5"))
    reference_size = int(os.environ.get("EPOCHS_REFERENCE_DATASET_SIZE", "500"))
    max_cap = int(os.environ.get("EPOCHS_MAX_CAP", "10"))

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


def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def log_data_hash(ds_train, ds_val, ds_test, env_vars: dict, logger, log_dataset_details: bool = False) -> None:
    def _check_for_duplicates(uids_with_hashes_train: List[UIDwithHash],
                              uids_with_hashes_valid: List[UIDwithHash],
                              uids_with_hashes_test: List[UIDwithHash],
                              where: str,
                              log_dataset_details: bool) -> None:
        def _check_separately_for_duplicates(uids_with_hashes: List[UIDwithHash], where: str, which: str, log_dataset_details: bool) -> None:
            if log_dataset_details:
                logger.info(
                    f'All {which} data {where}, UIDs with hashes:\n'
                    + '\n'.join([f'{item.uid}, {item.hash}' for item in uids_with_hashes])
                )

            hashes = [item.hash for item in uids_with_hashes]
            if len(hashes) != len(set(hashes)):
                logger.error(f'Duplicate {where} detected. This should not happen.')
                if log_dataset_details:
                    message = []
                    if where == 'image UIDs':
                        for uid_hash in uids_with_hashes:
                            count = hashes.count(uid_hash.hash)
                            if count > 1:
                                message.append(f'{uid_hash.uid} ({uid_hash.hash}) appears {count} times')
                    elif where == 'image data':
                        uids_for_hash = {}
                        for uid_hash in uids_with_hashes:
                            uids_for_hash.setdefault(uid_hash.hash, []).append(uid_hash.uid)
                        for image_hash, uids in uids_for_hash.items():
                            if len(uids) > 1:
                                message.append(
                                    f'Image data with hash {image_hash} appears {len(uids)} times: ' + ', '.join(sorted(uids))
                                )
                    message.sort()
                    logger.error(f'Duplicate {where}:\n' + '\n'.join(message))

        _check_separately_for_duplicates(uids_with_hashes_train, where, 'training', log_dataset_details)
        _check_separately_for_duplicates(uids_with_hashes_valid, where, 'validation', log_dataset_details)
        _check_separately_for_duplicates(uids_with_hashes_test, where, 'test', log_dataset_details)
        _check_separately_for_duplicates(
            uids_with_hashes_train + uids_with_hashes_valid + uids_with_hashes_test,
            where,
            'training ∪ validation ∪ test',
            log_dataset_details,
        )

    def _uid_hashes_for_dataset(ds) -> List[UIDwithHash]:
        return [UIDwithHash(uid, _hexdigest_string(uid)) for uid in ds.df['UID'].tolist()]

    def _build_records(ds):
        return [
            (row.UID, row.Institution, ds.get_image_path(row.UID, row.Institution))
            for row in ds.df[['UID', 'Institution']].itertuples(index=False)
        ]

    records_train = _build_records(ds_train)
    records_validation = _build_records(ds_val)
    records_test = _build_records(ds_test)

    imageuid_hashes_train = _uid_hashes_for_dataset(ds_train)
    imageuid_hashes_validation = _uid_hashes_for_dataset(ds_val)
    imageuid_hashes_test = _uid_hashes_for_dataset(ds_test)
    _check_for_duplicates(imageuid_hashes_train, imageuid_hashes_validation, imageuid_hashes_test, 'image UIDs', log_dataset_details)

    unique_records = {}
    for uid, institution, path_img in records_train + records_validation + records_test:
        unique_records[(institution, uid)] = path_img

    hash_cache_path = Path(env_vars['odelia_hash_cache_dir']) / 'image_hash_cache.json'
    hash_cache = _load_json_cache(hash_cache_path)
    new_cache_entries: Dict[str, str] = {}
    cache_hits = 0
    cache_misses = 0

    def _compute_record_hash(item):
        institution, uid = item
        path_img = unique_records[item]
        signature = _hash_cache_signature(path_img)
        cached_hash = hash_cache.get(signature)
        if cached_hash is not None:
            return item, signature, UIDwithHash(uid, cached_hash), True

        image = ds_train.load_img(path_img)
        image_data = np.ascontiguousarray(image.data.detach().cpu().numpy())
        return item, signature, UIDwithHash(uid, _hexdigest(image_data.data)), False

    unique_items = list(unique_records.keys())
    hash_workers = max(0, env_vars['odelia_hash_num_workers'])
    if hash_workers > 0:
        with ThreadPoolExecutor(max_workers=hash_workers) as executor:
            resolved_items = list(executor.map(_compute_record_hash, unique_items))
    else:
        resolved_items = [_compute_record_hash(item) for item in unique_items]

    resolved_hashes = {}
    for item, signature, uid_hash, cache_hit in resolved_items:
        resolved_hashes[item] = uid_hash
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
            new_cache_entries[signature] = uid_hash.hash

    if new_cache_entries:
        hash_cache.update(new_cache_entries)
        _save_json_cache(hash_cache_path, hash_cache)

    def _hashes_for_records(records) -> List[UIDwithHash]:
        return [resolved_hashes[(institution, uid)] for uid, institution, _ in records]

    imagedata_hashes_train = _hashes_for_records(records_train)
    imagedata_hashes_validation = _hashes_for_records(records_validation)
    imagedata_hashes_test = _hashes_for_records(records_test)
    _check_for_duplicates(imagedata_hashes_train, imagedata_hashes_validation, imagedata_hashes_test, 'image data', log_dataset_details)

    logger.info(
        "Hash cache stats — hits=%s, misses=%s, cache_path=%s",
        cache_hits,
        cache_misses,
        hash_cache_path,
    )

    imageuid_hashes_train = sorted(item.hash for item in imageuid_hashes_train)
    imageuid_hashes_validation = sorted(item.hash for item in imageuid_hashes_validation)
    imageuid_hashes_test = sorted(item.hash for item in imageuid_hashes_test)
    imagedata_hashes_train = sorted(item.hash for item in imagedata_hashes_train)
    imagedata_hashes_validation = sorted(item.hash for item in imagedata_hashes_validation)
    imagedata_hashes_test = sorted(item.hash for item in imagedata_hashes_test)

    hash_all = _hexdigest_string(
        ''.join(imageuid_hashes_train) + '|'
        + ''.join(imageuid_hashes_validation) + '|'
        + ''.join(imageuid_hashes_test) + '|'
        + ''.join(imagedata_hashes_train) + '|'
        + ''.join(imagedata_hashes_validation) + '|'
        + ''.join(imagedata_hashes_test)
    )
    logger.info(f"Data hash: {hash_all}")


def set_up_data_module(logger, env_vars: dict, log_dataset_details: bool = False):
    institution = os.environ.get('INSTITUTION', env_vars['site_name'])
    stage_timings: Dict[str, float] = {}

    def _log_dataset_stats(ds_train, ds_val, ds_test, path_run_dir, run_name, logger) -> None:
        def _log_label_distribution(ds, which: str, logger) -> None:
            classes_in_ds = list(ds.df['Lesion'])
            counts = {i: classes_in_ds.count(i) for i in set(classes_in_ds)}
            logger.info(f'Total samples in {which} set: {len(classes_in_ds)}')
            for i in range(3):  # TODO generalize if there is a different number of classes
                if i not in counts:
                    logger.warning(f'No Samples of class {i} in {which} set, please make sure this was intended.')
                else:
                    cclass = counts[i]
                    percentage = 100 * cclass / len(classes_in_ds)
                    logger.info(f'Samples in {which} set of class {i}: {cclass} ({percentage:.1f}%)')

        logger.info(f'Run directory: {path_run_dir}')
        logger.info(f'Run name: {run_name}')

        _log_label_distribution(ds_train, 'training', logger)
        _log_label_distribution(ds_val, 'validation', logger)
        _log_label_distribution(ds_test, 'test', logger)

    torch.set_float32_matmul_precision('high')

    manifests = build_odelia_manifests(env_vars)

    with timed_stage(logger, "UID discrepancy scan", stage_timings):
        ODELIA_Dataset3D.log_UID_discrepancies(
            logger,
            path_root=env_vars['data_dir'],
            institutions=[institution],
            fold=env_vars['fold'],
            log_dataset_details=log_dataset_details,
            manifests=manifests,
        )

    with timed_stage(logger, "No-augmentation dataset build", stage_timings):
        ds_train_woaug, ds_val_woaug, ds_test_woaug = prepare_odelia_dataset_without_augmentation(
            manifests=manifests,
            env_vars=env_vars,
        )

    with timed_stage(logger, "Data hash", stage_timings):
        log_data_hash(ds_train_woaug, ds_val_woaug, ds_test_woaug, env_vars, logger, log_dataset_details)

    with timed_stage(logger, "Augmented dataset build", stage_timings):
        ds_train, ds_val, ds_test, path_run_dir, run_name = prepare_odelia_dataset(
            logger,
            log_dataset_details,
            manifests=manifests,
            env_vars=env_vars,
        )

    _log_dataset_stats(ds_train, ds_val, ds_test, path_run_dir, run_name, logger)
    num_classes = sum(ds_train.class_labels_num)

    # ---------------------------------------------------------------------------
    # Compute class weights from training set label distribution
    # ---------------------------------------------------------------------------
    class_weights = None
    try:
        classes_in_train = list(ds_train.df['Lesion'])
        class_counts = {}
        for c in classes_in_train:
            class_counts[c] = class_counts.get(c, 0) + 1
        n_samples = len(classes_in_train)
        # Use the full number of classes (num_classes) rather than only the
        # classes present in this site's training split.  If a site is missing
        # a class, the weight for that class defaults to 1.0 (the fallback
        # value from torch.zeros replaced below).  This prevents a shape
        # mismatch when CrossEntropyLoss expects weights for all classes.
        class_weights = torch.ones(num_classes)
        for cls_idx in range(num_classes):
            count = class_counts.get(cls_idx, 0)
            if count > 0:
                class_weights[cls_idx] = n_samples / (num_classes * count)
        logger.info(f"Computed class weights from training set: {class_weights.tolist()}")
        logger.info(f"Class counts: {class_counts}")
    except Exception as e:
        logger.warning(f"Could not compute class weights: {e}. Using uniform weights.")
        class_weights = None

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        batch_size=1,
        pin_memory=True,
        weights=None,
        num_workers=env_vars['odelia_num_workers'],
    )

    loss_kwargs = {}
    if class_weights is not None:
        loss_kwargs['weight'] = class_weights

    log_stage_timing_summary(logger, stage_timings, "ODELIA setup stage timings:")
    return dm, path_run_dir, run_name, num_classes, loss_kwargs


def create_run_directory(env_vars):
    return generate_run_directory(
        env_vars['scratch_dir'],
        env_vars['task_data_name'],
        env_vars['model_name'],
        env_vars['local_compare_flag']
    )


def output_GT_and_classprobs_csv(model, data_module: DataModule, epoch: int, csv_filename_train, csv_filename_validation) -> None:
    def _determine_GT_and_classprobs(model, data_loader: torch.utils.data.dataloader.DataLoader, device: torch.device):
        results = []
        for batch in data_loader:
            source, target = batch['source'], batch['target']

            with torch.no_grad():
                logits = model(source.to(device))

            pred_prob = model.logits2probabilities(logits).detach().cpu()

            for b in range(pred_prob.size(0)):
                results.append({'GT': target[b].tolist(),
                                'pred_prob': pred_prob[b].tolist(),
                                })
        return results

    def output_csv(results, epoch: int, csv_filename) -> None:
        with open(csv_filename, 'a') as csvfile:
            datawriter = csv.writer(csvfile)
            for datapoint in results:
                output_data = [epoch, datapoint['GT'][0]] + datapoint['pred_prob']
                datawriter.writerow(output_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    results_train = _determine_GT_and_classprobs(model, train_loader, device)
    output_csv(results_train, epoch, csv_filename_train)
    results_validation = _determine_GT_and_classprobs(model, val_loader, device)
    output_csv(results_validation, epoch, csv_filename_validation)



class GT_PredProb_Output_Callback(Callback):
    def __init__(self, data_module, csv_filename_train, csv_filename_validation):
        self.data_module = data_module
        self.csv_filename_train = csv_filename_train
        self.csv_filename_validation = csv_filename_validation
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        output_GT_and_classprobs_csv(pl_module,
                                     self.data_module,
                                     trainer.current_epoch,
                                     self.csv_filename_train,
                                     self.csv_filename_validation)


def prepare_training(logger, max_epochs: int, site_name: str = None,
                     log_dataset_details: bool = False, model_variant: str = None,
                     weighted_epochs: bool = False):
    """
    Unified prepare_training that supports both ODELIA and challenge job patterns.

    Args:
        logger: Logger instance
        max_epochs: Maximum training epochs (used as-is for local/preflight,
                    or as base_epochs for weighted computation)
        site_name: Site name (used for logging, read from env if not provided)
        log_dataset_details: Whether to log detailed dataset information
        model_variant: Optional model variant override. If provided, overrides MODEL_NAME env var.
                       Can be 'MST', 'ResNet101', 'challenge', '1DivideAndConquer', etc.
        weighted_epochs: If True, compute per-round epoch count from training
                         data size via compute_weighted_epochs(). The max_epochs
                         parameter is ignored in this case; configuration comes
                         from EPOCHS_PER_ROUND / EPOCHS_REFERENCE_DATASET_SIZE
                         environment variables instead.
    """
    try:
        env_vars = load_environment_variables()
        configure_odelia_runtime(env_vars, logger)

        # If model_variant is explicitly provided, override the env var value
        if model_variant is not None:
            env_vars['model_name'] = model_variant

        data_module, path_run_dir, run_name, num_classes, loss_kwargs = set_up_data_module(
            logger,
            env_vars,
            log_dataset_details,
        )

        # Compute weighted epochs based on training data size
        if weighted_epochs:
            num_train_samples = len(data_module.ds_train)
            max_epochs = compute_weighted_epochs(num_train_samples, site_name or "")

        # Use the centralized model factory for all models
        model = create_model(
            logger=logger,
            model_name=env_vars['model_name'],
            num_classes=num_classes,
            loss_kwargs=loss_kwargs,
            env_vars=env_vars
        )

        logger.info(f"Using model: {env_vars['model_name']}")

        to_monitor, min_max = resolve_key_metric()
        logger.info(f"Local best checkpoint monitors {to_monitor} (mode={min_max})")
        log_every_n_steps = 50

        checkpointing = ModelCheckpoint(
            dirpath=str(path_run_dir),
            monitor=to_monitor,
            save_last=True,
            save_top_k=1,
            mode=min_max,
        )

        gt_predprob_output = GT_PredProb_Output_Callback(data_module,
                                                         path_run_dir/FILENAME_GT_PREDPROB_SITE_MODEL_TRAIN,
                                                         path_run_dir/FILENAME_GT_PREDPROB_SITE_MODEL_VALIDATION)

        callbacks = [checkpointing, gt_predprob_output]

        # FedProx proximal term: penalise local model deviation from global model.
        # Enabled via FEDPROX_MU env var (default 0 = disabled).
        fedprox_mu = float(os.environ.get("FEDPROX_MU", "0"))
        if fedprox_mu > 0:
            from fedprox_callback import FedProxCallback
            callbacks.append(FedProxCallback(mu=fedprox_mu))
            logger.info(f"FedProx enabled with mu={fedprox_mu}")

        # Gradient accumulation: with batch_size=1, accumulate 8 steps for
        # an effective batch size of 8.  This stabilises training considerably
        # compared to pure SGD updates every sample.
        accumulate_grad_batches = 8

        trainer = Trainer(
            accelerator='gpu',
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=1.0,
            precision='16-mixed',
            default_root_dir=str(path_run_dir),
            callbacks=callbacks,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=log_every_n_steps,
            max_epochs=max_epochs,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )

    except Exception as e:
        logger.error(f"Error in prepare_training: {e}")
        raise

    return data_module, model, checkpointing, trainer, path_run_dir, env_vars


def should_export_aggregated_predictions(current_round, total_rounds) -> bool:
    """Whether to export per-sample aggregated predictions this swarm round (#314).

    The per-round aggregated prediction export (full train+val inference at
    batch_size=1 plus CSV writes, in output_GT_and_classprobs_csv) dominates swarm
    round time -- it made the PR301 validation ~3.3x slower -- and is not
    informative more than once or twice. It is therefore throttled via the
    ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS environment variable:
        unset / 0 -> final round only (default)
        1         -> every round (legacy behaviour)
        N (>1)    -> every Nth round, plus the final round
    """
    try:
        every_n = int(os.environ.get("ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS", "0") or "0")
    except (TypeError, ValueError):
        every_n = 0
    is_final = total_rounds is not None and total_rounds > 0 and current_round >= total_rounds - 1
    if every_n <= 0:
        return is_final
    if every_n == 1:
        return True
    return is_final or (current_round % every_n == 0)


def validate_and_train(logger, data_module, model, trainer, path_run_dir, output_GT_and_classprob=True) -> None:
    stage_timings: Dict[str, float] = {}

    with timed_stage(logger, "Validation", stage_timings):
        logger.info("--- Validate global model ---")
        trainer.validate(model, datamodule=data_module)

    if output_GT_and_classprob:
        with timed_stage(logger, "Aggregated prediction export", stage_timings):
            output_GT_and_classprobs_csv(
                model,
                data_module,
                trainer.current_epoch,
                path_run_dir/FILENAME_GT_PREDPROB_AGGREGATED_MODEL_TRAIN,
                path_run_dir/FILENAME_GT_PREDPROB_AGGREGATED_MODEL_VALIDATION,
            )

    with timed_stage(logger, "Fit", stage_timings):
        logger.info("--- Train new model ---")
        trainer.fit(model, datamodule=data_module)

    log_stage_timing_summary(logger, stage_timings, "ODELIA execution stage timings:")


def finalize_training(logger, model, checkpointing, trainer, path_run_dir, env_vars) -> None:
    """Save best and latest checkpoints after training completes."""
    import shutil

    # Save best checkpoint (highest val/ACC)
    best_path = checkpointing.best_model_path
    if best_path:
        model.save_best_checkpoint(trainer.logger.log_dir, best_path)
        logger.info(f'Best model checkpoint: {best_path}')
    else:
        logger.warning('No best checkpoint found.')

    # Save latest (last) checkpoint — useful for resuming training or when
    # the best checkpoint was from an early round and the final aggregated
    # model is preferred for deployment.
    last_path = checkpointing.last_model_path
    if last_path:
        final_last = path_run_dir / "last_global_model.ckpt"
        shutil.copy(last_path, final_last)
        logger.info(f'Last model saved to: {final_last}')
    else:
        logger.warning('No last checkpoint found.')

    logger.info('Training completed successfully.')
