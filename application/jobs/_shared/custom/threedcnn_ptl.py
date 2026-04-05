import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodules import DataModule
from models.models_config import create_model
from env_config import load_environment_variables, prepare_odelia_dataset, prepare_odelia_dataset_without_augmentation, generate_run_directory
import torch.multiprocessing as mp
from hashlib import sha3_224 as hash_function
from typing import List, Tuple
from dataclasses import dataclass

import logging
import csv

FILENAME_GT_PREDPROB_AGGREGATED_MODEL_TRAIN = 'aggregated_model_gt_and_classprob_train.csv'
FILENAME_GT_PREDPROB_SITE_MODEL_TRAIN = 'site_model_gt_and_classprob_train.csv'

FILENAME_GT_PREDPROB_AGGREGATED_MODEL_VALIDATION = 'aggregated_model_gt_and_classprob_validation.csv'
FILENAME_GT_PREDPROB_SITE_MODEL_VALIDATION = 'site_model_gt_and_classprob_validation.csv'

import os


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
                                      excessively many epochs (default 20).
    """
    logger = logging.getLogger(__name__)
    base_epochs = int(os.environ.get("EPOCHS_PER_ROUND", "5"))
    reference_size = int(os.environ.get("EPOCHS_REFERENCE_DATASET_SIZE", "500"))
    max_cap = int(os.environ.get("EPOCHS_MAX_CAP", "20"))

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


def log_data_hash(dm: DataModule, logger, log_dataset_details: bool = False) -> None:
    @dataclass
    class UIDwithHash:
        uid: str
        hash: str

    def _hexdigest(data) -> str:
        return hash_function(data).hexdigest()

    def _hexdigest_string(data) -> str:
        return _hexdigest(data.encode('utf-8'))

    def _get_imageuid_hashes(dataloader) -> List[UIDwithHash]:
        hashes: List[UIDwithHash] = []
        for batch in dataloader:
            assert (len(batch['uid']) == 1)  # currently only implemented for batch size 1
            uid = batch['uid'][0]
            hashes.append(UIDwithHash(uid, _hexdigest_string(uid)))
        return hashes

    def _get_imagedata_hashes(dataloader) -> List[UIDwithHash]:
        hashes: List[UIDwithHash] = []
        for batch in dataloader:
            hashes.append(UIDwithHash(batch['uid'][0], _hexdigest(batch['source']['data'][0].detach().cpu().numpy().data)))
        return hashes

    def _check_for_duplicates(uids_with_hashes_train: List[UIDwithHash],
                              uids_with_hashes_valid: List[UIDwithHash],
                              uids_with_hashes_test: List[UIDwithHash],
                              where: str,
                              log_dataset_details: bool) -> None:
        def _check_separately_for_duplicates(uids_with_hashes: List[UIDwithHash], where: str, which: str, log_dataset_details: bool) -> None:
            if log_dataset_details:
                logger.info(f'All {which} data {where}, UIDs with hashes:\n' + \
                            '\n'.join([f'{i.uid}, {i.hash}' for i in uids_with_hashes]))

            hashes = [i.hash for i in uids_with_hashes]
            if len(hashes) != len(set(hashes)):
                logger.warning(f'Duplicate {where} detected. Please make sure this was intended')
                if log_dataset_details:
                    message = f'Duplicate {where}:\n'
                    if where == 'image UIDs':
                        multiplicity_messages = {}
                        for uh in uids_with_hashes:
                            count = hashes.count(uh.hash)
                            if count > 1:
                                multiplicity_messages[uh.uid] = f'{uh.uid} ({uh.hash}) appears {count} times'
                        message += '\n'.join(multiplicity_messages.values())

                    elif where == 'image data':
                        uids_for_hash = {}
                        for uh in uids_with_hashes:
                            if uh.hash not in uids_for_hash:
                                uids_for_hash[uh.hash] = []
                            count = hashes.count(uh.hash)
                            if count > 1:
                                uids_for_hash[uh.hash].append(uh.uid)

                        for hsh, uids in uids_for_hash.items():
                            if uids:
                                message += f'Image data with hash {hsh} appears {count} times: ' + ', '.join(uids) + '\n'
                    logger.info(message)

        _check_separately_for_duplicates(uids_with_hashes_train, where, 'training', log_dataset_details)
        _check_separately_for_duplicates(uids_with_hashes_valid, where, 'validation', log_dataset_details)
        _check_separately_for_duplicates(uids_with_hashes_test, where, 'test', log_dataset_details)
        _check_separately_for_duplicates(uids_with_hashes_train + uids_with_hashes_valid + uids_with_hashes_test, where, 'training ∪ validation ∪ test', log_dataset_details)

    def _get_imageuid_hashes_train_val_test(dm: DataModule, log_dataset_details: bool) -> Tuple[str, str, str]:
        imageuid_hashes_train = _get_imageuid_hashes(dm.train_dataloader())
        imageuid_hashes_validation = _get_imageuid_hashes(dm.val_dataloader())
        imageuid_hashes_test = _get_imageuid_hashes(dm.test_dataloader())
        _check_for_duplicates(imageuid_hashes_train, imageuid_hashes_validation, imageuid_hashes_test, 'image UIDs', log_dataset_details)

        imageuid_hashes_train = [i.hash for i in imageuid_hashes_train]
        imageuid_hashes_validation = [i.hash for i in imageuid_hashes_validation]
        imageuid_hashes_test = [i.hash for i in imageuid_hashes_test]
        imageuid_hashes_train.sort()
        imageuid_hashes_validation.sort()
        imageuid_hashes_test.sort()
        return ''.join(imageuid_hashes_train), ''.join(imageuid_hashes_validation), ''.join(imageuid_hashes_test)

    def _get_imagedata_hashes_train_val_test(dm: DataModule, log_dataset_details: bool) -> Tuple[str, str, str]:
        imagedata_hashes_train = _get_imagedata_hashes(dm.train_dataloader())
        imagedata_hashes_validation = _get_imagedata_hashes(dm.val_dataloader())
        imagedata_hashes_test = _get_imagedata_hashes(dm.test_dataloader())
        _check_for_duplicates(imagedata_hashes_train, imagedata_hashes_validation, imagedata_hashes_test, 'image data', log_dataset_details)

        imagedata_hashes_train = [i.hash for i in imagedata_hashes_train]
        imagedata_hashes_validation = [i.hash for i in imagedata_hashes_validation]
        imagedata_hashes_test = [i.hash for i in imagedata_hashes_test]
        imagedata_hashes_train.sort()
        imagedata_hashes_validation.sort()
        imagedata_hashes_test.sort()
        return ''.join(imagedata_hashes_train), ''.join(imagedata_hashes_validation), ''.join(imagedata_hashes_test)

    imageuid_hashes_train, imageuid_hashes_validation, imageuid_hashes_test = _get_imageuid_hashes_train_val_test(dm, log_dataset_details)
    imagedata_hashes_train, imagedata_hashes_validation, imagedata_hashes_test = _get_imagedata_hashes_train_val_test(dm, log_dataset_details)
    hash_all = _hexdigest_string(imageuid_hashes_train + imageuid_hashes_validation + imageuid_hashes_test + imagedata_hashes_train + imagedata_hashes_validation + imagedata_hashes_test)
    logger.info(f"Data hash: {hash_all}")


def set_up_data_module(logger, log_dataset_details: bool = False):
    def _log_dataset_hash(logger, log_dataset_details: bool) -> None:
        ds_train_woaug, ds_val_woaug, ds_test_woaug = prepare_odelia_dataset_without_augmentation()
        datamodule = DataModule(
            ds_train=ds_train_woaug,
            ds_val=ds_val_woaug,
            ds_test=ds_test_woaug,
            batch_size=1,
            pin_memory=True,
            weights=None,
            num_workers=mp.cpu_count(),
        )
        log_data_hash(datamodule, logger, log_dataset_details)

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
    _log_dataset_hash(logger, log_dataset_details)
    ds_train, ds_val, ds_test, path_run_dir, run_name = prepare_odelia_dataset(logger, log_dataset_details)
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
        n_classes = len(class_counts)
        # Inverse frequency weighting: w_i = n_samples / (n_classes * count_i)
        class_weights = torch.zeros(n_classes)
        for cls_idx in range(n_classes):
            count = class_counts.get(cls_idx, 1)  # avoid division by zero
            class_weights[cls_idx] = n_samples / (n_classes * count)
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
        num_workers=mp.cpu_count(),
    )

    loss_kwargs = {}
    if class_weights is not None:
        loss_kwargs['weight'] = class_weights

    return dm, path_run_dir, run_name, num_classes, loss_kwargs


def create_run_directory(env_vars):
    return generate_run_directory(
        env_vars['scratch_dir'],
        env_vars['task_data_name'],
        env_vars['model_name'],
        env_vars['local_compare_flag']
    )


def output_GT_and_classprobs_csv(model, data_module: DataModule, epoch: int, csv_filename_train, csv_filename_validation) -> None:
    def _determine_GT_and_classprobs(model, data_loader: torch.utils.data.dataloader.DataLoader):
        results = []
        device = torch.device('cuda')
        for batch in data_loader:
            source, target = batch['source'], batch['target']

            with torch.no_grad():
                logits = model.to(device)(source.to(device))

            pred_prob = model.logits2probabilities(logits)

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

    results_train = _determine_GT_and_classprobs(model, data_module.train_dataloader())
    output_csv(results_train, epoch, csv_filename_train)
    results_validation = _determine_GT_and_classprobs(model, data_module.val_dataloader())
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

        # If model_variant is explicitly provided, override the env var value
        if model_variant is not None:
            env_vars['model_name'] = model_variant

        data_module, path_run_dir, run_name, num_classes, loss_kwargs = set_up_data_module(logger, log_dataset_details)

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

        to_monitor = "val/ACC"
        min_max = "max"
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


def validate_and_train(logger, data_module, model, trainer, path_run_dir, output_GT_and_classprob=True) -> None:
    logger.info("--- Validate global model ---")
    trainer.validate(model, datamodule=data_module)
    if output_GT_and_classprob:
        output_GT_and_classprobs_csv(model, data_module, trainer.current_epoch,
                                     path_run_dir/FILENAME_GT_PREDPROB_AGGREGATED_MODEL_TRAIN,
                                     path_run_dir/FILENAME_GT_PREDPROB_AGGREGATED_MODEL_VALIDATION)

    logger.info("--- Train new model ---")
    trainer.fit(model, datamodule=data_module)


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
