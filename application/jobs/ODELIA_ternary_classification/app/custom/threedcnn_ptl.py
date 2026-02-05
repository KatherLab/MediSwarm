from sklearn.model_selection import train_test_split
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodules import DataModule
from models import ResNet, MST
from env_config import load_environment_variables, prepare_odelia_dataset, prepare_odelia_dataset_without_augmentation, generate_run_directory
import torch.multiprocessing as mp
from hashlib import sha3_224 as hash_function
from typing import List, Tuple

import logging
import csv

FILENAME_GT_PREDPROB_AGGREGATED_MODEL_TRAIN = 'aggregated_model_gt_and_classprob_train.csv'
FILENAME_GT_PREDPROB_SITE_MODEL_TRAIN = 'site_model_gt_and_classprob_train.csv'

FILENAME_GT_PREDPROB_AGGREGATED_MODEL_VALIDATION = 'aggregated_model_gt_and_classprob_validation.csv'
FILENAME_GT_PREDPROB_SITE_MODEL_VALIDATION = 'site_model_gt_and_classprob_validation.csv'

def get_num_epochs_per_round(site_name: str) -> int:
    NUM_EPOCHS_FOR_SITE = {
        "TUD_1": 2, "TUD_2": 4, "TUD_3": 8,
        "MEVIS_1": 2, "MEVIS_2": 4,
    }
    max_epochs = NUM_EPOCHS_FOR_SITE.get(site_name, 5)
    print(f"Site name: {site_name}")
    print(f"Max epochs set to: {max_epochs}")
    return max_epochs


def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def log_data_hash(dm: DataModule, logger) -> None:
    def _hexdigest(data) -> str:
        return hash_function(data).hexdigest()

    def _hexdigest_string(data) -> str:
        return _hexdigest(data.encode('utf-8'))

    def _get_imageuid_hashes(dataloader) -> List[str]:
        hashes = []
        for batch in dataloader:
            assert (len(batch['uid']) == 1)  # currently only implemented for batch size 1
            hashes.append(_hexdigest_string(batch['uid'][0]))
        return hashes

    def _get_imagedata_hashes(dataloader) -> List[str]:
        hashes = []
        for batch in dataloader:
            hashes.append(_hexdigest(batch['source']['data'][0].detach().cpu().numpy().data))
        return hashes

    def _check_for_duplicates(strings: List[str], where: str) -> None:
        if len(strings) != len(set(strings)):
            print(f"Duplicate {where} detected. Please make sure this was intended")

    def _get_imageuid_hashes_train_val(dm: DataModule) -> Tuple[str, str]:
        imageuid_hashes_train = _get_imageuid_hashes(dm.train_dataloader())
        imageuid_hashes_validation = _get_imageuid_hashes(dm.val_dataloader())
        _check_for_duplicates(imageuid_hashes_train + imageuid_hashes_validation, 'image UIDs')
        imageuid_hashes_train.sort()
        imageuid_hashes_validation.sort()
        return ''.join(imageuid_hashes_train), ''.join(imageuid_hashes_validation)

    def _get_imagedata_hashes_train_val(dm: DataModule) -> Tuple[str, str]:
        imagedata_hashes_train = _get_imagedata_hashes(dm.train_dataloader())
        imagedata_hashes_validation = _get_imagedata_hashes(dm.val_dataloader())
        _check_for_duplicates(imagedata_hashes_train + imagedata_hashes_validation, 'image data')
        imagedata_hashes_train.sort()
        imagedata_hashes_validation.sort()
        return ''.join(imagedata_hashes_train), ''.join(imagedata_hashes_validation)

    imageuid_hashes_train, imageuid_hashes_validation = _get_imageuid_hashes_train_val(dm)
    imagedata_hashes_train, imagedata_hashes_validation = _get_imagedata_hashes_train_val(dm)
    hash_all = _hexdigest_string(imageuid_hashes_train + imageuid_hashes_validation + imagedata_hashes_train + imagedata_hashes_validation)
    logger.info(f"Data hash: {hash_all}")


def set_up_data_module(logger):
    def _log_dataset_hash(logger) -> None:
        ds_train_woaug, ds_val_woaug = prepare_odelia_dataset_without_augmentation()
        datamodule = DataModule(
            ds_train=ds_train_woaug,
            ds_val=ds_val_woaug,
            batch_size=1,
            pin_memory=True,
            weights=None,
            num_workers=mp.cpu_count(),
        )
        log_data_hash(datamodule, logger)

    torch.set_float32_matmul_precision('high')
    _log_dataset_hash(logger)
    ds_train, ds_val, path_run_dir, run_name = prepare_odelia_dataset()
    num_classes = sum(ds_train.class_labels_num)
    logger.info(f"Dataset path: {ds_train}")
    logger.info(f"Run directory: {path_run_dir}")
    logger.info(f"Run name: {run_name}")
    # logger.info(f"Number of classes: {num_classes}")  # number of possible classes, not number of classes present, thus misleading
    logger.info(f"Length of train dataset: {len(ds_train)}")
    logger.info(f"Length of val dataset: {len(ds_val)}")

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,  # TODO shouldn't this remain unset?
        batch_size=1,
        pin_memory=True,
        weights=None,
        num_workers=mp.cpu_count(),
    )

    # # Log label distribution
    # distribution = dm.get_train_label_distribution(lambda sample: sample['label'])
    # logger.info(f"Total samples in training set: {distribution['total']}")
    # for label, pct in distribution['percentages'].items():
    #     logger.info(f"Label '{label}': {pct:.2f}% of training set, Count: {distribution['counts'][label]}")
    # logger.info(f"Number of unique labels: {len(distribution['counts'])}")

    loss_kwargs = {}

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


def prepare_training(logger, max_epochs: int, site_name: str):
    try:
        env_vars = load_environment_variables()
        data_module, path_run_dir, run_name, num_classes, loss_kwargs = set_up_data_module(logger)

        if not torch.cuda.is_available():
            raise RuntimeError("This example requires a GPU")

        logger.info(f"Running code version {env_vars['mediswarm_version']}")
        logger.info(f"Using GPU for training")

        model_name = env_vars['model_name']

        model = None
        if model_name in ['ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
            resnet_variant = int(model_name[6:])
            model = ResNet(n_input_channels=1,
                           num_classes=num_classes,
                           spatial_dims=3,
                           resnet_variant=resnet_variant,
                           loss_kwargs=loss_kwargs)
        elif model_name == 'MST':
            model = MST(n_input_channels=1,
                        num_classes=num_classes,
                        spatial_dims=3,
                        loss_kwargs=loss_kwargs)

        logger.info(f"Using model: {model_name}")

        to_monitor = "val/ACC"
        min_max = "max"
        log_every_n_steps = 50

        '''
        early_stopping = EarlyStopping(
            monitor=to_monitor,
            min_delta=0.0,
            patience=25,
            mode=min_max
        )
        '''
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

        trainer = Trainer(
            accelerator='gpu',
            accumulate_grad_batches=1,
            precision='16-mixed',
            default_root_dir=str(path_run_dir),
            callbacks=[checkpointing, gt_predprob_output],
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
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)

    logger.info('Prediction currently not implemented.')

    logger.info('Training completed successfully.')
