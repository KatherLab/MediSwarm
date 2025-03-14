import os
import logging
from collections import Counter

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data.datamodules import DataModule
from env_config import (
    load_environment_variables, load_prediction_modules,
    prepare_dataset, generate_run_directory
)
from models import ResNet

# Configuration Constants
BATCH_SIZE = 2
ACCUMULATE_GRAD_BATCHES = 1


def get_num_epochs_per_round(site_name: str) -> int:
    """
    Determines the number of epochs per training round based on the site name.

    :param site_name: Name of the dataset site
    :return: Number of epochs for training at the site
    """
    num_epochs_map = {
        "TUD_1": 2, "TUD_2": 4, "TUD_3": 8,
        "MEVIS_1": 2, "MEVIS_2": 4, "UKA": 2
    }
    max_epochs = num_epochs_map.get(site_name, 5)
    logging.info(f"Site: {site_name}, Max Epochs: {max_epochs}")
    return max_epochs


def set_up_logging() -> logging.Logger:
    """
    Sets up logging for the training script.

    :return: Configured logger instance
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def set_up_data_module(env_vars: dict, logger: logging.Logger, site_name: str) -> DataModule:
    """
    Prepares and returns a DataModule for training and validation.

    :param env_vars: Environment variables dictionary
    :param logger: Logger instance
    :param site_name: Site name for dataset preparation
    :return: Configured DataModule instance
    """
    ds, _ = prepare_dataset(env_vars['task_data_name'], env_vars['data_dir'], site_name)
    labels = ds.get_labels()

    indices = list(range(len(ds)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    ds_train, ds_val = Subset(ds, train_indices), Subset(ds, val_indices)

    train_labels = [labels[i] for i in train_indices]
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)

    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"Label '{label}': {percentage:.2f}% of training set (Count: {count})")

    logger.info(f"Train Size: {len(ds_train)}, Val Size: {len(ds_val)}")

    return DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=24
    )


def create_run_directory(env_vars: dict) -> str:
    """
    Creates and returns a directory for storing run data.

    :param env_vars: Environment variables dictionary
    :return: Path to the generated run directory
    """
    return generate_run_directory(env_vars['scratch_dir'], env_vars['task_data_name'],
                                  env_vars['model_name'], env_vars['local_compare_flag'])


def prepare_training(logger: logging.Logger, max_epochs: int, site_name: str):
    """
    Sets up the training environment including model, data, and trainer.

    :param logger: Logger instance
    :param max_epochs: Maximum number of training epochs
    :param site_name: Site name for dataset preparation
    :return: Tuple containing data_module, model, checkpointing, trainer, run directory, and env_vars
    """
    try:
        env_vars = load_environment_variables()
        path_run_dir = create_run_directory(env_vars)

        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for training")

        logger.info("Using GPU for training")
        data_module = set_up_data_module(env_vars, logger, site_name)

        model = ResNet(in_ch=1, model=18)

        to_monitor, min_max = "val/AUC_ROC", "max"
        log_steps = 50
        logger = WandbLogger(project='ODELIA', name=f"ResNet_{site_name}", log_model=False)

        early_stopping = EarlyStopping(monitor=to_monitor, patience=25, mode=min_max)
        checkpointing = ModelCheckpoint(dirpath=path_run_dir, monitor=to_monitor, save_top_k=1, mode=min_max)

        trainer = Trainer(
            accelerator='gpu',
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
            precision='16-mixed',
            default_root_dir=path_run_dir,
            callbacks=[checkpointing, early_stopping],
            check_val_every_n_epoch=1,
            log_every_n_steps=log_steps,
            max_epochs=max_epochs,
            num_sanity_val_steps=2,
            logger=logger
        )

    except Exception as e:
        logger.error(f"Error in prepare_training: {e}")
        raise

    return data_module, model, checkpointing, trainer, path_run_dir, env_vars


def validate_and_train(logger: logging.Logger, data_module: DataModule, model, trainer: Trainer):
    """
    Runs validation and training for the model.

    :param logger: Logger instance
    :param data_module: DataModule instance
    :param model: Model instance
    :param trainer: Trainer instance
    """
    logger.info("--- Validating Model ---")
    trainer.validate(model, datamodule=data_module)

    logger.info("--- Training Model ---")
    trainer.fit(model, datamodule=data_module)


def finalize_training(logger: logging.Logger, model, checkpointing, trainer: Trainer, path_run_dir: str,
                      env_vars: dict):
    """
    Saves the best checkpoint and runs prediction if test data is available.

    :param logger: Logger instance
    :param model: Model instance
    :param checkpointing: Checkpoint callback instance
    :param trainer: Trainer instance
    :param path_run_dir: Path to run directory
    :param env_vars: Environment variables dictionary
    """
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    predict, prediction_flag = load_prediction_modules(env_vars['prediction_flag'])

    test_data_path = os.path.join(env_vars['data_dir'], env_vars['task_data_name'], 'test')
    if os.path.exists(test_data_path):
        predict(path_run_dir, test_data_path, env_vars['model_name'], last_flag=False, prediction_flag=prediction_flag)
    else:
        logger.info("No test data found, skipping evaluation")

    logger.info("Training completed successfully")
