from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from collections import Counter
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodules import DataModule
from model_selector import select_model
from env_config import load_environment_variables, load_prediction_modules, prepare_dataset, generate_run_directory

import os
import logging


def get_num_epochs_per_round(site_name: str) -> int:
    #TODO: Set max_epochs based on the data set size
    NUM_EPOCHS_FOR_SITE = { "TUD_1":   2,
                            "TUD_2":   4,
                            "TUD_3":   8,
                            "MEVIS_1": 2,
                            "MEVIS_2": 4,
                            "UKA":     2,
                           }

    if site_name in NUM_EPOCHS_FOR_SITE.keys():
        MAX_EPOCHS = NUM_EPOCHS_FOR_SITE[site_name]
    else:
        MAX_EPOCHS = 5

    print(f"Site name: {site_name}")
    print(f"Max epochs set to: {MAX_EPOCHS}")

    return MAX_EPOCHS


def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def set_up_data_module(env_vars, logger, site_name: str):
    ds, task_data_name = prepare_dataset(env_vars['task_data_name'], env_vars['data_dir'], site_name=site_name)

    labels = ds.get_labels()

    # Generate indices and perform stratified split
    indices = list(range(len(ds)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

    # Create training and validation subsets
    ds_train = Subset(ds, train_indices)
    ds_val = Subset(ds, val_indices)

    # Extract training labels using the train_indices
    train_labels = [labels[i] for i in train_indices]
    label_counts = Counter(train_labels)

    # Calculate the total number of samples in the training set
    total_samples = len(train_labels)

    # Print the percentage of the training set for each label
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"Label '{label}': {percentage:.2f}% of the training set, Exact count: {count}")

    logger.info(f"Total number of different labels in the training set: {len(label_counts)}")

    ads_val_data = DataLoader(ds_val, batch_size=2, shuffle=False)
    logger.info(f'ads_val_data type: {type(ads_val_data)}')

    train_size = len(ds_train)
    val_size = len(ds_val)
    logger.info(f'Train size: {train_size}')
    logger.info(f'Val size: {val_size}')

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=1,
        num_workers=16,
        pin_memory=True,
    )

    return dm


def create_run_directory(env_vars):
    path_run_dir = generate_run_directory(env_vars['scratch_dir'], env_vars['task_data_name'], env_vars['model_name'], env_vars['local_compare_flag'])
    return path_run_dir


def prepare_training(logger, max_epochs:int , site_name: str):
    try:
        env_vars = load_environment_variables()
        path_run_dir = create_run_directory(env_vars)
        if not torch.cuda.is_available():
            raise(RuntimeError("This example does not work without GPU"))
        accelerator = 'gpu'
        logger.info(f"Using {accelerator} for training")

        data_module = set_up_data_module(env_vars, logger, site_name)

        # max_epochs = env_vars['max_epochs']
        # cal_max_epochs = cal_max_epochs(max_epochs, cal_weightage(train_size))
        # logger.info(f"Max epochs set to: {cal_max_epochs}")

        # Initialize the model
        model_name = env_vars['model_name']
        model = select_model(model_name)
        logger.info(f"Using model: {model_name}")

        to_monitor = "val/AUC_ROC"
        min_max = "max"
        log_every_n_steps = 1

        checkpointing = ModelCheckpoint(
            dirpath=str(path_run_dir),
            monitor=to_monitor,
            save_last=True,
            save_top_k=2,
            mode=min_max,
        )

        trainer = Trainer(
            accelerator=accelerator,
            precision=16,
            default_root_dir=str(path_run_dir),
            callbacks=[checkpointing],
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=log_every_n_steps,
            max_epochs=max_epochs,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )

    except Exception as e:
        logger.error(f"Error in set_up_training: {e}")
        raise

    return data_module, model, checkpointing, trainer, path_run_dir, env_vars


def validate_and_train(logger, data_module, model, trainer) -> None:
    logger.info("--- Validate global model ---")
    trainer.validate(model, datamodule=data_module)

    logger.info("--- Train new model ---")
    trainer.fit(model, datamodule=data_module)


def finalize_training(logger, model, checkpointing, trainer, path_run_dir, env_vars) -> None:
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    predict, prediction_flag = load_prediction_modules(env_vars['prediction_flag'])
    test_data_path = os.path.join(env_vars['data_dir'], env_vars['task_data_name'], 'test')
    if os.path.exists(test_data_path):
        predict(path_run_dir, os.path.join(env_vars['data_dir'], env_vars['task_data_name'], 'test'), env_vars['model_name'], last_flag=False, prediction_flag=prediction_flag)
    else:
        logger.info('No test data found, not running evaluation')
    logger.info('Training completed successfully')
