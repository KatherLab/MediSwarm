from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import Counter
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodules import DataModule
#from model_selector import select_model
from models import ResNet, MST, ResNetRegression, MSTRegression
from env_config import load_environment_variables, load_prediction_modules, prepare_odelia_dataset, generate_run_directory
import torch.multiprocessing as mp

import os
import logging


def get_num_epochs_per_round(site_name: str) -> int:
    NUM_EPOCHS_FOR_SITE = {
        "TUD_1": 2, "TUD_2": 4, "TUD_3": 8,
        "MEVIS_1": 2, "MEVIS_2": 4,
        "UKA": 2,
    }
    max_epochs = NUM_EPOCHS_FOR_SITE.get(site_name, 5)
    print(f"Site name: {site_name}")
    print(f"Max epochs set to: {max_epochs}")
    return max_epochs


def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def set_up_data_module(env_vars, logger, site_name: str):
    torch.set_float32_matmul_precision('high')
    ds_train, ds_val, path_run_dir, run_name, is_binary_task = prepare_odelia_dataset(
        env_vars['task_data_name'], env_vars['data_dir'], site_name=site_name
    )

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
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

    # ------------ Initialize Model ------------
    loss_kwargs = {}
    out_ch = len(ds_train.labels)
    if not is_binary_task:
        out_ch = sum(ds_train.class_labels_num)
        loss_kwargs = {'class_labels_num': ds_train.class_labels_num}


    return dm, path_run_dir, run_name, is_binary_task, out_ch, loss_kwargs


def create_run_directory(env_vars):
    return generate_run_directory(
        env_vars['scratch_dir'],
        env_vars['task_data_name'],
        env_vars['model_name'],
        env_vars['local_compare_flag']
    )


def prepare_training(logger, max_epochs: int, site_name: str):
    try:
        env_vars = load_environment_variables()
        data_module, path_run_dir, run_name, is_binary_task , out_ch, loss_kwargs= set_up_data_module(env_vars, logger, site_name)

        if not torch.cuda.is_available():
            raise RuntimeError("This example requires a GPU")

        logger.info(f"Using GPU for training")

        model_name = env_vars['model_name']

        model_map = {
            'ResNet101': ResNet if is_binary_task else ResNetRegression,
            'MST': MST if is_binary_task else MSTRegression
        }
        MODEL = model_map.get(model_name, None)
        model = MODEL(
            in_ch=1,
            out_ch=out_ch,
            loss_kwargs=loss_kwargs
        )
        logger.info(f"Using model: {model_name}")

        to_monitor = "val/AUC_ROC" if is_binary_task else "val/MAE"
        min_max = "max" if is_binary_task else "min"
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

        trainer = Trainer(
            accelerator='gpu',
            accumulate_grad_batches=1,
            precision='16-mixed',
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
        logger.error(f"Error in prepare_training: {e}")
        raise

    return data_module, model, checkpointing, trainer, path_run_dir, env_vars


def validate_and_train(logger, data_module, model, trainer) -> None:
    logger.info("--- Validate global model ---")
    trainer.validate(model, datamodule=data_module)

    logger.info("--- Train new model ---")
    trainer.fit(model, datamodule=data_module)


def finalize_training(logger, model, checkpointing, trainer, path_run_dir, env_vars) -> None:
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)

    logger.info('Prediction currently not implemented.')
    '''
    predict, prediction_flag = load_prediction_modules(env_vars['prediction_flag'])

    test_data_path = os.path.join(env_vars['data_dir'], env_vars['task_data_name'], 'test')
    if os.path.exists(test_data_path):
        predict(path_run_dir, test_data_path, env_vars['model_name'], last_flag=False, prediction_flag=prediction_flag)
    else:
        logger.info('No test data found, skipping evaluation.')
    '''

    logger.info('Training completed successfully.')
