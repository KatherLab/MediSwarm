from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from collections import Counter
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nvflare.client.lightning as flare
from data.datamodules import DataModule
from model_selector import select_model
from env_config import load_environment_variables, load_prediction_modules, prepare_dataset, generate_run_directory
import nvflare.client as flare_util

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

flare_util.init()

SITE_NAME=flare.get_site_name()

#TODO: Set max_epochs based on the data set size
if SITE_NAME == "manual_dl3":
    MAX_EPOCHS = 2
elif SITE_NAME == "manual_dl2":
    MAX_EPOCHS = 4
elif SITE_NAME == "manual_dl0":
    MAX_EPOCHS = 8

def main():
    """
    Main function for training and evaluating the model using NVFlare and PyTorch Lightning.
    """
    try:
        env_vars = load_environment_variables()
        logger.info(f'Model name: {env_vars["model_name"]}')

        predict, prediction_flag = load_prediction_modules(env_vars['prediction_flag'])
        ds, task_data_name = prepare_dataset(env_vars['task_data_name'], env_vars['data_dir'], site_name=SITE_NAME)
        path_run_dir = generate_run_directory(env_vars['scratch_dir'], env_vars['task_data_name'], env_vars['model_name'], env_vars['local_compare_flag'])

        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using {accelerator} for training")

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

        adsValData = DataLoader(ds_val, batch_size=2, shuffle=False)
        logger.info(f'adsValData type: {type(adsValData)}')

        train_size = len(ds_train)
        val_size = len(ds_val)
        logger.info(f'Train size: {train_size}')
        logger.info(f'Val size: {val_size}')

        max_epochs = env_vars['max_epochs']
        #cal_max_epochs = cal_max_epochs(max_epochs, cal_weightage(train_size))
        #logger.info(f"Max epochs set to: {cal_max_epochs}")

        dm = DataModule(
            ds_train=ds_train,
            ds_val=ds_val,
            batch_size=1,
            num_workers=16,
            pin_memory=True,
        )

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
            max_epochs=5,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )

        flare.patch(trainer)  # Patch trainer to enable swarm learning
        torch.autograd.set_detect_anomaly(True)

        logger.info(f"Site name: {flare.get_site_name()}")

        while flare.is_running():
            input_model = flare.receive()
            logger.info(f"Current round: {input_model.current_round}")

            logger.info("--- Validate global model ---")
            trainer.validate(model, datamodule=dm)

            logger.info("--- Train new model ---")
            trainer.fit(model, datamodule=dm)

        model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
        predict(path_run_dir, os.path.join(env_vars['data_dir'], task_data_name, 'test'), model_name, last_flag=False, prediction_flag=prediction_flag)
        logger.info('Training completed successfully')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
