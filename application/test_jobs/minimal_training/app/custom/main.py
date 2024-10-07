from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nvflare.client.lightning as flare
from data.datamodules import DataModule
from model_selector import select_model
from env_config import load_environment_variables, create_run_directory
import nvflare.client as flare_util

from data.datasets import MiniDatasetForTesting

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

flare_util.init()

SITE_NAME=flare.get_site_name()

def set_up_data_module(env_vars):
    ds = MiniDatasetForTesting()
    labels = ds.get_labels()

    # Generate indices and perform stratified split
    indices = list(range(len(ds)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

    # Create training and validation subsets
    ds_train = Subset(ds, train_indices)
    ds_val = Subset(ds, val_indices)

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=1,
        num_workers=16,
        pin_memory=True,
    )

    return dm


def main():
    """
    Main function for training and evaluating the model using NVFlare and PyTorch Lightning.
    """
    try:
        env_vars = load_environment_variables()
        path_run_dir = create_run_directory(env_vars['scratch_dir'])
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using {accelerator} for training")

        data_module = set_up_data_module(env_vars)

        # Initialize the model
        model = select_model()

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
            max_epochs=2,
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
            trainer.validate(model, datamodule=data_module)

            logger.info("--- Train new model ---")
            trainer.fit(model, datamodule=data_module)

        model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
        logger.info('Training completed successfully')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
