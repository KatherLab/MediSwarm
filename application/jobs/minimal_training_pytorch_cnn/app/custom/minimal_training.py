from datetime import datetime
import logging
import os
from pathlib import Path
import shutil

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext

from data.datamodules import DataModule
from data.datasets import MiniDatasetForTesting
from models import MiniCNNForTesting


def load_environment_variables():
    """Load environment variables and return them as a dictionary."""
    return {
        'scratch_dir': os.getenv('SCRATCH_DIR', '/scratch/'),
        'max_epochs': int(os.getenv('MAX_EPOCHS', 100)),
        'min_peers': int(os.getenv('MIN_PEERS', 2)),
        'max_peers': int(os.getenv('MAX_PEERS', 7)),
        'use_adaptive_sync': os.getenv('USE_ADAPTIVE_SYNC', 'False').lower() == 'true',
        'sync_frequency': int(os.getenv('SYNC_FREQUENCY', 1024)),
        'prediction_flag': os.getenv('PREDICT_FLAG', 'ext')
    }

def create_run_directory(scratch_dir):
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # make dir if not exist
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    return os.path.join(scratch_dir, f"{current_time}_minimal_training_pytorch_cnn")

def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def set_up_data_module(env_vars):
    ds = MiniDatasetForTesting()
    labels = ds.get_labels()

    # Generate indices and train/val/test split
    len_ds = len(ds)
    indices = list(range(len_ds))
    train_indices = indices[:len_ds//3]
    val_indices = indices[len_ds//3:2*len_ds//3]
    test_indices = indices[2*len_ds//3:]

    # Create training and validation subsets
    ds_train = Subset(ds, train_indices)
    ds_val = Subset(ds, val_indices)
    ds_test = Subset(ds, test_indices)

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        batch_size=1,
        num_workers=16,
        pin_memory=True,
    )

    return dm

def prepare_training(logger):
    try:
        env_vars = load_environment_variables()
        path_run_dir = create_run_directory(env_vars['scratch_dir'])
        if not torch.cuda.is_available():
            raise(RuntimeError("This example does not work without GPU"))
        accelerator = 'gpu'
        logger.info(f"Using {accelerator} for training")

        data_module = set_up_data_module(env_vars)

        # Initialize the model
        model = MiniCNNForTesting()

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

        callbacks = [checkpointing]

        trainer = Trainer(
            accelerator=accelerator,
            precision=16,
            default_root_dir=str(path_run_dir),
            callbacks=callbacks,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=log_every_n_steps,
            max_epochs=2,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )
    except Exception as e:
        logger.error(f"Error in set_up_training: {e}")
        raise

    return data_module, model, checkpointing, trainer

def validate_and_train(logger, data_module, model, trainer) -> None:
    logger.info("--- Validate global model ---")
    trainer.validate(model, datamodule=data_module)

    logger.info("--- Train new model ---")
    trainer.fit(model, datamodule=data_module)

def finalize_training(logger, model, checkpointing, trainer) -> None:
    # Save best checkpoint
    best_path = checkpointing.best_model_path
    if best_path:
        model.save_best_checkpoint(trainer.logger.log_dir, best_path)
        logger.info(f'Best model checkpoint: {best_path}')
    else:
        logger.warning('No best checkpoint found.')

    # Save latest (last) checkpoint
    last_path = checkpointing.last_model_path
    if last_path:
        final_last = os.path.join(os.path.dirname(last_path), "last_global_model.ckpt")
        shutil.copy(last_path, final_last)
        logger.info(f'Last model saved to: {final_last}')
    else:
        logger.warning('No last checkpoint found.')

    logger.info('Training completed successfully.')


def _evaluate_global_model_on_test_data(model_filename) -> None:
    dummy_logger = logging.getLogger('_evaluate_global_models_on_test_data')  # these log messages are not printed

    data_module, global_model, _, _ = prepare_training(dummy_logger)

    nvflare_run_dir = os.path.dirname(os.path.abspath(__file__)) + '/..'   # TODO is there a better way to obtain the run directory here?
    nvflare_run_dir = Path(os.path.abspath(nvflare_run_dir))
    global_model_path = nvflare_run_dir/model_filename                     #      Or the location of the (best) global model?

    print(f'Loading {model_filename}')
    loaded_pt = torch.load(global_model_path, weights_only=False)

    results = []
    with torch.autocast(device_type="cuda"):  # to avoid "RuntimeError: Input type (c10::Half) and bias type (float) should be the same" for this model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        global_model.load_state_dict(loaded_pt["model"], strict=False)
        global_model = global_model.to(device)
        global_model.eval()

        for batch in data_module.test_dataloader():
            source, target = batch['source'], batch['target']
            with torch.no_grad():
                prediction = global_model(source.to(device))
                results.append(prediction)
    print(f'Predicted batch for {model_filename}: {results}')


class EvaluatorOnTestData(FLComponent):
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.END_RUN:
            _evaluate_global_model_on_test_data('FL_global_model.pt')
            _evaluate_global_model_on_test_data('best_FL_global_model.pt')

class EvaluatorOnTestDataTwo(FLComponent):  # TODO figure out why both need to be run in different order for both outputs to appear reliably
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.END_RUN:
            _evaluate_global_model_on_test_data('best_FL_global_model.pt')
            _evaluate_global_model_on_test_data('FL_global_model.pt')
