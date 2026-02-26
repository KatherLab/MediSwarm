from sklearn.model_selection import train_test_split
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodules import DataModule
from models import ResNet, MST, Swin3D
from env_config import load_environment_variables, prepare_odelia_dataset, generate_run_directory
import torch.multiprocessing as mp

import logging


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


def set_up_data_module(logger, model_name: str = ''):
    torch.set_float32_matmul_precision('high')
    ds_train, ds_val, path_run_dir, run_name = prepare_odelia_dataset(model_name=model_name)
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

    loss_kwargs = {}

    return dm, path_run_dir, run_name, num_classes, loss_kwargs


def create_run_directory(env_vars):
    return generate_run_directory(
        env_vars['scratch_dir'],
        env_vars['task_data_name'],
        env_vars['model_name'],
        env_vars['local_compare_flag']
    )


def prepare_training(logger, max_epochs: int, site_name: str, model_variant: str | None = None):
    try:
        env_vars = load_environment_variables()

        if not torch.cuda.is_available():
            raise RuntimeError("This example requires a GPU")

        logger.info(f"Running code version {env_vars['mediswarm_version']}")
        logger.info(f"Using GPU for training")

        # Allow an explicit model_variant to override the configured env model name.
        model_name = model_variant if (model_variant is not None and model_variant != "") else os.environ.get('MODEL_NAME', '')

        data_module, path_run_dir, run_name, num_classes, loss_kwargs = set_up_data_module(logger, model_name)
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
        elif model_name == "Swin3D":
            print(f"Using Swin3D model:\nShould we include {loss_kwargs} here?")
            model = Swin3D(n_input_channels=1,
                           num_classes=num_classes,
                           spatial_dims=3)
        elif "challenge_" in model_name:
            # The challenge model folder starts with a digit (3agaldran), which
            # is not a valid Python identifier for standard imports. Load the
            # factory by file path using importlib to avoid renaming directories.
            import importlib.util
            import os
            team_name = model_name.split('_')[1] 
            if team_name == "1DvideAndConquer":
                # TODO not yet implemented
                model = None
                pass
            elif team_name == "2BCN_AIM":
                # TODO not yet implemented
                model = None
                pass
            elif team_name == "3agaldran":
                factory_path = os.path.join(
                    os.path.dirname(__file__),
                    "models",
                    "challenge",
                    "3agaldran",
                    "model_factory.py",
                )
                spec = importlib.util.spec_from_file_location("agaldran_model_factory", factory_path)
                agaldran_factory = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(agaldran_factory)
                model = agaldran_factory.model_factory(arch="mvit_v2_s", 
                                                    pretrained_path="application\\jobs\\ODELIA_ternary_classification\\app\\custom\\models\\challenge\\3agaldran\\mvit_v2_s-ae3be167.pth", 
                                                    num_classes=3, 
                                                    in_ch=3, 
                                                    seed=123)
            elif team_name == "4LME_ABMIL":
                model_creator_path = os.path.join(
                    os.path.dirname(__file__),
                    "models",
                    "challenge",
                    "abmil",
                    "model.py",
                )
                spec = importlib.util.spec_from_file_location("abmil_model", model_creator_path)
                abmil_model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(abmil_model_module)
                model = abmil_model_module.create_model(config_path="", in_ch=3, num_classes=num_classes)

            elif team_name == "5Pimed":
                # TODO not yet implemented
                model = None
                pass
            else:
                raise ValueError(f"Unknown challenge team name: {team_name}")
        else:
            raise ValueError(f"Unsupported model name: {model_name}.")

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

    logger.info('Training completed successfully.')
