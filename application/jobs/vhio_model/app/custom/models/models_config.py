import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add custom to sys path
from env_config import load_environment_variables
from pathlib import Path
import importlib.util
import logging
from models import ResNet, MST, MedicalNet

"""
Shared configuration for challenge models.
Used by both testing scripts and config updaters.
"""

CMP_MODELS = {
    "vhio": {
        "team_name": "vhio",
        "persistor_path": "medicalnet.MedicalNet",
        "persistor_args": {
            "num_classes": 3,
            "n_input_channels": 1,
            "spatial_dims": 3,  # must be three
            "pretrained_path": "medicalnet_resnet34_23dataset.pth"  
        }
    },
    "cam": {
        # TODO
        "team_name": "cam",
        "persistor_path": "challenge.3agaldran.model_factory.model_factory",
        "persistor_args": {
            "arch": "mvit_v2_s",
            "pretrained_path": "mvit_v2_s-ae3be167.pth", # will be converted into relative path by create_model
            "num_classes": 3,
            "in_ch": 1,
            "seed": 123
        }
    }
}
DEFAULT_MODEL = {
    "MST": {
        "persistor_path": "mst.MST",
        "persistor_args": {
            "n_input_channels": 1,
            "num_classes": 3, 
            "spatial_dims": 3
        }
    }
}

def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def get_model_config(logger, model_name: str):
    """Get configuration for a specific model."""
    if model_name in get_all_model_names():
        return CMP_MODELS.get(model_name) 
    else: 
        logger.info(f"No Configuration for model {model_name} found. Use default model <MST>")
        return DEFAULT_MODEL.get("MST")

def get_all_model_names():
    """Get list of all available model names."""
    return list(CMP_MODELS.keys())

def get_persistor_config(logger, model_name: str):
    """Get persistor configuration for a specific model."""
    config = get_model_config(logger, model_name)
    if config:
        return {
            "persistor_path": config["persistor_path"],
            "persistor_args": config["persistor_args"]
        }
    return None

def get_unified_model_name(logger, model_variant: str, env_vars):
    logger.info(f"Environment variables: {env_vars}")
    if model_variant is None:
        logger.info("No model variant defined. Read environment variables.")
        model_variant = env_vars.get('model_name', 'MST')
        logger.info(f"Using model variant {model_variant} (MST as default if MODEL_NAME has not been set es environmental variable).")
    
    model_name = model_variant
    logger.info(f"Use model variant as model name: {model_name}")

    return model_name

def create_model(logger=None, model_name: str = None, num_classes: int = 3, 
                 loss_kwargs: dict = None, env_vars = None):
    """
    Factory function to create any model.
    Can be called with explicit model_name or reads from MODEL_NAME env var.
    """
    if logger == None:
        logger = set_up_logging()

    if env_vars == None:
        env_vars = load_environment_variables()
    model_name = get_unified_model_name(logger, model_name, env_vars)

    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a GPU")

    logger.info(f"Running code version {env_vars['mediswarm_version']}")
    logger.info(f"Using GPU for training")
    logger.info(f"Model name: {model_name}")

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
    elif model_name == 'MedicalNet':
        model = MedicalNet(
            n_input_channels=1,
            num_classes=num_classes,
            spatial_dims=3,
            pretrained_path=os.environ.get('MEDICALNET_PRETRAINED_PATH'),
            loss_kwargs=loss_kwargs,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}.")
    return model
