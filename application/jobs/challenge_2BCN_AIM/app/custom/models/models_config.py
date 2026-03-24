import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add custom to sys path
from env_config import load_environment_variables
from pathlib import Path
import importlib.util
import logging
from models import ResNet, MST, Swin3D


def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def get_model_config(logger, model_name: str):
    """Get configuration for a specific model."""
    return {
        "team_name": "2BCN_AIM",
        "persistor_path": "models.swinunetr.create_model",
        "persistor_args": {
            "img_size": 224,
            "num_classes": 3,
            "n_input_channels": 1,
            "spatial_dims": 3
        }
    }

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
        model_variant = env_vars.get('model_name', 'challenge_2BCN_AIM')
        logger.info(f"Using model variant {model_variant} .")
    
    if model_variant == "2BCN_AIM":
        model_name = f"challenge_{model_variant}"
        logger.info(f"Model variant {model_variant}; Select challenge model: {model_name}.")
    else:
        model_name = model_variant
        logger.info(f"Use model variant as model name: {model_name}")

    return model_name
