import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add custom to sys path
from env_config import load_environment_variables
from pathlib import Path
import importlib.util
import logging


def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def get_model_config(logger, model_name: str):
    """Get configuration for a specific model."""
    return {
        "team_name": "3agaldran",
        "persistor_path": "models.model_factory.model_factory",
        "persistor_args": {
            "arch": "mvit_v2_s",
            "pretrained_path": "mvit_v2_s-ae3be167.pth", # will be converted into relative path by create_model
            "num_classes": 3,
            "in_ch": 1,
            "seed": 123
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
        model_variant = env_vars.get('model_name', 'challenge_3agaldran')
        logger.info(f"Using model variant {model_variant} (MST as default if MODEL_NAME has not been set es environmental variable).")
    
    if model_variant == "3agaldran":
        model_name = f"challenge_{model_variant}"
        logger.info(f"Model variant {model_variant}; Select challenge model: {model_name}.")
    else:
        model_name = model_variant
        logger.info(f"Use model variant as model name: {model_name}")

    return model_name