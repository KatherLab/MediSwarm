"""
STAMP model wrapper for NVFlare PTFileModelPersistor.

NVFlare's PTFileModelPersistor needs a callable that returns an nn.Module
whose state_dict defines the parameter space for federated aggregation.
This module creates a STAMP model from environment variables and wraps it
so NVFlare can serialize/deserialize its weights.
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_stamp_env():
    """Read STAMP-specific configuration from environment variables."""
    return {
        "task": os.environ.get("STAMP_TASK", "classification"),
        "model_name": os.environ.get("STAMP_MODEL_NAME", "vit"),
        "feature_type": os.environ.get("STAMP_FEATURE_TYPE", "tile"),
        # Feature dimension from H5 files (UNI2 = 1024, UNI = 1024, CTransPath = 768, etc.)
        "dim_input": int(os.environ.get("STAMP_DIM_INPUT", "1024")),
        "num_classes": int(os.environ.get("STAMP_NUM_CLASSES", "3")),
        # Model hyperparameters
        "bag_size": int(os.environ.get("STAMP_BAG_SIZE", "512")),
        "batch_size": int(os.environ.get("STAMP_BATCH_SIZE", "64")),
        "max_epochs": int(os.environ.get("STAMP_MAX_EPOCHS", "32")),
        "max_lr": float(os.environ.get("STAMP_MAX_LR", "1e-4")),
        "div_factor": float(os.environ.get("STAMP_DIV_FACTOR", "25.0")),
        # VIT-specific
        "vit_dim_model": int(os.environ.get("STAMP_VIT_DIM_MODEL", "512")),
        "vit_dim_feedforward": int(os.environ.get("STAMP_VIT_DIM_FEEDFORWARD", "512")),
        "vit_n_heads": int(os.environ.get("STAMP_VIT_N_HEADS", "8")),
        "vit_n_layers": int(os.environ.get("STAMP_VIT_N_LAYERS", "2")),
        "vit_dropout": float(os.environ.get("STAMP_VIT_DROPOUT", "0.25")),
        # MLP-specific
        "mlp_dim_hidden": int(os.environ.get("STAMP_MLP_DIM_HIDDEN", "512")),
        "mlp_num_layers": int(os.environ.get("STAMP_MLP_NUM_LAYERS", "2")),
        "mlp_dropout": float(os.environ.get("STAMP_MLP_DROPOUT", "0.25")),
    }


def _build_stamp_model(env: dict) -> nn.Module:
    """Build a STAMP Lightning model from environment configuration.

    This imports STAMP's model registry and instantiates the correct
    Lightning wrapper + backbone for the given (feature_type, task, model_name).
    """
    try:
        from stamp.modeling.registry import ModelName, load_model_class
    except ImportError:
        raise ImportError(
            "STAMP is not installed. Install it with: pip install stamp "
            "or ensure the STAMP source is on PYTHONPATH."
        )

    task = env["task"]
    feature_type = env["feature_type"]
    model_name_str = env["model_name"]
    dim_input = env["dim_input"]
    num_classes = env["num_classes"]

    # Resolve model name enum
    model_name = ModelName(model_name_str)

    # Load the correct Lightning wrapper and backbone class
    LitModelClass, ModelClass = load_model_class(task, feature_type, model_name)

    # Build model-specific parameters
    model_specific_params = {}
    if model_name == ModelName.VIT:
        model_specific_params = {
            "dim_model": env["vit_dim_model"],
            "dim_feedforward": env["vit_dim_feedforward"],
            "n_heads": env["vit_n_heads"],
            "n_layers": env["vit_n_layers"],
            "dropout": env["vit_dropout"],
        }
    elif model_name == ModelName.MLP:
        model_specific_params = {
            "dim_hidden": env["mlp_dim_hidden"],
            "num_layers": env["mlp_num_layers"],
            "dropout": env["mlp_dropout"],
        }
    elif model_name == ModelName.TRANS_MIL:
        model_specific_params = {
            "dim_hidden": int(os.environ.get("STAMP_TRANSMIL_DIM_HIDDEN", "512")),
        }

    # Build categories placeholder — actual categories come from training data
    # For the persistor, we just need the model architecture to be correct
    categories = [str(i) for i in range(num_classes)]

    # Calculate total_steps placeholder (actual value set during training)
    total_steps = 100  # placeholder, overridden at training time

    common_params = {
        "categories": categories,
        "category_weights": [],
        "dim_input": dim_input,
        "total_steps": total_steps,
        "max_lr": env["max_lr"],
        "div_factor": env["div_factor"],
        "model_name": model_name_str,
        # Metadata fields (no effect on model architecture)
        "ground_truth_label": None,
        "time_label": None,
        "status_label": None,
        "train_patients": [],
        "valid_patients": [],
        "clini_table": Path("/dev/null"),
        "slide_table": None,
        "feature_dir": Path("/dev/null"),
    }

    all_params = {**common_params, **model_specific_params}

    model = LitModelClass(model_class=ModelClass, **all_params)

    logger.info(
        f"Created STAMP model: {model_name_str} "
        f"(task={task}, feature_type={feature_type}, "
        f"dim_input={dim_input}, num_classes={num_classes})"
    )

    return model


def create_stamp_model(**kwargs) -> nn.Module:
    """Entry point for NVFlare PTFileModelPersistor.

    Called by NVFlare to instantiate the model whose state_dict defines
    the federated parameter space. Configuration is read from environment
    variables (STAMP_TASK, STAMP_MODEL_NAME, etc.).

    Any kwargs passed from config_fed_client.conf are merged with env vars,
    with kwargs taking precedence.
    """
    env = _get_stamp_env()
    # Allow config overrides from NVFlare conf
    env.update({k: v for k, v in kwargs.items() if v is not None})
    return _build_stamp_model(env)
