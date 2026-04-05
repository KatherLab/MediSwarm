"""
STAMP model wrapper for NVFlare PTFileModelPersistor.

NVFlare's PTFileModelPersistor needs a callable that returns an nn.Module
whose state_dict defines the parameter space for federated aggregation.
This module creates a STAMP model from environment variables and wraps it
so NVFlare can serialize/deserialize its weights.
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_stamp_env():
    """Read STAMP-specific configuration from environment variables.

    Model-specific hyperparameters (VIT dim_model, MLP dim_hidden, etc.) are
    read from STAMP 2.4.0's ``ModelParams`` defaults rather than environment
    variables — this avoids hardcoding parameter lists that break when STAMP
    adds new fields (e.g. VIT's ``use_alibi`` in 2.4.0).
    """
    return {
        "task": os.environ.get("STAMP_TASK", "classification"),
        "model_name": os.environ.get("STAMP_MODEL_NAME", "vit"),
        "feature_type": os.environ.get("STAMP_FEATURE_TYPE", "tile"),
        # Feature dimension from H5 files (UNI2 = 1024, UNI = 1024, CTransPath = 768, etc.)
        "dim_input": int(os.environ.get("STAMP_DIM_INPUT", "1024")),
        "num_classes": int(os.environ.get("STAMP_NUM_CLASSES", "3")),
        # Model hyperparameters
        "max_lr": float(os.environ.get("STAMP_MAX_LR", "1e-4")),
        "div_factor": float(os.environ.get("STAMP_DIV_FACTOR", "25.0")),
    }


def _build_stamp_model(env: dict) -> nn.Module:
    """Build a STAMP Lightning model from environment configuration.

    This imports STAMP's model registry and instantiates the correct
    Lightning wrapper + backbone for the given (feature_type, task, model_name).

    Uses STAMP 2.4.0's ``ModelParams`` defaults to get model-specific
    hyperparameters (e.g. VIT's ``use_alibi``), avoiding hardcoded parameter
    lists that break when STAMP adds new fields.
    """
    try:
        from stamp.modeling.registry import ModelName, load_model_class
        from stamp.modeling.config import ModelParams
    except ImportError:
        raise ImportError(
            "STAMP is not installed. Install from GitHub with: "
            'pip install "stamp @ git+https://github.com/KatherLab/STAMP.git@2.4.0" '
            "(NOTE: the PyPI 'stamp' package is an unrelated metagenomics tool)"
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

    # Get model-specific parameters from STAMP's ModelParams defaults.
    # This automatically picks up all fields (including new ones like
    # VIT's use_alibi) without hardcoding parameter lists.
    model_params = ModelParams()
    model_specific_params = (
        model_params.model_dump().get(model_name_str) or {}
    )

    # Build categories placeholder — actual categories come from training data
    # For the persistor, we just need the model architecture to be correct
    categories = [str(i) for i in range(num_classes)]

    # category_weights must be a Tensor matching len(categories) (STAMP 2.4.0
    # validates this in LitBaseClassifier.__init__).  Equal weights are fine
    # for the persistor — real weights are computed during training.
    category_weights = torch.ones(num_classes) / num_classes

    # Calculate total_steps placeholder (actual value set during training)
    total_steps = 100  # placeholder, overridden at training time

    common_params = {
        "categories": categories,
        "category_weights": category_weights,
        "dim_input": dim_input,
        "total_steps": total_steps,
        "max_lr": env["max_lr"],
        "div_factor": env["div_factor"],
        "model_name": model_name_str,
        # Metadata fields (no effect on model architecture, but
        # ground_truth_label must be a str for beartype validation)
        "ground_truth_label": "placeholder",
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
