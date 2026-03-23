import torch
from env_config import load_environment_variables
import os
from pathlib import Path
import importlib.util
import logging
from models import ResNet, MST, Swin3D

"""
Shared configuration for challenge models.
Used by both testing scripts and config updaters.
"""

CHALLENGE_MODELS = {
    "1DivideAndConquer": {
        "team_name": "1DivideAndConquer",
        "persistor_path": "challenge.1DivideAndConquer.model.create_model",
        "persistor_args": {
            "num_classes": 3,
            "n_input_channels": 1,
            "spatial_dims": 3,  # must be three
            "pretrained_path": "checkpoint_final.pth"  # "challenge.1DivideAndConquer.checkpoint_final.pth."
        }
    },
    "2BCN_AIM": {
        "team_name": "2BCN_AIM",
        "persistor_path": "challenge.2bcnaim.swinunetr.create_model",
        "persistor_args": {
            "img_size": 224,
            "num_classes": 3,
            "n_input_channels": 1,
            "spatial_dims": 3
        }
    },
    "3agaldran": {
        "team_name": "3agaldran",
        "persistor_path": "challenge.3agaldran.model_factory.model_factory",
        "persistor_args": {
            "arch": "mvit_v2_s",
            "pretrained_path": "mvit_v2_s-ae3be167.pth",
            "num_classes": 3,
            "in_ch": 1,
            "seed": 123
        }
    },
    "4LME_ABMIL": {
        "team_name": "4LME_ABMIL",
        "persistor_path": "challenge.4abmil.model.create_model",
        "persistor_args": {
            "model_type": "swin",
            "n_input_channels": 3,
            "num_classes": 3
        }
    },
    "5Pimed": {
        "team_name": "5Pimed",
        "persistor_path": "challenge.5pimed.model.create_model",
        "persistor_args": {
            "model_name": "resnet18",
            "num_classes": 3,
            "n_input_channels": 1,
            "spatial_dims": 3,
            "norm": "batch"
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
        return CHALLENGE_MODELS.get(model_name) 
    else: 
        logger.info(f"No Configuration for model {model_name} found. Use default model <MST>")
        return DEFAULT_MODEL.get("MST")

def get_all_model_names():
    """Get list of all available model names."""
    return list(CHALLENGE_MODELS.keys())

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
        model_name = env_vars.get('model_name', 'MST')
        logger.info(f"Using model {model_name} (MST as default if MODEL_NAME has not been set es environmental variable).")
    
    # normalize name
    if model_variant == "challenge":
        model_name = f"challenge_{get_all_model_names()[0]}"
        logger.info(f"Model variant {model_variant}; Select first challenge model as default: {model_name}.")
    elif model_variant in get_all_model_names():
        model_name = f"challenge_{model_variant}"
        logger.info(f"Model variant {model_variant}; Select challenge model: {model_name}.")
    else:
        model_name = model_variant
        logger.info(f"Use model variant as model name: {model_name}")

    return model_name

def create_model(logger=None, model_name: str = None, num_classes: int = 3, 
                 loss_kwargs: dict = None):
    """
    Factory function to create any model.
    Can be called with explicit model_name or reads from MODEL_NAME env var.
    """
    if logger == None:
        logger = set_up_logging()
        
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
    elif model_name == "Swin3D":
        print(f"Using Swin3D model:\nShould we include {loss_kwargs} here?")
        model = Swin3D(n_input_channels=1,
                        num_classes=num_classes,
                        spatial_dims=3)
    elif "challenge_" in model_name:
        team_name = "_".join(model_name.split('_')[1:])
        config = get_model_config(logger, team_name)
        
        persistor_path = config["persistor_path"]
        persistor_args = config["persistor_args"].copy()
        persistor_args["num_classes"] = num_classes  # allow overrides

        # Dynamic import: split into module path and function name
        module_path, func_name = persistor_path.rsplit(".", 1)
        
        # Use importlib by file path since module names start with digits
        base_dir = os.path.dirname(Path(__file__)) # is already in directory models
        # e.g. "challenge.1DivideAndConquer.model" -> file path
        file_path = os.path.join(base_dir, *module_path.split(".")) + ".py"
        
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if "pretrained_path" in persistor_args:
            rel_path = persistor_args["pretrained_path"].rstrip(".")
            persistor_args["pretrained_path"] = os.path.join(
                base_dir, "challenge", team_name, rel_path
            )
            logger.info(f'__________ model path is : {persistor_args["pretrained_path"]}')

        factory_fn = getattr(module, func_name)
        logger.info(f"Now access {persistor_args} from module {module}")
        return factory_fn(**persistor_args)
    else:
        raise ValueError(f"Unsupported model name: {model_name}.")
    return model
