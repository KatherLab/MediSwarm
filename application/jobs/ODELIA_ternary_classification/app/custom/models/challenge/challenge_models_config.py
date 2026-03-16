"""
Shared configuration for challenge models.
Used by both testing scripts and config updaters.
"""

CHALLENGE_MODELS = {
    "1DivideAndConquer": {
        "team_name": "1DivideAndConquer",
        "persistor_path": "models.challenge.1DivideAndConquer.model.create_model",
        "persistor_args": {
            "n_input_channels": 3,
            "num_classes": 3
        }
    },
    "2BCN_AIM": {
        "team_name": "2BCN_AIM",
        "persistor_path": "models.challenge.2bcnaim.swinunetr.create_model",
        "persistor_args": {
            "img_size": 224,
            "num_classes": 3,
            "n_input_channels": 1,
            "spatial_dims": 3
        }
    },
    "3agaldran": {
        "team_name": "3agaldran",
        "persistor_path": "models.challenge.3agaldran.model_factory.model_factory",
        "persistor_args": {
            "arch": "mvit_v2_s",
            "num_classes": 3,
            "in_ch": 1,
            "seed": 123
        }
    },
    "4LME_ABMIL": {
        "team_name": "4LME_ABMIL",
        "persistor_path": "models.challenge.4abmil.model.create_model",
        "persistor_args": {
            "n_input_channels": 3,
            "num_classes": 3
        }
    },
    "5Pimed": {
        "team_name": "5Pimed",
        "persistor_path": "models.challenge.5pimed.model.create_model",
        "persistor_args": {
            "model_name": "resnet18",
            "num_classes": 3,
            "n_input_channels": 1,
            "spatial_dims": 3,
            "norm": "batch"
        }
    }
}

# For backward compatibility - simple list of model names
MODEL_NAMES = list(CHALLENGE_MODELS.keys())

def get_model_config(model_name: str):
    """Get configuration for a specific model."""
    return CHALLENGE_MODELS.get(model_name)

def get_all_model_names():
    """Get list of all available model names."""
    return MODEL_NAMES

def get_persistor_config(model_name: str):
    """Get persistor configuration for a specific model."""
    config = get_model_config(model_name)
    if config:
        return {
            "persistor_path": config["persistor_path"],
            "persistor_args": config["persistor_args"]
        }
    return None

def create_challenge_model(model_name: str = None, **kwargs):
    """
    Factory function to create any challenge model.
    Can be called with explicit model_name or reads from MODEL_VARIANT env var.
    """
    import os
    import importlib.util
    from pathlib import Path

    if model_name is None:
        model_name = os.getenv("MODEL_VARIANT")
        if not model_name:
            raise ValueError("MODEL_VARIANT environment variable must be set")

    # Normalize model name
    CHALLENGE_TEAMS = ["1DivideAndConquer", "2BCN_AIM", "3agaldran", "4LME_ABMIL", "5Pimed"]
    if model_name in CHALLENGE_TEAMS:
        model_name = f"challenge_{model_name}"

    if not model_name.startswith("challenge_"):
        raise ValueError(f"Model name must start with 'challenge_': {model_name}")

    # Get model config
    team_name = "_".join(model_name.split('_')[1:])
    config = get_model_config(team_name)
    if not config:
        raise ValueError(f"Unknown challenge model: {team_name}")

    # Import and create model
    module_path = config["persistor_path"]
    module_parts = module_path.split('.')
    function_name = module_parts[-1]

    # Import the module
    module = __import__(module_path.rsplit('.', 1)[0], fromlist=[function_name])
    create_func = getattr(module, function_name)

    # Merge config args with any additional kwargs
    model_args = config["persistor_args"].copy()
    model_args.update(kwargs)

    return create_func(**model_args)
