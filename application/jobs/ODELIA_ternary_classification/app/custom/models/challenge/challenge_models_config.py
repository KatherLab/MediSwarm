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
