import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add custom to sys path
from env_config import load_environment_variables
from pathlib import Path
import importlib.util
import logging
from models import ResNet, MST, MedicalNet

# MedicalNet 3D-ResNet34 (VHIO model): the Tencent 23-dataset checkpoint ships in the image from
# 1.9.1 on. Elsewhere point MEDICALNET_PRETRAINED_PATH at resnet_34_23dataset.pth
# (https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34), or drop the file next to
# this module or at /MediSwarm/pretrained_weights/ (in the image from 1.9.1 on). Without it the model refuses
# to start unless MEDICALNET_ALLOW_RANDOM_INIT=1.
MEDICALNET_WEIGHTS_FILE = "resnet_34_23dataset.pth"


def resolve_medicalnet_weights(logger):
    candidates = [
        os.environ.get("MEDICALNET_PRETRAINED_PATH", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), MEDICALNET_WEIGHTS_FILE),
        os.path.join("/MediSwarm/pretrained_weights", MEDICALNET_WEIGHTS_FILE),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            logger.info(f"MedicalNet pretrained weights: {path}")
            return path
    # Like MST_SAMMed2D: the round-0 global comes from the starting client alone, so one site without
    # the file would start every site from a random backbone while the run still ends COMPLETED.
    if os.environ.get("MEDICALNET_ALLOW_RANDOM_INIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.warning(f"MedicalNet weights {MEDICALNET_WEIGHTS_FILE} not found; MEDICALNET_ALLOW_RANDOM_INIT is "
                       "set, so the backbone starts from RANDOM weights (tests only).")
        return None
    searched = ", ".join(p for p in candidates if p)
    raise RuntimeError(
        f"MedicalNet (the VHIO model) needs its pretrained weights {MEDICALNET_WEIGHTS_FILE} and found none "
        f"(searched: {searched}). The 1.9.1 image ships them at /MediSwarm/pretrained_weights/; on an older image "
        f"set MEDICALNET_PRETRAINED_PATH, or MEDICALNET_ALLOW_RANDOM_INIT=1 for a deliberate run from random weights."
    )

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
            "pretrained_path": "checkpoint_final.pth"  # will be converted into relative path by create_model
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
            "pretrained_path": "mvit_v2_s-ae3be167.pth", # will be converted into relative path by create_model
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

# Models create_model() builds itself (challenge models come from CHALLENGE_MODELS).
RESNET_MODEL_NAMES = ("ResNet10", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152")
BUILTIN_MODEL_NAMES = ("MST", "MST_SAMMed2D", "MedicalNet") + RESNET_MODEL_NAMES
CHALLENGE_PREFIX = "challenge_"


def set_up_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def get_model_config(logger, model_name: str):
    """Configuration of a challenge model, looked up by its team name (a key of CHALLENGE_MODELS).

    An unknown name raises. It used to fall back to an MST entry, so a misspelled challenge name
    (e.g. the job-directory spelling '4abmil') ended in a misleading relative-import error from
    loading mst.py by file path instead of saying that the name is wrong.
    """
    if model_name in CHALLENGE_MODELS:
        return CHALLENGE_MODELS[model_name]
    raise ValueError(
        f"No challenge model named {model_name!r}. Valid challenge names: {', '.join(get_all_model_names())}. "
        f"Built-in models ({', '.join(BUILTIN_MODEL_NAMES)}) have no challenge configuration."
    )

def get_all_model_names():
    """Team names of the challenge models (the keys of CHALLENGE_MODELS); see get_supported_model_names()."""
    return list(CHALLENGE_MODELS.keys())

def get_supported_model_names():
    """Every model name create_model() accepts after get_unified_model_name().

    A challenge model may also be given by its bare team name (MODEL_NAME=1DivideAndConquer) and
    'challenge' selects the first one; both are mapped onto the names listed here.
    """
    return list(BUILTIN_MODEL_NAMES) + [CHALLENGE_PREFIX + name for name in get_all_model_names()]

def unsupported_model_error(model_name) -> ValueError:
    """The error for a model name create_model() cannot build, listing every valid name."""
    return ValueError(
        f"Unsupported model name: {model_name!r}. Valid names: {', '.join(get_supported_model_names())} "
        f"(a challenge model may also be given without the '{CHALLENGE_PREFIX}' prefix, e.g. 1DivideAndConquer)."
    )

def get_persistor_config(logger, model_name: str):
    """Persistor path and arguments of a challenge model (team name); unknown names raise ValueError."""
    config = get_model_config(logger, model_name)
    return {
        "persistor_path": config["persistor_path"],
        "persistor_args": config["persistor_args"]
    }

def get_unified_model_name(logger, model_variant: str, env_vars):
    logger.info(f"Environment variables: {env_vars}")
    if model_variant is None:
        logger.info("No model variant defined. Read environment variables.")
        model_variant = env_vars.get('model_name', 'MST')
        logger.info(f"Using model variant {model_variant} (MST as default if MODEL_NAME has not been set es environmental variable).")
    
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
                 loss_kwargs: dict = None, env_vars = None):
    """
    Factory function to create any model.
    Can be called with explicit model_name or reads from MODEL_NAME env var.
    """
    if logger == None:
        logger = set_up_logging()

    if loss_kwargs is None:
        loss_kwargs = {}

    if env_vars == None:
        env_vars = load_environment_variables()
    model_name = get_unified_model_name(logger, model_name, env_vars)
    # Checked before anything slow or hardware-dependent, so a misspelled MODEL_NAME is reported
    # as such (with the valid names) rather than as a missing GPU or a failed import.
    if model_name not in get_supported_model_names():
        raise unsupported_model_error(model_name)

    allow_cpu_model = os.environ.get("MEDISWARM_ALLOW_CPU_MODEL", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not torch.cuda.is_available() and not allow_cpu_model:
        raise RuntimeError("This example requires a GPU")

    logger.info(f"Running code version {env_vars['mediswarm_version']}")
    if torch.cuda.is_available():
        logger.info(f"Using GPU for training")
    else:
        logger.info("Using CPU for model creation")
    logger.info(f"Model name: {model_name}")

    # Default LR scheduler: CosineAnnealingWarmRestarts gives a cyclical
    # learning rate that helps escape local minima.  T_0=10 restarts the
    # cosine schedule every 10 epochs; T_mult=2 doubles the period after
    # each restart (10, 20, 40, …).
    scheduler_kwargs = dict(
        lr_scheduler=CosineAnnealingWarmRestarts,
        lr_scheduler_kwargs={'T_0': 10, 'T_mult': 2},
    )

    model = None
    if model_name in RESNET_MODEL_NAMES:
        resnet_variant = int(model_name[6:])
        model = ResNet(n_input_channels=1,
                        num_classes=num_classes,
                        spatial_dims=3,
                        resnet_variant=resnet_variant,
                        loss_kwargs=loss_kwargs,
                        **scheduler_kwargs)
    elif model_name == 'MST':
        model = MST(n_input_channels=1,
                    num_classes=num_classes,
                    spatial_dims=3,
                    loss_kwargs=loss_kwargs,
                    **scheduler_kwargs)
    elif model_name == 'MST_SAMMed2D':
        # MST slice fusion on the SAM-Med2D ViT-B encoder (Cambridge, branch custom_cam_ZD)
        model = MST(n_input_channels=1,
                    num_classes=num_classes,
                    spatial_dims=3,
                    backbone_type="sammed2d",
                    loss_kwargs=loss_kwargs,
                    **scheduler_kwargs)
    elif model_name == 'MedicalNet':
        # VHIO model: MedicalNet 3D-ResNet34 with dilated stages 3 and 4 (original Tencent
        # backbone), global average pooling and a linear head; pretrained on 23 datasets.
        model = MedicalNet(n_input_channels=1,
                           num_classes=num_classes,
                           spatial_dims=3,
                           pretrained_path=resolve_medicalnet_weights(logger),
                           loss_kwargs=loss_kwargs,
                           **scheduler_kwargs)
    elif model_name.startswith(CHALLENGE_PREFIX):
        team_name = model_name[len(CHALLENGE_PREFIX):]
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
            # Primary: look relative to the model's challenge directory
            primary_path = os.path.join(
                base_dir, "challenge", team_name, rel_path
            )
            # Fallback: Docker image stores weights at /MediSwarm/pretrained_weights/
            # to avoid bloating NVFlare job transfers (see _cacheAndCopyPretrainedModelWeights.sh)
            fallback_path = os.path.join("/MediSwarm/pretrained_weights", rel_path)
            if os.path.isfile(primary_path):
                persistor_args["pretrained_path"] = primary_path
            elif os.path.isfile(fallback_path):
                persistor_args["pretrained_path"] = fallback_path
                logger.info(f'Using fallback pretrained weights from Docker image: {fallback_path}')
            else:
                persistor_args["pretrained_path"] = primary_path  # let it fail with original path
            logger.info(f'__________ model path is : {persistor_args["pretrained_path"]}')

        # Forward loss_kwargs (e.g. class weights) so the challenge model
        # registers the same buffers as the server-side persistor model.
        # Without this, _class_weight may appear as a buffer on one side
        # but not the other, causing state_dict mismatches during
        # federated aggregation.
        if loss_kwargs:
            persistor_args["loss_kwargs"] = loss_kwargs

        factory_fn = getattr(module, func_name)
        logger.info(f"Now access {persistor_args} from module {module}")
        return factory_fn(**persistor_args)
    else:
        # Only reachable if get_supported_model_names() lists a name no branch above builds.
        raise unsupported_model_error(model_name)
    return model
