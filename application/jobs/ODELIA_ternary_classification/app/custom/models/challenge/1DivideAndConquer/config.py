from pathlib import Path

# Pretrained model checkpoint
PRETRAINED_CHECKPOINT = Path(__file__).resolve().parent / "pretrained_weights" / "checkpoint_final.pth"

# Data configuration
NUM_CLASSES = 3  # healthy, benign, malignant
IN_CHANNELS = 3

# This is really just a dir with three .b2nd files for testing
DATA_DIR = Path(__file__).resolve().parent / "test_data"

# Training hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 100
WARMUP_EPOCHS = 10

# Optimizer parameters
INITIAL_LR = 1e-4
MAX_LR = 1e-2
WEIGHT_DECAY = 3e-5
MOMENTUM = 0.99
NESTEROV = True

# Scheduler parameters
POLY_EXPONENT = 0.9

# Model parameters
FINAL_LAYER_DROPOUT = 0.0
SPATIAL_DIMS = 3

# Data augmentation parameters
AUG_CONFIG = {
    "p_rotation": 0.2,
    "rotation_range_degrees": (-30.0, 30.0),
    "p_scaling": 0.2,
    "scaling": (0.7, 1.4),
    "p_noise": 0.1,
    "noise_variance": (0.0, 0.1),
    "p_brightness": 0.15,
    "brightness_range": (0.75, 1.25),
    "p_contrast": 0.15,
    "contrast_range": (0.75, 1.25),
    "mirror_axes": (0, 1, 2),
}
NUM_WORKERS = 3