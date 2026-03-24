#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(Path(__file__))
import os
import torch

import nvflare.client.lightning as flare
import nvflare.client as flare_util

import threedcnn_ptl

TRAINING_MODE = os.getenv("TRAINING_MODE")

TM_PREFLIGHT_CHECK = "preflight_check"
TM_LOCAL_TRAINING = "local_training"
TM_SWARM = "swarm"

if not TRAINING_MODE:
    raise ValueError("TRAINING_MODE environment variable must be set")

if TRAINING_MODE == TM_SWARM:
    flare_util.init()
    SITE_NAME = flare.get_site_name()
    NUM_EPOCHS = threedcnn_ptl.get_num_epochs_per_round(SITE_NAME)
    MODEL_NAME = os.getenv("MODEL_NAME", "challenge_5Pimed")
elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
    SITE_NAME = os.getenv("SITE_NAME")
    MODEL_NAME = os.getenv("MODEL_NAME", "challenge_5Pimed")  # e.g. 'mst or 'challenge_2BCN_AIM'
    if not SITE_NAME:
        raise ValueError("SITE_NAME environment variable must be set for local training")
    if not MODEL_NAME:
        raise ValueError("MODEL_NAME environment variable must be set for local training")
    try:
        NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))
    except ValueError:
        raise ValueError("NUM_EPOCHS must be an integer")
    print(f"Running in {TRAINING_MODE} mode with SITE_NAME={SITE_NAME}, MODEL_NAME={MODEL_NAME}, NUM_EPOCHS={NUM_EPOCHS}")
else:
    raise ValueError(f"Unsupported TRAINING_MODE: {TRAINING_MODE}")


def main():
    """
    Main function for training and evaluating the model using NVFlare and PyTorch Lightning.
    The following variables are expect to be set: 
    SITE_NAME
    MODEL_NAME: can be 
       - MST or 
       - challenge_<model name as defined in ./challenge/challenge_models_config.sh> or 
       - challenge (that will select the first mentioned model in ./challenge/challenge_models_config.sh.)
    NUM_EPOCHS
    """
    logger = threedcnn_ptl.set_up_logging()

    try:
        data_module, model, checkpointing, trainer, path_run_dir, env_vars = threedcnn_ptl.prepare_training(
            logger, NUM_EPOCHS, model_variant=MODEL_NAME
        )

        if TRAINING_MODE == TM_SWARM:
            flare.patch(trainer)  # Patch trainer to enable swarm learning
            torch.autograd.set_detect_anomaly(True)

            logger.info(f"Site name: {SITE_NAME}")

            while flare.is_running():
                input_model = flare.receive()
                logger.info(f"Current round: {input_model.current_round}")

                threedcnn_ptl.validate_and_train(logger, data_module, model, trainer, path_run_dir)

        elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
            threedcnn_ptl.validate_and_train(logger, data_module, model, trainer, path_run_dir)

        if TRAINING_MODE in [TM_LOCAL_TRAINING, TM_SWARM]:
            threedcnn_ptl.finalize_training(logger, model, checkpointing, trainer, path_run_dir, env_vars)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
