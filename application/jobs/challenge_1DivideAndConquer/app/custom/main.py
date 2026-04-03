#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(Path(__file__))
import os
import signal
import traceback
import torch

import nvflare.client.lightning as flare
import nvflare.client as flare_util

import threedcnn_ptl


def _signal_handler(signum, frame):
    """Catch signals and print a traceback before dying."""
    sig_name = signal.Signals(signum).name
    print(f"\n[SIGNAL] Received {sig_name} ({signum})", flush=True)
    traceback.print_stack(frame)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(128 + signum)


# Register signal handlers so we get a traceback instead of silent death
for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2):
    signal.signal(_sig, _signal_handler)

TRAINING_MODE = os.getenv("TRAINING_MODE")

TM_PREFLIGHT_CHECK = "preflight_check"
TM_LOCAL_TRAINING = "local_training"
TM_SWARM = "swarm"

if not TRAINING_MODE:
    raise ValueError("TRAINING_MODE environment variable must be set")

if TRAINING_MODE == TM_SWARM:
    flare_util.init(rank="0")
    SITE_NAME = flare.get_site_name()
    NUM_EPOCHS = threedcnn_ptl.get_num_epochs_per_round(SITE_NAME)
    MODEL_NAME = "challenge_1DivideAndConquer"  # Hardcoded: ignore global MODEL_NAME env var (defaults to MST)
elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
    SITE_NAME = os.getenv("SITE_NAME")
    MODEL_NAME = "challenge_1DivideAndConquer"  # e.g. 'mst or 'challenge_2BCN_AIM'
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
        logger.info("[DEBUG] Starting prepare_training...")
        sys.stdout.flush()
        data_module, model, checkpointing, trainer, path_run_dir, env_vars = threedcnn_ptl.prepare_training(
            logger, NUM_EPOCHS, model_variant=MODEL_NAME
        )
        logger.info("[DEBUG] prepare_training complete.")
        sys.stdout.flush()

        if TRAINING_MODE == TM_SWARM:
            flare.patch(trainer)  # Patch trainer to enable swarm learning
            torch.autograd.set_detect_anomaly(True)

            logger.info(f"Site name: {SITE_NAME}")
            sys.stdout.flush()

            while flare.is_running():
                logger.info("[DEBUG] flare.is_running() returned True, calling flare.receive()...")
                sys.stdout.flush()
                input_model = flare.receive()
                logger.info(f"[DEBUG] flare.receive() returned. Current round: {input_model.current_round}")
                logger.info(f"[DEBUG] input_model type: {type(input_model)}, params type: {type(input_model.params) if input_model.params else None}")
                if input_model.params:
                    logger.info(f"[DEBUG] input_model.params keys count: {len(input_model.params)}")
                sys.stdout.flush()

                logger.info("[DEBUG] Calling validate_and_train...")
                sys.stdout.flush()
                threedcnn_ptl.validate_and_train(logger, data_module, model, trainer, path_run_dir)
                logger.info("[DEBUG] validate_and_train complete.")
                sys.stdout.flush()

        elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
            threedcnn_ptl.validate_and_train(logger, data_module, model, trainer, path_run_dir)

        if TRAINING_MODE in [TM_LOCAL_TRAINING, TM_SWARM]:
            threedcnn_ptl.finalize_training(logger, model, checkpointing, trainer, path_run_dir, env_vars)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
