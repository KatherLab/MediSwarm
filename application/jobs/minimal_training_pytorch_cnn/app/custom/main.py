#!/usr/bin/env python3

import os

import nvflare.client as flare_util
import nvflare.client.lightning as flare
import torch

import minimal_training

TRAINING_MODE = os.getenv("TRAINING_MODE")
TRAINING_MODE = "swarm"
if TRAINING_MODE == "swarm":
    flare_util.init()
    SITE_NAME=flare.get_site_name()
elif TRAINING_MODE == "local_training":
    SITE_NAME="site_name_unset"
else:
    raise Exception(f"Illegal TRAINING_MODE {TRAINING_MODE}")


def main():
    """
    Main function for training and evaluating the model using NVFlare and PyTorch Lightning.
    """
    logger = minimal_training.set_up_logging()
    try:
        data_module, model, checkpointing, trainer = minimal_training.prepare_training(logger)

        if TRAINING_MODE == "swarm":
            flare.patch(trainer)  # Patch trainer to enable swarm learning
            torch.autograd.set_detect_anomaly(True)

            logger.info(f"Site name: {SITE_NAME}")

            while flare.is_running():
                logger.info('Waiting for input model from server...')
                input_model = flare.receive()

                if input_model is not None:
                    logger.info("==== Swarm model received ====")
                    logger.info(
                        f"input_model.params.keys() = {list(input_model.params.keys())[:10]} ... total = {len(input_model.params)}")
                    logger.info(
                        f"model.state_dict().keys() = {list(model.state_dict().keys())[:10]} ... total = {len(model.state_dict())}")


                logger.info(f"Current round: {input_model.current_round}")

                minimal_training.validate_and_train(logger, data_module, model, trainer)

        elif TRAINING_MODE == "preflight_check" or TRAINING_MODE == "local_training":
            minimal_training.validate_and_train(logger, data_module, model, trainer)

        minimal_training.finalize_training(logger, model, checkpointing, trainer)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
