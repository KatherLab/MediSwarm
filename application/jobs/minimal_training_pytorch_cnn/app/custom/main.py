#!/usr/bin/env python3

import nvflare.client.lightning as flare
import nvflare.client as flare_util
import torch

import minimal_training

FLARE_MODE = "swarm"

if FLARE_MODE == "swarm":
    flare_util.init()
    SITE_NAME=flare.get_site_name()
elif FLARE_MODE == "preflight_check" or FLARE_MODE == "local_training":
    SITE_NAME="site_name_unset"
else:
    raise Exception(f"Illegal FLARE_MODE {FLARE_MODE}")


def main():
    """
    Main function for training and evaluating the model using NVFlare and PyTorch Lightning.
    """
    logger = minimal_training.set_up_logging()
    try:
        data_module, model, checkpointing, trainer = minimal_training.prepare_training(logger)

        if FLARE_MODE == "swarm":
            flare.patch(trainer)  # Patch trainer to enable swarm learning
            torch.autograd.set_detect_anomaly(True)

            logger.info(f"Site name: {SITE_NAME}")

            while flare.is_running():
                input_model = flare.receive()
                logger.info(f"Current round: {input_model.current_round}")

                minimal_training.validate_and_train(logger, data_module, model, trainer)

        elif FLARE_MODE == "preflight_check" or FLARE_MODE == "local_training":
            minimal_training.validate_and_train(logger, data_module, model, trainer)

        minimal_training.finalize_training(logger, model, checkpointing, trainer)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
