import nvflare.client.lightning as flare
import nvflare.client as flare_util
import torch

import threedcnn_ptl

flare_util.init()

SITE_NAME = flare.get_site_name()
NUM_EPOCHS_PER_ROUND = threedcnn_ptl.get_num_epochs_per_round(SITE_NAME)


def main():
    """
    Main function for training and evaluating the model using NVFlare and PyTorch Lightning.
    """
    logger = threedcnn_ptl.set_up_logging()
    try:
        data_module, model, checkpointing, trainer, path_run_dir, env_vars = threedcnn_ptl.prepare_training(logger, NUM_EPOCHS_PER_ROUND, SITE_NAME)

        flare.patch(trainer)  # Patch trainer to enable swarm learning
        torch.autograd.set_detect_anomaly(True)

        logger.info(f"Site name: {SITE_NAME}")

        while flare.is_running():
            input_model = flare.receive()
            logger.info(f"Current round: {input_model.current_round}")

            threedcnn_ptl.validate_and_train(logger, data_module, model, trainer)

        threedcnn_ptl.finalize_training(logger, model, checkpointing, trainer, path_run_dir, env_vars)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
