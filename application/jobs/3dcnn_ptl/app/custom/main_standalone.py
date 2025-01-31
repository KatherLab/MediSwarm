#!/usr/bin/env python3

import threedcnn_ptl

SITE_NAME = ""

if __name__ == "__main__":
    max_epochs = threedcnn_ptl.get_max_num_of_epochs(SITE_NAME)
    logger = threedcnn_ptl.set_up_logging()
    data_module, model, checkpointing, trainer, path_run_dir, env_vars = threedcnn_ptl.prepare_training(logger, max_epochs, SITE_NAME)
    threedcnn_ptl.validate_and_train(logger, data_module, model, trainer)
    threedcnn_ptl.finalize_training(logger, model, checkpointing, trainer, path_run_dir, env_vars)
