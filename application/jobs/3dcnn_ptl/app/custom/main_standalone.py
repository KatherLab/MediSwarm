#!/usr/bin/env python3

import argparse
import threedcnn_ptl

if __name__ == "__main__":
    logger = threedcnn_ptl.set_up_logging()

    parser = argparse.ArgumentParser(description='Process some site information.')
    parser.add_argument('--site_name', type=str, required=True, help='Name of your site (matching the data folder name)')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs to be run in total')
    args = parser.parse_args()

    data_module, model, checkpointing, trainer, path_run_dir, env_vars = threedcnn_ptl.prepare_training(logger, args.num_epochs, args.site_name)
    threedcnn_ptl.validate_and_train(logger, data_module, model, trainer)
    threedcnn_ptl.finalize_training(logger, model, checkpointing, trainer, path_run_dir, env_vars)
