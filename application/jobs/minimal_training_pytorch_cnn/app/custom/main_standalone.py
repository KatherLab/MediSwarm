#!/usr/bin/env python3

import minimal_training

if __name__ == "__main__":
    logger = minimal_training.set_up_logging()
    data_module, model, checkpointing, trainer = minimal_training.prepare_training(logger)
    minimal_training.validate_and_train(logger, data_module, model, trainer)
    minimal_training.finalize_training(logger, model, checkpointing, trainer)
