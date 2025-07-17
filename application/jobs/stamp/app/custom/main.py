import logging
import os
import sys
from pathlib import Path

from modeling.train import train_categorical_model_

# Set up the logger
_logger = logging.getLogger("stamp")
_logger.setLevel(logging.DEBUG)
_formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

_stream_handler = logging.StreamHandler(sys.stderr)
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(_formatter)
_logger.addHandler(_stream_handler)


def _add_file_handle_(logger: logging.Logger, *, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(output_dir / "logfile.log")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


def main():
    output_dir = os.getenv("TRAINING_OUTPUT_DIR")
    _add_file_handle_(_logger, output_dir=Path(output_dir))
    _logger.info("Using training configuration from environment variables.")

    train_categorical_model_(
        output_dir=Path(output_dir),
        clini_table=Path(os.getenv("TRAINING_CLINI_TABLE")),
        slide_table=Path(os.getenv("TRAINING_SLIDE_TABLE")),
        feature_dir=Path(os.getenv("TRAINING_FEATURE_DIR")),
        patient_label=os.getenv("TRAINING_PATIENT_LABEL"),
        ground_truth_label=os.getenv("TRAINING_GROUND_TRUTH_LABEL"),
        filename_label=os.getenv("TRAINING_FILENAME_LABEL"),
        categories=os.getenv("TRAINING_CATEGORIES").split(","),
        # Dataset and loader parameters
        bag_size=int(os.getenv("TRAINING_BAG_SIZE")),
        num_workers=int(os.getenv("TRAINING_NUM_WORKERS")),
        # Training parameters
        batch_size=int(os.getenv("TRAINING_BATCH_SIZE")),
        max_epochs=int(os.getenv("TRAINING_MAX_EPOCHS")),
        patience=int(os.getenv("TRAINING_PATIENCE")),
        accelerator=os.getenv("TRAINING_ACCELERATOR"),
        # Experimental features
        use_vary_precision_transform=os.getenv("USE_VARY_PRECISION_TRANSFORM", "False").lower() == "true",
        use_alibi=os.getenv("USE_ALIBI", "False").lower() == "true",
    )


if __name__ == "__main__":
    main()
