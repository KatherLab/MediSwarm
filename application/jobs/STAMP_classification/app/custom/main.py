#!/usr/bin/env python3
"""
NVFlare entry point for STAMP swarm learning.

This follows the same pattern as the ODELIA main.py but uses STAMP's
data pipeline (H5 features + clinical tables) and model registry instead
of ODELIA's 3D CNN pipeline.

Supports three training modes:
- swarm:            NVFlare-managed federated training
- local_training:   Single-site training without federation
- preflight_check:  Quick smoke test with 1 epoch
"""

import logging
import os

import torch

import nvflare.client.lightning as flare
import nvflare.client as flare_util

import stamp_training

logger = logging.getLogger(__name__)

TRAINING_MODE = os.getenv("TRAINING_MODE")
TM_PREFLIGHT_CHECK = "preflight_check"
TM_LOCAL_TRAINING = "local_training"
TM_SWARM = "swarm"

if not TRAINING_MODE:
    raise ValueError("TRAINING_MODE environment variable must be set")

if TRAINING_MODE == TM_SWARM:
    flare_util.init(rank="0")
    SITE_NAME = flare.get_site_name()
    # Epoch count will be computed from training data size inside
    # prepare_training (weighted_epochs=True).
    NUM_EPOCHS = 1  # placeholder — replaced by weighted computation
    USE_WEIGHTED_EPOCHS = True
    # Total federated rounds — needed to size OneCycleLR scheduler steps.
    # Must match num_rounds in config_fed_server.conf (default 20).
    TOTAL_ROUNDS = int(os.getenv("STAMP_NUM_ROUNDS", "20"))
elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
    SITE_NAME = os.getenv("SITE_NAME")
    if not SITE_NAME:
        raise ValueError("SITE_NAME environment variable must be set for local training")
    # The two modes want very different epoch budgets:
    #   preflight_check  — a smoke test: 1 epoch is the point.
    #   local_training   — the site's *baseline*, which the swarm model is compared
    #                      against. It must train properly, so it follows
    #                      STAMP_MAX_EPOCHS (STAMP's own default: 32).
    # Defaulting both to NUM_EPOCHS=1 silently produced a 1-epoch baseline, making
    # the swarm-vs-local comparison meaningless.
    if TRAINING_MODE == TM_LOCAL_TRAINING:
        default_epochs = os.getenv("STAMP_MAX_EPOCHS", "32")
    else:
        default_epochs = "1"
    try:
        NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", default_epochs))
    except ValueError:
        raise ValueError("NUM_EPOCHS must be an integer")
    USE_WEIGHTED_EPOCHS = False
    TOTAL_ROUNDS = 1
else:
    raise ValueError(f"Unsupported TRAINING_MODE: {TRAINING_MODE}")


def main():
    """Main function for STAMP training with NVFlare swarm learning."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [{SITE_NAME}] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(f"STAMP training starting — mode={TRAINING_MODE}, site={SITE_NAME}")

    try:
        # Load STAMP environment configuration
        env = stamp_training.load_stamp_environment()
        # Override with runtime settings
        env["site_name"] = SITE_NAME

        # Prepare training (data, model, trainer)
        train_dl, valid_dl, model, checkpointing, trainer, output_dir, metric_callback = (
            stamp_training.prepare_training(
                env, NUM_EPOCHS,
                weighted_epochs=USE_WEIGHTED_EPOCHS,
                total_rounds=TOTAL_ROUNDS,
            )
        )

        if TRAINING_MODE == TM_SWARM:
            flare.patch(trainer)
            torch.autograd.set_detect_anomaly(True)

            logger.info(f"Entering swarm learning loop — site: {SITE_NAME}")

            while flare.is_running():
                input_model = flare.receive()
                logger.info(f"Round {input_model.current_round}")

                stamp_training.validate_and_train(
                    train_dl, valid_dl, model, trainer,
                    output_dir=output_dir, current_round=input_model.current_round,
                )

                # Report validation metric to NVFlare for model selection
                if metric_callback.last_val_loss is not None:
                    logger.info(
                        f"Round {input_model.current_round} — "
                        f"val_loss={metric_callback.last_val_loss:.4f}"
                        + (f", val_auroc={metric_callback.last_val_auroc:.4f}"
                           if metric_callback.last_val_auroc is not None else "")
                    )

                # Consolidate checkpoints every round. When the swarm loop ends,
                # the launcher (launch_once=true) SIGTERMs this subprocess before
                # the post-loop finalize can run, so finalize must happen in-loop
                # (#480). It is idempotent — copies the latest best/last checkpoints
                # — so repeating it each round is cheap, and the final round leaves
                # the correct best/last artifacts behind.
                stamp_training.finalize_training(model, checkpointing, trainer, output_dir)

        elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
            stamp_training.validate_and_train(
                train_dl, valid_dl, model, trainer,
                output_dir=output_dir,
            )

        # Swarm mode finalizes per-round inside the loop above (see #480); only
        # local training reaches this post-loop path reliably.
        if TRAINING_MODE == TM_LOCAL_TRAINING:
            stamp_training.finalize_training(model, checkpointing, trainer, output_dir)

    except Exception as e:
        logger.error(f"STAMP training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
