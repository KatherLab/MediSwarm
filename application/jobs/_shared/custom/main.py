#!/usr/bin/env python3

import os
import torch

import nvflare.client.lightning as flare
import nvflare.client as flare_util

import round_hooks
import threedcnn_ptl

TRAINING_MODE = os.getenv("TRAINING_MODE")
TM_PREFLIGHT_CHECK = "preflight_check"
TM_LOCAL_TRAINING = "local_training"
TM_SWARM = "swarm"
LOG_DATASET_DETAILS = 'LOG_DATASET_DETAILS' in os.environ

if not TRAINING_MODE:
    raise ValueError("TRAINING_MODE environment variable must be set")

if TRAINING_MODE == TM_SWARM:
    flare_util.init(rank="0")
    SITE_NAME = flare.get_site_name()
    # Epoch count will be computed from training data size inside
    # prepare_training (weighted_epochs=True).  The placeholder value here
    # is overridden before the Trainer is created.
    NUM_EPOCHS = 1  # placeholder — replaced by weighted computation
    USE_WEIGHTED_EPOCHS = True
elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
    SITE_NAME = os.getenv("SITE_NAME")
    if not SITE_NAME:
        raise ValueError("SITE_NAME environment variable must be set for local training")
    try:
        NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))
    except ValueError:
        raise ValueError("NUM_EPOCHS must be an integer")
    USE_WEIGHTED_EPOCHS = False
else:
    raise ValueError(f"Unsupported TRAINING_MODE: {TRAINING_MODE}")


def _job_id() -> str:
    """Job id for the round hooks (e.g. per-job capture directories); "" when the client API cannot tell."""
    try:
        return flare.get_job_id() or ""
    except Exception:
        return ""


def _receive_round(ctx: round_hooks.RoundContext) -> None:
    """Receive this round's global and fill the round fields of the context."""
    input_model = flare.receive()
    ctx.input_model = input_model
    ctx.current_round = input_model.current_round
    ctx.total_rounds = getattr(input_model, "total_rounds", None)
    ctx.is_final_round = threedcnn_ptl.is_final_round(ctx.current_round, ctx.total_rounds)
    # Evaluation-only final round (ODELIA_FINAL_EVAL_ROUND=1, release contract 3.3) is not implemented
    # yet: every round trains. Its implementation sets this flag for the last round.
    ctx.is_eval_only_round = False


def _run_training_round(ctx: round_hooks.RoundContext, hooks) -> None:
    """Validate the received global, train locally (FLCallback sends the result), finalize in the last round.

    Every object is read from ctx at its point of use, so a hook that replaced one is honoured.
    """
    # #314: the per-round aggregated prediction export is expensive and
    # dominates swarm round time; throttle it (default: final round only).
    export_predictions = threedcnn_ptl.should_export_aggregated_predictions(ctx.current_round, ctx.total_rounds)
    stage_timings = {}
    threedcnn_ptl.validate_global_model(
        ctx.logger, ctx.data_module, ctx.model, ctx.trainer, ctx.path_run_dir,
        output_GT_and_classprob_aggregated_model=export_predictions, stage_timings=stage_timings,
    )
    round_hooks.call_hooks(hooks, "before_fit", ctx)
    threedcnn_ptl.train_local_model(ctx.logger, ctx.data_module, ctx.model, ctx.trainer, stage_timings=stage_timings)
    threedcnn_ptl.log_stage_timing_summary(ctx.logger, stage_timings, threedcnn_ptl.STAGE_TIMING_HEADING)
    round_hooks.call_hooks(hooks, "after_round", ctx)

    if ctx.is_final_round:
        threedcnn_ptl.finalize_training(
            ctx.logger, ctx.data_module, ctx.model, ctx.checkpointing, ctx.trainer, ctx.path_run_dir, ctx.env_vars
        )


def _run_eval_only_round(ctx: round_hooks.RoundContext, hooks) -> None:
    """Evaluation-only final round (release contract 3.3): evaluate the received final global on the
    validation and test split, finalize, and send the received weights back unchanged. No training."""
    raise NotImplementedError("The evaluation-only final round (ODELIA_FINAL_EVAL_ROUND=1) is not implemented yet")


def run_swarm(logger, data_module, model, checkpointing, trainer, path_run_dir, env_vars, hooks) -> None:
    flare.patch(trainer, load_state_dict_strict=False)  # Patch trainer to enable swarm learning; strict=False allows _class_weight buffer mismatch between server persistor (no class weights) and client models (with class weights computed from training data)
    torch.autograd.set_detect_anomaly(True)

    logger.info(f"Site name: {SITE_NAME}")

    ctx = round_hooks.RoundContext(
        logger=logger, site_name=SITE_NAME, job_id=_job_id(), env_vars=env_vars,
        model=model, trainer=trainer, data_module=data_module,
        checkpointing=checkpointing, path_run_dir=path_run_dir,
    )
    round_hooks.start_hooks(hooks, ctx, fl_callback_cls=flare.FLCallback)

    ended = False
    while flare.is_running():
        _receive_round(ctx)
        logger.info(f"Current round: {ctx.current_round} (of {ctx.total_rounds})")
        round_hooks.call_hooks(hooks, "after_receive", ctx)

        if ctx.is_eval_only_round:
            _run_eval_only_round(ctx, hooks)
        else:
            _run_training_round(ctx, hooks)

        # on_end runs inside the last round rather than after the loop: once the job is over the
        # subprocess may be stopped while it waits for the next task.
        if ctx.is_final_round and not ended:
            round_hooks.call_hooks(hooks, "on_end", ctx)
            ended = True

    if not ended:  # the job ended without a recognisable final round (total_rounds unknown)
        round_hooks.call_hooks(hooks, "on_end", ctx)


def main():
    """
    Main function for training and evaluating the model using NVFlare and PyTorch Lightning.
    """
    logger = threedcnn_ptl.set_up_logging()

    try:
        # Before prepare_training: a misconfigured hook list fails in seconds, not after data hashing.
        hooks = round_hooks.load_hooks(os.environ, logger)

        data_module, model, checkpointing, trainer, path_run_dir, env_vars = threedcnn_ptl.prepare_training(
            logger, NUM_EPOCHS, SITE_NAME, LOG_DATASET_DETAILS,
            weighted_epochs=USE_WEIGHTED_EPOCHS,
        )

        if TRAINING_MODE == TM_SWARM:
            run_swarm(logger, data_module, model, checkpointing, trainer, path_run_dir, env_vars, hooks)

        elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
            threedcnn_ptl.validate_and_train(logger, data_module, model, trainer, path_run_dir, output_GT_and_classprob_aggregated_model=False)
            threedcnn_ptl.finalize_training(logger, data_module, model, checkpointing, trainer, path_run_dir, env_vars)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
