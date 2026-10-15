#!/usr/bin/env python3

import os
import torch

import nvflare.client.lightning as flare
import nvflare.client as flare_util
from nvflare.app_common.abstract.fl_model import FLModel, MetaKey, ParamsType
# The module whose send()/clear() FLCallback itself uses; the evaluation-only round answers through them.
from nvflare.app_opt.lightning import api as fl_lightning_api

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


def _receive_round(ctx: round_hooks.RoundContext, final_eval_round: bool = False) -> None:
    """Receive this round's global and fill the round fields of the context."""
    input_model = flare.receive()
    ctx.input_model = input_model
    ctx.current_round = input_model.current_round
    ctx.total_rounds = getattr(input_model, "total_rounds", None)
    ctx.is_final_round = threedcnn_ptl.is_final_round(ctx.current_round, ctx.total_rounds)
    # ODELIA_FINAL_EVAL_ROUND=1 (release contract 3.3): the job's last round evaluates the final global
    # instead of training. Needs total_rounds; without it no round is recognised as the last one.
    ctx.is_eval_only_round = final_eval_round and ctx.is_final_round


def _check_received_structure(ctx: round_hooks.RoundContext) -> None:
    """#541: refuse a received global whose parameters do not fit this site's model.

    FLCallback loads the global non-strictly (load_state_dict_strict=False, for the _class_weight buffer)
    and only stops on shape mismatches or zero overlap, so a global from a partially overlapping
    architecture (another MODEL_NAME at the starting client, a stale warm-start file) would load in part
    and train on silently. Checked on the first global a site receives; every later one has the same keys.
    """
    from models.base_model import state_dict_mismatch  # job code; imported here to keep main.py light

    params = getattr(ctx.input_model, "params", None)
    if not params:
        return
    problem = state_dict_mismatch(ctx.model.state_dict(), params)
    model_name = (ctx.env_vars or {}).get("model_name", "?")
    if problem:
        raise RuntimeError(
            f"The global received in round {ctx.current_round} does not fit this site's model "
            f"(MODEL_NAME={model_name}): {problem}. Every site and the job's persistor must build the same "
            f"model (#541): pin MODEL_NAME in the job's launcher and persistor (build_consortium_job.py does)."
        )
    ctx.logger.info(f"Received global fits this site's model (MODEL_NAME={model_name}, {len(params)} tensors)")


def _run_training_round(ctx: round_hooks.RoundContext, hooks) -> None:
    """Validate the received global, train locally (FLCallback sends the result), finalize in the last round.

    Every object is read from ctx at its point of use, so a hook that replaced one is honoured.
    """
    # #314: the per-round aggregated prediction export is expensive and
    # dominates swarm round time; throttle it (default: final round only).
    export_predictions = threedcnn_ptl.should_export_aggregated_predictions(ctx.current_round, ctx.total_rounds)
    stage_timings = {}
    # The received global streams as global_val/* (contract 3.4); val/* is the local model inside the fit.
    threedcnn_ptl.validate_global_model(
        ctx.logger, ctx.data_module, ctx.model, ctx.trainer, ctx.path_run_dir,
        output_GT_and_classprob_aggregated_model=export_predictions, stage_timings=stage_timings,
        metric_prefix=threedcnn_ptl.GLOBAL_VAL_PREFIX, current_round=ctx.current_round,
    )
    round_hooks.call_hooks(hooks, "before_fit", ctx)
    threedcnn_ptl.train_local_model(ctx.logger, ctx.data_module, ctx.model, ctx.trainer, stage_timings=stage_timings,
                                    current_round=ctx.current_round)
    threedcnn_ptl.log_stage_timing_summary(ctx.logger, stage_timings, threedcnn_ptl.STAGE_TIMING_HEADING)
    round_hooks.call_hooks(hooks, "after_round", ctx)

    if ctx.is_final_round:
        threedcnn_ptl.finalize_training(
            ctx.logger, ctx.data_module, ctx.model, ctx.checkpointing, ctx.trainer, ctx.path_run_dir, ctx.env_vars
        )


def _fl_callback(trainer):
    for callback in trainer.callbacks:
        if isinstance(callback, flare.FLCallback):
            return callback
    return None


def _return_received_global(ctx: round_hooks.RoundContext, metrics: dict) -> None:
    """Answer the evaluation-only round's train task with the received weights, unchanged.

    Why the final global stays identical: the jobs exchange differences (params_transfer_type = "DIFF").
    Inside this process the client API turns the FULL params sent here into "sent - received", against the
    very task this round received (ModelRegistry._prepare_param_diff), so sending the received tensors
    themselves is an exact zero update for every key. InTimeAccumulateWeightedAggregator averages the zero
    updates (weight NUM_STEPS_CURRENT_ROUND = 1; any positive weight averages zeros to zero) and
    SimpleModelShareableGenerator adds that zero to the round's global, so the final global the clients
    persist (FL_global_model.pt) is the global received here. (Bitwise, a stored -0.0 would become +0.0.)

    The metrics are those of the received global's validation, exactly what FLCallback records for every
    round: the client job rejects a train result without metrics (train_with_evaluation = true), and CCWF's
    IntimeModelSelector reads their key_metric ("val/AUC_ROC") to decide whether this final global is also
    the best global of the run.
    """
    fl_callback = _fl_callback(ctx.trainer)
    if fl_callback is not None and fl_callback.metrics:
        metrics = fl_callback.metrics
    result = FLModel(params_type=ParamsType.FULL, params=ctx.input_model.params, metrics=metrics,
                     meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1})
    # clear_cache=False as in FLCallback._send_model: the received params stay readable for on_end hooks.
    fl_lightning_api.send(result, clear_cache=False)
    # What FLCallback.reset_state() does after its own send: forget this round's validation metrics (a later
    # validate() then loads the next task's model) and drop the answered task. Its fit-loop bookkeeping is
    # skipped: nothing trains after this round, and it would build a train dataloader just to count batches.
    if fl_callback is not None:
        fl_callback.metrics = None
    fl_lightning_api.clear()
    key_metric = threedcnn_ptl.resolve_key_metric()[0]
    ctx.logger.info(
        f"Round {ctx.current_round}: returned the received global unchanged (NUM_STEPS_CURRENT_ROUND=1) "
        f"with the metrics of its validation ({key_metric}={metrics.get(key_metric)})"
    )


def _run_eval_only_round(ctx: round_hooks.RoundContext, hooks) -> None:
    """Evaluation-only final round (ODELIA_FINAL_EVAL_ROUND=1, release contract 3.3); no training.

    The received global is the final global of the job. It is validated (final_val/*) and tested
    (final_test/*), its local CSVs are written and, with ODELIA_RETURN_PER_CASE=1, its per-case rows are
    returned; finalize_training() exports the local best/last checkpoints on the test split as before; then
    the received weights go back unchanged so CCWF completes the round with an identical final global.

    Hooks see after_receive (with ctx.is_eval_only_round=True) and on_end, never before_fit or after_round:
    there is no fit, hence no update for DP to noise and no local model to capture. The per-epoch CSV
    callback does not run either.

    The weights are sent last on purpose. Once every site has answered, CCWF ends the job and the client
    stops this subprocess, so evaluation work placed after the send could be cut off. Only the validation of
    the received global can fail the round; the test split, the CSVs and finalize_training() are outputs and
    run best effort (threedcnn_ptl.best_effort): up to 1.8 finalize ran after the last send, where a failure
    could no longer hold up the job either.
    """
    stage_timings = {}
    export_predictions = threedcnn_ptl.should_export_aggregated_predictions(ctx.current_round, ctx.total_rounds)
    metrics = threedcnn_ptl.evaluate_final_global_model(
        ctx.logger, ctx.data_module, ctx.model, ctx.trainer, ctx.path_run_dir, ctx.current_round,
        output_GT_and_classprob_aggregated_model=export_predictions, stage_timings=stage_timings,
    )

    def _finalize():
        with threedcnn_ptl.timed_stage(ctx.logger, "Local checkpoint test export (finalize)", stage_timings):
            threedcnn_ptl.finalize_training(
                ctx.logger, ctx.data_module, ctx.model, ctx.checkpointing, ctx.trainer, ctx.path_run_dir,
                ctx.env_vars,
            )

    threedcnn_ptl.best_effort(ctx.logger, "Local checkpoint test export (finalize)", _finalize)
    with threedcnn_ptl.timed_stage(ctx.logger, "Return received global", stage_timings):
        _return_received_global(ctx, metrics)
    threedcnn_ptl.log_stage_timing_summary(ctx.logger, stage_timings, threedcnn_ptl.STAGE_TIMING_HEADING)


def run_swarm(logger, data_module, model, checkpointing, trainer, path_run_dir, env_vars, hooks,
              final_eval_round: bool = False) -> None:
    flare.patch(trainer, load_state_dict_strict=False)  # Patch trainer to enable swarm learning; strict=False allows _class_weight buffer mismatch between server persistor (no class weights) and client models (with class weights computed from training data)
    torch.autograd.set_detect_anomaly(True)

    logger.info(f"Site name: {SITE_NAME}")
    if final_eval_round:
        logger.info(
            f"{threedcnn_ptl.FINAL_EVAL_ROUND_ENV}=1: the job's last round evaluates the final global on the "
            f"validation and test split instead of training (training rounds = num_rounds - 1)"
        )

    ctx = round_hooks.RoundContext(
        logger=logger, site_name=SITE_NAME, job_id=_job_id(), env_vars=env_vars,
        model=model, trainer=trainer, data_module=data_module,
        checkpointing=checkpointing, path_run_dir=path_run_dir,
    )
    round_hooks.start_hooks(hooks, ctx, fl_callback_cls=flare.FLCallback)

    ended = False
    structure_checked = False
    while flare.is_running():
        _receive_round(ctx, final_eval_round)
        if not structure_checked:
            _check_received_structure(ctx)
            structure_checked = True
        logger.info(f"Current round: {ctx.current_round} (of {ctx.total_rounds})"
                    + (" -- evaluation only" if ctx.is_eval_only_round else ""))
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
        # Before prepare_training: a misconfigured hook list or switch fails in seconds, not after data hashing.
        hooks = round_hooks.load_hooks(os.environ, logger)
        final_eval_round = threedcnn_ptl.final_eval_round_enabled()
        threedcnn_ptl.export_train_predictions_enabled()  # raises on a value that is not a 0/1 switch

        data_module, model, checkpointing, trainer, path_run_dir, env_vars = threedcnn_ptl.prepare_training(
            logger, NUM_EPOCHS, SITE_NAME, LOG_DATASET_DETAILS,
            weighted_epochs=USE_WEIGHTED_EPOCHS,
        )

        if TRAINING_MODE == TM_SWARM:
            run_swarm(logger, data_module, model, checkpointing, trainer, path_run_dir, env_vars, hooks,
                      final_eval_round=final_eval_round)

        elif TRAINING_MODE in [TM_PREFLIGHT_CHECK, TM_LOCAL_TRAINING]:
            if final_eval_round:
                logger.warning(f"{threedcnn_ptl.FINAL_EVAL_ROUND_ENV}=1 only applies to swarm training; ignored")
            threedcnn_ptl.validate_and_train(logger, data_module, model, trainer, path_run_dir, output_GT_and_classprob_aggregated_model=False)
            threedcnn_ptl.finalize_training(logger, data_module, model, checkpointing, trainer, path_run_dir, env_vars)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
