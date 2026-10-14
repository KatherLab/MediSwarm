"""Round hooks: fixed extension points of the swarm training loop in main.py.

Why this exists
---------------
Several experiments must act inside the training subprocess at fixed points of every swarm round:
keep the received global and the sent local model (MS6), add calibrated noise to the update before it
leaves the site (DP), restrict training to an actively selected subset (active learning), check the
round-0 global against a warm-start digest. Each feature lives in its own module next to this file and
is switched on per job from the launcher line in config_fed_client.conf, for example

    script = "ODELIA_ROUND_HOOKS=ms6_capture,dp_noise ODELIA_DP_SIGMA=... python3 custom/{app_script} ..."

so main.py stays one short loop and no experiment needs a new image or kit: this code travels with the
job. The SubprocessLauncher turns the KEY=VALUE prefixes into environment variables of the training
subprocess only; components in config_fed_client.conf never see them.

Rules
-----
* ``ODELIA_ROUND_HOOKS`` is a comma-separated list of module names under ``custom/``. Each module
  defines ``create_hook(env, logger) -> RoundHook | None`` (None = not active for this site/config).
  A listed module that cannot be imported, or that has no ``create_hook``, is a hard error and never a
  silent skip.
* Safety net: a feature variable (``ODELIA_MS6_*``, ``ODELIA_DP_*``, ``ODELIA_AL_*``,
  ``ODELIA_TRAIN_INCLUDE_LIST``) that is present while the module acting on it is not listed fails the
  job before any data is touched. A requested feature that silently does nothing must be impossible;
  for DP it would mean updates leaving the site without the noise the protocol promised.
* Hooks are active only in ``TRAINING_MODE=swarm``. Preflight and local training get no hooks (a
  feature that needs local mode has to extend ``load_hooks`` and the local path in main.py).
* Hooks run in list order. An exception in a hook fails the round: nothing is swallowed here, so a
  feature that wants best-effort behaviour (e.g. MS6 capture on a full disk) catches its own errors.

Call points, in the order main.py reaches them
----------------------------------------------
    on_start(ctx)             once, after prepare_training() and flare.patch(), before round 0
    lightning_callbacks(ctx)  once, after every hook's on_start; the returned callbacks are inserted
                              into trainer.callbacks immediately BEFORE NVFlare's FLCallback, so their
                              on_train_end runs before FLCallback.on_train_end sends the local model
    after_receive(ctx)        each round, after flare.receive(), before the global is validated/loaded
                              (ctx.model still holds last round's local weights; ctx.input_model.params
                              is the received global)
    before_fit(ctx)           each round, after the received global was validated (it is now loaded
                              into ctx.model), before trainer.fit()
    after_round(ctx)          each round, after trainer.fit() returned (the model has been sent)
    on_end(ctx)               once, after the last round (in the final round itself, so it also runs
                              when the job ends right after it)

main.py reads model, trainer, data_module, checkpointing and path_run_dir from the RoundContext at every
use, so a hook may replace any of them (e.g. a data module restricted to a selected subset) and the loop
uses the replacement from then on. A hook that replaces the trainer must flare.patch() the new one and
re-insert its callbacks with insert_before_fl_callback().

What FLCallback does to the weights (NVFlare fork, nvflare/app_opt/lightning/api.py): it loads the
received global in on_validation_start of the round's first validate() AND again in on_train_start,
because the client API keeps returning the current task's model until it is answered. Weight changes made
in before_fit are therefore overwritten when training starts; change the update in a callback's
on_train_end instead (it runs before the send), or the data/trainer setup in before_fit.
"""

import importlib
import logging
import os
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

HOOKS_ENV = "ODELIA_ROUND_HOOKS"
SWARM_MODE = "swarm"

# Feature variables and the hook module that must be listed whenever one of them is present.
# Matching is by prefix, so a future ODELIA_DP_* variable is covered without touching this table.
FEATURE_VARIABLES: Tuple[Tuple[str, str], ...] = (
    ("ODELIA_MS6_", "ms6_capture"),
    ("ODELIA_DP_", "dp_noise"),
    ("ODELIA_AL_", "al_subset"),
    ("ODELIA_TRAIN_INCLUDE_LIST", "al_subset"),
)

# Methods of RoundHook that call_hooks() may dispatch to (lightning_callbacks is collected by start_hooks).
HOOK_POINTS = ("on_start", "after_receive", "before_fit", "after_round", "on_end")

_LOG = logging.getLogger(__name__)


class RoundHookError(RuntimeError):
    """Round-hook configuration that must stop the job (unimportable module, unlisted feature, ...)."""


class RoundContext:
    """Plain attribute bag shared by main.py and the hooks; one instance per training subprocess.

    Fields:
        logger, site_name, job_id (``flare.get_job_id()`` or ""), env_vars (dict from prepare_training)
        model, trainer, data_module, checkpointing, path_run_dir: what the loop uses; always read from here
        current_round, total_rounds: ints from the received FLModel (None before round 0; total may be None)
        is_final_round: last round of the job (total_rounds - 1)
        is_eval_only_round: True only in the evaluation-only final round (ODELIA_FINAL_EVAL_ROUND=1)
        input_model: FLModel received this round (params = the received global state dict)
        state: dict the hooks share across rounds; key it by hook name to avoid collisions
    """

    def __init__(
        self,
        logger=None,
        site_name: str = "",
        job_id: str = "",
        env_vars: Optional[dict] = None,
        model=None,
        trainer=None,
        data_module=None,
        checkpointing=None,
        path_run_dir=None,
    ):
        self.logger = logger
        self.site_name = site_name
        self.job_id = job_id
        self.env_vars = env_vars if env_vars is not None else {}
        self.model = model
        self.trainer = trainer
        self.data_module = data_module
        self.checkpointing = checkpointing
        self.path_run_dir = path_run_dir
        self.current_round: Optional[int] = None
        self.total_rounds: Optional[int] = None
        self.is_final_round = False
        self.is_eval_only_round = False
        self.input_model = None
        self.state: dict = {}

    def __repr__(self) -> str:
        # Deliberately short: the default repr of model/trainer would print the whole network.
        return (
            f"RoundContext(site={self.site_name!r}, job={self.job_id!r}, round={self.current_round}"
            f"/{self.total_rounds}, final={self.is_final_round}, eval_only={self.is_eval_only_round})"
        )


class RoundHook:
    """Base class of a round hook. Every method is optional; the defaults do nothing."""

    name = "base"

    def on_start(self, ctx: RoundContext) -> None:
        pass

    def after_receive(self, ctx: RoundContext) -> None:
        pass

    def before_fit(self, ctx: RoundContext) -> None:
        pass

    def lightning_callbacks(self, ctx: RoundContext) -> list:
        return []

    def after_round(self, ctx: RoundContext) -> None:
        pass

    def on_end(self, ctx: RoundContext) -> None:
        pass


def parse_hook_names(value: Optional[str]) -> List[str]:
    """Split ODELIA_ROUND_HOOKS into module names; reject anything that is not a plain module name.

    Duplicates are an error rather than collapsed: listing dp_noise twice would add the noise twice.
    """
    names: List[str] = []
    for raw in (value or "").split(","):
        name = raw.strip()
        if not name:
            continue
        if not name.isidentifier():
            raise RoundHookError(f"{HOOKS_ENV}: '{name}' is not a module name under custom/ (expected e.g. dp_noise)")
        if name in names:
            raise RoundHookError(f"{HOOKS_ENV}: '{name}' is listed twice; every hook may run only once per call point")
        names.append(name)
    return names


def feature_variables(env: Mapping[str, str]) -> List[Tuple[str, str]]:
    """Return (variable, required module) for every feature variable present in ``env``.

    Presence counts, even with an empty value: an empty ODELIA_DP_SIGMA usually means a template value
    was lost on its way into the launcher line, which is exactly the case that has to stop the job.
    """
    found = []
    for var in sorted(env):
        for prefix, module in FEATURE_VARIABLES:
            if var.startswith(prefix):
                found.append((var, module))
                break
    return found


def check_feature_variables(env: Mapping[str, str], names: Sequence[str]) -> None:
    """Raise RoundHookError when a feature variable is set but its module is not listed in ``names``."""
    missing = {}
    for var, module in feature_variables(env):
        if module not in names:
            missing.setdefault(module, []).append(var)
    if missing:
        details = "; ".join(f"{', '.join(variables)} -> '{module}'" for module, variables in sorted(missing.items()))
        raise RoundHookError(
            f"Feature variables are set but their round-hook module is not listed in "
            f"{HOOKS_ENV}={env.get(HOOKS_ENV, '')!r}: {details}. The feature would silently not run "
            f"(for DP: updates would leave the site without noise). Add the module to {HOOKS_ENV} "
            f"or remove the variables from the launcher line."
        )


def _import_hook_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:  # ImportError, SyntaxError or anything the module raises at import time
        raise RoundHookError(
            f"{HOOKS_ENV} lists '{name}', but custom/{name}.py cannot be imported ({exc!r}). "
            f"A listed hook is never skipped: fix the module or remove it from {HOOKS_ENV}."
        ) from exc


def load_hooks(env: Optional[Mapping[str, str]] = None, logger=None) -> List[RoundHook]:
    """Build the hooks listed in ODELIA_ROUND_HOOKS, in list order (see module docstring for the rules).

    ``env`` defaults to os.environ. main.py calls this before prepare_training(), so a misconfigured job
    fails in seconds instead of after the dataset has been hashed.
    """
    env = os.environ if env is None else env
    log = logger or _LOG

    mode = env.get("TRAINING_MODE", "")
    if mode != SWARM_MODE:
        ignored = ([HOOKS_ENV] if env.get(HOOKS_ENV, "").strip() else []) + [var for var, _ in feature_variables(env)]
        if ignored:
            log.warning(f"Round hooks only run in swarm mode; ignoring {', '.join(ignored)} in TRAINING_MODE={mode!r}")
        return []

    names = parse_hook_names(env.get(HOOKS_ENV, ""))
    check_feature_variables(env, names)

    hooks: List[RoundHook] = []
    for name in names:
        module = _import_hook_module(name)
        factory = getattr(module, "create_hook", None)
        if not callable(factory):
            raise RoundHookError(f"{HOOKS_ENV} lists '{name}', but custom/{name}.py defines no create_hook(env, logger)")
        hook = factory(env, log)
        if hook is None:
            log.info(f"Round hook module '{name}' is not active for this site/config")
            continue
        if not isinstance(hook, RoundHook):
            raise RoundHookError(
                f"{name}.create_hook() returned {type(hook).__name__}; expected a round_hooks.RoundHook or None"
            )
        hooks.append(hook)

    if hooks:
        log.info(f"Round hooks active, in call order: {', '.join(hook.name for hook in hooks)}")
    return hooks


def call_hooks(hooks: Iterable[RoundHook], point: str, ctx: RoundContext) -> None:
    """Call ``point`` on every hook in list order. Exceptions propagate unchanged and fail the round."""
    if point not in HOOK_POINTS:
        raise ValueError(f"Unknown round-hook point {point!r}; expected one of {HOOK_POINTS}")
    for hook in hooks:
        try:
            getattr(hook, point)(ctx)
        except Exception:
            # Name the hook in the log; the original exception is re-raised untouched.
            (ctx.logger or _LOG).error(f"Round hook '{hook.name}' failed in {point} (round {ctx.current_round})")
            raise


def insert_before_fl_callback(trainer, callbacks: Sequence, fl_callback_cls=None) -> None:
    """Insert ``callbacks`` into ``trainer.callbacks`` immediately before NVFlare's FLCallback.

    Lightning calls a hook such as on_train_end on the callbacks in list order, so these run before
    FLCallback.on_train_end sends the local model. Lightning (2.4) reorders trainer.callbacks only in
    _CallbackConnector._reorder_callbacks: at Trainer construction (before flare.patch() appends
    FLCallback) and again at fit/validate if the LightningModule defines configure_callbacks. That
    reorder is a stable partition (tuner callbacks first, Checkpoint callbacks last), so it keeps
    ordinary callbacks before FLCallback, but it would move a Checkpoint or tuner callback to the other
    side. Those are therefore refused here rather than silently ending up after the send.
    """
    from pytorch_lightning.callbacks import BatchSizeFinder, Callback, Checkpoint, LearningRateFinder

    if fl_callback_cls is None:
        from nvflare.app_opt.lightning.api import FLCallback as fl_callback_cls

    callbacks = list(callbacks)
    for cb in callbacks:
        if not isinstance(cb, Callback):
            raise TypeError(f"Round-hook callback {cb!r} is not a pytorch_lightning Callback")
        if isinstance(cb, fl_callback_cls):
            raise ValueError("A round hook must not add another FLCallback: every FLCallback sends the model")
        if isinstance(cb, (Checkpoint, BatchSizeFinder, LearningRateFinder)):
            raise ValueError(
                f"Round-hook callback {type(cb).__name__} would be moved by Lightning's callback reordering "
                f"and could end up after FLCallback; use a plain Callback"
            )

    current = list(trainer.callbacks)
    positions = [i for i, cb in enumerate(current) if isinstance(cb, fl_callback_cls)]
    if len(positions) != 1:
        raise RuntimeError(
            f"Expected exactly one FLCallback in trainer.callbacks (call flare.patch(trainer) first), "
            f"found {len(positions)}"
        )
    i = positions[0]
    trainer.callbacks = current[:i] + callbacks + current[i:]


def start_hooks(hooks: Sequence[RoundHook], ctx: RoundContext, fl_callback_cls=None) -> list:
    """Run on_start of every hook, then insert their Lightning callbacks before FLCallback.

    Callbacks are collected after all on_start calls, so a hook that replaces ctx.trainer in on_start gets
    the callbacks in the trainer that is actually used. Returns the inserted callbacks in order.
    """
    call_hooks(hooks, "on_start", ctx)
    callbacks = []
    for hook in hooks:
        returned = hook.lightning_callbacks(ctx) or []
        if not isinstance(returned, (list, tuple)):
            raise TypeError(f"Round hook '{hook.name}'.lightning_callbacks() must return a list")
        callbacks.extend(returned)
    if callbacks:
        insert_before_fl_callback(ctx.trainer, callbacks, fl_callback_cls)
        (ctx.logger or _LOG).info(
            f"Round-hook callbacks inserted before FLCallback: {', '.join(type(cb).__name__ for cb in callbacks)}"
        )
    return callbacks
