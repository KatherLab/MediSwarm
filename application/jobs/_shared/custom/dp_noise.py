"""DP noise round hook: clip each site's per-round update and add calibrated Gaussian noise before it
leaves the site (T3.3, #593). Client-level differential privacy; docs/DIFFERENTIAL_PRIVACY.md has the
guarantee, its limits and how to run it.

Switched on per job from the launcher line in config_fed_client.conf (round_hooks.py), for example

    script = "ODELIA_ROUND_HOOKS=dp_noise ODELIA_DP_SIGMA=2.4 ODELIA_DP_CLIP_NORM=0.5 ODELIA_DP_TARGET_EPSILON=10
              ODELIA_RETURN_PER_CASE=0 KEY_METRIC=val/AUC_ROC python3 custom/{app_script} {app_config}"

Modes (exactly one; anything ambiguous stops the job before training)
---------------------------------------------------------------------
noise          ODELIA_DP_SIGMA (noise multiplier, > 0) with ODELIA_DP_CLIP_NORM (C, > 0); optional
               ODELIA_DP_DELTA (default 1e-5), ODELIA_DP_TARGET_EPSILON, ODELIA_DP_EQUAL_WEIGHTS=1.
               Refuses to start while ODELIA_RETURN_PER_CASE=1: per-case predictions of the local model
               would leave the site outside the guarantee, so DP jobs set ODELIA_RETURN_PER_CASE=0.
measure        ODELIA_DP_MEASURE_ONLY=1: sends the update unchanged (NO privacy guarantee) and streams its
               L2 norm every round, to calibrate C on public data. ODELIA_DP_CLIP_NORM may be given to see
               how often a candidate C would clip. SIGMA, DELTA and TARGET_EPSILON are refused here.
equal weights  ODELIA_DP_EQUAL_WEIGHTS=1 alone: no noise, only the unweighted aggregation the DP runs use
               (a matched non-DP baseline).

Where the noise enters
----------------------
NVFlare's FLCallback sends ``pl_module.state_dict()`` in on_train_end. The hook's Lightning callback sits
immediately before FLCallback (round_hooks.insert_before_fl_callback), so in on_train_end of every
training round it rewrites the model in place, and FLCallback then sends the rewritten model:

    U  = W - G                       W: local model after training; G: the global received this round
                                     (ctx.input_model.params); over the trainable floating-point
                                     parameters that G contains
    U' = U * min(1, C / ||U||_2)     one L2 norm over all those tensors together
    W' = G + U' + N(0, (sigma*C)^2)  independent noise per coordinate

With ``params_transfer_type = "DIFF"`` (every ODELIA job template) the client API transmits
W' - G = U' + noise, and every aggregate and global model built from it is post-processing of that
noised release. The arithmetic runs on CPU in float32 (float64 for float64 parameters); the training
precision "16-mixed" is autocast only and keeps the parameters in float32.

Buffers and everything else
---------------------------
Every other entry the state dict carries that G also has (buffers of any dtype, frozen parameters) is
copied back from G (bit-exact for integer buffers), so no local change to it leaves the site. Integer
buffers are never noised or cast. For MST this changes nothing: all its parameters train, and its only
float buffer (slice_fusion.rotary_pos_emb.inv_freq) is a constant. For a model with BatchNorm the running
statistics never leave the site, so the global keeps the round-0 statistics for the whole DP run.
Entries that G does not contain (the ``_class_weight`` buffer, computed from the site's label counts,
because the persistor's model is built without class weights) are not sent at all under DIFF transfer;
under a FULL transfer they would be, so the hook refuses that combination.

Accounting and budget
---------------------
After each release the epsilon of this site's releases so far (privacy_accounting.compute_epsilon,
add/remove-site adjacency, sampling rate 1) is logged and streamed with the other metrics:

    dp/epsilon       epsilon after this release (noise mode)
    dp/update_norm   L2 norm of U before clipping (also in measure mode)
    dp/clipped       1 if the update was scaled down to C, else 0 (measure mode: would have been)
    dp/noise_std     sigma * C, the per-coordinate noise standard deviation (noise mode)
    dp/released_norm L2 norm of what leaves the site, U' + noise (noise mode)

dp/update_norm and dp/clipped are computed before the noise. Like the validation metrics they go to the
coordinator's metric stream and the site's own logs, not to the other sites, and they are outside the
guarantee (docs/DIFFERENTIAL_PRIVACY.md, threat model).

Releases are counted conservatively: the release of round r is counted as release
max(previous count + 1, r + 1). The count is also saved after every release in ``dp_noise_state.json``
in the working directory of the training subprocess, which is the job's app directory on the client
(the SubprocessLauncher starts the script there), so it is job-scoped. A training subprocess restarted
within the job continues that count, so a round repeated after its update was already sent counts as a
new release. Saving is best effort (a warning; the round-index bound still holds); a count file that
exists but cannot be read, or that was written with another sigma, C or delta, stops the job, because
the number of releases already made would be unknown. With ODELIA_DP_TARGET_EPSILON the hook refuses
(raises PrivacyBudgetExceeded) before a release that would exceed the target: at the start, when the
job's configured training rounds cannot fit (fail fast instead of after hours of training), before each
round's training, and again immediately before the noised update would be written. Nothing is released
by a refused round. The evaluation-only final round (ODELIA_FINAL_EVAL_ROUND=1) sends the received
global back unchanged: no noise, no release counted.

Noise source: a torch CPU generator per training subprocess, seeded from os.urandom. It is never seeded
from ODELIA_SEED or the global RNG (other code seeds those); tests pass their own seeded generator.
"""

import json
import logging
import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from pytorch_lightning.callbacks import Callback

from privacy_accounting import PrivacyAccountant, PrivacyBudgetExceeded, compute_epsilon, max_rounds_for_budget
from round_hooks import RoundContext, RoundHook

ENV_SIGMA = "ODELIA_DP_SIGMA"
ENV_CLIP_NORM = "ODELIA_DP_CLIP_NORM"
ENV_DELTA = "ODELIA_DP_DELTA"
ENV_TARGET_EPSILON = "ODELIA_DP_TARGET_EPSILON"
ENV_MEASURE_ONLY = "ODELIA_DP_MEASURE_ONLY"
ENV_EQUAL_WEIGHTS = "ODELIA_DP_EQUAL_WEIGHTS"
DP_VARIABLES = (ENV_SIGMA, ENV_CLIP_NORM, ENV_DELTA, ENV_TARGET_EPSILON, ENV_MEASURE_ONLY, ENV_EQUAL_WEIGHTS)
DP_PREFIX = "ODELIA_DP_"
# The pre-1.9.0 accountant read this name; nothing ever set or enforced it. Refused so nobody relies on it.
LEGACY_TARGET_VARIABLE = "PRIVACY_EPSILON_TARGET"
ENV_RETURN_PER_CASE = "ODELIA_RETURN_PER_CASE"
ENV_FINAL_EVAL_ROUND = "ODELIA_FINAL_EVAL_ROUND"

DEFAULT_DELTA = 1e-5
# Release count of this job, in the training subprocess's working directory (the job's app dir).
STATE_FILE_NAME = "dp_noise_state.json"
MODE_NOISE = "noise"
MODE_MEASURE = "measure"
MODE_EQUAL_WEIGHTS = "equal_weights"

# NVFlare names (nvflare.app_opt.lightning.api / nvflare.app_common.abstract.fl_model); imported lazily
# with these fallbacks so the module also imports where NVFlare is absent (unit tests of the math).
FL_META_ATTRIBUTE = "__fl_meta__"
NUM_STEPS_KEY = "NUM_STEPS_CURRENT_ROUND"
TRANSFER_DIFF = "DIFF"

_LOG = logging.getLogger(__name__)


class DPConfigError(ValueError):
    """The ODELIA_DP_* variables do not describe exactly one valid mode."""


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DPConfig:
    mode: str
    sigma: Optional[float] = None
    clip_norm: Optional[float] = None
    delta: float = DEFAULT_DELTA
    target_epsilon: Optional[float] = None
    equal_weights: bool = False

    @property
    def noise_std(self) -> float:
        """Per-coordinate standard deviation of the added noise (0 outside noise mode)."""
        return self.sigma * self.clip_norm if self.mode == MODE_NOISE else 0.0


def _positive_float(env: Mapping[str, str], name: str) -> Optional[float]:
    if name not in env:
        return None
    raw = env[name].strip()
    if not raw:
        raise DPConfigError(f"{name} is set but empty (a template value lost on its way into the launcher line?)")
    try:
        value = float(raw)
    except ValueError:
        raise DPConfigError(f"{name}={env[name]!r} is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise DPConfigError(f"{name}={env[name]!r} must be a finite number > 0")
    return value


def _flag(env: Mapping[str, str], name: str) -> bool:
    if name not in env:
        return False
    raw = env[name].strip()
    if raw not in ("0", "1"):
        raise DPConfigError(f"{name}={env[name]!r}: expected 0 or 1")
    return raw == "1"


def parse_config(env: Mapping[str, str]) -> DPConfig:
    """Read the ODELIA_DP_* variables into one DPConfig; raise DPConfigError on anything ambiguous.

    Strict on purpose: a typo (ODELIA_DP_SIGAM), an empty value or a combination whose intent is unclear
    (noise and measure-only together) must stop the job, because the alternative is a run that leaves
    the sites without the noise the protocol promised.
    """
    unknown = sorted(name for name in env if name.startswith(DP_PREFIX) and name not in DP_VARIABLES)
    if unknown:
        raise DPConfigError(f"Unknown DP variable(s) {', '.join(unknown)}; known: {', '.join(DP_VARIABLES)}")

    sigma = _positive_float(env, ENV_SIGMA)
    clip_norm = _positive_float(env, ENV_CLIP_NORM)
    target = _positive_float(env, ENV_TARGET_EPSILON)
    delta = _positive_float(env, ENV_DELTA)
    if delta is not None and not delta < 1:
        raise DPConfigError(f"{ENV_DELTA}={env[ENV_DELTA]!r} must be < 1")
    measure = _flag(env, ENV_MEASURE_ONLY)
    equal_weights = _flag(env, ENV_EQUAL_WEIGHTS)

    if sigma is not None:
        if measure:
            raise DPConfigError(
                f"{ENV_SIGMA} and {ENV_MEASURE_ONLY}=1 are both set: noise or a measurement run without noise? "
                f"Set exactly one."
            )
        if clip_norm is None:
            raise DPConfigError(f"{ENV_SIGMA} needs {ENV_CLIP_NORM}: the noise std is sigma * C")
        return DPConfig(MODE_NOISE, sigma, clip_norm, DEFAULT_DELTA if delta is None else delta, target, equal_weights)

    noise_only = [name for name, value in ((ENV_DELTA, delta), (ENV_TARGET_EPSILON, target)) if value is not None]
    if measure:
        if noise_only:
            raise DPConfigError(f"{', '.join(noise_only)} only apply with {ENV_SIGMA}; {ENV_MEASURE_ONLY}=1 adds no noise")
        return DPConfig(MODE_MEASURE, clip_norm=clip_norm, equal_weights=equal_weights)
    if clip_norm is not None or noise_only:
        given = ([ENV_CLIP_NORM] if clip_norm is not None else []) + noise_only
        raise DPConfigError(
            f"{', '.join(given)} set without {ENV_SIGMA}: no noise would be added. Set {ENV_SIGMA} for DP noise "
            f"or {ENV_MEASURE_ONLY}=1 for a calibration run."
        )
    if equal_weights:
        return DPConfig(MODE_EQUAL_WEIGHTS, equal_weights=True)
    raise DPConfigError(
        f"dp_noise is listed in ODELIA_ROUND_HOOKS but no mode is configured: set {ENV_SIGMA} and {ENV_CLIP_NORM} "
        f"(noise), {ENV_MEASURE_ONLY}=1 (calibration) or {ENV_EQUAL_WEIGHTS}=1 (unweighted aggregation only)"
    )


# --------------------------------------------------------------------------------------------------
# The math (pure functions on tensors; the hook below applies them to the model)
# --------------------------------------------------------------------------------------------------


def new_generator() -> torch.Generator:
    """CPU generator seeded from os.urandom: unpredictable, independent of ODELIA_SEED and the global RNG."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1))
    return generator


def l2_norm(tensors: Iterable[torch.Tensor]) -> float:
    """One L2 norm over all given tensors, accumulated in float64."""
    total = 0.0
    for tensor in tensors:
        total += float(torch.sum(torch.square(tensor.detach().to(torch.float64))))
    return math.sqrt(total)


def clip_and_noise(
    update: Mapping[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, torch.Tensor], float, float, bool]:
    """Clip ``update`` (all tensors together) to L2 norm ``clip_norm`` and add N(0, (sigma*C)^2) per coordinate.

    Returns (released tensors, norm before clipping, norm of the released tensors, whether it clipped).
    The input tensors are not modified. ``generator`` defaults to a fresh os.urandom-seeded one; only
    tests pass a seeded generator. The tensors must be on CPU (the generator is a CPU generator).
    """
    if not (clip_norm > 0 and math.isfinite(clip_norm)):
        raise ValueError(f"clip_norm must be a finite number > 0, got {clip_norm!r}")
    if not (noise_multiplier >= 0 and math.isfinite(noise_multiplier)):
        raise ValueError(f"noise_multiplier must be a finite number >= 0, got {noise_multiplier!r}")
    std = noise_multiplier * clip_norm
    if std > 0 and generator is None:
        generator = new_generator()

    norm = l2_norm(update.values())
    clipped = norm > clip_norm
    scale = clip_norm / norm if clipped else 1.0
    released: Dict[str, torch.Tensor] = {}
    squares = 0.0
    for name, tensor in update.items():
        value = tensor * scale  # a new tensor: the caller's update stays as it was
        if std > 0:
            value.add_(torch.randn(tensor.shape, generator=generator, dtype=value.dtype), alpha=std)
        released[name] = value
        squares += float(torch.sum(torch.square(value.to(torch.float64))))
    return released, norm, math.sqrt(squares), clipped


@dataclass
class StatePartition:
    """How the hook treats each entry of the model's state dict, given the received global's keys."""

    trainable: List[Tuple[str, torch.Tensor]] = field(default_factory=list)  # clipped + noised
    restored: List[Tuple[str, torch.Tensor]] = field(default_factory=list)  # copied back from the global
    local_only: List[str] = field(default_factory=list)  # not in the global: not sent under DIFF transfer
    trainable_not_in_global: List[str] = field(default_factory=list)

    @property
    def coordinates(self) -> int:
        return sum(tensor.numel() for _, tensor in self.trainable)


def partition_state(module: torch.nn.Module, global_params: Mapping[str, object]) -> StatePartition:
    """Split the state dict FLCallback will send into noised, restored and local-only entries.

    Trainable = an nn.Parameter with requires_grad and a floating-point dtype that the global contains.
    A tensor reachable under two names (tied weights) is handled once, under its first name.
    """
    partition = StatePartition()
    seen = set()
    for name, tensor in module.state_dict(keep_vars=True).items():
        trainable = isinstance(tensor, torch.nn.Parameter) and tensor.requires_grad and tensor.is_floating_point()
        if name not in global_params:
            partition.local_only.append(name)
            if trainable:
                partition.trainable_not_in_global.append(name)
            continue
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        (partition.trainable if trainable else partition.restored).append((name, tensor))
    return partition


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """float32 for half-precision parameters, otherwise the parameter's own dtype."""
    return torch.promote_types(dtype, torch.float32)


def _cpu_tensor(value, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = torch.as_tensor(value)  # the received global may hold numpy arrays (numpy exchange format)
    return tensor.to(device="cpu", dtype=dtype) if dtype is not None else tensor.to(device="cpu")


def compute_update(trainable: List[Tuple[str, torch.Tensor]], global_params: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    """U = W - G per trainable tensor, on CPU, in the compute dtype; W and G are not modified."""
    update: Dict[str, torch.Tensor] = {}
    for name, param in trainable:
        dtype = _compute_dtype(param.dtype)
        local = param.detach().to(device="cpu", dtype=dtype)
        received = _cpu_tensor(global_params[name], dtype)
        if received.shape != local.shape:
            raise RuntimeError(
                f"DP: '{name}' has shape {tuple(received.shape)} in the received global but {tuple(local.shape)} "
                f"in the model; the update is undefined"
            )
        update[name] = local - received
    return update


def write_release(partition: StatePartition, global_params: Mapping[str, object], released: Mapping[str, torch.Tensor]) -> None:
    """In place: trainable tensors become G + released, every other shared entry is copied back from G."""
    with torch.no_grad():
        for name, param in partition.trainable:
            value = _cpu_tensor(global_params[name], released[name].dtype) + released[name]
            param.copy_(value)  # copy_ casts to the parameter's dtype and moves to its device
        for name, tensor in partition.restored:
            tensor.copy_(_cpu_tensor(global_params[name]))


def set_equal_weight_meta(pl_module) -> None:
    """Make FLCallback report NUM_STEPS_CURRENT_ROUND=1, so the weighted aggregator averages unweighted.

    FLCallback.on_train_end reads the dict ``pl_module.__fl_meta__`` and only fills NUM_STEPS_CURRENT_ROUND
    (with trainer.estimated_stepping_batches) when the key is missing. A dict that is already there (from
    another hook) is kept and only this key is set. NUM_STEPS is otherwise a dataset-size proxy that every
    update carries in clear.
    """
    meta = getattr(pl_module, FL_META_ATTRIBUTE, None)
    meta = dict(meta) if isinstance(meta, dict) else {}
    meta[_num_steps_key()] = 1
    setattr(pl_module, FL_META_ATTRIBUTE, meta)


def _num_steps_key() -> str:
    try:
        from nvflare.app_common.abstract.fl_model import MetaKey

        return MetaKey.NUM_STEPS_CURRENT_ROUND
    except ImportError:
        return NUM_STEPS_KEY


def _read_transfer_type() -> Optional[str]:
    """params_transfer_type of the client API config ("DIFF"/"FULL"), or None when it cannot be read."""
    try:
        from nvflare.client.api import get_config
        from nvflare.client.config import ConfigKey

        value = (get_config() or {}).get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.TRANSFER_TYPE)
    except Exception:
        return None
    value = getattr(value, "value", value)
    return str(value).upper() if value else None


def _is_fl_callback(callback) -> bool:
    # By class name along the MRO: also matches subclasses, and needs no NVFlare import here.
    return any(klass.__name__ == "FLCallback" for klass in type(callback).__mro__)


def fl_callback_will_send(trainer) -> bool:
    """Whether FLCallback will send a model at the end of this fit (it only does for a training task).

    Reads FLCallback's own flag, set when it received the task in on_train_start. Without an FLCallback
    the answer is True: treating an unknown case as a release adds noise and counts it, the safe side.
    """
    for callback in getattr(trainer, "callbacks", None) or []:
        if _is_fl_callback(callback):
            return bool(getattr(callback, "_is_training", True))
    return True


def check_placement(trainer, callback) -> None:
    """Raise unless ``callback`` is in trainer.callbacks and runs before the (single) FLCallback."""
    callbacks = list(getattr(trainer, "callbacks", None) or [])
    own = [i for i, cb in enumerate(callbacks) if cb is callback]
    if not own:
        raise RuntimeError(
            "DP: the dp_noise callback is not in trainer.callbacks, so the update would leave the site without "
            "noise. A hook that replaces the trainer must re-insert the hook callbacks "
            "(round_hooks.insert_before_fl_callback)."
        )
    fl = [i for i, cb in enumerate(callbacks) if _is_fl_callback(cb)]
    if len(fl) != 1:
        raise RuntimeError(f"DP: expected exactly one FLCallback in trainer.callbacks, found {len(fl)}")
    if own[0] > fl[0]:
        raise RuntimeError("DP: the dp_noise callback runs after FLCallback, so the model would be sent before the noise")


# --------------------------------------------------------------------------------------------------
# Hook and callback
# --------------------------------------------------------------------------------------------------


@dataclass
class ReleaseStats:
    """What the hook did to one round's update."""

    round: Optional[int]
    mode: str
    update_norm: Optional[float] = None  # L2 norm of U before clipping
    clipped: Optional[bool] = None
    released_norm: Optional[float] = None  # L2 norm of what leaves the site (noise mode: U' + noise)
    noise_std: float = 0.0
    coordinates: int = 0
    releases: int = 0
    epsilon: Optional[float] = None

    def metrics(self) -> Dict[str, float]:
        if self.mode == MODE_NOISE:
            return {
                "dp/epsilon": self.epsilon,
                "dp/update_norm": self.update_norm,
                "dp/clipped": float(self.clipped),
                "dp/noise_std": self.noise_std,
                "dp/released_norm": self.released_norm,
            }
        if self.mode == MODE_MEASURE:
            metrics = {"dp/update_norm": self.update_norm}
            if self.clipped is not None:
                metrics["dp/clipped"] = float(self.clipped)
            return metrics
        return {}


class DPNoiseCallback(Callback):
    """Runs the hook's release step in on_train_end, i.e. immediately before FLCallback sends the model."""

    def __init__(self, hook: "DPNoiseHook", ctx: RoundContext):
        super().__init__()
        self.hook = hook
        self.ctx = ctx

    def on_train_start(self, trainer, pl_module) -> None:
        check_placement(trainer, self)

    def on_train_end(self, trainer, pl_module) -> None:
        self.hook.release(trainer, pl_module, self.ctx)


class DPNoiseHook(RoundHook):
    """Round hook of the DP modes (module docstring). Exceptions are never swallowed: a DP step that
    cannot be done fails the round, because the alternative is an update leaving without its noise."""

    name = "dp_noise"

    def __init__(self, config: DPConfig, logger=None, final_eval_round: bool = False,
                 generator: Optional[torch.Generator] = None, state_path: Optional[str] = None):
        self.config = config
        self.logger = logger or _LOG
        self.final_eval_round = final_eval_round
        # Tests pass a seeded generator; production never does (create_hook), see module docstring.
        self.generator = generator if generator is not None else new_generator()
        self.accountant = (
            PrivacyAccountant(config.sigma, config.delta, config.target_epsilon) if config.mode == MODE_NOISE else None
        )
        # Where the release count is saved (noise mode); None = in memory only (tests that need no file).
        self.state_path = state_path if self.accountant is not None else None
        self.callback: Optional[DPNoiseCallback] = None
        self.transfer_type: Optional[str] = None
        self.history: List[ReleaseStats] = []
        self._plan_checked = False
        self._fit_round = object()  # round whose on_train_end the callback handled last (sentinel: none yet)
        self._warned_local_only = False
        if self.state_path:
            self._resume_count()

    # ---------------------------------------------------------------- round_hooks call points
    def on_start(self, ctx: RoundContext) -> None:
        self.transfer_type = _read_transfer_type()
        self.logger.info(self._banner())
        if self.config.mode == MODE_NOISE and self.transfer_type != TRANSFER_DIFF:
            self.logger.warning(
                f"DP: params_transfer_type is {self.transfer_type or 'unknown'}, not DIFF: a release is refused if the "
                f"model has entries the received global lacks (they would be sent without noise)"
            )
        if ctx.model is not None:
            self._describe_model(ctx.model)

    def lightning_callbacks(self, ctx: RoundContext) -> list:
        self.callback = DPNoiseCallback(self, ctx)
        return [self.callback]

    def after_receive(self, ctx: RoundContext) -> None:
        if ctx.is_eval_only_round or self.accountant is None:
            return
        if not self._plan_checked:
            self._check_plan(ctx)
            self._plan_checked = True
        # Refuse before this round's validation and training, not only at the release: no compute is
        # spent on a round whose update could not be sent anyway.
        self._refuse_if_over_budget(ctx, self._planned_releases(ctx))

    def before_fit(self, ctx: RoundContext) -> None:
        if ctx.is_eval_only_round:
            return
        check_placement(ctx.trainer, self.callback)

    def after_round(self, ctx: RoundContext) -> None:
        if ctx.is_eval_only_round:
            return
        if self._fit_round != ctx.current_round:
            raise RuntimeError(
                f"DP: the dp_noise callback did not run at the end of round {ctx.current_round}'s training. Either "
                f"trainer.fit() ended without training (then FLCallback sent nothing either), or the callback is not "
                f"attached to the trainer that trained (then the update left the site without DP processing)."
            )

    def on_end(self, ctx: RoundContext) -> None:
        if self.accountant is not None:
            acc = self.accountant
            self.logger.info(
                f"DP summary: {acc.steps} release(s), epsilon {acc.epsilon():.4f} at delta {acc.delta:g}"
                f"{self._target_text()}; client-level (epsilon, delta)-DP, add/remove-site adjacency, for the model "
                f"updates this site sent; NOT record-level"
            )
        elif self.config.mode == MODE_MEASURE:
            norms = [stats.update_norm for stats in self.history if stats.update_norm is not None]
            if norms:
                self.logger.info(
                    f"DP measure-only summary: update L2 norm per round {[round(n, 6) for n in norms]}, "
                    f"median {statistics.median(norms):.6g} (candidate ODELIA_DP_CLIP_NORM; calibrate on public data)"
                )

    # ---------------------------------------------------------------- the release (callback)
    def release(self, trainer, pl_module, ctx: RoundContext) -> Optional[ReleaseStats]:
        """Apply the configured mode to the model FLCallback is about to send. Returns None if it sends nothing."""
        self._fit_round = ctx.current_round
        if ctx.is_eval_only_round:
            self.logger.info(
                f"DP: round {ctx.current_round} is the evaluation-only round: the received global goes back "
                f"unchanged, no noise, no release counted"
            )
            return None
        if not fl_callback_will_send(trainer):
            self.logger.info(f"DP: round {ctx.current_round} is not a training task; FLCallback sends no model")
            return None
        if (getattr(trainer, "world_size", 1) or 1) > 1:
            raise RuntimeError("DP noise supports single-process training only (every ODELIA site trains on one GPU)")

        mode = self.config.mode
        stats = ReleaseStats(round=ctx.current_round, mode=mode, noise_std=self.config.noise_std)
        if mode in (MODE_NOISE, MODE_MEASURE):
            global_params = _received_params(ctx)
            partition = partition_state(pl_module, global_params)
            if not partition.trainable:
                raise RuntimeError(
                    "DP: no trainable floating-point parameter of the model is in the received global; nothing to "
                    "clip, so nothing could be protected"
                )
            self._check_entries(partition)
            stats.coordinates = partition.coordinates
            if mode == MODE_NOISE:
                planned = self._planned_releases(ctx)
                self._refuse_if_over_budget(ctx, planned)  # before the model is touched
                update = compute_update(partition.trainable, global_params)
                released, stats.update_norm, stats.released_norm, stats.clipped = clip_and_noise(
                    update, self.config.clip_norm, self.config.sigma, self.generator
                )
                del update
                write_release(partition, global_params, released)
                self.accountant.step(planned - self.accountant.steps)
                # Saved now, before FLCallback sends: a release that leaves the site is always counted.
                self._save_count(ctx)
                stats.releases = self.accountant.steps
                stats.epsilon = self.accountant.epsilon()
            else:
                stats.update_norm = l2_norm(compute_update(partition.trainable, global_params).values())
                stats.released_norm = stats.update_norm
                if self.config.clip_norm is not None:
                    stats.clipped = stats.update_norm > self.config.clip_norm
        if self.config.equal_weights:
            set_equal_weight_meta(pl_module)

        self.history.append(stats)
        self._report(trainer, stats)
        return stats

    # ---------------------------------------------------------------- helpers
    def _planned_releases(self, ctx: RoundContext) -> int:
        """Release count after this round's release: never below round index + 1 (restart-safe)."""
        by_round = ctx.current_round + 1 if ctx.current_round is not None else 0
        return max(self.accountant.steps + 1, by_round)

    def _count_parameters(self) -> Dict[str, float]:
        return {"sigma": self.config.sigma, "clip_norm": self.config.clip_norm, "delta": self.config.delta}

    def _resume_count(self) -> None:
        """Continue the release count an earlier training subprocess of this job saved (_save_count).

        No file is the normal start of a job. A file that cannot be read, or that was written for other
        parameters, is an error: the releases already made would be unknown, and guessing low would
        under-state epsilon.
        """
        try:
            with open(self.state_path) as f:
                saved = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"DP: the release-count file {self.state_path} exists but cannot be read ({exc!r}); the number of "
                f"updates this site already released in this job is unknown, so no further release can be accounted"
            ) from exc
        releases = saved.get("releases") if isinstance(saved, dict) else None
        if not isinstance(releases, int) or isinstance(releases, bool) or releases < 0:
            raise RuntimeError(f"DP: the release-count file {self.state_path} holds no valid release count: {saved!r}")
        mismatched = {k: (saved.get(k), v) for k, v in self._count_parameters().items() if saved.get(k) != v}
        if mismatched:
            raise RuntimeError(
                f"DP: the release-count file {self.state_path} was written with other parameters "
                f"(saved, now: {mismatched}); one job must keep sigma, C and delta"
            )
        self.accountant.steps = releases
        self.logger.warning(
            f"DP: continuing this job's release count from {self.state_path}: {releases} release(s) already made "
            f"(last in round {saved.get('last_round')}), epsilon {self.accountant.epsilon():.4f} so far"
        )

    def _save_count(self, ctx: RoundContext) -> None:
        """Save the release count for a restarted training subprocess (best effort, see module docstring)."""
        if not self.state_path:
            return
        record = {
            "releases": self.accountant.steps,
            "last_round": ctx.current_round,
            "epsilon": self.accountant.epsilon(),
            **self._count_parameters(),
        }
        tmp_path = f"{self.state_path}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(record, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.state_path)  # atomic: a reader sees the old or the new count
        except OSError as exc:
            self.logger.warning(
                f"DP: could not save the release count to {self.state_path} ({exc!r}); a restarted training "
                f"subprocess would count from the round index alone"
            )

    def _refuse_if_over_budget(self, ctx: RoundContext, planned: int) -> None:
        acc = self.accountant
        if acc.epsilon_target is None:
            return
        eps_after = compute_epsilon(acc.noise_multiplier, planned, acc.delta)
        if eps_after > acc.epsilon_target:
            raise PrivacyBudgetExceeded(
                f"privacy budget exhausted: the update of round {ctx.current_round} would be release {planned} and "
                f"bring epsilon to {eps_after:.4f} > {ENV_TARGET_EPSILON}={acc.epsilon_target:g} (sigma "
                f"{acc.noise_multiplier:g}, delta {acc.delta:g}); spent so far: {acc.epsilon():.4f} after "
                f"{acc.steps} release(s). Nothing is sent for this round."
            )

    def _check_plan(self, ctx: RoundContext) -> None:
        """Fail fast when the job's configured training rounds cannot fit the target."""
        acc = self.accountant
        if acc.epsilon_target is None or ctx.total_rounds is None:
            return
        training_rounds = ctx.total_rounds - (1 if self.final_eval_round else 0)
        allowed = max_rounds_for_budget(acc.noise_multiplier, acc.delta, acc.epsilon_target)
        if training_rounds > allowed:
            raise PrivacyBudgetExceeded(
                f"privacy budget too small for this job: {training_rounds} training round(s) would bring epsilon to "
                f"{compute_epsilon(acc.noise_multiplier, training_rounds, acc.delta):.4f} > {ENV_TARGET_EPSILON}="
                f"{acc.epsilon_target:g}; at sigma {acc.noise_multiplier:g} and delta {acc.delta:g} the target allows "
                f"{allowed}. Rebuild the job with at most {allowed} training rounds (refusing now, before training)."
            )
        self.logger.info(
            f"DP budget: {training_rounds} training round(s) end at epsilon "
            f"{compute_epsilon(acc.noise_multiplier, training_rounds, acc.delta):.4f}, within the target "
            f"{acc.epsilon_target:g} (which allows up to {allowed})"
        )

    def _check_entries(self, partition: StatePartition) -> None:
        diff = self.transfer_type == TRANSFER_DIFF
        if partition.local_only and self.config.mode == MODE_NOISE and not diff:
            raise RuntimeError(
                f"DP: state-dict entries that the received global does not have ({_short_list(partition.local_only)}) "
                f"would be sent without noise under params_transfer_type={self.transfer_type or 'unknown'}. DP jobs "
                f"need params_transfer_type = \"DIFF\" (as in every ODELIA job template): it sends only the entries "
                f"the global has."
            )
        if partition.local_only and not self._warned_local_only:
            self._warned_local_only = True
            fate = "not sent (DIFF transfer)" if diff else f"sent as they are ({self.transfer_type or 'unknown'} transfer)"
            self.logger.info(f"DP: entries not in the received global, {fate}: {_short_list(partition.local_only)}")
            if partition.trainable_not_in_global:
                self.logger.warning(
                    f"DP: trainable parameters missing from the received global are neither noised nor aggregated: "
                    f"{_short_list(partition.trainable_not_in_global)}"
                )

    def _report(self, trainer, stats: ReleaseStats) -> None:
        self.logger.info(self._describe(stats))
        metrics = stats.metrics()
        if not metrics:
            return
        step = stats.round if stats.round is not None else len(self.history) - 1
        # Best effort: the line above carries the same numbers, and the release itself has already happened.
        for metric_logger in list(getattr(trainer, "loggers", None) or []):
            try:
                metric_logger.log_metrics(metrics, step=step)
            except Exception as exc:
                self.logger.warning(f"DP metrics not written to {type(metric_logger).__name__}: {exc!r}")

    def _describe(self, stats: ReleaseStats) -> str:
        weights = "; NUM_STEPS_CURRENT_ROUND=1 (equal weights)" if self.config.equal_weights else ""
        if stats.mode == MODE_NOISE:
            return (
                f"DP release, round {stats.round}: update L2 norm {stats.update_norm:.6g} "
                f"({'clipped to' if stats.clipped else 'within'} C={self.config.clip_norm:g}), Gaussian noise std "
                f"{stats.noise_std:.6g} on {stats.coordinates} coordinates, released norm {stats.released_norm:.6g}; "
                f"epsilon {stats.epsilon:.4f} after {stats.releases} release(s) at delta {self.config.delta:g}"
                f"{self._target_text()}{weights}"
            )
        if stats.mode == MODE_MEASURE:
            clip = ""
            if stats.clipped is not None:
                clip = f"; C={self.config.clip_norm:g} would {'clip' if stats.clipped else 'not clip'} it"
            return (
                f"DP measure-only, round {stats.round}: update L2 norm {stats.update_norm:.6g} over "
                f"{stats.coordinates} coordinates, sent WITHOUT clipping or noise{clip}{weights}"
            )
        return f"DP equal weights, round {stats.round}: NUM_STEPS_CURRENT_ROUND=1, no noise"

    def _target_text(self) -> str:
        target = self.config.target_epsilon
        return f", target {target:g}" if target is not None else ""

    def _banner(self) -> str:
        c = self.config
        weights = f", equal aggregation weights {'on' if c.equal_weights else 'off'}"
        if c.mode == MODE_NOISE:
            allowed = (
                f" (allows {max_rounds_for_budget(c.sigma, c.delta, c.target_epsilon)} training rounds)"
                if c.target_epsilon is not None else ""
            )
            return (
                f"DP ACTIVE: noise multiplier sigma={c.sigma:g}, clip norm C={c.clip_norm:g}, noise std per coordinate "
                f"{c.noise_std:g}, delta={c.delta:g}, target epsilon "
                f"{c.target_epsilon if c.target_epsilon is not None else 'none'}{allowed}{weights}; "
                f"client-level (epsilon, delta)-DP for the model updates this site sends (add/remove-site adjacency), "
                f"NOT record-level; params_transfer_type={self.transfer_type or 'unknown'}"
            )
        if c.mode == MODE_MEASURE:
            return (
                f"DP MEASURE-ONLY: updates are sent WITHOUT clipping or noise, this run has NO privacy guarantee; "
                f"dp/update_norm per round calibrates {ENV_CLIP_NORM}{weights}"
            )
        return "DP noise OFF: equal aggregation weights only (NUM_STEPS_CURRENT_ROUND=1)"

    def _describe_model(self, model) -> None:
        params = [p for p in model.parameters() if p.requires_grad and p.is_floating_point()]
        dtypes = sorted({str(p.dtype) for p in params})
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad and p.is_floating_point()}
        others = [name for name in model.state_dict() if name not in trainable_names]
        self.logger.info(
            f"DP: {len(params)} trainable parameter tensors ({sum(p.numel() for p in params)} coordinates, "
            f"dtypes {dtypes}) are clipped and noised; the other state-dict entries are sent at the received "
            f"global's value, or not at all if the global lacks them ({_short_list(others) or 'none'})"
        )
        if any(dtype != "torch.float32" for dtype in dtypes):
            self.logger.warning(f"DP: trainable parameters are not all float32 ({dtypes}); noise is added in float32")
        batch_norms = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats
        ]
        if batch_norms and self.config.mode == MODE_NOISE:
            self.logger.warning(
                f"DP: {len(batch_norms)} BatchNorm layer(s) ({_short_list(batch_norms)}): their running statistics "
                f"never leave the site in DP mode, so the global keeps the round-0 statistics"
            )


def _received_params(ctx: RoundContext) -> Mapping[str, object]:
    params = getattr(ctx.input_model, "params", None)
    if not params:
        raise RuntimeError(
            "DP: the received global (ctx.input_model.params) is not available, so the update cannot be formed"
        )
    return params


def _short_list(names: List[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f", ... ({len(names) - limit} more)" if len(names) > limit else "")


def create_hook(env: Mapping[str, str], logger=None) -> DPNoiseHook:
    """round_hooks entry point. Never returns None: a listed dp_noise without a valid mode is an error."""
    config = parse_config(env)
    if LEGACY_TARGET_VARIABLE in env:
        raise DPConfigError(f"{LEGACY_TARGET_VARIABLE} is not read; the budget variable is {ENV_TARGET_EPSILON}")
    if config.mode == MODE_NOISE and env.get(ENV_RETURN_PER_CASE, "") == "1":
        raise DPConfigError(
            f"{ENV_RETURN_PER_CASE}=1 with DP noise: per-case predictions of the local model would leave the site "
            f"outside the guarantee. DP jobs set {ENV_RETURN_PER_CASE}=0 in the launcher line."
        )
    final_eval_round = env.get(ENV_FINAL_EVAL_ROUND, "0").strip() == "1"
    # The SubprocessLauncher starts the training script in the job's app directory: job-scoped, and kept
    # when the training subprocess is restarted within the job.
    state_path = os.path.join(os.getcwd(), STATE_FILE_NAME) if config.mode == MODE_NOISE else None
    return DPNoiseHook(config, logger=logger, final_eval_round=final_eval_round, state_path=state_path)
