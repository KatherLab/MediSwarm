"""reproducibility: the pins of a swarm run that load_hooks applies by itself (release 1.9.0).

Two launcher variables are not optional experiments but statements about the run: it starts from exactly
these weights, and with exactly this random state. The job builder (scripts/admin/build_consortium_job.py,
``--warm-start-digest``, ``--seed``) sets them without listing a round-hook module, so
round_hooks.IMPLICIT_HOOKS adds this module whenever one of them is present, ahead of every listed hook.
A pinned seed or digest that silently does nothing is therefore impossible.

ODELIA_WARM_START_SHA256  after_receive, round 0: the state digest (state_digest.py, contract 3.4) of the
                          received global must equal this value at every site, or the job stops before any
                          site trains. The starting client's persistor verifies the FILE it loads
                          (source_sha256 in warm_continue.py); this check verifies what every site actually
                          trains from, so it also catches a starting client that seeded round 0 from another
                          file, or from none (a fresh model, e.g. warm_start_mode=auto with the file missing).
ODELIA_SEED               on_start: seed_training() below, which documents what becomes deterministic and what
                          does not.
"""

import logging
from typing import Mapping, Optional

import state_digest
from round_hooks import RoundHook

ENV_WARM_START_SHA256 = "ODELIA_WARM_START_SHA256"
ENV_TRAIN_SEED = "ODELIA_SEED"
MAX_SEED = 2**32 - 1  # numpy's limit; seed_everything would silently fall back to 0 above it
STATE_KEY = "reproducibility"

_LOG = logging.getLogger(__name__)


class ReproducibilityConfigError(ValueError):
    """ODELIA_SEED / ODELIA_WARM_START_SHA256 is malformed; raised at load_hooks(), before any data is touched."""


class WarmStartDigestMismatch(RuntimeError):
    """The round-0 global is not the warm start the job was built for."""


def _value(env: Mapping[str, str], name: str) -> Optional[str]:
    """Stripped value, None when absent. Present but empty is an error: a template value got lost."""
    if name not in env:
        return None
    value = str(env[name]).strip()
    if not value:
        raise ReproducibilityConfigError(
            f"{name} is set but empty; give it a value or remove it from the launcher line"
        )
    return value


def parse_seed(env: Mapping[str, str]) -> Optional[int]:
    raw = _value(env, ENV_TRAIN_SEED)
    if raw is None:
        return None
    try:
        seed = int(raw)
    except ValueError:
        raise ReproducibilityConfigError(f"{ENV_TRAIN_SEED} must be an integer, got {raw!r}") from None
    if not 0 <= seed <= MAX_SEED:
        raise ReproducibilityConfigError(f"{ENV_TRAIN_SEED} must be in [0, {MAX_SEED}], got {seed}")
    return seed


def parse_warm_start_digest(env: Mapping[str, str]) -> Optional[str]:
    try:
        return state_digest.normalise_hex_digest(_value(env, ENV_WARM_START_SHA256), what=ENV_WARM_START_SHA256)
    except ReproducibilityConfigError:
        raise
    except ValueError as exc:
        raise ReproducibilityConfigError(str(exc)) from None


def seed_training(ctx, seed: int, logger=None) -> None:
    """ODELIA_SEED: make the training subprocess's own randomness a function of ``seed``.

    Deterministic afterwards, for the same data, worker count, code and hardware: the python/numpy/torch RNGs
    of the training process (dropout, augmentation drawn in the main process when ODELIA_NUM_WORKERS=0), the
    order the RandomSampler visits the training cases (the DataModule's generator, which is otherwise always
    seed 0), and the augmentation in loader workers (torch seeds each worker from that generator: base seed +
    worker id, python and numpy included).

    Not covered: the round-0 global (built by the starting client's persistor in the client job process;
    fixed by a warm start, random for a fresh start); GPU kernels (cuDNN algorithm choice, atomic adds in
    backward, mixed precision), whose last-bit differences grow over training; the swarm aggregate, which
    mixes every site's update and arrives in a timing-dependent order; a changed ODELIA_NUM_WORKERS (it moves
    cases between worker RNG streams); the active-learning random arms (ODELIA_AL_SEED, independent of this
    seed by design, so arms and training replicates can be varied separately). Replicate runs with different
    seeds therefore measure training noise; they do not remove the GPU part of it.

    Runs in on_start, after prepare_training() built the model and the DataModule: model initialisation is
    not seeded here, which does not matter in a swarm (the round-0 global replaces every site's weights).
    """
    import pytorch_lightning as pl

    log = logger or _LOG
    pl.seed_everything(seed, workers=True)
    data_module = ctx.data_module
    if data_module is not None and hasattr(data_module, "seed"):
        data_module.seed = seed
        loader = getattr(data_module, "_train_loader", None)
        if loader is not None:
            # A loader built before now holds a generator seeded with the old value.
            release_loader(loader)
            data_module._train_loader = None
    ctx.state.setdefault(STATE_KEY, {})["train_seed"] = seed
    log.info(f"{ENV_TRAIN_SEED}={seed}: training RNGs, sampler order and loader-worker augmentation seeded")


def release_loader(loader) -> None:
    """Stop the worker processes a (persistent-worker) DataLoader may still hold."""
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass
    if loader is not None and hasattr(loader, "_iterator"):
        loader._iterator = None


def check_round0_global(params: Optional[Mapping], expected: str, site_name: str = "") -> str:
    """State digest of the round-0 global ``params``; raises WarmStartDigestMismatch unless it is ``expected``."""
    if not params:
        raise WarmStartDigestMismatch(f"{ENV_WARM_START_SHA256}: the round-0 task carries no model parameters")
    actual = state_digest.state_digest(params)
    if actual != expected:
        raise WarmStartDigestMismatch(
            f"{ENV_WARM_START_SHA256} mismatch at {site_name or 'this site'}: the round-0 global has state digest "
            f"{actual}, the job expects {expected}. The starting client did not seed round 0 from the intended "
            f"warm-start checkpoint (another file, or none: a fresh model). Stopping before any training; check the "
            f"starting client's persistor log (source_ckpt_file_full_name, source_sha256)."
        )
    return actual


class ReproducibilityHook(RoundHook):
    name = "reproducibility"

    def __init__(self, train_seed: Optional[int] = None, warm_start_sha256: Optional[str] = None, logger=None):
        self.train_seed = train_seed
        self.warm_start_sha256 = warm_start_sha256
        self.logger = logger or _LOG
        self.first_round: Optional[int] = None

    def _log(self, ctx):
        return ctx.logger or self.logger

    def on_start(self, ctx) -> None:
        if self.train_seed is not None:
            seed_training(ctx, self.train_seed, self._log(ctx))

    def after_receive(self, ctx) -> None:
        first = self.first_round is None
        if first:
            self.first_round = ctx.current_round
        if self.warm_start_sha256 is None:
            return
        if ctx.current_round != 0:
            if first:
                self._log(ctx).warning(
                    f"{ENV_WARM_START_SHA256}: this process started in round {ctx.current_round}; the round-0 "
                    f"global is gone, so the warm-start digest cannot be checked here"
                )
            return
        params = getattr(ctx.input_model, "params", None)
        actual = check_round0_global(params, self.warm_start_sha256, ctx.site_name)
        ctx.state.setdefault(STATE_KEY, {})["round0_global_digest"] = actual
        self._log(ctx).info(
            f"{ENV_WARM_START_SHA256}: round-0 global verified ({len(params)} tensors, digest {actual[:12]})"
        )


def create_hook(env, logger):
    seed = parse_seed(env)
    digest = parse_warm_start_digest(env)
    if seed is None and digest is None:
        return None
    return ReproducibilityHook(train_seed=seed, warm_start_sha256=digest, logger=logger)
