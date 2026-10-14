"""Warm-continue / central checkpointing for swarm runs (issues #347, #160).

Problem: when a run aborts, the latest aggregated global lives only in the
rotating aggregator's per-run directory -- often on the very node that crashed,
and at a path that changes every run (it contains the job id). So progress is
lost, and resuming would require shipping a ~689 MB checkpoint to every client
(bundling it in the job overruns the deploy timeout -- see #347).

``WarmStartablePTFileModelPersistor`` solves both with a uniform, run-independent
mirror on each client:

* Every time a new global is saved, it is also copied to a fixed local path
  (``latest_global_path``, default ``/scratch/mediswarm_latest_global.pt``). Best
  globals are mirrored too, but continuation is based on the latest completed
  global so jobs without a selected best metric can still continue. That path
  persists on the host scratch mount across runs, so the *next* run finds the
  prior run's global locally -- no bundling, no per-site staging.
* On start, if ``source_ckpt_file_full_name`` points at an absolute path that does
  not exist yet (the first run of a chain), it is disabled so the run initializes
  fresh instead of ``system_panic``; on later runs the mirror exists and the run
  warm-starts from it.
* ``warm_start_mode`` lets the admin make that choice explicit per submitted job:
  ``auto`` keeps the current best-effort behavior, ``fresh`` ignores the mirror
  without deleting it, and ``require`` aborts if the mirror is missing.

Wiring it into every job's client persistor with
``source_ckpt_file_full_name = latest_global_path`` makes a chain of short runs
auto-resume: each block continues from the previous block's global. Collect the
mirrors from all sites with ``scripts/collect_swarm_globals.sh`` (#160).

Warm starts from one pinned checkpoint (release 1.9.0, D2.5/D3.2)
-----------------------------------------------------------------
``require`` needs the file at every site, because every client loads its persistor
at configure. In a swarm only the starting client's persistor seeds round 0 (the
others are overwritten by the round-0 global), so an experiment warm-started from
one confidential checkpoint pins that file by digest instead:

* ``source_sha256``: sha256 of ``source_ckpt_file_full_name`` (``sha256sum``). With
  it, ``auto`` (what scripts/admin/build_consortium_job.py writes for
  ``--warm-start source:<path>:<sha256>``) and ``source`` both mean: the starting
  client must hold the file and it must match, or configure fails (and with it the
  run, before round 0); every other client does not read the file at all and
  initialises from its own model (logged at INFO), which the round-0 global then
  replaces. ``auto`` therefore never falls back to a fresh start at the starting
  client. The starting client is read from the CCWF configure task
  (``cwf.start_client``); a client that cannot tell is treated as the starting
  client. With ``require`` the file must exist and match at every client.
* ``warm_start_mode = "source"``: the explicit form; it refuses a job without
  ``source_sha256``. ``fresh`` with ``source_sha256`` is refused as contradictory.
* A pinned source is never the mirror (``latest_global_path``): the run would
  overwrite the file it is pinned to.
* ``model_name`` / ``data_config``: the model (MODEL_NAME) and data format (CONFIG)
  this job trains, set next to the launcher pins. The persistor runs in the client
  job process, which does not see the launcher's KEY=VALUE pins; without these args
  it falls back to the container's environment, which is the kit default (MST,
  unilateral), not necessarily this job's. Both are written to the mirror's
  provenance sidecar and checked against it.
* ``latest_global_path`` per experiment (the job builder's default is
  ``/scratch/mediswarm_latest_global_<experiment>.pt``), so experiments do not
  overwrite each other's mirror or the shared default.

The persistor verifies the file; every site's training subprocess then verifies what
it actually received, the round-0 global, against ``ODELIA_WARM_START_SHA256``
(reproducibility.py, state_digest.py).
"""

import hashlib
import json
import os
import shutil

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import WorkspaceConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

DEFAULT_LATEST_GLOBAL_PATH = "/scratch/mediswarm_latest_global.pt"
WARM_START_MODE_AUTO = "auto"
WARM_START_MODE_FRESH = "fresh"
WARM_START_MODE_REQUIRE = "require"
WARM_START_MODE_SOURCE = "source"
WARM_START_MODES = {
    WARM_START_MODE_AUTO,
    WARM_START_MODE_FRESH,
    WARM_START_MODE_REQUIRE,
    WARM_START_MODE_SOURCE,
}
WARM_START_REQUIRED_MISSING = "WARM_START_REQUIRED_MISSING"
WARM_START_MODEL_MISMATCH = "WARM_START_MODEL_MISMATCH"
WARM_START_SOURCE_MISSING = "WARM_START_SOURCE_MISSING"
WARM_START_SOURCE_DIGEST_MISMATCH = "WARM_START_SOURCE_DIGEST_MISMATCH"
WARM_START_DATA_CONFIG_MISMATCH = "WARM_START_DATA_CONFIG_MISMATCH"

# env_config's default CONFIG; also what a provenance sidecar written before data_config existed is
# taken to mean (every mirror of that time was written by a unilateral run).
DEFAULT_DATA_CONFIG = "unilateral"

# Where the CCWF configure task names the starting client (nvflare.app_common.ccwf.common.Constant and
# FLContextKey.TASK_DATA). Spelled out so this module imports without the CCWF package.
_CCWF_CONFIG_KEY = "cwf.config"
_CCWF_START_CLIENT_KEY = "cwf.start_client"
_TASK_DATA_KEY = getattr(FLContextKey, "TASK_DATA", "__task_data__")

ROLE_STARTING = "starting"
ROLE_OTHER = "other"
ROLE_UNKNOWN = "unknown"


class WarmStartRefused(RuntimeError):
    """Raised (after system_panic) when the source of a pinned warm start is missing or not the pinned file.

    Raising makes the CCWF configure task fail, so the server stops the run at once ("starting client did
    not configure") instead of waiting for the start task of a client that has already aborted.
    """



def sidecar_path(ckpt_path: str) -> str:
    """Provenance file that travels with a mirrored checkpoint."""
    return f"{ckpt_path}.provenance.json"


def current_model_name() -> str:
    """The model this process is training, as the job's launcher set it."""
    return os.environ.get("MODEL_NAME", "") or "unknown"


def read_provenance(ckpt_path: str):
    """Return the recorded provenance dict, or None if there is none."""
    try:
        with open(sidecar_path(ckpt_path)) as fh:
            return json.load(fh)
    except Exception:
        return None


def write_provenance(ckpt_path: str, model_name: str, job_id: str = "", digest: str = "", data_config=None, extra=None):
    """Write the sidecar. ``data_config`` (CONFIG of the run) and ``extra`` (lineage) are optional."""
    record = {"model_name": model_name, "job_id": job_id, "sha256": digest}
    if data_config is not None:
        record["data_config"] = data_config
    for key, value in (extra or {}).items():
        record.setdefault(key, value)
    try:
        with open(sidecar_path(ckpt_path), "w") as fh:
            json.dump(record, fh)
    except Exception:
        # provenance is a guard, not a feature -- never fail a run over it
        pass


def file_digest(path: str) -> str:
    """sha256 of a checkpoint, or "" if it cannot be read.

    Read in blocks: these files are ~700 MB and must not be loaded into memory
    just to be identified.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return ""


def normalize_warm_start_mode(warm_start_mode: str) -> str:
    mode = (warm_start_mode or WARM_START_MODE_AUTO).strip().lower()
    if mode not in WARM_START_MODES:
        expected = ", ".join(sorted(WARM_START_MODES))
        raise ValueError(f"Invalid warm_start_mode '{warm_start_mode}'. Expected one of: {expected}")
    return mode


def resolve_source_checkpoint(
    source_ckpt_file_full_name: str,
    warm_start_mode: str,
    exists=os.path.exists,
    validate_required: bool = True,
):
    """Return the source checkpoint path the base persistor should use.

    ``auto`` preserves the historical behavior: missing absolute paths are
    disabled so the run starts fresh, while existing paths are passed through.
    ``fresh`` always disables warm-start. ``require`` passes through only an
    existing source and raises a sentinel-bearing error otherwise, unless
    ``validate_required`` is disabled so the runtime persistor can report the
    failure through NVFlare's normal ``system_panic`` path after client config.
    ``source`` passes the path through unchecked: the file has to exist only at
    the starting client, which is known only at configure time.
    """

    mode = normalize_warm_start_mode(warm_start_mode)
    source_ckpt = source_ckpt_file_full_name

    if mode == WARM_START_MODE_FRESH:
        return None

    if mode == WARM_START_MODE_SOURCE:
        return source_ckpt or None

    if not source_ckpt:
        if mode == WARM_START_MODE_REQUIRE:
            if not validate_required:
                return None
            raise FileNotFoundError(f"{WARM_START_REQUIRED_MISSING}: no warm-start checkpoint path configured")
        return None

    if mode == WARM_START_MODE_REQUIRE:
        if not exists(source_ckpt):
            if not validate_required:
                return source_ckpt
            raise FileNotFoundError(
                f"{WARM_START_REQUIRED_MISSING}: required warm-start checkpoint missing: {source_ckpt}"
            )
        return source_ckpt

    if os.path.isabs(source_ckpt) and not exists(source_ckpt):
        return None

    return source_ckpt



_LAST_CHECKPOINT_READ_ERROR = None


def checkpoint_keys(ckpt_path: str):
    """Parameter names stored in a checkpoint, and the model label it carries, if any.

    Returns ``(keys, label)``. ``keys`` is a set of state-dict names or ``None``
    when the file cannot be read as a torch checkpoint; ``label`` is the model
    name recorded in the checkpoint's own ``train_conf`` (E2's prescription:
    read it out of the file) or ``None``. The label is advisory only -- it is a
    class name, not a MODEL_NAME, so it must not be compared as one.

    Reads with ``mmap=True`` where torch supports it so a multi-hundred-MB
    mirror is not pulled into memory just to list its keys.
    """
    global _LAST_CHECKPOINT_READ_ERROR
    _LAST_CHECKPOINT_READ_ERROR = None
    try:
        import torch
    except Exception as e:
        _LAST_CHECKPOINT_READ_ERROR = f"torch unavailable: {e}"
        return None, None
    data = None
    last = None
    for kwargs in ({"mmap": True}, {}):
        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False, **kwargs)
            break
        except TypeError as e:
            last = e          # torch without the mmap keyword: try again without it
            continue
        except Exception as e:
            last = e
            break
    if data is None:
        _LAST_CHECKPOINT_READ_ERROR = f"{type(last).__name__}: {str(last)[:160]}" if last else "unreadable"
        return None, None

    label = None
    if isinstance(data, dict):
        tc = data.get("train_conf")
        if isinstance(tc, str):
            try:
                import ast
                tc = ast.literal_eval(tc)
            except Exception:
                tc = None
        if isinstance(tc, dict) and isinstance(tc.get("train"), dict):
            label = tc["train"].get("model")
        sd = data.get("model", data.get("state_dict", data))
    else:
        sd = getattr(data, "state_dict", lambda: None)()
    try:
        return set(sd.keys()), label
    except Exception:
        return None, label


def normalize_sha256(value, what: str = "source_sha256"):
    """Lower-cased 64-hex SHA-256, None for None/empty; ValueError for anything else (a typo must not pass)."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{what} must be a 64-character hex SHA-256 (sha256sum output), got {value!r}")
    return text


class WarmStartablePTFileModelPersistor(PTFileModelPersistor):
    def __init__(
        self,
        latest_global_path: str = DEFAULT_LATEST_GLOBAL_PATH,
        warm_start_mode: str = WARM_START_MODE_AUTO,
        source_sha256: str = None,
        model_name: str = None,
        data_config: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latest_global_path = latest_global_path
        self.warm_start_mode = normalize_warm_start_mode(warm_start_mode)
        self.source_sha256 = normalize_sha256(source_sha256)
        self.model_name = (model_name or "").strip() or None
        self.data_config = (data_config or "").strip() or None
        self._source_role = None  # ROLE_* once load_model decided, for a pinned source

        sc = self.source_ckpt_file_full_name
        resolved_sc = resolve_source_checkpoint(sc, self.warm_start_mode, validate_required=False)
        if self.source_sha256:
            self._validate_pinned_source(sc)
            # Pinned by digest: never turn an absent file into a fresh start here. load_model decides, once the
            # configure task says which client starts, whether this client needs the file at all.
            resolved_sc = sc
        elif self.warm_start_mode == WARM_START_MODE_SOURCE:
            raise ValueError("warm_start_mode=source needs source_sha256: the starting client must verify the file")
        self.source_ckpt_file_full_name = resolved_sc

        if self._pins_source():
            self.logger.info(
                f"WarmStart: warm start pinned to {sc} (sha256 {self.source_sha256[:12]}..., mode="
                f"{self.warm_start_mode}); the starting client seeds round 0 from it, other clients do not need it"
            )
        elif self.warm_start_mode == WARM_START_MODE_FRESH:
            self.logger.info("WarmStart: warm_start_mode=fresh; initializing fresh and ignoring any local checkpoint")
        elif resolved_sc and (self.warm_start_mode != WARM_START_MODE_REQUIRE or os.path.exists(resolved_sc)):
            self.logger.info(f"WarmStart: will warm-start from checkpoint {resolved_sc} (mode={self.warm_start_mode})")
        elif resolved_sc and self.warm_start_mode == WARM_START_MODE_REQUIRE:
            self.logger.info(f"WarmStart: require mode will validate checkpoint at runtime: {resolved_sc}")
        elif sc:
            self.logger.info(f"WarmStart: source checkpoint {sc} not present yet; initializing fresh")

    def _validate_pinned_source(self, source):
        """Configuration errors of a digest-pinned warm start; raised at construction, at every client."""
        if self.warm_start_mode == WARM_START_MODE_FRESH:
            raise ValueError("source_sha256 pins a warm-start file, but warm_start_mode=fresh never reads one")
        if not source:
            raise ValueError("source_sha256 needs source_ckpt_file_full_name (the pinned file at the starting client)")
        if os.path.abspath(str(self.latest_global_path)) == os.path.abspath(str(source)):
            raise ValueError(
                f"latest_global_path and source_ckpt_file_full_name are both {source}: the run would overwrite "
                f"the checkpoint it is pinned to. Give the experiment its own mirror path."
            )

    def _pins_source(self) -> bool:
        """Only the starting client needs the file: 'source' mode, or 'auto' pinned by source_sha256."""
        return bool(getattr(self, "source_sha256", None)) and getattr(self, "warm_start_mode", None) in (
            WARM_START_MODE_SOURCE,
            WARM_START_MODE_AUTO,
        )

    def _model_name(self) -> str:
        """The model this job trains: the persistor's model_name arg, else the container env (kit default)."""
        return getattr(self, "model_name", None) or current_model_name()

    def _data_config(self):
        """(CONFIG this job reads, 'arg' | 'env'); the env fallback mirrors env_config's default."""
        explicit = getattr(self, "data_config", None)
        if explicit:
            return explicit, "arg"
        return (os.environ.get("CONFIG", "") or "").strip() or DEFAULT_DATA_CONFIG, "env"

    def _refuse(self, reason: str, fl_ctx: FLContext):
        self.system_panic(reason=reason, fl_ctx=fl_ctx)
        raise WarmStartRefused(reason)

    def _ccwf_role(self, fl_ctx: FLContext):
        """(ROLE_*, starting client name) from the CCWF configure task this load runs in, if any."""
        try:
            task_data = fl_ctx.get_prop(_TASK_DATA_KEY)
        except Exception:
            task_data = None
        config = task_data.get(_CCWF_CONFIG_KEY) if isinstance(task_data, dict) else None
        start_client = config.get(_CCWF_START_CLIENT_KEY) if isinstance(config, dict) else None
        get_identity = getattr(fl_ctx, "get_identity_name", None)
        me = get_identity() if callable(get_identity) else None
        if not start_client or not me:
            return ROLE_UNKNOWN, start_client
        return (ROLE_STARTING if start_client == me else ROLE_OTHER), start_client

    def _verify_source_digest(self, ckpt_path: str, fl_ctx: FLContext):
        """Refuse unless the file's sha256 is source_sha256. Streams the file (~700 MB): once, at configure."""
        actual = file_digest(ckpt_path)
        if not actual:
            self._refuse(f"{WARM_START_SOURCE_DIGEST_MISMATCH}: cannot read {ckpt_path} to verify its sha256", fl_ctx)
        if actual != self.source_sha256:
            self._refuse(
                f"{WARM_START_SOURCE_DIGEST_MISMATCH}: {ckpt_path} has sha256 {actual}, but this job is pinned to "
                f"{self.source_sha256}. Refusing to warm-start from another file (replaced, truncated or a "
                f"different checkpoint). Put the pinned file there or rebuild the job with this file's digest.",
                fl_ctx,
            )
        self.log_info(fl_ctx, f"WarmStart: sha256 of {ckpt_path} verified ({actual[:12]}...)")

    def _load_pinned_source(self, fl_ctx: FLContext):
        if self._source_role is None:
            self._source_role, start_client = self._ccwf_role(fl_ctx)
        else:
            start_client = None
        ckpt_path = self._runtime_source_checkpoint_path(fl_ctx)
        if self._source_role == ROLE_OTHER:
            self.log_info(
                fl_ctx,
                f"WarmStart: this client is not the starting client ({start_client} is). Round 0 starts "
                f"from the starting client's global, which replaces this model, so {ckpt_path} is not read here.",
            )
            self.source_ckpt_file_full_name = None  # the base persistor then initialises from the model
            return super().load_model(fl_ctx)

        if self._source_role == ROLE_UNKNOWN:
            self.log_info(fl_ctx, "WarmStart: starting client unknown here; requiring the pinned source file")
        if not ckpt_path or not os.path.exists(ckpt_path):
            self._refuse(
                f"{WARM_START_SOURCE_MISSING}: this is the starting client of a pinned warm start, but {ckpt_path} "
                f"does not exist here. Round 0 would start from an untrained model. Copy the pinned checkpoint "
                f"there (sha256 {self.source_sha256}) or rebuild the job with the client that holds it as "
                f"starting_client.",
                fl_ctx,
            )
        self._verify_source_digest(ckpt_path, fl_ctx)
        if not self._check_provenance(fl_ctx):
            raise WarmStartRefused(f"warm-start source {ckpt_path} refused by the provenance/structure check")
        self.log_info(fl_ctx, f"WarmStart: starting client seeds round 0 from the pinned {ckpt_path}")
        return super().load_model(fl_ctx)

    def _runtime_source_checkpoint_path(self, fl_ctx: FLContext):
        source_ckpt = self.source_ckpt_file_full_name
        if not source_ckpt:
            return None
        if os.path.isabs(source_ckpt):
            return source_ckpt

        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        if not app_root:
            return source_ckpt
        return os.path.join(app_root, WorkspaceConstants.CUSTOM_FOLDER_NAME, source_ckpt)

    def load_model(self, fl_ctx: FLContext):
        if self._pins_source():
            return self._load_pinned_source(fl_ctx)

        if self.warm_start_mode == WARM_START_MODE_REQUIRE:
            ckpt_path = self._runtime_source_checkpoint_path(fl_ctx)
            if not ckpt_path:
                self.system_panic(
                    reason=f"{WARM_START_REQUIRED_MISSING}: no warm-start checkpoint path configured",
                    fl_ctx=fl_ctx,
                )
                return None
            if not os.path.exists(ckpt_path):
                self.system_panic(
                    reason=f"{WARM_START_REQUIRED_MISSING}: required warm-start checkpoint missing: {ckpt_path}",
                    fl_ctx=fl_ctx,
                )
                return None

        if getattr(self, "source_sha256", None):
            # require mode (fresh is refused at construction, auto/source are pinned sources): the file exists
            # here, checked just above, and must be the pinned one.
            self._verify_source_digest(self._runtime_source_checkpoint_path(fl_ctx), fl_ctx)

        if not self._check_provenance(fl_ctx):
            return None                     # refused: system_panic already raised
        return super().load_model(fl_ctx)

    def _check_provenance(self, fl_ctx: FLContext):
        """Refuse a checkpoint that a different model produced.

        All six ODELIA jobs read and write one mirror path, so the file found
        there may have been written by any of five architectures. Loading a
        state dict from the wrong one is E2 in docs/EVALUATION_PITFALLS.md --
        "wrong architecture loads silently" -- and shape compatibility is not a
        safe proxy for correctness.

        Two checks, in order. The provenance sidecar names the model that wrote
        the mirror and is refused on a mismatch. Then, whether or not a sidecar
        exists, the checkpoint's parameter names are intersected with this run's
        model: an empty intersection is another architecture, and is refused.

        The structural check is what closes the gap an unlabelled mirror leaves.
        Every mirror written before provenance existed has no sidecar, so those
        used to be accepted with only a warning -- and on 2026-09-11 a 722 MB
        1DivideAndConquer mirror left on a test host by a failed run was loaded
        into an MST client that way. It trained, and then failed as aggregator
        with "none of the 187 incoming parameters matched the local model's 450":
        E2, one round late. The names were disjoint from the first byte; nothing
        needed to be trained to know it.
        """
        ckpt_path = self._runtime_source_checkpoint_path(fl_ctx)
        if not ckpt_path or not os.path.exists(ckpt_path):
            return True

        recorded = read_provenance(ckpt_path)
        expected = self._model_name()

        if recorded is None:
            self.log_warning(
                fl_ctx,
                f"WarmStart: {ckpt_path} carries no provenance; cannot confirm it was "
                f"produced by '{expected}'. Written before provenance was recorded, or by "
                "another job. Checking its parameter names against this model instead.",
            )
        else:
            found = recorded.get("model_name", "unknown")
            if expected != "unknown" and found != "unknown" and found != expected:
                self.system_panic(
                    reason=(
                        f"{WARM_START_MODEL_MISMATCH}: {ckpt_path} was written by model "
                        f"'{found}' but this run trains '{expected}'. Refusing to warm-start "
                        "from another architecture's weights. Use warm_start_mode=fresh, or "
                        "point source_ckpt_file_full_name at that model's own mirror."
                    ),
                    fl_ctx=fl_ctx,
                )
                return False
            self.log_info(fl_ctx, f"WarmStart: provenance OK -- checkpoint was written by '{found}'")
            if not self._check_data_config(recorded, ckpt_path, fl_ctx):
                return False

        return self._check_structure(ckpt_path, fl_ctx, expected)

    def _check_data_config(self, recorded: dict, ckpt_path: str, fl_ctx: FLContext):
        """Refuse a mirror written by a run on another data format (CONFIG), when this job states its own.

        Crops of different formats (unilateral vs unilateral_BD) have identical tensor shapes, so a global
        trained on one loads and aggregates on the other without any error; the BreastDivider benchmark
        measured 0.06-0.18 AUROC lost that way. A sidecar from before data_config was recorded counts as
        'unilateral' (every mirror of that time). Without the persistor's data_config arg, the container's
        CONFIG is only a guess at what the training subprocess reads, so a mismatch is logged, not refused.
        """
        expected, source = self._data_config()
        found = recorded.get("data_config")
        legacy = found is None
        if legacy:
            found = DEFAULT_DATA_CONFIG
        if found in (expected, "unknown"):
            return True
        assumed = " (assumed: the sidecar predates data_config)" if legacy else ""
        message = (
            f"{WARM_START_DATA_CONFIG_MISMATCH}: {ckpt_path} was written by a run on data config '{found}'"
            f"{assumed}, but this run reads '{expected}'. Mixed crop formats aggregate silently. Use "
            f"warm_start_mode=fresh, or a mirror path of this data config."
        )
        if source != "arg":
            self.log_warning(fl_ctx, f"{message} Not refused: this job does not set the persistor's data_config.")
            return True
        self.system_panic(reason=message, fl_ctx=fl_ctx)
        return False

    def _check_structure(self, ckpt_path: str, fl_ctx: FLContext, expected: str):
        """Refuse a checkpoint whose parameter names share nothing with the model.

        Independent of the sidecar on purpose: a sidecar can be absent (every
        pre-provenance mirror) or wrong (copied, hand-edited). Parameter names
        cannot be. A partial overlap is normal -- the Lightning wrapper adds a
        few buffers the persistor's model lacks -- so only a total miss refuses.
        """
        model = getattr(self, "model", None)
        try:
            import torch
            if not isinstance(model, torch.nn.Module):
                # Not built yet, or not a torch model: nothing to compare against.
                # Say so, so a skipped check is visible in the log rather than
                # indistinguishable from a passed one.
                self.log_info(fl_ctx, "WarmStart: no torch model to compare against; structural check skipped.")
                return True
            model_keys = set(model.state_dict().keys())
        except Exception as e:
            self.log_warning(fl_ctx, f"WarmStart: could not read this model's parameter names ({e}); structural check skipped.")
            return True

        ckpt_keys, ckpt_label = checkpoint_keys(ckpt_path)
        if ckpt_keys is None:
            self.log_warning(
                fl_ctx,
                f"WarmStart: could not read parameter names from {ckpt_path} "
                f"({_LAST_CHECKPOINT_READ_ERROR or 'not a torch checkpoint'}); structural check skipped.",
            )
            return True

        shared = model_keys & ckpt_keys
        if not shared:
            labelled = f" and names its own model '{ckpt_label}'" if ckpt_label else ""
            self.system_panic(
                reason=(
                    f"{WARM_START_MODEL_MISMATCH}: {ckpt_path} shares no parameter names "
                    f"with this run's model '{expected}'. The checkpoint holds "
                    f"{len(ckpt_keys)} parameters (e.g. {sorted(ckpt_keys)[:2]}){labelled}; "
                    f"the model has {len(model_keys)} (e.g. {sorted(model_keys)[:2]}). "
                    "Refusing to warm-start from another architecture's weights. Use "
                    "warm_start_mode=fresh, or point source_ckpt_file_full_name at this "
                    "model's own mirror."
                ),
                fl_ctx=fl_ctx,
            )
            return False

        self.log_info(
            fl_ctx,
            f"WarmStart: structure OK -- {len(shared)}/{len(model_keys)} parameter names "
            f"shared with {ckpt_path}",
        )
        return True

    def _mirror_checkpoint(self, src: str, fl_ctx: FLContext, label: str):
        try:
            if src and os.path.exists(src):
                dst = os.path.abspath(self.latest_global_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                self.log_info(fl_ctx, f"WarmStart: mirrored {label} global -> {dst}")
                self._record_provenance(dst, fl_ctx)
        except Exception as e:
            # mirroring is best-effort; never let it break the run
            self.log_warning(fl_ctx, f"WarmStart: failed to mirror {label} global to {self.latest_global_path}: {e}")

    def _record_provenance(self, dst: str, fl_ctx: FLContext):
        """Label a mirrored checkpoint with the model that produced it.

        Deliberately separate from the copy, and separately guarded: provenance
        is a safety net, and a failure to write it must never make a successful
        mirror look like a failed one.
        """
        try:
            model_name = self._model_name()
            data_config, data_config_source = self._data_config()
            get_job_id = getattr(fl_ctx, "get_job_id", None)
            job_id = (get_job_id() or "") if callable(get_job_id) else ""
            lineage = {
                "model_name_source": "arg" if getattr(self, "model_name", None) else "env",
                "data_config_source": data_config_source,
                "warm_start_mode": getattr(self, "warm_start_mode", None),
            }
            pinned = getattr(self, "source_sha256", None)
            if pinned:
                # What this run started from: the file the starting client verified (every client of a source
                # warm start descends from it, whether or not it held the file itself).
                lineage["warm_start_source_sha256"] = pinned
            write_provenance(dst, model_name, job_id, file_digest(dst), data_config=data_config, extra=lineage)
            self.log_info(fl_ctx, f"WarmStart: recorded provenance model={model_name} data_config={data_config}")
        except Exception as e:
            self.log_warning(fl_ctx, f"WarmStart: could not record provenance for {dst}: {e}")

    def save_model(self, ml, fl_ctx: FLContext):
        super().save_model(ml, fl_ctx)
        self._mirror_checkpoint(getattr(self, "_ckpt_save_path", None), fl_ctx, "latest")

    def handle_event(self, event: str, fl_ctx: FLContext):
        # let the base persistor do its normal save/init first
        super().handle_event(event, fl_ctx)

        if event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            self._mirror_checkpoint(getattr(self, "_best_ckpt_save_path", None), fl_ctx, "best")
