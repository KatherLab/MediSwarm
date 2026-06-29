"""Warm-continue / central checkpointing for swarm runs (issues #347, #160).

Problem: when a run aborts, the latest aggregated global lives only in the
rotating aggregator's per-run directory -- often on the very node that crashed,
and at a path that changes every run (it contains the job id). So progress is
lost, and resuming would require shipping a ~689 MB checkpoint to every client
(bundling it in the job overruns the deploy timeout -- see #347).

``WarmStartablePTFileModelPersistor`` solves both with a uniform, run-independent
mirror on each client:

* Every time a new best global is saved, it is also copied to a fixed local path
  (``latest_global_path``, default ``/scratch/mediswarm_latest_global.pt``). That
  path persists on the host scratch mount across runs, so the *next* run finds the
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
"""

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
WARM_START_MODES = {
    WARM_START_MODE_AUTO,
    WARM_START_MODE_FRESH,
    WARM_START_MODE_REQUIRE,
}
WARM_START_REQUIRED_MISSING = "WARM_START_REQUIRED_MISSING"


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
    """

    mode = normalize_warm_start_mode(warm_start_mode)
    source_ckpt = source_ckpt_file_full_name

    if mode == WARM_START_MODE_FRESH:
        return None

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


class WarmStartablePTFileModelPersistor(PTFileModelPersistor):
    def __init__(
        self,
        latest_global_path: str = DEFAULT_LATEST_GLOBAL_PATH,
        warm_start_mode: str = WARM_START_MODE_AUTO,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latest_global_path = latest_global_path
        self.warm_start_mode = normalize_warm_start_mode(warm_start_mode)

        sc = self.source_ckpt_file_full_name
        resolved_sc = resolve_source_checkpoint(sc, self.warm_start_mode, validate_required=False)
        self.source_ckpt_file_full_name = resolved_sc

        if self.warm_start_mode == WARM_START_MODE_FRESH:
            self.logger.info("WarmStart: warm_start_mode=fresh; initializing fresh and ignoring any local checkpoint")
        elif resolved_sc and (self.warm_start_mode != WARM_START_MODE_REQUIRE or os.path.exists(resolved_sc)):
            self.logger.info(f"WarmStart: will warm-start from checkpoint {resolved_sc} (mode={self.warm_start_mode})")
        elif resolved_sc and self.warm_start_mode == WARM_START_MODE_REQUIRE:
            self.logger.info(f"WarmStart: require mode will validate checkpoint at runtime: {resolved_sc}")
        elif sc:
            self.logger.info(f"WarmStart: source checkpoint {sc} not present yet; initializing fresh")

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

        return super().load_model(fl_ctx)

    def handle_event(self, event: str, fl_ctx: FLContext):
        # let the base persistor do its normal save/init first
        super().handle_event(event, fl_ctx)

        if event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            # mirror the just-saved best global to the uniform, run-independent
            # path so the next run can warm-start from it locally (no bundling).
            try:
                src = getattr(self, "_best_ckpt_save_path", None)
                if src and os.path.exists(src):
                    dst = os.path.abspath(self.latest_global_path)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    self.log_info(fl_ctx, f"WarmStart: mirrored best global -> {dst}")
            except Exception as e:
                # mirroring is best-effort; never let it break the run
                self.log_warning(fl_ctx, f"WarmStart: failed to mirror best global to {self.latest_global_path}: {e}")
