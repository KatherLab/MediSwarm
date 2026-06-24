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

Wiring it into every job's client persistor with
``source_ckpt_file_full_name = latest_global_path`` makes a chain of short runs
auto-resume: each block continues from the previous block's global. Collect the
mirrors from all sites with ``scripts/collect_swarm_globals.sh`` (#160).
"""

import os
import shutil

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

DEFAULT_LATEST_GLOBAL_PATH = "/scratch/mediswarm_latest_global.pt"


class WarmStartablePTFileModelPersistor(PTFileModelPersistor):
    def __init__(self, latest_global_path: str = DEFAULT_LATEST_GLOBAL_PATH, **kwargs):
        super().__init__(**kwargs)
        self.latest_global_path = latest_global_path
        # Graceful warm-start: if the configured source checkpoint is an absolute
        # path that doesn't exist yet (first run of a chain), disable it so we
        # init fresh instead of system_panic. On later runs the prior run's
        # mirror is present and we warm-start from it.
        sc = self.source_ckpt_file_full_name
        if sc and os.path.isabs(sc):
            if os.path.exists(sc):
                self.logger.info(f"WarmStart: will warm-start from existing checkpoint {sc}")
            else:
                self.logger.info(f"WarmStart: source checkpoint {sc} not present yet; initializing fresh")
                self.source_ckpt_file_full_name = None

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
