"""Return per-case predictions to the coordinator (#526, #527, #528).

Why this exists
---------------
Four deliverables converge on the same missing capability. D2.5 needs paired
per-case scores to compare a regionally fine-tuned model against the
pan-European one; D3.2 needs them to rank cases by uncertainty; D3.3's
regulatory testing node is defined by receiving per-site figures; and D3.4's
analysis needs case-level results to say anything about robustness.

The alternative was moving a 722 MB checkpoint out of each hospital, which is
what D2.5 is currently blocked on and which nobody here has the access to do.
Per-site *summary* metrics already travel (see ``per_site_metrics.py``); this
extends the same path to the rows behind them.

What actually leaves a site
---------------------------
One line per validation case:

    row_index,ground_truth,prob_class_0,prob_class_1,prob_class_2

**No case identifier, no patient identifier, no image.** The row index is a
position within that site's own validation split and carries no meaning
outside it. Pairing two models' predictions -- which is what D2.5 needs --
works from the index because the validation loader is not shuffled, so the
n-th row is the same case in both runs.

Sending a real uid would have been easier and is what the first draft of the
consortium slide promised; an index is strictly less information and does the
same job, so it is what this sends.

Off by default
--------------
This is patient-derived data, even pseudonymised, and returning it needs the
consortium's agreement rather than a maintainer's. The client half emits
nothing unless ``ODELIA_RETURN_PER_CASE=1`` is set in the site's environment,
which is a decision each site makes for itself.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Dict, List

from nvflare.apis.analytix import AnalyticsData
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver

PER_CASE_PREFIX = "per_case/"
DEFAULT_OUTPUT_DIR = "cross_site_val/per_case_predictions"
CSV_HEADER = "row_index,ground_truth,prob_class_0,prob_class_1,prob_class_2"


class PerCasePredictionCollector(AnalyticsReceiver):
    """Reassemble per-case prediction chunks and write one CSV per site.

    Chunks arrive as ``AnalyticsDataType.TEXT`` records whose ``path`` looks
    like ``per_case/<split>/<seq>.csv``. They are ordered by ``<seq>`` on
    write rather than on arrival: the transport gives no ordering guarantee,
    and a silently reordered file would break the index-based pairing that is
    the whole point of sending an index.
    """

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, max_chunks_per_site: int = 4096):
        super().__init__()
        self._output_dir = output_dir
        self._max_chunks = max_chunks_per_site
        self._chunks: Dict[str, Dict[int, str]] = {}
        self._dropped: Dict[str, int] = {}
        self._lock = Lock()

    def initialize(self, fl_ctx: FLContext):
        with self._lock:
            self._chunks = {}
            self._dropped = {}

    @staticmethod
    def _sequence_of(path: str) -> int | None:
        """Extract the chunk sequence from ``per_case/<split>/<seq>.csv``."""
        try:
            return int(os.path.splitext(os.path.basename(path))[0])
        except (TypeError, ValueError):
            return None

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        try:
            data = AnalyticsData.from_dxo(from_shareable(shareable))
        except Exception as exc:
            self.log_warning(fl_ctx, f"Unparseable record from {record_origin}: {exc}", fire_event=False)
            return
        if not data or not data.path or not str(data.path).startswith(PER_CASE_PREFIX):
            return

        seq = self._sequence_of(str(data.path))
        if seq is None:
            self.log_warning(
                fl_ctx, f"Ignoring per-case chunk from {record_origin} with unparseable path {data.path!r}",
                fire_event=False,
            )
            return
        if not isinstance(data.value, str):
            return

        with self._lock:
            site = self._chunks.setdefault(record_origin, {})
            # A site sending unbounded chunks must not exhaust coordinator memory.
            if len(site) >= self._max_chunks and seq not in site:
                self._dropped[record_origin] = self._dropped.get(record_origin, 0) + 1
                return
            site[seq] = data.value

    def finalize(self, fl_ctx: FLContext):
        with self._lock:
            collected = {s: dict(c) for s, c in self._chunks.items()}
            dropped = dict(self._dropped)

        if not collected:
            self.log_info(
                fl_ctx,
                "No per-case predictions received. This is the expected state unless sites "
                "set ODELIA_RETURN_PER_CASE=1; the capability is opt-in per site.",
                fire_event=False,
            )
            return

        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
        out_dir = os.path.join(run_dir, self._output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for site, chunks in collected.items():
            body = "".join(chunks[k] for k in sorted(chunks))
            rows = sum(1 for line in body.splitlines() if line.strip())
            path = os.path.join(out_dir, f"{site}.csv")
            try:
                with open(path, "w") as fh:
                    fh.write(CSV_HEADER + "\n")
                    fh.write(body if body.endswith("\n") or not body else body + "\n")
            except Exception as exc:
                self.log_error(fl_ctx, f"Could not write per-case predictions for {site}: {exc}",
                               fire_event=False)
                continue
            msg = f"Wrote {rows} per-case rows for {site} -> {path}"
            if dropped.get(site):
                # Say so loudly: a truncated file still looks like a valid result.
                msg += f" (WARNING: {dropped[site]} chunk(s) dropped at the {self._max_chunks} cap; file is INCOMPLETE)"
            self.log_info(fl_ctx, msg, fire_event=False)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # Same reason as per_site_metrics: publish before END_RUN, where other
        # components write their own output and component order would decide
        # whether this ran first.
        if event_type == EventType.ABOUT_TO_END_RUN:
            self._handle_end_run_event(fl_ctx)
            return
        super().handle_event(event_type, fl_ctx)


# --------------------------------------------------------------------------
# Client-side helper
# --------------------------------------------------------------------------

def per_case_rows(ground_truth: List[int], probabilities: List[List[float]]) -> List[str]:
    """Format prediction rows. Index is positional; nothing identifying is included."""
    rows = []
    for i, (gt, probs) in enumerate(zip(ground_truth, probabilities)):
        rows.append(",".join([str(i), str(int(gt))] + [f"{float(p):.6f}" for p in probs]))
    return rows


def chunk_rows(rows: List[str], max_bytes: int = 64 * 1024) -> List[str]:
    """Split rows into payloads of at most ``max_bytes``, never splitting a row."""
    out, cur, size = [], [], 0
    for r in rows:
        line = r + "\n"
        n = len(line.encode("utf-8"))
        if cur and size + n > max_bytes:
            out.append("".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += n
    if cur:
        out.append("".join(cur))
    return out


def emit_per_case_predictions(writer, ground_truth, probabilities, split: str = "val") -> int:
    """Send per-case predictions if the site has opted in. Returns chunks sent.

    Opt-in is checked here rather than by the caller so that there is exactly
    one place where the decision to transmit patient-derived data is made.
    """
    if os.environ.get("ODELIA_RETURN_PER_CASE", "") != "1":
        return 0
    chunks = chunk_rows(per_case_rows(ground_truth, probabilities))
    for seq, payload in enumerate(chunks):
        writer.log_text(payload, f"{PER_CASE_PREFIX}{split}/{seq:06d}.csv")
    return len(chunks)
