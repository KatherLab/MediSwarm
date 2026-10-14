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
One line per case of the returned split:

    row_index,ground_truth,prob_class_0,prob_class_1,prob_class_2

**No case identifier, no patient identifier, no image.** The row index is a
position within that site's own split and carries no meaning outside it.
Pairing two models' predictions -- which is what D2.5 needs -- works from the
index because the validation and test loaders are not shuffled, so the n-th
row is the same case in both runs.

Sending a real uid would have been easier and is what the first draft of the
consortium slide promised; an index is strictly less information and does the
same job, so it is what this sends.

Which model and which split (release 1.9.0)
-------------------------------------------
Every emission is tagged with the split (``val`` or ``test``), the kind of model
(``site_local``: the site's own model after a local epoch; ``final_global``: the
job's final global, scored in the evaluation-only round) and the swarm round.
The tags travel in the chunk path, never in the rows:

    per_case/<split>/<model_kind>/round_<r>/epoch_<e>/emission_<n>/<seq>_of_<count>.csv

The coordinator keeps, per (site, split, model kind), the newest *complete*
emission and writes it to its own file,
``cross_site_val/per_case_predictions/<SITE>__<split>__<model_kind>.csv``, with
a ``#`` provenance header (job id, site, split, model kind, round, epoch, rows)
and the tags repeated as columns. Rows of different splits or model kinds are
never mixed; before 1.9.0 they were all written into one ``<site>.csv`` and the
last emission won chunk by chunk, so test rows could overwrite validation rows.
Read the files with ``pandas.read_csv(path, comment="#")``.

Off by default
--------------
This is patient-derived data, even pseudonymised, and returning it needs the
consortium's agreement rather than a maintainer's. The client half emits
nothing unless ``ODELIA_RETURN_PER_CASE=1`` is set in the site's environment,
which is a decision each site makes for itself.
"""

from __future__ import annotations

import itertools
import os
import re
from threading import Lock
from typing import Dict, List, Optional, Tuple

from nvflare.apis.analytix import AnalyticsData
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver

PER_CASE_PREFIX = "per_case/"
DEFAULT_OUTPUT_DIR = "cross_site_val/per_case_predictions"
# What leaves the site, per case.
ROW_COLUMNS = "row_index,ground_truth,prob_class_0,prob_class_1,prob_class_2"
# What the coordinator writes: the same columns, each row tagged with its emission's split, model kind, round.
CSV_HEADER = ROW_COLUMNS + ",split,model_kind,round"

SPLITS = ("val", "test")
MODEL_KINDS = ("site_local", "final_global")

_CHUNK_PATH_RE = re.compile(
    r"^per_case/(?P<split>[a-z]+)/(?P<model_kind>[a-z_]+)/round_(?P<round>\d+|na)/epoch_(?P<epoch>\d+|na)"
    r"/emission_(?P<emission>\d+)/(?P<seq>\d+)_of_(?P<count>\d+)\.csv$"
)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.@-]")


def chunk_path(split: str, model_kind: str, round_num: Optional[int], epoch: Optional[int], emission: int,
               seq: int, count: int) -> str:
    """Path of one chunk of an emission; the coordinator recovers every tag from it (parse_chunk_path)."""
    r = "na" if round_num is None else str(int(round_num))
    e = "na" if epoch is None else str(int(epoch))
    return f"{PER_CASE_PREFIX}{split}/{model_kind}/round_{r}/epoch_{e}/emission_{emission}/{seq:06d}_of_{count:06d}.csv"


def parse_chunk_path(path) -> Optional[dict]:
    """Inverse of chunk_path(); None for anything that is not a well-formed per-case chunk path."""
    match = _CHUNK_PATH_RE.match(str(path or ""))
    if not match:
        return None
    fields = match.groupdict()
    if fields["split"] not in SPLITS or fields["model_kind"] not in MODEL_KINDS:
        return None
    seq, count = int(fields["seq"]), int(fields["count"])
    if count <= 0 or seq >= count:
        return None
    return {
        "split": fields["split"],
        "model_kind": fields["model_kind"],
        "round": None if fields["round"] == "na" else int(fields["round"]),
        "epoch": None if fields["epoch"] == "na" else int(fields["epoch"]),
        "emission": int(fields["emission"]),
        "seq": seq,
        "count": count,
    }


class _Emission:
    """Chunks of one emission (one model, one split, one moment) as they arrive."""

    def __init__(self, count: int, round_num: Optional[int], epoch: Optional[int]):
        self.count = count
        self.round = round_num
        self.epoch = epoch
        self.chunks: Dict[int, str] = {}

    @property
    def complete(self) -> bool:
        return len(self.chunks) == self.count

    def body(self) -> str:
        return "".join(self.chunks[k] for k in sorted(self.chunks))


class PerCasePredictionCollector(AnalyticsReceiver):
    """Reassemble per-case prediction chunks and write one CSV per (site, split, model kind).

    Chunks arrive as ``AnalyticsDataType.TEXT`` records whose ``path`` is built by ``chunk_path``. They are
    ordered by sequence number on write rather than on arrival: the transport gives no ordering guarantee,
    and a silently reordered file would break the index-based pairing that is the whole point of sending an
    index. Sites emit the same key many times (the local model's validation rows after every epoch), so an
    emission is identified by (round, emission number) and only the newest complete one is written.
    """

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, max_chunks_per_site: int = 4096):
        super().__init__()
        self._output_dir = output_dir
        self._max_chunks = max_chunks_per_site
        # (site, split, model_kind) -> {(round or -1, emission number): _Emission}
        self._emissions: Dict[Tuple[str, str, str], Dict[Tuple[int, int], _Emission]] = {}
        self._dropped: Dict[str, int] = {}
        self._published = False
        self._lock = Lock()

    def initialize(self, fl_ctx: FLContext):
        with self._lock:
            self._emissions = {}
            self._dropped = {}
            self._published = False

    def _chunks_held(self, site: str) -> int:
        return sum(
            len(emission.chunks)
            for (s, _, _), emissions in self._emissions.items() if s == site
            for emission in emissions.values()
        )

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        try:
            data = AnalyticsData.from_dxo(from_shareable(shareable))
        except Exception as exc:
            self.log_warning(fl_ctx, f"Unparseable record from {record_origin}: {exc}", fire_event=False)
            return
        if not data or not data.path or not str(data.path).startswith(PER_CASE_PREFIX):
            return

        tags = parse_chunk_path(data.path)
        if tags is None:
            self.log_warning(
                fl_ctx, f"Ignoring per-case chunk from {record_origin} with unparseable path {data.path!r}",
                fire_event=False,
            )
            return
        if not isinstance(data.value, str):
            return

        key = (record_origin, tags["split"], tags["model_kind"])
        emission_id = (-1 if tags["round"] is None else tags["round"], tags["emission"])
        with self._lock:
            emissions = self._emissions.setdefault(key, {})
            complete = [eid for eid, emission in emissions.items() if emission.complete]
            if complete and emission_id < max(complete):
                return  # a late chunk of an emission that a newer complete one already replaced

            emission = emissions.get(emission_id)
            if emission is None or tags["seq"] not in emission.chunks:
                # A site sending unbounded chunks must not exhaust coordinator memory.
                if self._chunks_held(record_origin) >= self._max_chunks:
                    self._dropped[record_origin] = self._dropped.get(record_origin, 0) + 1
                    return
            if emission is None:
                emission = emissions[emission_id] = _Emission(tags["count"], tags["round"], tags["epoch"])
            elif emission.count != tags["count"]:
                self.log_warning(
                    fl_ctx, f"Ignoring per-case chunk from {record_origin}: chunk count {tags['count']} "
                            f"contradicts {emission.count} of the same emission ({data.path})",
                    fire_event=False,
                )
                return
            emission.chunks[tags["seq"]] = data.value

            if emission.complete:
                # Older emissions of this key can never be written any more.
                for eid in [eid for eid in emissions if eid < emission_id]:
                    del emissions[eid]

    @staticmethod
    def _choose(emissions: Dict[Tuple[int, int], _Emission]) -> Optional[_Emission]:
        """Newest complete emission; failing that, the newest one (written and flagged incomplete)."""
        complete = [eid for eid, emission in emissions.items() if emission.complete]
        if complete:
            return emissions[max(complete)]
        return emissions[max(emissions)] if emissions else None

    def finalize(self, fl_ctx: FLContext):
        # Called on ABOUT_TO_END_RUN and again on END_RUN; the files are written once.
        with self._lock:
            if self._published:
                return
            self._published = True
            chosen = {key: self._choose(emissions) for key, emissions in self._emissions.items()}
            chosen = {key: emission for key, emission in chosen.items() if emission is not None}
            dropped = dict(self._dropped)

        if not chosen:
            self.log_info(
                fl_ctx,
                "No per-case predictions received. This is the expected state unless sites "
                "set ODELIA_RETURN_PER_CASE=1; the capability is opt-in per site.",
                fire_event=False,
            )
            return

        job_id = fl_ctx.get_job_id()
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(job_id)
        out_dir = os.path.join(run_dir, self._output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for (site, split, model_kind), emission in sorted(chosen.items()):
            path = os.path.join(out_dir, f"{_SAFE_NAME_RE.sub('_', site)}__{split}__{model_kind}.csv")
            round_tag = "" if emission.round is None else str(emission.round)
            lines = [line for line in emission.body().splitlines() if line.strip()]
            header = [
                "# ODELIA per-case predictions, returned by opt-in (ODELIA_RETURN_PER_CASE=1). row_index is the "
                "position in the site's own split; no identifiers.",
                f"# job_id={job_id}",
                f"# site={site}",
                f"# split={split}",
                f"# model_kind={model_kind}",
                f"# round={round_tag or 'unknown'}",
                f"# epoch={'n/a' if emission.epoch is None else emission.epoch}",
                f"# rows={len(lines)}",
                f"# complete={'true' if emission.complete else 'false'} "
                f"(chunks {len(emission.chunks)} of {emission.count})",
            ]
            if dropped.get(site):
                header.append(f"# dropped_chunks={dropped[site]} (per-site cap {self._max_chunks})")
            try:
                with open(path, "w") as fh:
                    fh.write("\n".join(header) + "\n")
                    fh.write(CSV_HEADER + "\n")
                    for line in lines:
                        fh.write(f"{line},{split},{model_kind},{round_tag}\n")
            except Exception as exc:
                self.log_error(fl_ctx, f"Could not write per-case predictions for {site}: {exc}",
                               fire_event=False)
                continue
            msg = (f"Wrote {len(lines)} per-case {split} rows of {site}'s {model_kind} model "
                   f"(round {round_tag or '?'}) -> {path}")
            if not emission.complete:
                # Say so loudly: a truncated file still looks like a valid result.
                msg += f" (WARNING: only {len(emission.chunks)} of {emission.count} chunks arrived; file is INCOMPLETE)"
            if dropped.get(site):
                msg += f" (WARNING: {dropped[site]} chunk(s) of {site} dropped at the {self._max_chunks} cap)"
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

# Numbers this process's emissions, so the coordinator can tell a newer emission of the same key from an
# older one whatever order their chunks arrive in.
_EMISSION_COUNTER = itertools.count()


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


def emit_per_case_predictions(writer, ground_truth, probabilities, split: str = "val",
                              model_kind: str = "site_local", round_num: Optional[int] = None,
                              epoch: Optional[int] = None) -> int:
    """Send per-case predictions if the site has opted in. Returns chunks sent.

    Opt-in is checked here rather than by the caller so that there is exactly
    one place where the decision to transmit patient-derived data is made.
    """
    if os.environ.get("ODELIA_RETURN_PER_CASE", "") != "1":
        return 0
    if split not in SPLITS:
        raise ValueError(f"per-case split must be one of {SPLITS}, got {split!r}")
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"per-case model_kind must be one of {MODEL_KINDS}, got {model_kind!r}")
    chunks = chunk_rows(per_case_rows(ground_truth, probabilities))
    if not chunks:
        return 0
    emission = next(_EMISSION_COUNTER)
    for seq, payload in enumerate(chunks):
        writer.log_text(payload, chunk_path(split, model_kind, round_num, epoch, emission, seq, len(chunks)))
    return len(chunks)
