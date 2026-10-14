"""MS6 global archive (#612): every global of a swarm run, kept at the coordinator (TUD).

Why this exists
---------------
In a CCWF swarm the coordinator never holds a model. Each round's global is built at a randomly chosen
aggregating client (swarm_client_ctl.py: random.choice per round), saved there as FL_global_model.pt and
overwritten at that client's next save; the server job store holds no .pt file and is wiped when the
coordinator container is recreated (#607). The MS6 attack package needs every per-round global
(docs/MS6_WHITE_HAT_PROTOCOL.md section 3), TUD's white-box probe needs them, and they are the reference
against which the attacking sites' own G_r captures (ms6_capture.py) are checked.

Two job components, present but disabled in the job templates; the job builder flips ``enabled``:

    client, config_fed_client.conf (client job process):
        { id = "ms6_global_streamer", path = "ms6_archive.MS6GlobalStreamer", args { enabled = false } }
    server, config_fed_server.conf (server job process):
        { id = "ms6_global_archive", path = "ms6_archive.MS6GlobalArchive",
          args { enabled = false, archive_root = "/archive" } }

Both must be enabled together: a streamer without the archive logs an ERROR per round (training goes on).

Who streams what
----------------
* Round r's aggregating client, right after the swarm controller recorded the aggregated global
  (ClientSideController.record_last_result, the call that writes FL_global_model.pt), copies that file and
  streams it as round_<r+1>: the global every site receives in round r+1. The take-over aggregator of
  #595 goes through the same call. The global after the last round, the final model, is round_<num_rounds>.
* The starting client streams the initial global as round_<start_round> (round_00 for a fresh run) when
  its first learn task starts (ROUND_STARTED).
round_<rr>_global.pt here and an attacking site's capture round_<rr>_global.pt hold the same tensors; both
manifests carry the same "tensor_sha256" for them (ms6_capture.tensor_digest).

The streamer hooks in by wrapping record_last_result on the controller instance announced by the CCWF
EXECUTOR_INITIALIZED event. With enabled = false nothing is wrapped and the job runs exactly as before.

Transport, memory, failures
---------------------------
NVFlare's FileStreamer (app_common/streamers/file_streamer.py) over the client job's object streamer to
the server job, in chunks of ``chunk_size`` (1 MB), one file at a time, from a background thread: memory
stays at one chunk whatever the model size (the MST_SAMMed2D global is 1.07 GB). The aggregator first
copies FL_global_model.pt into its outbox (<app dir>/ms6_outbox/; about a second for 1 GB) because the
persistor rewrites that file in place at its next save. The final global is streamed BEFORE the controller
broadcasts the final results, i.e. while the server job is certainly still running; earlier rounds stream
in the background while training goes on. Nothing here ever fails training: errors are logged as ERROR,
a failed stream is retried, and a global that could not be delivered stays in the outbox for recovery.

The server checks the received bytes against the sha256 and size the sender declared, takes the sending
site and the job id from the authenticated peer context (never from client-supplied fields), refuses
streams larger than ``max_file_bytes`` or than the free disk, never unpickles anything, and writes
<archive_root>/<job_id>/round_<rr>_global.pt, manifest.json and SHA256SUMS (``sha256sum -c`` compatible).
When archive_root is not writable it falls back to <server run dir>/ms6_global_archive, which download_job
returns but which disappears with the job store: point archive_root at a host-mounted directory for real
runs. A copy of the manifest is always written to <server run dir>/ms6_global_archive.json.
"""

import collections
import datetime
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReservedKey, ReturnCode, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.streaming import ConsumerFactory, StreamContextKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.streamers.file_streamer import FileStreamer
from nvflare.app_common.streamers.streamer_base import (
    KEY_DATA,
    KEY_DATA_SIZE,
    KEY_EOF,
    KEY_FILE_SIZE,
    BaseChunkConsumer,
)

CHANNEL = "odelia_ms6"
TOPIC = "global_archive"

# Stream-context fields the sender declares (sent once, with the first chunk).
CTX_ROUND = "ms6.round"
CTX_KIND = "ms6.kind"
CTX_SHA256 = "ms6.sha256"
CTX_SIZE = "ms6.size"
CTX_TENSOR_SHA256 = "ms6.tensor_sha256"
CTX_N_TENSORS = "ms6.n_tensors"
CTX_MODEL_NAME = "ms6.model_name"
CTX_NUM_ROUNDS = "ms6.num_rounds"
CTX_START_ROUND = "ms6.start_round"
# Receiver-side fields (set by the consumer, never sent).
_CTX_PART = "ms6.server.part_path"
_CTX_RECEIVED_SHA256 = "ms6.server.sha256"
_CTX_RECEIVED_SIZE = "ms6.server.size"

KIND_INITIAL = "initial"
KIND_AGGREGATED = "aggregated"

ARCHIVE_FORMAT = "odelia-ms6-archive/1"
MANIFEST_NAME = "manifest.json"
SUMS_NAME = "SHA256SUMS"
FALLBACK_DIR_NAME = "ms6_global_archive"
POINTER_NAME = "ms6_global_archive.json"
OUTBOX_DIR_NAME = "ms6_outbox"
MAX_ROUND = 100000

# Reply header in which the server's stream_done callback result travels back to the sender
# (nvflare/private/stream_runner.py HeaderKey.END_RESULT).
END_RESULT_HEADER = "ObjectStreamer.END_RESULT"

STATUS_ARCHIVED = "archived"
STATUS_DUPLICATE = "duplicate"
STATUS_CONFLICT = "conflict"
STATUS_REJECTED = "rejected"
DELIVERED = (STATUS_ARCHIVED, STATUS_DUPLICATE, STATUS_CONFLICT)

# Free space that must remain after a file is written, at the aggregator's outbox and at the coordinator.
MIN_FREE_RESERVE_BYTES = 1024**3

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def archive_file_name(round_number: int) -> str:
    return f"round_{round_number:02d}_global.pt"


def _safe_name(name) -> str:
    return _SAFE_NAME_RE.sub("_", str(name or "unknown"))[:128] or "unknown"


def _as_round(value) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= MAX_ROUND else None


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _sha256_file(path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


# ==================================================================================================
# Client side: MS6GlobalStreamer
# ==================================================================================================


class MS6GlobalStreamer(FLComponent):
    """Client-job component: streams each global this site records as aggregator to the coordinator.

    Args:
        enabled: off by default; the job builder switches it on together with the server's archive.
        chunk_size: bytes per streamed chunk (memory per stream is one chunk).
        chunk_timeout: seconds to wait for the server's reply to one chunk.
        max_attempts: streams per global before it is given up (and left in the outbox).
        retry_interval: seconds between attempts.
        final_stream_timeout: how long the final aggregator holds back the broadcast of the final results
            until every global it queued has reached the coordinator.
        end_run_timeout: how long END_RUN waits for streams still in flight.
        outbox_dir: where the copies wait for streaming (default <dir of FL_global_model.pt>/ms6_outbox).
        model_name: label recorded in the archive manifest. Default: the persistor's ``model_name`` arg (the
            job builder pins it there, because the client job does not see the launcher's MODEL_NAME),
            else MODEL_NAME of the client job's environment, if set.
    """

    def __init__(
        self,
        enabled: bool = False,
        chunk_size: int = 1024 * 1024,
        chunk_timeout: float = 120.0,
        max_attempts: int = 3,
        retry_interval: float = 30.0,
        final_stream_timeout: float = 1800.0,
        end_run_timeout: float = 300.0,
        outbox_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__()
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")
        for name, value in (
            ("chunk_timeout", chunk_timeout),
            ("final_stream_timeout", final_stream_timeout),
            ("end_run_timeout", end_run_timeout),
        ):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        if not isinstance(retry_interval, (int, float)) or retry_interval < 0:
            raise ValueError(f"retry_interval must be >= 0, got {retry_interval!r}")
        self.enabled = bool(enabled)
        self.chunk_size = chunk_size
        self.chunk_timeout = float(chunk_timeout)
        self.max_attempts = max_attempts
        self.retry_interval = float(retry_interval)
        self.final_stream_timeout = float(final_stream_timeout)
        self.end_run_timeout = float(end_run_timeout)
        self.outbox_dir = outbox_dir
        self.model_name = model_name or None

        self._controller = None
        self._engine = None
        self._job_id = ""
        self._site = ""
        self._cond = threading.Condition()
        self._queue = collections.deque()
        self._pending = 0
        self._stopping = False
        self._worker = None
        self._submitted = set()  # archive rounds already queued from this site
        self._undelivered = []

    # ---- events --------------------------------------------------------------------------------------

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if not self.enabled:
            return
        try:
            if event_type == Constant.EXECUTOR_INITIALIZED:
                self._attach(fl_ctx)
            elif event_type == AppEventType.ROUND_STARTED:
                self._on_round_started(fl_ctx)
            elif event_type == EventType.END_RUN:
                self._on_end_run()
        except Exception as exc:  # never let the archive break the job
            self.logger.error(f"MS6 archive: handling {event_type} failed: {exc!r}")

    def _attach(self, fl_ctx: FLContext) -> None:
        controller = fl_ctx.get_prop(Constant.EXECUTOR)
        if controller is None or not callable(getattr(controller, "record_last_result", None)):
            self.logger.warning("MS6 archive: EXECUTOR_INITIALIZED without a swarm controller; nothing to stream")
            return
        if self._controller is not None:
            if controller is not self._controller:
                self.logger.warning("MS6 archive: a second swarm controller in this job is not archived")
            return
        self._controller = controller
        self._engine = fl_ctx.get_engine()
        self._job_id = fl_ctx.get_job_id() or ""
        self._site = fl_ctx.get_identity_name() or ""

        original = controller.record_last_result
        streamer = self

        def record_last_result(fl_ctx, round_num, result):
            original(fl_ctx, round_num, result)
            streamer._on_global_recorded(round_num)

        record_last_result.__wrapped__ = original
        controller.record_last_result = record_last_result

        self._worker = threading.Thread(target=self._run_worker, name="ms6_global_streamer", daemon=True)
        self._worker.start()
        self.logger.info(
            f"MS6 archive: {self._site} streams every global it aggregates to the coordinator's archive "
            f"(the starting client also the initial global)"
        )

    # ---- what gets queued ------------------------------------------------------------------------

    def _config(self, name, default=None):
        getter = getattr(self._controller, "get_config_prop", None)
        return getter(name, default) if callable(getter) else default

    def _is_final_round(self, round_num) -> bool:
        num_rounds = self._config(AppConstants.NUM_ROUNDS)
        start_round = self._config(Constant.START_ROUND, 0) or 0
        if not isinstance(num_rounds, int) or not isinstance(round_num, int):
            return False
        return round_num - start_round + 1 >= num_rounds

    def _model_name(self) -> Optional[str]:
        persistor = getattr(self._controller, "persistor", None)
        pinned = getattr(persistor, "model_name", None)
        return self.model_name or (pinned if isinstance(pinned, str) else None) or os.environ.get("MODEL_NAME") or None

    def _source_path(self) -> Optional[str]:
        persistor = getattr(self._controller, "persistor", None)
        return getattr(persistor, "_ckpt_save_path", None)

    def _outbox(self) -> Path:
        if self.outbox_dir:
            outbox = Path(self.outbox_dir)
        else:
            source = self._source_path()
            outbox = Path(os.path.dirname(source)) / OUTBOX_DIR_NAME if source else Path.cwd() / OUTBOX_DIR_NAME
        outbox.mkdir(parents=True, exist_ok=True)
        return outbox

    def _claim(self, archive_round: int) -> bool:
        with self._cond:
            if archive_round in self._submitted:
                return False
            self._submitted.add(archive_round)
            return True

    def _enough_space(self, directory: Path, nbytes: int, archive_round: int) -> bool:
        free = shutil.disk_usage(directory).free
        if free < nbytes + max(MIN_FREE_RESERVE_BYTES, nbytes):
            self.logger.error(
                f"MS6 archive: round {archive_round} global NOT archived: {free} bytes free in {directory}, "
                f"{nbytes} needed plus reserve"
            )
            return False
        return True

    def _on_global_recorded(self, round_num) -> None:
        """Called (wrapped) right after the controller recorded round ``round_num``'s aggregated global."""
        try:
            archive_round = int(round_num) + 1
            if not self._claim(archive_round):
                return
            source = self._source_path()
            if not source or not os.path.isfile(source):
                self.logger.error(
                    f"MS6 archive: round {archive_round} global NOT archived: the persistor wrote no file ({source})"
                )
                return
            outbox = self._outbox()
            if not self._enough_space(outbox, os.path.getsize(source), archive_round):
                return
            target = outbox / archive_file_name(archive_round)
            part = outbox / f".{target.name}.part"
            # A copy now: the persistor rewrites FL_global_model.pt in place at its next save.
            shutil.copyfile(source, part)
            os.replace(part, target)
            self._enqueue({"round": archive_round, "kind": KIND_AGGREGATED, "path": str(target)})
        except Exception as exc:
            self.logger.error(f"MS6 archive: round {int(round_num) + 1} global NOT archived: {exc!r}")
            return
        if self._is_final_round(round_num):
            self.logger.info(
                f"MS6 archive: holding the final-result broadcast until the queued globals reached the coordinator "
                f"(at most {self.final_stream_timeout:.0f} s)"
            )
            if not self.wait_until_idle(self.final_stream_timeout):
                self.logger.error(
                    f"MS6 archive: globals still not delivered after {self.final_stream_timeout:.0f} s; the final "
                    f"results are broadcast now, the stream continues in the background"
                )

    def _on_round_started(self, fl_ctx: FLContext) -> None:
        """The starting client archives the initial global when its first learn task starts."""
        controller = self._controller
        if controller is None or not getattr(controller, "is_starting_client", False):
            return
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        start_round = self._config(Constant.START_ROUND, 0) or 0
        if current_round != start_round or not self._claim(start_round):
            return
        try:
            import torch

            learnable = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
            weights = learnable.get("weights") if isinstance(learnable, dict) else None
            if not weights:
                self.logger.error("MS6 archive: initial global NOT archived: no weights in the round's global model")
                return
            with warnings.catch_warnings():
                # torch warns about read-only numpy arrays; the tensors are only serialised, never written to.
                warnings.simplefilter("ignore", UserWarning)
                state = collections.OrderedDict((key, torch.as_tensor(value)) for key, value in weights.items())
            outbox = self._outbox()
            nbytes = sum(t.numel() * t.element_size() for t in state.values())
            if not self._enough_space(outbox, nbytes, start_round):
                return
            target = outbox / archive_file_name(start_round)
            part = outbox / f".{target.name}.part"
            torch.save({"model": state}, part)
            os.replace(part, target)
            self._enqueue({"round": start_round, "kind": KIND_INITIAL, "path": str(target)})
        except Exception as exc:
            self.logger.error(f"MS6 archive: initial global (round {start_round}) NOT archived: {exc!r}")

    # ---- worker ----------------------------------------------------------------------------------

    def _enqueue(self, item: dict) -> None:
        with self._cond:
            self._queue.append(item)
            self._pending += 1
            self._cond.notify_all()

    def wait_until_idle(self, timeout: float) -> bool:
        """Wait until every queued global was handled (delivered or given up). False on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def _run_worker(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait()
                if not self._queue:
                    return
                item = self._queue.popleft()
            try:
                self._deliver(item)
            except Exception as exc:
                self.logger.error(f"MS6 archive: round {item.get('round')} global NOT archived: {exc!r}")
                self._undelivered.append(item)
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()

    def _describe(self, path: str) -> dict:
        """sha256, size and the format-independent tensor digest of an outbox file."""
        info = {CTX_SIZE: os.path.getsize(path), CTX_SHA256: _sha256_file(path)}
        try:
            import torch

            from ms6_capture import tensor_digest

            # mmap: the tensors are read from the page cache one by one, not loaded at once (1 GB models).
            data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            state = data.get("model", data) if isinstance(data, dict) else None
            if isinstance(state, dict):
                info[CTX_TENSOR_SHA256], info[CTX_N_TENSORS] = tensor_digest(state)
            del data, state
        except Exception as exc:
            self.logger.warning(f"MS6 archive: no tensor digest for {path}: {exc!r}")
        return info

    def _deliver(self, item: dict) -> None:
        path = item["path"]
        archive_round = item["round"]
        stream_ctx = {
            CTX_ROUND: archive_round,
            CTX_KIND: item["kind"],
            CTX_MODEL_NAME: self._model_name(),
            CTX_NUM_ROUNDS: self._config(AppConstants.NUM_ROUNDS),
            CTX_START_ROUND: self._config(Constant.START_ROUND, 0),
        }
        stream_ctx.update(self._describe(path))
        reason = ""
        for attempt in range(1, self.max_attempts + 1):
            status, retry, reason = self._stream_once(path, dict(stream_ctx))
            if status in DELIVERED:
                os.remove(path)
                self.logger.info(
                    f"MS6 archive: round {archive_round} global ({stream_ctx[CTX_SIZE]} bytes, sha256 "
                    f"{stream_ctx[CTX_SHA256][:16]}) {status} at the coordinator"
                )
                return
            self.logger.warning(
                f"MS6 archive: streaming round {archive_round} global, attempt {attempt}/{self.max_attempts}: {reason}"
            )
            if not retry:
                break
            if attempt < self.max_attempts:
                time.sleep(self.retry_interval)
        self._undelivered.append(item)
        self.logger.error(
            f"MS6 archive: round {archive_round} global NOT archived ({reason}); the copy stays at {path}"
        )

    def _stream_once(self, path: str, stream_ctx: dict):
        """One stream to the server. Returns (status, retry_makes_sense, reason)."""
        # Own context with an abort signal nobody triggers (as JobLogStreamer does): the job's run abort
        # signal fires at the end of the job, which would cut the last stream short.
        fl_ctx = self._engine.new_context()
        fl_ctx.put(key=ReservedKey.RUN_ABORT_SIGNAL, value=Signal(), private=True, sticky=False)
        rc, result = FileStreamer.stream_file(
            channel=CHANNEL,
            topic=TOPIC,
            stream_ctx=stream_ctx,
            targets=[SiteType.SERVER],
            file_name=path,
            fl_ctx=fl_ctx,
            chunk_size=self.chunk_size,
            chunk_timeout=self.chunk_timeout,
        )
        if rc != ReturnCode.OK or not isinstance(result, dict) or not result:
            return None, True, f"stream ended with rc={rc}"
        reply = next(iter(result.values()))
        outcome = reply.get_header(END_RESULT_HEADER) if isinstance(reply, Shareable) else None
        if not isinstance(outcome, dict):
            return None, True, "the coordinator sent no archive status"
        status = outcome.get("status")
        return status, bool(outcome.get("retry", status not in DELIVERED)), outcome.get("reason") or status

    def _on_end_run(self) -> None:
        if self._worker is None:
            return
        idle = self.wait_until_idle(self.end_run_timeout)
        with self._cond:
            waiting = [item["round"] for item in self._queue]
            self._stopping = True
            self._cond.notify_all()
        lost = sorted(set(waiting + [item["round"] for item in self._undelivered]))
        if not idle or lost:
            self.logger.error(
                f"MS6 archive: globals of rounds {lost or 'in flight'} did not reach the coordinator; their copies "
                f"stay in the outbox"
            )


# ==================================================================================================
# Server side: MS6GlobalArchive
# ==================================================================================================


class _ArchiveConsumer(BaseChunkConsumer):
    """Writes one incoming stream to a .part file, hashing as it goes; enforces the declared size."""

    def __init__(self, stream_ctx, incoming_dir: Path, max_bytes: int):
        super().__init__()
        self.file = None
        expected = stream_ctx.get(KEY_FILE_SIZE)
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise ValueError(f"MS6 archive: invalid declared file size {expected!r}")
        if expected > max_bytes:
            raise ValueError(f"MS6 archive: declared size {expected} exceeds max_file_bytes {max_bytes}")
        free = shutil.disk_usage(incoming_dir).free
        if free < expected + MIN_FREE_RESERVE_BYTES:
            raise ValueError(f"MS6 archive: {free} bytes free in {incoming_dir}, {expected} needed plus reserve")
        self.expected = expected
        self.received = 0
        self.sha256 = hashlib.sha256()
        self.path = incoming_dir / f"{uuid.uuid4().hex}.part"
        self.file = open(self.path, "wb")
        stream_ctx[_CTX_PART] = str(self.path)

    def consume(self, shareable: Shareable, stream_ctx, fl_ctx):
        data = shareable.get(KEY_DATA)
        size = shareable.get(KEY_DATA_SIZE)
        self._validate_chunk(data, size)
        if data:
            if self.received + size > self.expected:
                raise ValueError(f"MS6 archive: more than the declared {self.expected} bytes received")
            self.file.write(data)
            self.sha256.update(data)
            self.received += size
        if shareable.get(KEY_EOF):
            if self.received != self.expected:
                raise ValueError(f"MS6 archive: received {self.received} bytes, {self.expected} declared")
            stream_ctx[_CTX_RECEIVED_SHA256] = self.sha256.hexdigest()
            stream_ctx[_CTX_RECEIVED_SIZE] = self.received
            return False, make_reply(ReturnCode.OK)
        return True, make_reply(ReturnCode.OK)

    def finalize(self, stream_ctx, fl_ctx):
        if self.file is not None:
            self.file.close()
            self.file = None


class _ArchiveConsumerFactory(ConsumerFactory):
    def __init__(self, archive: "MS6GlobalArchive"):
        self.archive = archive

    def get_consumer(self, stream_ctx, fl_ctx):
        return _ArchiveConsumer(stream_ctx, self.archive.incoming_dir, self.archive.max_file_bytes)


class MS6GlobalArchive(FLComponent):
    """Server-job component: receives the globals streamed by MS6GlobalStreamer and archives them.

    Args:
        enabled: off by default; the job builder switches it on together with the clients' streamer.
        archive_root: files go to <archive_root>/<job_id>/; falls back to <server run dir>/ms6_global_archive
            when archive_root cannot be written.
        max_file_bytes: largest file accepted (default 16 GiB).
    """

    def __init__(self, enabled: bool = False, archive_root: str = "/archive", max_file_bytes: int = 16 * 1024**3):
        super().__init__()
        if not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError(f"max_file_bytes must be a positive int, got {max_file_bytes!r}")
        self.enabled = bool(enabled)
        self.archive_root = archive_root
        self.max_file_bytes = max_file_bytes
        self.job_id = ""
        self.run_dir: Optional[Path] = None
        self.archive_dir: Optional[Path] = None
        self.incoming_dir: Optional[Path] = None
        self.manifest: dict = {}
        self._lock = threading.Lock()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if not self.enabled:
            return
        try:
            if event_type == EventType.START_RUN:
                self._start(fl_ctx)
            elif event_type == EventType.END_RUN:
                self._finish()
        except Exception as exc:
            self.logger.error(f"MS6 archive: handling {event_type} failed: {exc!r}")

    # ---- set-up --------------------------------------------------------------------------------------

    @staticmethod
    def _usable(directory: Path) -> Optional[str]:
        """None if ``directory`` can be created and written, else the reason."""
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o750)
            probe = directory / f".probe-{uuid.uuid4().hex}"
            probe.write_bytes(b"")
            probe.unlink()
            return None
        except Exception as exc:
            return repr(exc)

    def _start(self, fl_ctx: FLContext) -> None:
        self.job_id = _safe_name(fl_ctx.get_job_id())
        engine = fl_ctx.get_engine()
        self.run_dir = Path(engine.get_workspace().get_run_dir(fl_ctx.get_job_id()))
        preferred = Path(self.archive_root) / self.job_id if self.archive_root else None
        fallback = self.run_dir / FALLBACK_DIR_NAME
        problem = self._usable(preferred) if preferred is not None else "archive_root is empty"
        if problem is None:
            self.archive_dir = preferred
        else:
            fallback_problem = self._usable(fallback)
            if fallback_problem is not None:
                self.logger.error(
                    f"MS6 archive DISABLED: neither {preferred} ({problem}) nor {fallback} ({fallback_problem}) "
                    f"is writable"
                )
                return
            self.logger.warning(
                f"MS6 archive: archive_root {self.archive_root!r} not usable ({problem}); archiving into the job's "
                f"run directory {fallback}, which is deleted with the job store. Mount a host directory for real runs."
            )
            self.archive_dir = fallback
        self.incoming_dir = self.archive_dir / ".incoming"
        self.incoming_dir.mkdir(exist_ok=True)
        self.manifest = {
            "format": ARCHIVE_FORMAT,
            "job_id": self.job_id,
            "archive_dir": str(self.archive_dir),
            "fallback_dir": self.archive_dir == fallback,
            "tensor_digest": "tensor_sha256 is computed by the sending client with ms6_capture.tensor_digest",
            "rounds": {},
            "conflicts": [],
            "rejected": [],
            "expected": {},
            "started_at": _now(),
        }
        self._resume_manifest()
        self._write_manifest()
        engine.register_stream_processing(
            channel=CHANNEL,
            topic=TOPIC,
            factory=_ArchiveConsumerFactory(self),
            stream_done_cb=self._stream_done,
        )
        self.logger.info(f"MS6 archive: receiving the per-round globals of job {self.job_id} into {self.archive_dir}")

    def _resume_manifest(self) -> None:
        """Keep what an earlier start of the same job archived here (a restarted server job)."""
        path = self.archive_dir / MANIFEST_NAME
        if not path.exists():
            return
        try:
            previous = json.loads(path.read_text())
        except Exception as exc:
            self.logger.warning(f"MS6 archive: unreadable {path} ({exc!r}); starting a new manifest")
            return
        if previous.get("job_id") != self.job_id:
            return
        for key in ("rounds", "conflicts", "rejected", "expected"):
            if key in previous:
                self.manifest[key] = previous[key]

    # ---- receiving -----------------------------------------------------------------------------------

    def _write_manifest(self) -> None:
        self.manifest["updated_at"] = _now()
        _write_json_atomic(self.archive_dir / MANIFEST_NAME, self.manifest)
        lines = [
            f"{entry['sha256']}  {entry['file']}\n"
            for _, entry in sorted(self.manifest["rounds"].items(), key=lambda item: int(item[0]))
        ]
        lines += [f"{entry['sha256']}  {entry['file']}\n" for entry in self.manifest["conflicts"]]
        tmp = self.archive_dir / f".{SUMS_NAME}.tmp-{os.getpid()}"
        tmp.write_text("".join(lines))
        os.replace(tmp, self.archive_dir / SUMS_NAME)

    @staticmethod
    def _remove(path) -> None:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    def _reject(self, part, site: str, round_value, reason: str, retry: bool) -> dict:
        self._remove(part)
        self.logger.error(f"MS6 archive: refused round {round_value!r} global from {site}: {reason}")
        with self._lock:
            self.manifest["rejected"].append({"round": round_value, "sender": site, "reason": reason, "at": _now()})
            self._write_manifest()
        return {"status": STATUS_REJECTED, "reason": reason, "retry": retry}

    def _stream_done(self, stream_ctx, fl_ctx: FLContext):
        """Called by the object streamer when a stream ended; the return value goes back to the sender."""
        part = stream_ctx.get(_CTX_PART)
        peer_ctx = fl_ctx.get_peer_context()
        site = _safe_name(peer_ctx.get_identity_name() if peer_ctx else None)
        peer_job = peer_ctx.get_job_id() if peer_ctx else None
        round_value = stream_ctx.get(CTX_ROUND)
        try:
            rc = stream_ctx.get(StreamContextKey.RC)
            if rc != ReturnCode.OK:
                return self._reject(part, site, round_value, f"stream ended with {rc}", retry=True)
            if _safe_name(peer_job) != self.job_id:
                return self._reject(part, site, round_value, f"stream from job {peer_job!r}", retry=False)
            round_number = _as_round(round_value)
            if round_number is None:
                return self._reject(part, site, round_value, "round is not an integer in range", retry=False)
            sha256 = stream_ctx.get(_CTX_RECEIVED_SHA256)
            size = stream_ctx.get(_CTX_RECEIVED_SIZE)
            if stream_ctx.get(CTX_SHA256) != sha256 or stream_ctx.get(CTX_SIZE) != size:
                return self._reject(
                    part, site, round_number,
                    f"received {size} bytes with sha256 {sha256}, sender declared {stream_ctx.get(CTX_SIZE)} "
                    f"bytes with sha256 {stream_ctx.get(CTX_SHA256)}",
                    retry=True,
                )
            entry = {
                "round": round_number,
                "kind": _safe_name(stream_ctx.get(CTX_KIND)),
                "sender": site,
                "bytes": size,
                "sha256": sha256,
                "tensor_sha256": stream_ctx.get(CTX_TENSOR_SHA256),
                "n_tensors": stream_ctx.get(CTX_N_TENSORS),
                "model_name": _safe_name(stream_ctx.get(CTX_MODEL_NAME)) if stream_ctx.get(CTX_MODEL_NAME) else None,
                "received_at": _now(),
            }
            with self._lock:
                status = self._store(part, entry)
                self._note_expected(stream_ctx)
                self._write_manifest()
            return {"status": status, "round": round_number, "sha256": sha256, "file": entry["file"]}
        except Exception as exc:
            self._remove(part)
            self.logger.error(f"MS6 archive: round {round_value!r} global from {site} not archived: {exc!r}")
            return {"status": STATUS_REJECTED, "reason": repr(exc), "retry": True}

    def _store(self, part, entry: dict) -> str:
        round_number = entry["round"]
        key = str(round_number)
        target = self.archive_dir / archive_file_name(round_number)
        existing = self.manifest["rounds"].get(key)
        if existing is None and target.exists():
            existing = {"sha256": _sha256_file(target)}  # left over from an earlier start of this job
        if existing is not None and existing.get("sha256") == entry["sha256"]:
            self._remove(part)
            entry["file"] = target.name
            if key not in self.manifest["rounds"]:
                self.manifest["rounds"][key] = entry
            self.logger.info(f"MS6 archive: round {round_number} global from {entry['sender']} already archived")
            return STATUS_DUPLICATE
        if existing is not None:
            # Two different globals for one round (e.g. an aggregator replaced mid-round, #595): keep both.
            conflict = self.archive_dir / (
                f"round_{round_number:02d}_global.conflict-{entry['sender']}-{entry['sha256'][:12]}.pt"
            )
            os.replace(part, conflict)
            os.chmod(conflict, 0o640)
            entry["file"] = conflict.name
            self.manifest["conflicts"].append(entry)
            self.logger.warning(
                f"MS6 archive: a DIFFERENT round {round_number} global arrived from {entry['sender']}; kept as "
                f"{conflict.name} next to {target.name}"
            )
            return STATUS_CONFLICT
        os.replace(part, target)
        os.chmod(target, 0o640)
        entry["file"] = target.name
        self.manifest["rounds"][key] = entry
        self.logger.info(
            f"MS6 archive: round {round_number} global from {entry['sender']} archived as {target} "
            f"({entry['bytes']} bytes, sha256 {entry['sha256'][:16]})"
        )
        return STATUS_ARCHIVED

    def _note_expected(self, stream_ctx) -> None:
        """Remember the run length the senders report, for the missing-round check at the end."""
        num_rounds = _as_round(stream_ctx.get(CTX_NUM_ROUNDS))
        start_round = _as_round(stream_ctx.get(CTX_START_ROUND))
        if num_rounds and start_round is not None and not self.manifest["expected"]:
            self.manifest["expected"] = {"start_round": start_round, "num_rounds": num_rounds}

    # ---- end of job ----------------------------------------------------------------------------------

    def _finish(self) -> None:
        if self.archive_dir is None:
            return
        with self._lock:
            stale = sorted(self.incoming_dir.glob("*.part")) if self.incoming_dir else []
            for path in stale:
                self._remove(path)
            if stale:
                self.logger.warning(f"MS6 archive: removed {len(stale)} incomplete stream(s) at the end of the job")
            expected = self.manifest.get("expected") or {}
            received = sorted(int(key) for key in self.manifest["rounds"])
            if expected:
                start = expected["start_round"]
                wanted = range(start, start + expected["num_rounds"] + 1)
                missing = [r for r in wanted if str(r) not in self.manifest["rounds"]]
            else:
                missing = []
            self.manifest["missing_rounds"] = missing
            self.manifest["finished_at"] = _now()
            self._write_manifest()
            if self.run_dir is not None and self.run_dir.is_dir():
                pointer = dict(self.manifest)
                _write_json_atomic(self.run_dir / POINTER_NAME, pointer)
        if missing:
            self.logger.error(
                f"MS6 archive: job {self.job_id} ended with rounds {missing} MISSING from {self.archive_dir} "
                f"(have {received})"
            )
        elif not received:
            self.logger.warning(f"MS6 archive: job {self.job_id} ended without any archived global")
        else:
            self.logger.info(
                f"MS6 archive: job {self.job_id} complete: rounds {received[0]}..{received[-1]} in {self.archive_dir}"
            )
