"""MS6 per-round capture (#612): keep the global a site receives (G_r) and the local model it releases (W_r).

Why this exists
---------------
The MS6 white-hat attacks (docs/MS6_WHITE_HAT_PROTOCOL.md) run at the attacking sites against their own
update Delta_r = W_r - G_r. That difference is exactly what the round's aggregating peer receives: the
client API sends W_r - G_r over the keys of the received global (params_transfer_type = DIFF), and no
task_result_filter runs in the CCWF learn path. Before 1.9.0 nothing kept per-round models: Lightning
overwrites last.ckpt every epoch and the persistor keeps only the final globals.

This module is a round hook (round_hooks.py). It is switched on per job from the launcher line in
config_fed_client.conf, so no site has to change anything:

    script = "... ODELIA_ROUND_HOOKS=ms6_capture ODELIA_MS6_CAPTURE_SITES=CAM_1,RUMC_1 python3 custom/{app_script} ..."

Variables (training subprocess):
    ODELIA_MS6_CAPTURE_SITES      comma list of site names, or "all". Empty: nothing is captured anywhere.
    ODELIA_MS6_CAPTURE_ROUNDS     "all" (default) or a comma list of 0-based rounds.
    ODELIA_MS6_CAPTURE_DIR        root directory, default /scratch/ms6. Each job writes to <root>/<job_id>,
                                  so the default is /scratch/ms6/<job_id>. The job id is always appended: it
                                  exists only once the job is submitted, so a launcher line cannot name it.
                                  Never put "{...}" into the launcher line: NVFlare formats that line with
                                  str.format and keeps it entirely unformatted (custom/{app_script} included)
                                  when a name is unknown, so the trainer would not start.
    ODELIA_MS6_CAPTURE_OPTIMIZER  1 = also keep the optimizer and LR-scheduler state at the start of local
                                  training (what attack B needs to replay a round exactly). Default 0.

Files, in <root>/<job_id>/<SITE>/ (rr = 0-based NVFlare round, two digits):
    round_<rr>_global.pt     G_r: the global received in round r, {"model": state_dict, "train_conf": ..., "ms6": meta}
    round_<rr>_local.pt      W_r: the local model as released at the end of round r (after DP noise)
    round_<rr>_optimizer.pt  optimizer/LR-scheduler state at the start of round r's local training
    manifest.json            job id, site, model name and, per file, round, bytes, sha256 and tensor digest
The model files have the persistor's layout ({"model": ..., "train_conf": {"train": {"model": MODEL_NAME}}}), so
they load wherever an NVFlare global loads (predict.py, --external_validation) and predict.py can check the
architecture.
round_<rr>_global.pt holds the same tensors as the coordinator's archive file round_<rr>_global.pt
(ms6_archive.py); the "tensor_sha256" fields of both manifests make that checkable without loading either.

Where the copies are taken
--------------------------
G_r in after_receive, from ctx.input_model.params, before the global is loaded into the model.

W_r in after_round, right after trainer.fit() returned. By then FLCallback.on_train_end has sent
pl_module.cpu().state_dict(), and nothing has changed the weights since: FLCallback.reset_state and
RestoreState only touch loop counters and optimizer copies, and the ODELIA LightningModule defines no
on_train_end/on_fit_end. The file therefore holds exactly the released tensors, including whatever a hook's
Lightning callback did to them in on_train_end before the send (DP clipping and noise), whatever the order of
ODELIA_ROUND_HOOKS. A W_r callback would not have that property: round_hooks inserts hook callbacks before
FLCallback in list order, so with the job builder's order al_subset,ms6_capture,dp_noise such a callback
would run before dp_noise's and keep the model BEFORE the noise (it would need dp_noise listed first).
Taking W_r after the send also keeps the write out of the round's critical path. The one ordering rule left:
a hook whose after_round changes ctx.model must come after ms6_capture (none does; the builder's order
satisfies it).

The optimizer state is taken by a Lightning callback in on_train_start, after NVFlare's RestoreState has put
back (in on_fit_start) the state the previous round ended with.

Nothing is kept in the evaluation-only final round (release contract 3.3): nothing is trained or released
there, and its received global is the job's final global, which every site holds as FL_global_model.pt
anyway and the coordinator's archive (ms6_archive.py) holds too.

Privacy and safety
------------------
* The capture directory must not lie under $SCRATCH_DIR/runs or the trainer's run directory: live_sync
  uploads those trees to TUD, which would hand TUD the sites' local models. create_hook/on_start refuse such
  a path. The default /scratch/ms6 is invisible to live_sync and to the heartbeat.
* Best effort: a full disk or a failing write never fails training. Before every write the free space is
  checked against the file size plus a reserve for the checkpoints Lightning still has to write; a capture
  that does not fit is skipped with a WARNING and listed under "skipped" in the manifest. Configuration
  errors (bad variable values, a forbidden directory) fail the job before round 0.
* Capturing neither consumes random numbers nor changes the weights: training is the same with and without.
"""

import datetime
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

from pytorch_lightning.callbacks import Callback

from round_hooks import RoundHook

SITES_ENV = "ODELIA_MS6_CAPTURE_SITES"
ROUNDS_ENV = "ODELIA_MS6_CAPTURE_ROUNDS"
DIR_ENV = "ODELIA_MS6_CAPTURE_DIR"
OPTIMIZER_ENV = "ODELIA_MS6_CAPTURE_OPTIMIZER"

DEFAULT_CAPTURE_ROOT = "/scratch/ms6"
UNKNOWN_JOB_ID = "unknown_job"
MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "odelia-ms6-capture/1"
ALL = "all"

KIND_GLOBAL = "global"
KIND_LOCAL = "local"
KIND_OPTIMIZER = "optimizer"

# Non-parameter buffers left out of the tensor digest (release contract 3.4): the per-site class weights
# differ between the persistor's model and every trainer, so including them would make equal models differ.
DIGEST_IGNORED_KEYS = ("_class_weight",)
TENSOR_DIGEST_DEFINITION = (
    "sha256 over the state dict in sorted key order; per entry b'<key>|<dtype>|<shape>|' followed by the raw "
    "bytes of the contiguous CPU tensor (dtype without the 'torch.' prefix, shape as comma-separated sizes); "
    f"keys {list(DIGEST_IGNORED_KEYS)} (also as a dotted suffix) are left out"
)

# Free space that must remain after a capture write: the file itself, plus room for RESERVE_FACTOR more
# files of its size (a Lightning checkpoint holds the weights and two Adam moments), and never less than
# MIN_FREE_RESERVE_BYTES. Lightning writes last.ckpt and the best checkpoint to the same scratch disk; a
# capture must never be the reason those writes fail.
MIN_FREE_RESERVE_BYTES = 2 * 1024**3
RESERVE_FACTOR = 3

_SITE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}
_LOG = logging.getLogger(__name__)


class MS6CaptureConfigError(ValueError):
    """MS6 capture configuration that must stop the job before round 0."""


# --------------------------------------------------------------------------------------------------
# Configuration parsing
# --------------------------------------------------------------------------------------------------


def parse_sites(value: Optional[str]):
    """ODELIA_MS6_CAPTURE_SITES -> None (capture nowhere), ALL, or a frozenset of site names."""
    names = [token.strip() for token in (value or "").split(",") if token.strip()]
    if not names:
        return None
    if any(name.lower() == ALL for name in names):
        if len(names) > 1:
            raise MS6CaptureConfigError(f"{SITES_ENV}={value!r}: 'all' cannot be combined with site names")
        return ALL
    for name in names:
        if not _SITE_NAME_RE.match(name):
            raise MS6CaptureConfigError(f"{SITES_ENV}={value!r}: {name!r} is not a site name")
    return frozenset(names)


def parse_rounds(value: Optional[str]):
    """ODELIA_MS6_CAPTURE_ROUNDS -> None (every round) or a frozenset of 0-based round numbers."""
    text = (value or "").strip()
    if not text or text.lower() == ALL:
        return None
    rounds = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            number = int(token)
        except ValueError:
            raise MS6CaptureConfigError(
                f"{ROUNDS_ENV}={value!r}: {token!r} is not a round number (expected 'all' or e.g. '0,9,19')"
            ) from None
        if number < 0:
            raise MS6CaptureConfigError(f"{ROUNDS_ENV}={value!r}: rounds are 0-based, {number} is negative")
        rounds.add(number)
    if not rounds:
        raise MS6CaptureConfigError(f"{ROUNDS_ENV}={value!r} lists no round")
    return frozenset(rounds)


def parse_flag(name: str, value: Optional[str]) -> bool:
    text = (value or "").strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise MS6CaptureConfigError(f"{name}={value!r}: expected 0 or 1")


def resolve_capture_dir(configured: Optional[str], job_id: str) -> Path:
    """Job-level capture directory: <ODELIA_MS6_CAPTURE_DIR or /scratch/ms6>/<job_id>."""
    root = (configured or "").strip() or DEFAULT_CAPTURE_ROOT
    if any(char in root for char in "{}<>"):
        raise MS6CaptureConfigError(
            f"{DIR_ENV}={configured!r}: give the root directory without placeholders; the job id is appended"
        )
    if not os.path.isabs(root):
        # A relative path would resolve against the subprocess's working directory, i.e. inside the job
        # workspace, where nobody looks for it and where it is deleted with the workspace.
        raise MS6CaptureConfigError(f"{DIR_ENV}={configured!r} must be an absolute path")
    return Path(root) / (job_id or UNKNOWN_JOB_ID)


def _is_within(path, root) -> bool:
    path = os.path.realpath(str(path))
    root = os.path.realpath(str(root))
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def check_capture_dir_allowed(path, scratch_dir: Optional[str] = None, run_dir=None) -> None:
    """Refuse capture directories that live_sync would upload to TUD (it syncs $SCRATCH_DIR/runs)."""
    scratch_dir = scratch_dir or os.environ.get("SCRATCH_DIR") or "/scratch"
    forbidden = [Path(scratch_dir) / "runs"]
    if run_dir:
        forbidden.append(Path(run_dir))
    for root in forbidden:
        if _is_within(path, root):
            raise MS6CaptureConfigError(
                f"MS6 capture directory {path} lies inside {root}, which live_sync uploads to TUD: the site's "
                f"local models would leave the site. Use a directory outside it (default {DEFAULT_CAPTURE_ROOT})."
            )


# --------------------------------------------------------------------------------------------------
# Digests and sizes
# --------------------------------------------------------------------------------------------------


def _is_ignored(key: str, ignored: Iterable[str]) -> bool:
    return any(key == name or key.endswith("." + name) for name in ignored)


def _as_cpu_tensor(value):
    import numpy as np
    import torch

    if isinstance(value, np.ndarray):
        # Not np.ascontiguousarray: it turns a 0-d array into shape (1,), which would change the digest.
        array = value if value.flags["C_CONTIGUOUS"] else value.copy(order="C")
        with warnings.catch_warnings():  # read-only arrays are fine: the tensor is only read
            warnings.simplefilter("ignore", UserWarning)
            return torch.from_numpy(array)
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        return tensor.cpu() if tensor.device.type != "cpu" else tensor
    return None


def tensor_digest(state: Mapping[str, Any], ignored: Iterable[str] = DIGEST_IGNORED_KEYS) -> Tuple[str, int]:
    """Format-independent digest of a state dict (see TENSOR_DIGEST_DEFINITION). Returns (hexdigest, n).

    Two files holding the same tensors give the same digest however they were saved (an NVFlare persistor
    file, a capture file, a numpy learnable), which is what lets a site's G_r be matched against TUD's
    archive. Works tensor by tensor, so the extra memory is one tensor (for GPU or non-contiguous input).
    """
    import torch

    digest = hashlib.sha256()
    count = 0
    for key in sorted(state):
        if _is_ignored(key, ignored):
            continue
        tensor = _as_cpu_tensor(state[key])
        if tensor is None:  # not a tensor: fold in its repr so the digest still covers it
            digest.update(f"{key}|object|{type(state[key]).__name__}|{state[key]!r}".encode("utf-8"))
            count += 1
            continue
        tensor = tensor.contiguous()
        dtype = str(tensor.dtype).replace("torch.", "")
        shape = ",".join(str(size) for size in tensor.shape)
        digest.update(f"{key}|{dtype}|{shape}|".encode("utf-8"))
        flat = tensor.reshape(-1)
        if flat.dtype == torch.bfloat16:  # numpy has no bfloat16; the bytes are what matters
            flat = flat.view(torch.int16)
        if flat.numel():
            digest.update(flat.numpy())
        count += 1
    return digest.hexdigest(), count


def file_sha256(path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_nbytes(obj) -> int:
    """Bytes held by the tensors/arrays in a (nested) state structure."""
    import numpy as np
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, Mapping):
        return sum(tensor_nbytes(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_nbytes(value) for value in obj)
    return 0


def required_free_bytes(nbytes: int) -> int:
    return nbytes + max(MIN_FREE_RESERVE_BYTES, RESERVE_FACTOR * nbytes)


def _to_cpu(obj):
    """Copy of a nested optimizer state with every tensor on the CPU (so files load without map_location)."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, Mapping):
        return {key: _to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(value) for value in obj)
    return obj


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _format_bytes(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1000 or unit == "TB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1000.0
    return f"{nbytes:.1f} TB"


def _save_file(obj, path) -> None:
    import torch

    torch.save(obj, path)


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def round_file_name(round_number: int, kind: str) -> str:
    return f"round_{round_number:02d}_{kind}.pt"


# --------------------------------------------------------------------------------------------------
# Writer: one site directory, its manifest, best-effort writes
# --------------------------------------------------------------------------------------------------


class CaptureWriter:
    """Writes capture files into one site directory and keeps manifest.json in step with them."""

    def __init__(self, site_dir: Path, header: dict, logger=None):
        self.site_dir = Path(site_dir)
        self.logger = logger or _LOG
        self.manifest_path = self.site_dir / MANIFEST_NAME
        self.manifest = {
            "format": MANIFEST_FORMAT,
            "tensor_digest": TENSOR_DIGEST_DEFINITION,
            "files": [],
            "skipped": [],
            **header,
        }

    def open(self) -> None:
        """Create the directory and the manifest; keep the entries of an earlier start of the same job."""
        self.site_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            with open(self.manifest_path) as handle:
                previous = json.load(handle)
            if previous.get("job_id") != self.manifest.get("job_id"):
                raise MS6CaptureConfigError(
                    f"{self.manifest_path} belongs to job {previous.get('job_id')!r}, this is job "
                    f"{self.manifest.get('job_id')!r}; capturing here would mix two runs."
                )
            self.manifest["files"] = previous.get("files", [])
            self.manifest["skipped"] = previous.get("skipped", [])
        self.manifest["updated_at"] = _now()
        _write_json_atomic(self.manifest_path, self.manifest)

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.site_dir).free

    def _record(self, section: str, entry: dict) -> None:
        key = (entry["round"], entry["kind"])
        entries = [e for e in self.manifest[section] if (e.get("round"), e.get("kind")) != key]
        entries.append(entry)
        entries.sort(key=lambda e: (e["round"], e["kind"]))
        self.manifest[section] = entries
        if section == "files":  # a later success replaces an earlier skip of the same file
            self.manifest["skipped"] = [
                e for e in self.manifest["skipped"] if (e.get("round"), e.get("kind")) != key
            ]
        self.manifest["updated_at"] = _now()
        _write_json_atomic(self.manifest_path, self.manifest)

    def _skip(self, round_number: int, kind: str, reason: str, level=logging.WARNING) -> None:
        self.logger.log(level, f"MS6 capture: round {round_number} {kind} NOT kept: {reason}")
        try:
            self._record("skipped", {"round": round_number, "kind": kind, "reason": reason, "at": _now()})
        except Exception as exc:  # the manifest itself may be on the full disk
            self.logger.warning(f"MS6 capture: could not record the skip in {self.manifest_path}: {exc!r}")

    def save(self, round_number: int, kind: str, payload: dict, nbytes: int, meta: dict) -> Optional[dict]:
        """Save ``payload`` (plus meta under "ms6") as round_<rr>_<kind>.pt. Never raises; None if not kept."""
        name = round_file_name(round_number, kind)
        path = self.site_dir / name
        tmp = self.site_dir / f".{name}.tmp-{os.getpid()}"
        try:
            free = self.free_bytes()
            need = required_free_bytes(nbytes)
            if free < need:
                self._skip(
                    round_number, kind,
                    f"not enough free space in {self.site_dir}: {_format_bytes(free)} free, {_format_bytes(need)} "
                    f"needed ({_format_bytes(nbytes)} file plus the reserve for the training checkpoints)",
                )
                return None
            meta = dict(meta, round=round_number, kind=kind, saved_at=_now())
            _save_file({**payload, "ms6": meta}, tmp)
            os.replace(tmp, path)
            entry = dict(meta)
            entry.update(file=name, bytes=path.stat().st_size, sha256=file_sha256(path))
            self._record("files", entry)
            self.logger.info(
                f"MS6 capture: kept round {round_number} {kind} ({_format_bytes(entry['bytes'])}) as {path}, "
                f"sha256 {entry['sha256'][:16]}"
            )
            return entry
        except Exception as exc:
            try:
                tmp.unlink()
            except OSError:
                pass
            disk_full = isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT)
            self._skip(
                round_number, kind, f"write failed: {exc!r}", level=logging.WARNING if disk_full else logging.ERROR
            )
            return None


# --------------------------------------------------------------------------------------------------
# The round hook and its Lightning callback
# --------------------------------------------------------------------------------------------------


def _sent_num_steps(model, trainer):
    """The NUM_STEPS_CURRENT_ROUND that FLCallback sent with W_r (the weight the aggregator gave the update).

    FLCallback writes it into the module's ``__fl_meta__`` when the module has one (a hook may have set it,
    e.g. equal-weight aggregation). Otherwise it sends ``trainer.estimated_stepping_batches`` and, in
    reset_state right after the send, keeps the same number as ``total_local_steps`` (read before it extends
    the fit loop for the next round). Reading it after fit therefore needs no callback.
    """
    meta = getattr(model, "__fl_meta__", None)
    if isinstance(meta, dict) and "NUM_STEPS_CURRENT_ROUND" in meta:
        return meta["NUM_STEPS_CURRENT_ROUND"]
    for callback in getattr(trainer, "callbacks", None) or []:
        if type(callback).__name__ == "FLCallback":
            return getattr(callback, "total_local_steps", None)
    return None


class MS6CaptureCallback(Callback):
    """Keeps the optimizer and LR-scheduler state at the start of local training (ODELIA_MS6_CAPTURE_OPTIMIZER=1).

    on_train_start runs after RestoreState.on_fit_start put back the state the previous round ended with, so
    the file holds what this round's local training starts from (attack B replays the round from it). The
    callback only reads, so its place among the hook callbacks does not matter.
    """

    def __init__(self, hook: "MS6CaptureHook", ctx):
        super().__init__()
        self.hook = hook
        self.ctx = ctx

    def on_train_start(self, trainer, pl_module):
        ctx = self.ctx
        if not (self.hook.active and self.hook.capture_optimizer):
            return
        if ctx.is_eval_only_round or not self.hook.selects_round(ctx.current_round):
            return
        round_number = ctx.current_round
        try:
            state = {
                "optimizer_states": [_to_cpu(optimizer.state_dict()) for optimizer in trainer.optimizers],
                "lr_scheduler_states": [
                    _to_cpu(config.scheduler.state_dict()) for config in getattr(trainer, "lr_scheduler_configs", [])
                ],
            }
            scaler = getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
            if scaler is not None:
                state["grad_scaler_state"] = scaler.state_dict()
            meta = self.hook.file_meta(self.ctx)
            meta.update(epoch=trainer.current_epoch, global_step=trainer.global_step)
            self.hook.writer.save(round_number, KIND_OPTIMIZER, state, tensor_nbytes(state), meta)
        except Exception as exc:  # best effort: never fail training
            self.hook.logger.error(f"MS6 capture: optimizer state of round {round_number} not kept: {exc!r}")


class MS6CaptureHook(RoundHook):
    """Round hook that keeps G_r, W_r (and optionally the optimizer state) at the selected sites."""

    name = "ms6_capture"

    def __init__(self, sites, rounds, configured_dir: Optional[str], capture_optimizer: bool, env, logger=None):
        self.sites = sites
        self.rounds = rounds
        self.configured_dir = configured_dir
        self.capture_optimizer = capture_optimizer
        self.env = dict(env)
        self.logger = logger or _LOG
        self.model_name = self.env.get("MODEL_NAME") or "unknown"
        self.active = False
        self.writer: Optional[CaptureWriter] = None
        self.site_dir: Optional[Path] = None

    # selection ---------------------------------------------------------------------------------------

    def selects_site(self, site: str) -> bool:
        return self.sites == ALL or (bool(site) and site in self.sites)

    def selects_round(self, round_number) -> bool:
        if round_number is None:
            return False
        return self.rounds is None or round_number in self.rounds

    def model_payload(self, state) -> dict:
        """{"model": state} plus the persistor's train_conf record of the architecture, when it is known."""
        payload = {"model": state}
        if self.model_name and self.model_name != "unknown":
            payload["train_conf"] = {"train": {"model": self.model_name}}
        return payload

    def file_meta(self, ctx) -> dict:
        return {
            "job_id": ctx.job_id,
            "site": ctx.site_name,
            "model_name": self.model_name,
            "total_rounds": ctx.total_rounds,
            "is_final_round": bool(ctx.is_final_round),
        }

    # hook points -------------------------------------------------------------------------------------

    def on_start(self, ctx) -> None:
        site = ctx.site_name
        if not self.selects_site(site):
            self.logger.info(f"MS6 capture: {site or 'this site'} is not in {SITES_ENV}; nothing is kept here")
            return
        if not _SITE_NAME_RE.match(site or ""):
            raise MS6CaptureConfigError(f"MS6 capture: site name {site!r} cannot be used as a directory name")
        if not ctx.job_id:
            self.logger.warning("MS6 capture: the client API reports no job id; files go under " + UNKNOWN_JOB_ID)

        site_dir = resolve_capture_dir(self.configured_dir, ctx.job_id) / site
        check_capture_dir_allowed(site_dir, self.env.get("SCRATCH_DIR"), ctx.path_run_dir)

        env_vars = ctx.env_vars or {}
        self.model_name = self.env.get("MODEL_NAME") or env_vars.get("model_name") or "unknown"
        header = {
            "job_id": ctx.job_id or UNKNOWN_JOB_ID,
            "site": site,
            "model_name": self.model_name,
            "mediswarm_version": self.env.get("MEDISWARM_VERSION") or env_vars.get("mediswarm_version") or "unset",
            "capture_rounds": ALL if self.rounds is None else sorted(self.rounds),
            "capture_optimizer": self.capture_optimizer,
            "started_at": _now(),
        }
        writer = CaptureWriter(site_dir, header, self.logger)
        try:
            writer.open()
        except MS6CaptureConfigError:
            raise
        except Exception as exc:  # e.g. /scratch not writable: training goes on without capture
            self.logger.error(f"MS6 capture DISABLED at {site}: cannot use {site_dir}: {exc!r}")
            return
        self.writer = writer
        self.site_dir = site_dir
        self.active = True
        try:
            free = _format_bytes(writer.free_bytes())
        except OSError:
            free = "unknown"
        rounds = ALL if self.rounds is None else ",".join(str(r) for r in sorted(self.rounds))
        self.logger.info(
            f"MS6 capture active at {site}: rounds {rounds}, optimizer state "
            f"{'yes' if self.capture_optimizer else 'no'}, directory {site_dir} ({free} free)"
        )

    def lightning_callbacks(self, ctx) -> list:
        # Only the optimizer state needs a callback; G_r and W_r are taken at the round-hook points.
        return [MS6CaptureCallback(self, ctx)] if self.active and self.capture_optimizer else []

    def after_receive(self, ctx) -> None:
        if not self.active or not self.selects_round(ctx.current_round):
            return
        if ctx.is_eval_only_round:
            self.logger.info(
                f"MS6 capture: round {ctx.current_round} is the evaluation-only round; nothing is kept (its global "
                f"is the final global, FL_global_model.pt)"
            )
            return
        try:
            params = getattr(ctx.input_model, "params", None)
            if not params:
                self.logger.warning(f"MS6 capture: round {ctx.current_round} brought no global weights to keep")
                return
            digest, count = tensor_digest(params)
            meta = self.file_meta(ctx)
            meta.update(tensor_sha256=digest, n_tensors=count)
            payload = self.model_payload(dict(params))
            self.writer.save(ctx.current_round, KIND_GLOBAL, payload, tensor_nbytes(params), meta)
        except Exception as exc:  # best effort: never fail training
            self.logger.error(f"MS6 capture: global of round {ctx.current_round} not kept: {exc!r}")

    def after_round(self, ctx) -> None:
        """Keep W_r: the weights FLCallback.on_train_end has just sent (see the module docstring for why here)."""
        if not self.active or ctx.is_eval_only_round or not self.selects_round(ctx.current_round):
            return
        try:
            # FLCallback sent pl_module.cpu().state_dict(), so the module is on the CPU and .cpu() copies nothing;
            # it only guards the file against a module that stayed on the GPU (then the file still loads anywhere).
            state = {key: value.detach().cpu() for key, value in ctx.model.state_dict().items()}
            digest, count = tensor_digest(state)
            meta = self.file_meta(ctx)
            trainer = ctx.trainer
            meta.update(
                tensor_sha256=digest, n_tensors=count, num_steps=_sent_num_steps(ctx.model, trainer),
                epoch=getattr(trainer, "current_epoch", None), global_step=getattr(trainer, "global_step", None),
            )
            self.writer.save(ctx.current_round, KIND_LOCAL, self.model_payload(state), tensor_nbytes(state), meta)
        except Exception as exc:  # best effort: never fail training
            self.logger.error(f"MS6 capture: local model of round {ctx.current_round} not kept: {exc!r}")

    def on_end(self, ctx) -> None:
        if not self.active:
            return
        files = self.writer.manifest.get("files", [])
        skipped = self.writer.manifest.get("skipped", [])
        total = sum(entry.get("bytes", 0) for entry in files)
        message = (
            f"MS6 capture finished at {ctx.site_name}: {len(files)} files ({_format_bytes(total)}) in {self.site_dir}"
        )
        if skipped:
            self.logger.warning(f"{message}; {len(skipped)} NOT kept, see {self.writer.manifest_path}")
        else:
            self.logger.info(message)


def create_hook(env, logger=None):
    """Round-hook factory (round_hooks.load_hooks). None when no site is selected."""
    log = logger or _LOG
    sites = parse_sites(env.get(SITES_ENV, ""))
    rounds = parse_rounds(env.get(ROUNDS_ENV, ALL))
    capture_optimizer = parse_flag(OPTIMIZER_ENV, env.get(OPTIMIZER_ENV, "0"))
    configured_dir = env.get(DIR_ENV, "").strip() or None
    if configured_dir:
        # The configured path can be checked now, before the data is loaded; on_start checks the resolved one.
        check_capture_dir_allowed(resolve_capture_dir(configured_dir, UNKNOWN_JOB_ID), env.get("SCRATCH_DIR"))
    if sites is None:
        log.info(f"ms6_capture is listed but {SITES_ENV} names no site: nothing is captured")
        return None
    return MS6CaptureHook(sites, rounds, configured_dir, capture_optimizer, env, log)
