"""Tests of the MS6 per-round capture hook (application/jobs/_shared/custom/ms6_capture.py), CPU only.

The end-to-end tests run main.py for real on a tiny LightningModule with a real Lightning Trainer, the real
threedcnn_ptl round functions and NVFlare's real flare.patch()/FLCallback/RestoreState; only the client
transport is replaced (FakeClientRuntime hands out one global per round and records what FLCallback sends).
That is what makes "W_r is the model as released" checkable: the file must equal what FLCallback sent.
"""

import errno
import importlib
import json
import logging
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")
nvflare_api = pytest.importorskip("nvflare.app_opt.lightning.api")
flare_lightning = pytest.importorskip("nvflare.client.lightning")

import nvflare.client as flare_client  # noqa: E402
from nvflare.app_common.abstract.fl_model import FLModel  # noqa: E402
from nvflare.client.config import ConfigKey  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from conftest import import_module_from_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
JOB_ID = "job-0001"


# --------------------------------------------------------------------------------------------------
# Tiny Lightning stack and fake transport
# --------------------------------------------------------------------------------------------------


class TinyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 3)
        # A buffer named like the ODELIA class weights: left out of the tensor digest.
        self.register_buffer("_class_weight", torch.ones(3))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.cross_entropy(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val/loss", torch.nn.functional.cross_entropy(self.layer(x), y))

    def configure_optimizers(self):
        # Momentum gives the optimizer a state that RestoreState carries from round to round.
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)


class TinyData(pl.LightningDataModule):
    def __init__(self, n=8):
        super().__init__()
        generator = torch.Generator().manual_seed(n)
        self.data = TensorDataset(
            torch.randn(n, 4, generator=generator), torch.randint(0, 3, (n,), generator=generator)
        )

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=4)


class FakeClientRuntime:
    """One task per round; what FLCallback sends becomes the next global (no aggregation partner)."""

    def __init__(self, total_rounds, initial_params):
        self.total_rounds = total_rounds
        self.global_params = {k: v.detach().clone() for k, v in initial_params.items()}
        self.sent = []
        self.handed_out = []
        self.current = None
        self.round = None
        self.next_round = 0

    def is_running(self):
        if self.current is None:
            if self.next_round >= self.total_rounds:
                return False
            self.round = self.next_round
            self.next_round += 1
            self.current = FLModel(
                params={k: v.clone() for k, v in self.global_params.items()},
                current_round=self.round,
                total_rounds=self.total_rounds,
            )
            self.handed_out.append(self.current)
        return True

    def receive(self, *args, **kwargs):
        if self.current is None:
            self.is_running()
        return self.current

    def send(self, model, clear_cache=True):
        # Snapshot: the module's tensors keep changing after the send.
        self.sent.append(FLModel(params={k: v.detach().clone() for k, v in model.params.items()}, meta=model.meta))
        # Next global = the sent weights plus a fixed offset, so G_{r+1} differs from W_r like a real aggregate.
        self.global_params = {k: v.detach().clone() + 0.25 for k, v in model.params.items()}
        if clear_cache:
            self.current = None

    def clear(self):
        self.current = None


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Everything main.py needs, with the client transport and the ODELIA data stack replaced."""
    torch.manual_seed(0)
    model = TinyModule()
    runtime = FakeClientRuntime(total_rounds=3, initial_params=model.state_dict())

    for name, value in {
        "init": lambda *args, **kwargs: None,
        "get_config": lambda: {ConfigKey.TASK_EXCHANGE: {ConfigKey.TRAIN_WITH_EVAL: True}},
        "receive": runtime.receive,
        "send": runtime.send,
        "clear": runtime.clear,
        "is_train": lambda: True,
        "is_evaluate": lambda: False,
        "is_submit_model": lambda: False,
    }.items():
        monkeypatch.setattr(nvflare_api, name, value)
    monkeypatch.setattr(flare_lightning, "receive", runtime.receive)
    monkeypatch.setattr(flare_lightning, "is_running", runtime.is_running)
    monkeypatch.setattr(flare_lightning, "get_site_name", lambda: "SITE_A")
    monkeypatch.setattr(flare_lightning, "get_job_id", lambda: JOB_ID)
    monkeypatch.setattr(flare_client, "init", lambda *args, **kwargs: None)
    anomaly_mode_before = torch.is_anomaly_enabled()

    for variable in ("ODELIA_ROUND_HOOKS", "ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS", "ODELIA_MS6_CAPTURE_SITES",
                     "ODELIA_MS6_CAPTURE_ROUNDS", "ODELIA_MS6_CAPTURE_DIR", "ODELIA_MS6_CAPTURE_OPTIMIZER"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TRAINING_MODE", "swarm")
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path / "scratch"))
    monkeypatch.setenv("MODEL_NAME", "Tiny")

    # threedcnn_ptl imports the ODELIA data/model packages at module level; stub them like test_main_round_loop.
    data_pkg = types.ModuleType("data")
    data_datasets = types.ModuleType("data.datasets")
    data_datamodules = types.ModuleType("data.datamodules")
    env_config = types.ModuleType("env_config")
    models_pkg = types.ModuleType("models")
    models_config = types.ModuleType("models.models_config")
    data_datasets.ODELIA_Dataset3D = object
    data_datamodules.DataModule = object
    for attr in ("build_odelia_manifests", "generate_run_directory", "load_environment_variables",
                 "prepare_odelia_dataset", "prepare_odelia_dataset_without_augmentation"):
        setattr(env_config, attr, lambda *args, **kwargs: None)
    models_config.create_model = lambda *args, **kwargs: None
    for name, module in {"data": data_pkg, "data.datasets": data_datasets, "data.datamodules": data_datamodules,
                         "env_config": env_config, "models": models_pkg, "models.models_config": models_config}.items():
        monkeypatch.setitem(sys.modules, name, module)
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    monkeypatch.syspath_prepend(str(hook_dir))  # test doubles (dp_noise) shadow real modules
    for name in ("threedcnn_ptl", "round_hooks", "ms6_capture", "dp_noise", "odelia_main_under_test"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    threedcnn_ptl = import_module_from_path("threedcnn_ptl", SHARED_CUSTOM_DIR / "threedcnn_ptl.py")
    ms6_capture = importlib.import_module("ms6_capture")
    # The free-space reserve is about real checkpoints on a real scratch disk; keep it out of the way here.
    monkeypatch.setattr(ms6_capture, "MIN_FREE_RESERVE_BYTES", 0)

    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, default_root_dir=str(tmp_path),
        callbacks=[ModelCheckpoint(dirpath=str(tmp_path / "ckpt"), save_last=True)],
    )
    objects = types.SimpleNamespace(model=model, trainer=trainer, data_module=TinyData(),
                                    checkpointing=trainer.checkpoint_callback, path_run_dir=tmp_path / "run",
                                    env_vars={"model_name": "Tiny"})
    events = []

    def prepare_training(logger, num_epochs, site_name, log_dataset_details, weighted_epochs=False):
        events.append("prepare_training")
        return (objects.data_module, objects.model, objects.checkpointing, objects.trainer,
                objects.path_run_dir, objects.env_vars)

    monkeypatch.setattr(threedcnn_ptl, "set_up_logging", lambda: logging.getLogger("test_ms6_capture"))
    monkeypatch.setattr(threedcnn_ptl, "prepare_training", prepare_training)
    monkeypatch.setattr(threedcnn_ptl, "finalize_training", lambda *args, **kwargs: events.append("finalize"))
    monkeypatch.setattr(threedcnn_ptl, "output_GT_and_classprobs_csv_train_val", lambda *args, **kwargs: None)

    def write_hook(name, body):
        (hook_dir / f"{name}.py").write_text(textwrap.dedent(body))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()

    def run_main():
        main = import_module_from_path("odelia_main_under_test", SHARED_CUSTOM_DIR / "main.py")
        main.main()
        return main

    capture_root = tmp_path / "ms6"
    yield types.SimpleNamespace(runtime=runtime, objects=objects, events=events, run_main=run_main,
                                write_hook=write_hook, ms6_capture=ms6_capture, capture_root=capture_root,
                                site_dir=capture_root / JOB_ID / "SITE_A", tmp_path=tmp_path)
    torch.autograd.set_detect_anomaly(anomaly_mode_before)
    for name in ("odelia_main_under_test", "threedcnn_ptl", "round_hooks", "ms6_capture", "dp_noise"):
        sys.modules.pop(name, None)


def _enable_capture(monkeypatch, harness, sites="SITE_A", rounds=None, optimizer=False, hooks="ms6_capture"):
    monkeypatch.setenv("ODELIA_ROUND_HOOKS", hooks)
    monkeypatch.setenv("ODELIA_MS6_CAPTURE_SITES", sites)
    monkeypatch.setenv("ODELIA_MS6_CAPTURE_DIR", str(harness.capture_root))  # files go to <root>/<job_id>/<SITE>
    if rounds is not None:
        monkeypatch.setenv("ODELIA_MS6_CAPTURE_ROUNDS", rounds)
    if optimizer:
        monkeypatch.setenv("ODELIA_MS6_CAPTURE_OPTIMIZER", "1")


def _load_model(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def _assert_same_tensors(a, b):
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


# --------------------------------------------------------------------------------------------------
# End to end through main.py
# --------------------------------------------------------------------------------------------------


def test_capture_keeps_received_global_and_released_local_model_every_round(harness, monkeypatch):
    _enable_capture(monkeypatch, harness, optimizer=True)
    runtime, site_dir, ms6 = harness.runtime, harness.site_dir, harness.ms6_capture
    local_file_at_send = []

    def send(model, clear_cache=True):
        local_file_at_send.append((site_dir / f"round_{runtime.round:02d}_local.pt").exists())
        runtime.send(model, clear_cache)

    monkeypatch.setattr(nvflare_api, "send", send)
    harness.run_main()

    assert len(runtime.sent) == 3
    # W_r is written after FLCallback sent it: the capture never delays the update.
    assert local_file_at_send == [False, False, False]
    for r in range(3):
        # G_r is exactly the global handed to the trainer, W_r exactly what FLCallback sent.
        _assert_same_tensors(_load_model(site_dir / f"round_{r:02d}_global.pt"), runtime.handed_out[r].params)
        _assert_same_tensors(_load_model(site_dir / f"round_{r:02d}_local.pt"), runtime.sent[r].params)
        assert (site_dir / f"round_{r:02d}_optimizer.pt").exists()

    manifest = json.loads((site_dir / "manifest.json").read_text())
    assert manifest["job_id"] == JOB_ID and manifest["site"] == "SITE_A" and manifest["model_name"] == "Tiny"
    assert manifest["skipped"] == []
    entries = {(e["round"], e["kind"]): e for e in manifest["files"]}
    assert set(entries) == {(r, k) for r in range(3) for k in ("global", "local", "optimizer")}
    for (r, kind), entry in entries.items():
        path = site_dir / entry["file"]
        assert entry["sha256"] == ms6.file_sha256(path)
        assert entry["bytes"] == path.stat().st_size
        assert entry["job_id"] == JOB_ID and entry["total_rounds"] == 3
        if kind != "optimizer":
            assert entry["tensor_sha256"] == ms6.tensor_digest(_load_model(path))[0]
    # NUM_STEPS is what FLCallback reports to the aggregator (its aggregation weight).
    for r in range(3):
        assert entries[(r, "local")]["num_steps"] == runtime.sent[r].meta["NUM_STEPS_CURRENT_ROUND"]
    # The file is self-describing too.
    saved = torch.load(site_dir / "round_01_local.pt", weights_only=False)
    meta = saved["ms6"]
    assert meta["round"] == 1 and meta["kind"] == "local" and meta["site"] == "SITE_A"
    # The persistor's architecture record, which predict.py checks (MODEL_NAME of the training subprocess).
    assert saved["train_conf"] == {"train": {"model": "Tiny"}}

    # Optimizer state at the START of local training: empty in round 0, the momentum RestoreState put back
    # from round 0 in round 1.
    opt0 = torch.load(site_dir / "round_00_optimizer.pt", weights_only=False)["optimizer_states"][0]
    opt1 = torch.load(site_dir / "round_01_optimizer.pt", weights_only=False)["optimizer_states"][0]
    assert opt0["state"] == {}
    assert opt1["state"] and all(s["momentum_buffer"].abs().sum() > 0 for s in opt1["state"].values())

    names = [type(cb).__name__ for cb in harness.objects.trainer.callbacks]
    assert names.index("MS6CaptureCallback") == names.index("FLCallback") - 1


def test_capture_does_not_change_training(harness, monkeypatch, tmp_path):
    """Same seed, same data: the updates sent with capture equal the ones sent without it."""
    _enable_capture(monkeypatch, harness, optimizer=True)
    initial = {k: v.clone() for k, v in harness.runtime.global_params.items()}
    harness.run_main()
    with_capture = [m.params for m in harness.runtime.sent]

    # Second run on a fresh module with the same initial weights and no hooks.
    monkeypatch.delenv("ODELIA_ROUND_HOOKS")
    for variable in ("ODELIA_MS6_CAPTURE_SITES", "ODELIA_MS6_CAPTURE_DIR", "ODELIA_MS6_CAPTURE_OPTIMIZER"):
        monkeypatch.delenv(variable)
    model = TinyModule()
    model.load_state_dict(initial)
    runtime = FakeClientRuntime(total_rounds=3, initial_params=initial)
    for name, value in {"receive": runtime.receive, "send": runtime.send, "clear": runtime.clear}.items():
        monkeypatch.setattr(nvflare_api, name, value)
    monkeypatch.setattr(flare_lightning, "receive", runtime.receive)
    monkeypatch.setattr(flare_lightning, "is_running", runtime.is_running)
    harness.objects.model = model
    harness.objects.trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, default_root_dir=str(tmp_path / "second"),
        callbacks=[ModelCheckpoint(dirpath=str(tmp_path / "ckpt2"), save_last=True)],
    )
    harness.objects.checkpointing = harness.objects.trainer.checkpoint_callback
    sys.modules.pop("odelia_main_under_test", None)
    harness.run_main()

    assert len(runtime.sent) == 3
    for a, b in zip(with_capture, (m.params for m in runtime.sent)):
        _assert_same_tensors(a, b)


def test_only_selected_sites_and_rounds_are_captured(harness, monkeypatch):
    _enable_capture(monkeypatch, harness, rounds="1")
    harness.run_main()
    assert sorted(p.name for p in harness.site_dir.iterdir()) == [
        "manifest.json", "round_01_global.pt", "round_01_local.pt",
    ]
    assert len(harness.runtime.sent) == 3
    # G_r and W_r are taken at the round-hook points; only the optimizer state needs a Lightning callback.
    assert "MS6CaptureCallback" not in [type(cb).__name__ for cb in harness.objects.trainer.callbacks]


def test_a_site_that_is_not_selected_keeps_nothing(harness, monkeypatch):
    _enable_capture(monkeypatch, harness, sites="CAM_1,RUMC_1")
    harness.run_main()
    assert not harness.capture_root.exists()
    assert "MS6CaptureCallback" not in [type(cb).__name__ for cb in harness.objects.trainer.callbacks]
    assert len(harness.runtime.sent) == 3


DP_NOISE_DOUBLE = """
import torch
import pytorch_lightning as pl
from round_hooks import RoundHook


class _Noise(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        with torch.no_grad():
            for p in pl_module.parameters():
                p.add_(1000.0)


class _Hook(RoundHook):
    name = "dp_noise"

    def lightning_callbacks(self, ctx):
        return [_Noise()]


def create_hook(env, logger):
    return _Hook()
"""


@pytest.mark.parametrize("hooks", ["ms6_capture,dp_noise", "dp_noise,ms6_capture"])
def test_with_dp_noise_the_capture_is_the_noisy_released_model_in_either_order(harness, monkeypatch, hooks):
    """ms6_capture,dp_noise is the job builder's order (al_subset, ms6_capture, dp_noise).

    A capture callback would run before dp_noise's callback in that order and keep the pre-noise model; W_r
    taken in after_round is what FLCallback sent in both orders.
    """
    harness.write_hook("dp_noise", DP_NOISE_DOUBLE)
    _enable_capture(monkeypatch, harness, optimizer=True, hooks=hooks)
    harness.run_main()
    for r in range(3):
        local = _load_model(harness.site_dir / f"round_{r:02d}_local.pt")
        _assert_same_tensors(local, harness.runtime.sent[r].params)
        assert local["layer.bias"].min() > 500  # the noise is in what was kept, as in what was sent


def test_a_failing_write_never_fails_training(harness, monkeypatch, caplog):
    def disk_full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    _enable_capture(monkeypatch, harness)
    monkeypatch.setattr(harness.ms6_capture, "_save_file", disk_full)
    with caplog.at_level(logging.WARNING):
        harness.run_main()
    assert len(harness.runtime.sent) == 3
    assert harness.events[-1] == "finalize"
    manifest = json.loads((harness.site_dir / "manifest.json").read_text())
    assert manifest["files"] == []
    skipped = {(e["round"], e["kind"]) for e in manifest["skipped"]}
    assert skipped == {(r, k) for r in range(3) for k in ("global", "local")}
    assert not [p for p in harness.site_dir.iterdir() if p.name != "manifest.json"]  # no partial files
    assert "NOT kept" in caplog.text


def test_not_enough_free_space_skips_with_a_warning(harness, monkeypatch, caplog):
    _enable_capture(monkeypatch, harness)
    monkeypatch.setattr(harness.ms6_capture.CaptureWriter, "free_bytes", lambda self: 10)
    with caplog.at_level(logging.WARNING):
        harness.run_main()
    assert len(harness.runtime.sent) == 3
    manifest = json.loads((harness.site_dir / "manifest.json").read_text())
    assert manifest["files"] == [] and len(manifest["skipped"]) == 6
    assert all("not enough free space" in e["reason"] for e in manifest["skipped"])
    assert any(r.levelno == logging.WARNING and "not enough free space" in r.getMessage() for r in caplog.records)


def test_capture_dir_inside_the_live_sync_tree_is_refused(harness, monkeypatch):
    _enable_capture(monkeypatch, harness)
    monkeypatch.setenv("ODELIA_MS6_CAPTURE_DIR", str(harness.tmp_path / "scratch" / "runs" / "SITE_A" / "ms6"))
    with pytest.raises(harness.ms6_capture.MS6CaptureConfigError, match="live_sync"):
        harness.run_main()
    assert harness.events == []


# --------------------------------------------------------------------------------------------------
# Hook pieces in isolation
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def ms6(monkeypatch):
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    for name in ("round_hooks", "ms6_capture"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("ms6_capture")
    monkeypatch.setattr(module, "MIN_FREE_RESERVE_BYTES", 0)
    yield module
    for name in ("round_hooks", "ms6_capture"):
        sys.modules.pop(name, None)


def test_parse_sites(ms6):
    assert ms6.parse_sites("") is None and ms6.parse_sites(" , ") is None
    assert ms6.parse_sites("all") == ms6.ALL and ms6.parse_sites("ALL") == ms6.ALL
    assert ms6.parse_sites("CAM_1, RUMC_1") == frozenset({"CAM_1", "RUMC_1"})
    for bad in ("all,CAM_1", "../etc", "CAM 1"):
        with pytest.raises(ms6.MS6CaptureConfigError):
            ms6.parse_sites(bad)


def test_parse_rounds(ms6):
    assert ms6.parse_rounds(None) is None and ms6.parse_rounds("all") is None and ms6.parse_rounds("") is None
    assert ms6.parse_rounds("0, 9,19") == frozenset({0, 9, 19})
    for bad in ("1,two", "-1", ",", "1.5"):
        with pytest.raises(ms6.MS6CaptureConfigError):
            ms6.parse_rounds(bad)


def test_parse_flag(ms6):
    assert ms6.parse_flag("X", "1") and ms6.parse_flag("X", "true")
    assert not ms6.parse_flag("X", "0") and not ms6.parse_flag("X", "") and not ms6.parse_flag("X", None)
    with pytest.raises(ms6.MS6CaptureConfigError):
        ms6.parse_flag("X", "maybe")


def test_capture_dir_resolution_and_guard(ms6, tmp_path):
    assert ms6.resolve_capture_dir(None, "abc") == Path("/scratch/ms6/abc")
    assert ms6.resolve_capture_dir("/data/ms6", "abc") == Path("/data/ms6/abc")  # the job id is always appended
    assert ms6.resolve_capture_dir("/data/ms6", "") == Path("/data/ms6/unknown_job")
    with pytest.raises(ms6.MS6CaptureConfigError, match="absolute"):
        ms6.resolve_capture_dir("relative/ms6", "abc")
    # NVFlare str.format()s the launcher line and leaves it unformatted on an unknown {name}: no placeholders.
    for placeholder in ("/data/ms6/{job_id}", "/data/ms6/<job_id>"):
        with pytest.raises(ms6.MS6CaptureConfigError, match="placeholders"):
            ms6.resolve_capture_dir(placeholder, "abc")
    scratch = tmp_path / "scratch"
    ms6.check_capture_dir_allowed(scratch / "ms6" / "job", str(scratch))
    ms6.check_capture_dir_allowed(scratch / "runs_other", str(scratch))
    for forbidden in (scratch / "runs", scratch / "runs" / "SITE" / "x"):
        with pytest.raises(ms6.MS6CaptureConfigError):
            ms6.check_capture_dir_allowed(forbidden, str(scratch))
    with pytest.raises(ms6.MS6CaptureConfigError):
        ms6.check_capture_dir_allowed(tmp_path / "run" / "ms6", str(scratch), run_dir=tmp_path / "run")


def test_create_hook(ms6, tmp_path):
    base = {"ODELIA_ROUND_HOOKS": "ms6_capture", "SCRATCH_DIR": str(tmp_path)}
    assert ms6.create_hook(dict(base)) is None  # no site: nothing to do
    hook = ms6.create_hook(dict(base, ODELIA_MS6_CAPTURE_SITES="CAM_1", ODELIA_MS6_CAPTURE_ROUNDS="0,19"))
    assert hook.selects_site("CAM_1") and not hook.selects_site("RUMC_1")
    assert hook.selects_round(19) and not hook.selects_round(1) and not hook.selects_round(None)
    assert not hook.capture_optimizer
    assert ms6.create_hook(dict(base, ODELIA_MS6_CAPTURE_SITES="all")).selects_site("ANY_1")
    with pytest.raises(ms6.MS6CaptureConfigError):
        ms6.create_hook(dict(base, ODELIA_MS6_CAPTURE_SITES="CAM_1", ODELIA_MS6_CAPTURE_ROUNDS="x"))
    for order in ("al_subset,ms6_capture,dp_noise", "al_subset,dp_noise,ms6_capture"):  # W_r is order-independent
        assert ms6.create_hook(dict(base, ODELIA_ROUND_HOOKS=order, ODELIA_MS6_CAPTURE_SITES="CAM_1")) is not None
    with pytest.raises(ms6.MS6CaptureConfigError, match="live_sync"):
        ms6.create_hook(dict(base, ODELIA_MS6_CAPTURE_SITES="CAM_1",
                             ODELIA_MS6_CAPTURE_DIR=str(tmp_path / "runs")))


def test_tensor_digest(ms6):
    np = pytest.importorskip("numpy")
    state = {
        "b": torch.arange(6, dtype=torch.float32).reshape(2, 3), "a": torch.tensor(3), "_class_weight": torch.ones(3)
    }
    digest, count = ms6.tensor_digest(state)
    assert count == 2
    # Independent of dict order, of ignored buffers, and of how the tensors are held (numpy, non-contiguous).
    assert ms6.tensor_digest({"a": torch.tensor(3), "b": state["b"].t().contiguous().t()})[0] == digest
    assert ms6.tensor_digest({"a": np.array(3), "b": state["b"].numpy()})[0] == digest
    assert ms6.tensor_digest(dict(state, _class_weight=torch.zeros(3)))[0] == digest
    assert ms6.tensor_digest({"model._class_weight": torch.zeros(1), **state})[0] == digest
    # Sensitive to values, dtype, shape and names.
    assert ms6.tensor_digest(dict(state, a=torch.tensor(4)))[0] != digest
    assert ms6.tensor_digest(dict(state, b=state["b"].double()))[0] != digest
    assert ms6.tensor_digest(dict(state, b=state["b"].reshape(3, 2)))[0] != digest
    assert ms6.tensor_digest({"a": torch.tensor(3), "c": state["b"]})[0] != digest
    assert ms6.tensor_digest({"x": torch.ones(2, dtype=torch.bfloat16)})[1] == 1


class _Ctx:
    """Minimal RoundContext for direct hook calls."""

    def __init__(self, tmp_path, round_number=0, eval_only=False):
        self.site_name = "SITE_A"
        self.job_id = JOB_ID
        self.env_vars = {"model_name": "Tiny"}
        self.path_run_dir = tmp_path / "run"
        self.current_round = round_number
        self.total_rounds = 3
        self.is_final_round = round_number == 2
        self.is_eval_only_round = eval_only
        self.input_model = FLModel(params={"w": torch.ones(2, 2)}, current_round=round_number)
        self.model = None
        self.trainer = None


def _hook(ms6, tmp_path, **env):
    values = {"ODELIA_ROUND_HOOKS": "ms6_capture", "ODELIA_MS6_CAPTURE_SITES": "SITE_A",
              "ODELIA_MS6_CAPTURE_DIR": str(tmp_path / "ms6"), "SCRATCH_DIR": str(tmp_path / "scratch")}
    values.update(env)
    return ms6.create_hook(values)


def test_the_eval_only_round_keeps_nothing(ms6, tmp_path):
    """Contract 3.3: no MS6 capture in the evaluation-only round; its global is the final global."""
    hook = _hook(ms6, tmp_path, ODELIA_MS6_CAPTURE_OPTIMIZER="1")
    ctx = _Ctx(tmp_path, round_number=2, eval_only=True)
    hook.on_start(ctx)
    (callback,) = hook.lightning_callbacks(ctx)
    hook.after_receive(ctx)
    ctx.model = torch.nn.Linear(2, 2)
    ctx.trainer = types.SimpleNamespace(optimizers=[torch.optim.SGD(ctx.model.parameters(), lr=0.1)],
                                        lr_scheduler_configs=[], current_epoch=0, global_step=0, callbacks=[])
    callback.on_train_start(ctx.trainer, ctx.model)
    hook.after_round(ctx)  # main.py does not call it in that round; harmless if something did
    hook.on_end(ctx)
    site_dir = tmp_path / "ms6" / JOB_ID / "SITE_A"
    assert sorted(p.name for p in site_dir.iterdir()) == ["manifest.json"]
    manifest = json.loads((site_dir / "manifest.json").read_text())
    assert manifest["files"] == [] and manifest["skipped"] == []


def test_sent_num_steps(ms6):
    """The aggregation weight FLCallback sent: the module's __fl_meta__ first, else FLCallback's own record."""

    class FLCallback:
        total_local_steps = 7

    trainer = types.SimpleNamespace(callbacks=[object(), FLCallback()])
    model = types.SimpleNamespace()
    assert ms6._sent_num_steps(model, trainer) == 7
    model.__fl_meta__ = {"NUM_STEPS_CURRENT_ROUND": 1}  # e.g. equal-weight aggregation set by a hook
    assert ms6._sent_num_steps(model, trainer) == 1
    assert ms6._sent_num_steps(types.SimpleNamespace(), types.SimpleNamespace(callbacks=[])) is None


def test_a_manifest_of_another_job_is_not_overwritten(ms6, tmp_path):
    site_dir = tmp_path / "ms6" / JOB_ID / "SITE_A"
    site_dir.mkdir(parents=True)
    (site_dir / "manifest.json").write_text(json.dumps({"job_id": "older-job", "files": [{"round": 0}]}))
    hook = _hook(ms6, tmp_path)
    with pytest.raises(ms6.MS6CaptureConfigError, match="older-job"):
        hook.on_start(_Ctx(tmp_path))
    assert json.loads((site_dir / "manifest.json").read_text())["job_id"] == "older-job"


def test_a_restart_of_the_same_job_keeps_the_earlier_entries(ms6, tmp_path):
    hook = _hook(ms6, tmp_path)
    hook.on_start(_Ctx(tmp_path))
    hook.after_receive(_Ctx(tmp_path, round_number=0))
    again = _hook(ms6, tmp_path)
    again.on_start(_Ctx(tmp_path))
    again.after_receive(_Ctx(tmp_path, round_number=1))
    manifest = json.loads((tmp_path / "ms6" / JOB_ID / "SITE_A" / "manifest.json").read_text())
    assert [(e["round"], e["kind"]) for e in manifest["files"]] == [(0, "global"), (1, "global")]


def test_an_unwritable_capture_dir_disables_capture_without_failing(ms6, tmp_path, caplog):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    hook = _hook(ms6, tmp_path, ODELIA_MS6_CAPTURE_DIR=str(blocker))
    with caplog.at_level(logging.ERROR):
        hook.on_start(_Ctx(tmp_path))
    assert not hook.active and hook.lightning_callbacks(_Ctx(tmp_path)) == []
    hook.after_receive(_Ctx(tmp_path))  # no exception
    assert "DISABLED" in caplog.text
