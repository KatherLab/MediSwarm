"""Tests of the swarm round loop in application/jobs/_shared/custom/main.py (CPU only).

main.py runs for real on a tiny LightningModule with a real Lightning Trainer, the real
threedcnn_ptl round functions and NVFlare's real flare.patch()/FLCallback. Only the client transport
(receive/send/is_running and friends) is replaced by FakeClientRuntime, which hands out one task per
round and records what comes back. prepare_training() and finalize_training() are stubbed: they need
the ODELIA data stack.

Checks: without hooks the loop does what it always did (validate the received global, train, the
FLCallback sends once per round, finalize in the last round); with hooks every call point is reached in
the documented order, hook callbacks run before the send, and objects a hook replaces in the
RoundContext are the ones the loop uses afterwards.
"""

import importlib
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


# --------------------------------------------------------------------------------------------------
# Tiny Lightning stack
# --------------------------------------------------------------------------------------------------


class TinyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.cross_entropy(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val/loss", torch.nn.functional.cross_entropy(self.layer(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class TinyData(pl.LightningDataModule):
    def __init__(self, n=8):
        super().__init__()
        generator = torch.Generator().manual_seed(n)
        self.data = TensorDataset(torch.randn(n, 4, generator=generator), torch.randint(0, 2, (n,), generator=generator))
        self.train_loader_calls = 0

    def train_dataloader(self):
        self.train_loader_calls += 1
        return DataLoader(self.data, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=4)


class FakeClientRuntime:
    """Stands in for the NVFlare client API: one task per round; what FLCallback sends becomes the next global."""

    def __init__(self, total_rounds, initial_params, events):
        self.total_rounds = total_rounds
        self.global_params = {k: v.detach().clone() for k, v in initial_params.items()}
        self.events = events
        self.sent = []
        self.handed_out = []  # one FLModel per round, as main.py's flare.receive() returns it
        self.current = None
        self.round = None
        self.next_round = 0

    # main.py side
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
        # Like the client API: the same task until it is answered (FLCallback receives it again).
        if self.current is None:
            self.is_running()
        return self.current

    # FLCallback side
    def send(self, model, clear_cache=True):
        self.events.append(("send", self.round))
        self.sent.append(model)
        self.global_params = {k: v.detach().clone() for k, v in model.params.items()}
        if clear_cache:
            self.current = None

    def clear(self):
        self.current = None


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Everything main.py needs, with the client transport and the ODELIA data stack replaced."""
    events = []
    model = TinyModule()
    runtime = FakeClientRuntime(total_rounds=3, initial_params=model.state_dict(), events=events)

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
    monkeypatch.setattr(flare_lightning, "get_job_id", lambda: "job-0001")
    monkeypatch.setattr(flare_client, "init", lambda *args, **kwargs: None)
    # main.py switches anomaly detection on process-wide (Lightning uses the same function as a context
    # manager, so it cannot be stubbed); the previous mode is restored after the test.
    anomaly_mode_before = torch.is_anomaly_enabled()

    for variable in ("ODELIA_ROUND_HOOKS", "ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TRAINING_MODE", "swarm")

    # threedcnn_ptl imports the ODELIA data/model packages at module level; stub them like test_threedcnn_ptl.
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
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    for name in ("threedcnn_ptl", "round_hooks", "odelia_main_under_test"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    threedcnn_ptl = import_module_from_path("threedcnn_ptl", SHARED_CUSTOM_DIR / "threedcnn_ptl.py")

    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, default_root_dir=str(tmp_path),
        callbacks=[ModelCheckpoint(dirpath=str(tmp_path / "ckpt"), save_last=True)],
    )
    objects = types.SimpleNamespace(
        model=model, trainer=trainer, data_module=TinyData(), checkpointing=trainer.checkpoint_callback,
        path_run_dir=tmp_path / "run", env_vars={"model_name": "Tiny"},
        validate_calls=[], fit_calls=[], finalize_calls=[],
    )

    def prepare_training(logger, num_epochs, site_name, log_dataset_details, weighted_epochs=False):
        events.append(("prepare_training", site_name, weighted_epochs))
        return (objects.data_module, objects.model, objects.checkpointing, objects.trainer,
                objects.path_run_dir, objects.env_vars)

    real_validate, real_train = threedcnn_ptl.validate_global_model, threedcnn_ptl.train_local_model

    def validate_global_model(logger, data_module, model, trainer, path_run_dir, *args, **kwargs):
        events.append(("validate", runtime.round))
        objects.validate_calls.append((data_module, model, trainer))
        real_validate(logger, data_module, model, trainer, path_run_dir, *args, **kwargs)

    def train_local_model(logger, data_module, model, trainer, *args, **kwargs):
        events.append(("fit", runtime.round))
        objects.fit_calls.append((data_module, model, trainer))
        real_train(logger, data_module, model, trainer, *args, **kwargs)

    def finalize_training(logger, data_module, model, checkpointing, trainer, path_run_dir, env_vars):
        events.append(("finalize", runtime.round))
        objects.finalize_calls.append((data_module, model, checkpointing, trainer, path_run_dir, env_vars))

    monkeypatch.setattr(threedcnn_ptl, "set_up_logging", lambda: logging.getLogger("test_main_round_loop"))
    monkeypatch.setattr(threedcnn_ptl, "prepare_training", prepare_training)
    monkeypatch.setattr(threedcnn_ptl, "validate_global_model", validate_global_model)
    monkeypatch.setattr(threedcnn_ptl, "train_local_model", train_local_model)
    monkeypatch.setattr(threedcnn_ptl, "finalize_training", finalize_training)
    monkeypatch.setattr(threedcnn_ptl, "output_GT_and_classprobs_csv_train_val",
                        lambda *args, **kwargs: events.append(("export", runtime.round)))

    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    monkeypatch.syspath_prepend(str(hook_dir))
    written = []

    def write_hook(name, body):
        (hook_dir / f"{name}.py").write_text(textwrap.dedent(body))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        written.append(name)

    def run_main():
        main = import_module_from_path("odelia_main_under_test", SHARED_CUSTOM_DIR / "main.py")
        main.main()
        return main

    yield types.SimpleNamespace(events=events, runtime=runtime, objects=objects, write_hook=write_hook,
                                run_main=run_main)
    torch.autograd.set_detect_anomaly(anomaly_mode_before)
    # Drop the modules imported against stubs; monkeypatch then restores any earlier entries.
    for name in written + ["odelia_main_under_test", "threedcnn_ptl", "round_hooks"]:
        sys.modules.pop(name, None)


# A hook module that logs every call point into the shared event list (set by the test as EVENTS),
# returns a Lightning callback that records on_train_end and changes the model before the send, and
# records what the RoundContext holds at after_receive.
RECORDING_HOOK = """
import torch
import pytorch_lightning as pl
from round_hooks import RoundHook

EVENTS = []
SEEN = []


class Marker(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        EVENTS.append(("hook_callback.on_train_end", None))
        with torch.no_grad():
            pl_module.layer.bias.fill_(42.0)


class _Hook(RoundHook):
    name = "recording"

    def on_start(self, ctx):
        EVENTS.append(("on_start", ctx.current_round))

    def lightning_callbacks(self, ctx):
        EVENTS.append(("lightning_callbacks", ctx.current_round))
        return [Marker()]

    def after_receive(self, ctx):
        EVENTS.append(("after_receive", ctx.current_round))
        SEEN.append(dict(round=ctx.current_round, total=ctx.total_rounds, final=ctx.is_final_round,
                         eval_only=ctx.is_eval_only_round, input_model=ctx.input_model,
                         site=ctx.site_name, job=ctx.job_id))

    def before_fit(self, ctx):
        EVENTS.append(("before_fit", ctx.current_round))

    def after_round(self, ctx):
        EVENTS.append(("after_round", ctx.current_round))

    def on_end(self, ctx):
        EVENTS.append(("on_end", ctx.current_round))


def create_hook(env, logger):
    return _Hook()
"""


def test_loop_without_hooks_is_unchanged_and_fl_callback_sends_every_round(harness):
    torch.autograd.set_detect_anomaly(False)
    harness.run_main()

    assert torch.is_anomaly_enabled()  # as before: main.py switches it on in swarm mode
    assert harness.events == [
        ("prepare_training", "SITE_A", True),
        ("validate", 0), ("fit", 0), ("send", 0),
        ("validate", 1), ("fit", 1), ("send", 1),
        # default export throttle: the aggregated-model export runs in the final round only
        ("validate", 2), ("export", 2), ("fit", 2), ("send", 2),
        ("finalize", 2),
    ]
    runtime = harness.runtime
    assert len(runtime.sent) == 3
    for sent in runtime.sent:
        assert set(sent.params) == set(harness.objects.model.state_dict())
        assert "val/loss" in sent.metrics
    # The trainer carries exactly NVFlare's own callbacks after the checkpoint: nothing was inserted.
    names = [type(cb).__name__ for cb in harness.objects.trainer.callbacks]
    assert names[-3:] == ["ModelCheckpoint", "FLCallback", "RestoreState"]


def test_hooks_run_at_the_defined_points(harness, monkeypatch):
    harness.write_hook("recording_hook", RECORDING_HOOK)
    hook_module = importlib.import_module("recording_hook")
    hook_module.EVENTS = harness.events  # one timeline for hook calls, loop steps and sends
    monkeypatch.setenv("ODELIA_ROUND_HOOKS", "recording_hook")

    harness.run_main()

    def round_events(r, final=False):
        events = [("after_receive", r), ("validate", r)]
        if final:
            events.append(("export", r))
        events += [("before_fit", r), ("fit", r), ("hook_callback.on_train_end", None), ("send", r),
                   ("after_round", r)]
        if final:
            events += [("finalize", r), ("on_end", r)]
        return events

    assert harness.events == [
        ("prepare_training", "SITE_A", True),
        ("on_start", None),
        ("lightning_callbacks", None),
        *round_events(0), *round_events(1), *round_events(2, final=True),
    ]
    # What the hook's callback did in on_train_end is what FLCallback sent.
    for sent in harness.runtime.sent:
        assert torch.all(sent.params["layer.bias"] == 42.0)
    names = [type(cb).__name__ for cb in harness.objects.trainer.callbacks]
    assert names.index("Marker") == names.index("FLCallback") - 1

    seen = hook_module.SEEN
    assert [s["round"] for s in seen] == [0, 1, 2]
    assert [s["total"] for s in seen] == [3, 3, 3]
    assert [s["final"] for s in seen] == [False, False, True]
    assert [s["eval_only"] for s in seen] == [False, False, False]
    assert len(seen) == len(harness.runtime.handed_out)
    assert all(s["input_model"] is handed for s, handed in zip(seen, harness.runtime.handed_out))
    assert {(s["site"], s["job"]) for s in seen} == {("SITE_A", "job-0001")}


REPLACING_HOOK = """
import types
from round_hooks import RoundHook

NEW = types.SimpleNamespace(data_module=None, path_run_dir=None)


class _Hook(RoundHook):
    name = "replacing"

    def __init__(self, data_module_factory):
        self.data_module_factory = data_module_factory

    def before_fit(self, ctx):
        # Like an active-learning subset: from round 0's fit on, train on a different data module.
        if ctx.current_round == 0:
            NEW.data_module = self.data_module_factory()
            ctx.data_module = NEW.data_module

    def after_receive(self, ctx):
        if ctx.is_final_round:
            NEW.path_run_dir = ctx.path_run_dir / "replaced"
            ctx.path_run_dir = NEW.path_run_dir


FACTORY = None


def create_hook(env, logger):
    return _Hook(FACTORY)
"""


def test_objects_replaced_in_the_context_are_used_by_the_loop(harness, monkeypatch):
    harness.write_hook("replacing_hook", REPLACING_HOOK)
    hook_module = importlib.import_module("replacing_hook")
    hook_module.FACTORY = lambda: TinyData(n=4)
    monkeypatch.setenv("ODELIA_ROUND_HOOKS", "replacing_hook")

    harness.run_main()

    new_dm = hook_module.NEW.data_module
    objects = harness.objects
    assert new_dm is not None and new_dm is not objects.data_module
    # Round 0 validated the global on the original data module; its fit and every later round used the
    # replacement, and the real trainer.fit() really pulled its loaders from it.
    assert [call[0] for call in objects.validate_calls] == [objects.data_module, new_dm, new_dm]
    assert [call[0] for call in objects.fit_calls] == [new_dm, new_dm, new_dm]
    assert new_dm.train_loader_calls >= 1
    assert [call[4] for call in objects.finalize_calls] == [hook_module.NEW.path_run_dir]
    assert len(harness.runtime.sent) == 3


def test_safety_net_stops_the_job_before_prepare_training(harness, monkeypatch):
    monkeypatch.setenv("ODELIA_DP_SIGMA", "2.4")
    with pytest.raises(RuntimeError, match="dp_noise"):
        harness.run_main()
    assert harness.events == []
    assert harness.runtime.sent == []


def test_failing_hook_fails_the_round(harness, monkeypatch):
    harness.write_hook(
        "failing_hook",
        """
        from round_hooks import RoundHook

        class _Hook(RoundHook):
            name = "failing"

            def before_fit(self, ctx):
                raise RuntimeError("privacy budget exhausted")

        def create_hook(env, logger):
            return _Hook()
        """,
    )
    monkeypatch.setenv("ODELIA_ROUND_HOOKS", "failing_hook")
    with pytest.raises(RuntimeError, match="privacy budget exhausted"):
        harness.run_main()
    assert ("fit", 0) not in harness.events
    assert harness.runtime.sent == []


def test_local_training_ignores_hooks(harness, monkeypatch):
    monkeypatch.setenv("TRAINING_MODE", "local_training")
    monkeypatch.setenv("SITE_NAME", "SITE_LOCAL")
    monkeypatch.setenv("ODELIA_ROUND_HOOKS", "no_such_hook_module_xyz")
    harness.run_main()
    assert harness.events == [
        ("prepare_training", "SITE_LOCAL", False),
        ("validate", None), ("fit", None),
        ("finalize", None),
    ]
