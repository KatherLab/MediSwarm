"""Evaluation-only final round, honest metric labels and export behaviour (release 1.9.0, contract 3.3/3.4).

CPU only. What runs for real: the BasicClassifier metric/logging code every ODELIA model inherits, a real
Lightning Trainer, main.py's swarm loop with NVFlare's real flare.patch()/FLCallback, the server-side
collectors (PerSiteMetricCollector, PerCasePredictionCollector), and NVFlare's own diff, aggregation,
shareable-generator and IntimeModelSelector classes. Faked: the client transport (FakeRuntime hands out one
task per round and records the answers), the ODELIA data stack (a 12-case in-memory data module) and the
streaming logger (RecordingLogger receives exactly what ClientLogger would stream to the coordinator).

What is pinned:
* the received global streams as global_val/*, the final global as final_val/* and final_test/*, the local
  model as val/*, and the coordinator files them under three owners;
* "val/AUC_ROC" still reaches FLCallback's metrics, which CCWF's best-global selection reads, and the
  selector accepts the evaluation round's result with it (and rejects a result without it);
* the evaluation-only round trains nothing, sends the received tensors back with NUM_STEPS_CURRENT_ROUND=1,
  and NVFlare turns that into an exact zero update and an identical final global;
* per-case rows are tagged by split, model kind and round and reproduce the streamed metrics;
* every export runs in eval mode and restores each submodule's mode (#618); the train-split pass can be
  switched off; per-step validation/test losses (per-case values) are no longer streamed.
"""

import csv
import importlib
import logging
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")
pytest.importorskip("torchmetrics")
nvflare_api = pytest.importorskip("nvflare.app_opt.lightning.api")
flare_lightning = pytest.importorskip("nvflare.client.lightning")

import nvflare.client as flare_client  # noqa: E402
from nvflare.app_common.abstract.fl_model import FLModel, MetaKey, ParamsType  # noqa: E402
from nvflare.client.config import ConfigKey  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers.logger import Logger  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from conftest import MODELS_DIR, SHARED_CUSTOM_DIR, import_module_from_path  # noqa: E402

base_model = import_module_from_path("_eval_round_base_model", MODELS_DIR / "base_model.py")

N_CASES = 12
SITE = "SITE_A"


# --------------------------------------------------------------------------------------------------
# A tiny ODELIA-shaped model and data
# --------------------------------------------------------------------------------------------------


class TinyClassifier(base_model.BasicClassifier):
    """The real BasicClassifier (metrics, support, logging, capture) around one linear layer."""

    def __init__(self):
        super().__init__(in_ch=4, out_ch=3, spatial_dims=1, optimizer_kwargs={"lr": 0.05})
        self.layer = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.layer(x.flatten(1))


def _cases(seed):
    generator = torch.Generator().manual_seed(seed)
    labels = [i % 3 for i in range(N_CASES)]
    x = torch.randn(N_CASES, 4, generator=generator)
    x[torch.arange(N_CASES), torch.tensor(labels)] += 1.0  # some signal, not separable
    return [{"uid": f"case_{seed}_{i}", "source": x[i], "target": torch.tensor([labels[i]])} for i in range(N_CASES)]


class TinyData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train, self.val, self.test = _cases(1), _cases(2), _cases(3)
        self.train_loader_calls = 0

    def train_dataloader(self):
        self.train_loader_calls += 1
        return DataLoader(self.train, batch_size=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1)


class RecordingLogger(Logger):
    """Stands in for NVFlare's ClientLogger: what reaches it is what streams to the coordinator."""

    def __init__(self, round_of=lambda: None, log_dir=None):
        super().__init__()
        self.records = []
        self._round_of = round_of
        self._log_dir = log_dir

    @property
    def name(self):
        return "recording"

    @property
    def version(self):
        return "0"

    @property
    def log_dir(self):
        return self._log_dir

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def log_metrics(self, metrics, step=None):
        self.records.append((self._round_of(), dict(metrics), step))

    def keys(self, round_num="any"):
        return {key for r, metrics, _ in self.records if round_num == "any" or r == round_num for key in metrics}


class FakeMetricWriter:
    """Stands in for ClientLogger's MLflowWriter, which carries the per-case chunks."""

    def __init__(self):
        self.texts = []

    def log_text(self, text, artifact_file_path):
        self.texts.append((artifact_file_path, text))


def _auroc_over_scorable_classes(rows, num_classes=3):
    """The model's own definition (#617): macro one-vs-rest AUROC over classes present and absent."""
    from torchmetrics.classification import MulticlassAUROC

    probs = torch.tensor([r["pred_prob"] for r in rows])
    target = torch.tensor([r["GT"] for r in rows])
    per_class = MulticlassAUROC(num_classes=num_classes, average=None)(probs, target)
    counts = [int((target == k).sum()) for k in range(num_classes)]
    scorable = [k for k, c in enumerate(counts) if 0 < c < len(rows)]
    return per_class[scorable].mean().item()


def _trainer(tmp_path, loggers, callbacks=(), max_epochs=1):
    return pl.Trainer(accelerator="cpu", devices=1, max_epochs=max_epochs, logger=list(loggers),
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                      default_root_dir=str(tmp_path), callbacks=list(callbacks), log_every_n_steps=1,
                      enable_checkpointing=any(isinstance(cb, ModelCheckpoint) for cb in callbacks))


# --------------------------------------------------------------------------------------------------
# BasicClassifier: labels, per-step losses, capture
# --------------------------------------------------------------------------------------------------


def test_relabelled_validation_streams_only_the_new_prefix_but_keeps_val_for_selection(tmp_path):
    torch.manual_seed(0)
    model, data, streamed = TinyClassifier(), TinyData(), RecordingLogger()
    trainer = _trainer(tmp_path, [streamed])
    model.set_metric_labels({"val": "global_val"}, {"round": 3})
    trainer.validate(model, datamodule=data, verbose=False)
    model.clear_metric_labels()

    keys = streamed.keys()
    assert {"global_val/AUC_ROC", "global_val/ACC", "global_val/n", "global_val/loss", "global_val/round"} <= keys
    assert not any(k.startswith("val/") for k in keys), "the global must not be filed under the local model"
    callback = trainer.callback_metrics
    # FLCallback sends these as initial_metrics; IntimeModelSelector selects on key_metric = "val/AUC_ROC".
    assert callback["val/AUC_ROC"].item() == pytest.approx(callback["global_val/AUC_ROC"].item())
    final_record = [m for _, m, _ in streamed.records if "global_val/round" in m][-1]
    assert final_record["global_val/round"] == 3.0


def test_unlabelled_validation_logs_val_as_before(tmp_path):
    model, data, streamed = TinyClassifier(), TinyData(), RecordingLogger()
    _trainer(tmp_path, [streamed]).validate(model, datamodule=data, verbose=False)
    keys = streamed.keys()
    assert {"val/AUC_ROC", "val/ACC", "val/n", "val/loss"} <= keys
    assert not any(k.startswith("global_val/") or k.endswith("/round") for k in keys)


def test_validation_and_test_losses_are_not_streamed_per_case(tmp_path):
    """At batch size 1 a per-step val/test loss is one case's loss; only training keeps per-step values."""
    model, data, streamed = TinyClassifier(), TinyData(), RecordingLogger()
    trainer = _trainer(tmp_path, [streamed])
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data, verbose=False)
    keys = streamed.keys()
    assert "train/loss_step" in keys and "val/loss" in keys and "test/loss" in keys
    assert not any(k.startswith(("val/", "test/")) and k.endswith(("_step", "_epoch")) for k in keys)


def test_captured_rows_are_the_forward_pass_the_metrics_come_from(tmp_path):
    torch.manual_seed(0)
    model, data, streamed = TinyClassifier(), TinyData(), RecordingLogger()
    trainer = _trainer(tmp_path, [streamed])
    model.set_metric_labels({"test": "final_test"}, {"round": 5})
    model.start_prediction_capture(("test",))
    trainer.test(model, datamodule=data, verbose=False)
    rows = model.stop_prediction_capture()["test"]
    model.clear_metric_labels()

    assert [r["GT"] for r in rows] == [i % 3 for i in range(N_CASES)], "loader order, one row per case"
    assert all(sum(r["pred_prob"]) == pytest.approx(1.0, abs=1e-6) for r in rows)
    streamed_auc = [m for _, m, _ in streamed.records if "final_test/AUC_ROC" in m][-1]["final_test/AUC_ROC"]
    assert _auroc_over_scorable_classes(rows) == pytest.approx(streamed_auc, abs=1e-6)
    assert model._prediction_capture is None


# --------------------------------------------------------------------------------------------------
# threedcnn_ptl helpers (imported against stubs of the ODELIA data stack)
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def threedcnn_ptl(monkeypatch):
    data_pkg = types.ModuleType("data")
    data_datasets = types.ModuleType("data.datasets")
    data_datamodules = types.ModuleType("data.datamodules")
    env_config = types.ModuleType("env_config")
    models_pkg = types.ModuleType("models")
    models_pkg.__path__ = []
    models_config = types.ModuleType("models.models_config")
    models_base_model = types.ModuleType("models.base_model")
    models_base_model.state_dict_mismatch = lambda *args, **kwargs: ""
    data_datasets.ODELIA_Dataset3D = object
    data_datamodules.DataModule = object
    for attr in ("build_odelia_manifests", "generate_run_directory", "load_environment_variables",
                 "prepare_odelia_dataset", "prepare_odelia_dataset_without_augmentation"):
        setattr(env_config, attr, lambda *args, **kwargs: None)
    models_config.create_model = lambda *args, **kwargs: None
    for name, module in {"data": data_pkg, "data.datasets": data_datasets, "data.datamodules": data_datamodules,
                         "env_config": env_config, "models": models_pkg, "models.models_config": models_config,
                         "models.base_model": models_base_model}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    for name in ("threedcnn_ptl", "per_case_predictions"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for variable in ("ODELIA_FINAL_EVAL_ROUND", "ODELIA_EXPORT_TRAIN_PREDICTIONS", "ODELIA_RETURN_PER_CASE"):
        monkeypatch.delenv(variable, raising=False)
    module = import_module_from_path("threedcnn_ptl", SHARED_CUSTOM_DIR / "threedcnn_ptl.py")
    yield module
    sys.modules.pop("threedcnn_ptl", None)
    sys.modules.pop("per_case_predictions", None)


class NormDropNet(torch.nn.Module):
    """BatchNorm + Dropout (#618: both behave differently in train mode) and one submodule frozen in eval."""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(4)
        self.drop = torch.nn.Dropout(0.5)
        self.frozen = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5))
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.drop(self.frozen(self.norm(x.flatten(1)))))

    @staticmethod
    def logits2probabilities(logits):
        return torch.softmax(logits, dim=1)


def test_exports_run_in_eval_mode_and_restore_every_submodule(threedcnn_ptl):
    torch.manual_seed(0)
    net = NormDropNet()
    with torch.no_grad():  # non-trivial running statistics
        net.norm.running_mean.fill_(0.3)
        net.norm.running_var.fill_(2.0)
    net.train()
    net.frozen.eval()  # deliberately frozen; must stay so
    flags = {name: module.training for name, module in net.named_modules()}
    stats = (net.norm.running_mean.clone(), net.norm.running_var.clone(), net.norm.num_batches_tracked.clone())
    loader = DataLoader(_cases(4), batch_size=4)

    exported = threedcnn_ptl._determine_GT_and_classprobs(net, loader, torch.device("cpu"))

    assert {name: module.training for name, module in net.named_modules()} == flags
    for before, after in zip(stats, (net.norm.running_mean, net.norm.running_var, net.norm.num_batches_tracked)):
        assert torch.equal(before, after), "an export must not update BatchNorm running statistics"
    reference = NormDropNet()
    reference.load_state_dict(net.state_dict())
    reference.eval()
    with torch.no_grad():
        expected = torch.cat([reference.logits2probabilities(reference(b["source"])) for b in loader])
    assert torch.allclose(torch.tensor([r["pred_prob"] for r in exported]), expected, atol=1e-6)
    assert [r["GT"] for r in exported] == [[i % 3] for i in range(N_CASES)]


class _ExportData:
    """Data module stand-in for the per-epoch export callback; fails if the train split is touched."""

    def __init__(self, allow_train):
        self.allow_train = allow_train

    def train_dataloader(self):
        if not self.allow_train:
            raise AssertionError("train split exported although ODELIA_EXPORT_TRAIN_PREDICTIONS=0")
        return DataLoader(_cases(5), batch_size=1)

    def val_dataloader(self):
        return DataLoader(_cases(6), batch_size=1)


@pytest.mark.parametrize("export_train", ["0", "1"])
def test_per_epoch_export_train_switch_and_tagged_return(threedcnn_ptl, monkeypatch, tmp_path, export_train):
    monkeypatch.setenv("ODELIA_EXPORT_TRAIN_PREDICTIONS", export_train)
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")
    writer = FakeMetricWriter()
    callback = threedcnn_ptl.GT_PredProb_Output_Callback(_ExportData(allow_train=export_train == "1"),
                                                         tmp_path / "train.csv", tmp_path / "val.csv", writer)
    callback.current_round = 4
    net = NormDropNet()
    net.train()
    callback.on_train_epoch_end(types.SimpleNamespace(current_epoch=7), net)

    assert (tmp_path / "train.csv").exists() == (export_train == "1")
    val_rows = list(csv.reader(open(tmp_path / "val.csv")))
    assert len(val_rows) == N_CASES and all(row[0] == "7" for row in val_rows)
    assert all(module.training for module in net.modules()), "train mode restored after the export"
    # The returned rows are the validation rows just written (one pass), tagged as the local model.
    import per_case_predictions

    tags = [per_case_predictions.parse_chunk_path(path) for path, _ in writer.texts]
    assert {(t["split"], t["model_kind"], t["round"], t["epoch"]) for t in tags} == {("val", "site_local", 4, 7)}
    returned = [line.split(",") for _, text in writer.texts for line in text.splitlines()]
    assert [r[1] for r in returned] == [row[1] for row in val_rows]
    assert [float(r[4]) for r in returned] == pytest.approx([float(row[4]) for row in val_rows], abs=1e-6)


@pytest.mark.parametrize("value,expected", [(None, False), ("", False), ("0", False), ("1", True),
                                            ("true", True), ("off", False)])
def test_final_eval_round_switch(threedcnn_ptl, monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("ODELIA_FINAL_EVAL_ROUND", value)
    assert threedcnn_ptl.final_eval_round_enabled() is expected


@pytest.mark.parametrize("variable", ["ODELIA_FINAL_EVAL_ROUND", "ODELIA_EXPORT_TRAIN_PREDICTIONS"])
def test_a_switch_that_is_not_0_or_1_stops_the_job(threedcnn_ptl, monkeypatch, variable):
    monkeypatch.setenv(variable, "l")
    with pytest.raises(ValueError, match=variable):
        threedcnn_ptl.final_eval_round_enabled()
        threedcnn_ptl.export_train_predictions_enabled()


def test_export_train_predictions_defaults_on(threedcnn_ptl):
    assert threedcnn_ptl.export_train_predictions_enabled() is True


def test_finalize_without_a_local_checkpoint_skips_the_local_exports(threedcnn_ptl, tmp_path, caplog):
    """A job whose only round is the evaluation round trained nothing: there is no local model to export."""
    checkpointing = types.SimpleNamespace(best_model_path="", last_model_path="")
    model = types.SimpleNamespace(load_checkpoint_from_file=None, load_best_checkpoint=None)
    with caplog.at_level(logging.INFO):
        threedcnn_ptl.finalize_training(logging.getLogger("t"), None, model, checkpointing, None, tmp_path,
                                        {"model_name": "MST"})
    assert "skipping the test-split export of the local models" in caplog.text
    assert "Training completed successfully." in caplog.text


# --------------------------------------------------------------------------------------------------
# main.py swarm loop with the evaluation-only round
# --------------------------------------------------------------------------------------------------


class FakeRuntime:
    """NVFlare client transport: one task per round; what is sent becomes the next round's global."""

    def __init__(self, total_rounds, initial_params, events):
        self.total_rounds = total_rounds
        self.global_params = {k: v.detach().clone() for k, v in initial_params.items()}
        self.events = events
        self.sent, self.handed_out = [], []
        self.current, self.round, self.next_round = None, None, 0

    def is_running(self):
        if self.current is None:
            if self.next_round >= self.total_rounds:
                return False
            self.round, self.next_round = self.next_round, self.next_round + 1
            self.current = FLModel(params={k: v.clone() for k, v in self.global_params.items()},
                                   current_round=self.round, total_rounds=self.total_rounds)
            self.handed_out.append(self.current)
        return True

    def receive(self, *args, **kwargs):
        if self.current is None:
            self.is_running()
        return self.current

    def send(self, model, clear_cache=True):
        self.events.append(("send", self.round))
        self.sent.append(model)
        self.global_params = {k: v.detach().clone() for k, v in model.params.items()}
        if clear_cache:
            self.current = None

    def clear(self):
        self.current = None


@pytest.fixture
def swarm(monkeypatch, tmp_path, threedcnn_ptl):
    events = []
    torch.manual_seed(0)
    model, data = TinyClassifier(), TinyData()
    runtime = FakeRuntime(total_rounds=3, initial_params=model.state_dict(), events=events)
    for name, value in {
        "init": lambda *args, **kwargs: None,
        "get_config": lambda: {ConfigKey.TASK_EXCHANGE: {ConfigKey.TRAIN_WITH_EVAL: True}},
        "receive": runtime.receive, "send": runtime.send, "clear": runtime.clear,
        "is_train": lambda: True, "is_evaluate": lambda: False, "is_submit_model": lambda: False,
    }.items():
        monkeypatch.setattr(nvflare_api, name, value)
    monkeypatch.setattr(flare_lightning, "receive", runtime.receive)
    monkeypatch.setattr(flare_lightning, "is_running", runtime.is_running)
    monkeypatch.setattr(flare_lightning, "get_site_name", lambda: SITE)
    monkeypatch.setattr(flare_lightning, "get_job_id", lambda: "job-eval")
    monkeypatch.setattr(flare_client, "init", lambda *args, **kwargs: None)
    # A hook module imported by a test and main.py must share one round_hooks module (isinstance check).
    monkeypatch.delitem(sys.modules, "round_hooks", raising=False)
    anomaly_mode_before = torch.is_anomaly_enabled()
    for variable in ("ODELIA_ROUND_HOOKS", "ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS", "KEY_METRIC"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TRAINING_MODE", "swarm")
    monkeypatch.setenv("ODELIA_FINAL_EVAL_ROUND", "1")
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    streamed = RecordingLogger(round_of=lambda: runtime.round, log_dir=str(tmp_path / "logs"))
    writer = FakeMetricWriter()
    csv_callback = threedcnn_ptl.GT_PredProb_Output_Callback(
        data, run_dir / threedcnn_ptl.FILENAME_GT_PREDPROB_SITE_MODEL_TRAIN,
        run_dir / threedcnn_ptl.FILENAME_GT_PREDPROB_SITE_MODEL_VALIDATION, metric_writer=writer)
    checkpointing = ModelCheckpoint(dirpath=str(run_dir), monitor="val/AUC_ROC", mode="max", save_last=True,
                                    save_top_k=1)
    trainer = _trainer(tmp_path, [streamed], [checkpointing, csv_callback], max_epochs=2)
    objects = types.SimpleNamespace(model=model, data=data, trainer=trainer, checkpointing=checkpointing,
                                    run_dir=run_dir, streamed=streamed, writer=writer)

    def prepare_training(logger, num_epochs, site_name, log_dataset_details, weighted_epochs=False):
        return data, model, checkpointing, trainer, run_dir, {"model_name": "Tiny"}

    def recorded(name, function):
        def wrapper(*args, **kwargs):
            events.append((name, runtime.round))
            return function(*args, **kwargs)
        return wrapper

    def export(model_, data_module, epoch, csv_train, csv_val):
        events.append(("export:" + Path(csv_val).name.split("_model")[0], runtime.round, csv_train is not None))
        return real_export(model_, data_module, epoch, csv_train, csv_val)

    real_export = threedcnn_ptl.output_GT_and_classprobs_csv_train_val
    monkeypatch.setattr(threedcnn_ptl, "set_up_logging", lambda: logging.getLogger("test_eval_round"))
    monkeypatch.setattr(threedcnn_ptl, "prepare_training", prepare_training)
    for name, short in (("validate_global_model", "validate"), ("train_local_model", "fit")):
        monkeypatch.setattr(threedcnn_ptl, name, recorded(short, getattr(threedcnn_ptl, name)))
    monkeypatch.setattr(trainer, "test", recorded("test", trainer.test))
    # The real finalize reloads Lightning checkpoints through the model's constructor, which this tiny model
    # does not support; it is covered on its own above.
    monkeypatch.setattr(threedcnn_ptl, "finalize_training", lambda *args, **kwargs: events.append(("finalize", runtime.round)))
    monkeypatch.setattr(threedcnn_ptl, "output_GT_and_classprobs_csv_train_val", export)

    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    monkeypatch.syspath_prepend(str(hook_dir))

    def write_hook(name, body):
        (hook_dir / f"{name}.py").write_text(textwrap.dedent(body))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()

    def run_main():
        main = import_module_from_path("odelia_main_eval_round", SHARED_CUSTOM_DIR / "main.py")
        main.main()

    yield types.SimpleNamespace(events=events, runtime=runtime, objects=objects, run_main=run_main,
                                write_hook=write_hook, ptl=threedcnn_ptl)
    torch.autograd.set_detect_anomaly(anomaly_mode_before)
    for name in ("odelia_main_eval_round", "round_hooks", "recording_hook"):
        sys.modules.pop(name, None)


def _rounds(events, name):
    return [event[1] for event in events if event[0] == name]


def test_eval_round_trains_nothing_and_returns_the_received_global_unchanged(swarm):
    swarm.run_main()
    events, runtime, objects = swarm.events, swarm.runtime, swarm.objects

    assert _rounds(events, "fit") == [0, 1]
    assert _rounds(events, "validate") == [0, 1, 2]
    assert _rounds(events, "test") == [2]
    assert _rounds(events, "send") == [0, 1, 2]
    assert _rounds(events, "finalize") == [2]
    # #542: the default final-round aggregated export moved to the evaluation round
    assert [e[1] for e in events if e[0] == "export:aggregated"] == [2]
    assert events.index(("finalize", 2)) < events.index(("send", 2)), "evaluate everything before answering"

    received, result = runtime.handed_out[2], runtime.sent[2]
    assert result.params is received.params, "the received tensors themselves go back"
    assert result.params_type == ParamsType.FULL
    assert result.meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 1
    # What the client API makes of it (ModelRegistry._prepare_param_diff uses this function): zeros.
    from nvflare.client.utils import numerical_params_diff

    diff = numerical_params_diff(original=received.params, new=result.params)
    assert set(diff) == set(received.params) and all(not torch.any(v) for v in diff.values())
    # Nothing was trained in round 2: the model still holds exactly the received global.
    state = objects.model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in received.params.items())

    # The result carries the global's validation metrics; val/AUC_ROC is the selector's key.
    assert result.metrics["val/AUC_ROC"] == pytest.approx(result.metrics["final_val/AUC_ROC"])
    assert result.metrics["final_val/round"] == 2.0
    for sent in runtime.sent[:2]:
        assert sent.metrics["val/AUC_ROC"] == pytest.approx(sent.metrics["global_val/AUC_ROC"])


def test_streamed_metrics_land_under_honest_owners(swarm):
    swarm.run_main()
    streamed = swarm.objects.streamed
    assert {k for k in streamed.keys(2) if "/" in k} == {
        k for k in streamed.keys(2) if k.startswith(("final_val/", "final_test/"))}
    assert {"global_val/AUC_ROC", "val/AUC_ROC", "train/loss_step"} <= streamed.keys(1)
    assert not any(k.startswith(("val/", "test/")) and k.endswith("_step") for k in streamed.keys())

    collector_module = import_module_from_path("_eval_round_per_site", SHARED_CUSTOM_DIR / "per_site_metrics.py")
    from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
    from nvflare.apis.dxo import from_shareable

    collector = collector_module.PerSiteMetricCollector()
    collector.log_info = collector.log_warning = lambda *args, **kwargs: None
    published = {}

    def fire_event(event_type, fl_ctx):
        published[fl_ctx.props["_model_owner_"]] = from_shareable(fl_ctx.props["_validation_result_"]).data

    collector.fire_event = fire_event
    ctx = types.SimpleNamespace(props={})
    ctx.set_prop = lambda key, value, private=True, sticky=False: ctx.props.__setitem__(key, value)
    collector.initialize(ctx)
    for _, metrics, step in streamed.records:  # what ClientLogger -> MLflowWriter puts on the wire
        record = AnalyticsData(key="metrics", value=metrics, data_type=AnalyticsDataType.METRICS, global_step=step)
        collector.save(ctx, record.to_dxo().to_shareable(), SITE)
    collector.finalize(ctx)

    assert set(published) == {"site_local_model", "swarm_global_received", "swarm_global"}
    assert published["site_local_model"]["val/round"] == 1.0          # last training round's local model
    assert published["swarm_global_received"]["global_val/round"] == 1.0
    final = published["swarm_global"]
    assert final["final_val/round"] == 2.0 and final["final_test/round"] == 2.0
    assert final["final_test/n"] == float(N_CASES)
    # the published final_val is what the selector saw for this global
    assert final["final_val/AUC_ROC"] == pytest.approx(swarm.runtime.sent[2].metrics["val/AUC_ROC"])


def test_per_case_rows_reach_the_coordinator_by_split_and_kind(swarm, tmp_path):
    swarm.run_main()
    run_dir = swarm.objects.run_dir
    collector_module = import_module_from_path("_eval_round_per_case", SHARED_CUSTOM_DIR / "per_case_predictions.py")
    from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName

    collector = collector_module.PerCasePredictionCollector()
    collector.log_info = collector.log_warning = collector.log_error = lambda *args, **kwargs: None
    collector.initialize(None)
    for path, text in swarm.objects.writer.texts:
        record = AnalyticsData(key="text", value=text, data_type=AnalyticsDataType.TEXT,
                               sender=LogWriterName.MLFLOW, path=path)
        collector.save(None, record.to_dxo().to_shareable(), SITE)
    workspace = types.SimpleNamespace(get_run_dir=lambda job_id: str(tmp_path / "server" / job_id))
    ctx = types.SimpleNamespace(get_job_id=lambda: "job-eval",
                                get_engine=lambda: types.SimpleNamespace(get_workspace=lambda: workspace))
    collector.finalize(ctx)

    out = tmp_path / "server" / "job-eval" / "cross_site_val" / "per_case_predictions"
    assert sorted(p.name for p in out.iterdir()) == [
        f"{SITE}__test__final_global.csv", f"{SITE}__val__final_global.csv", f"{SITE}__val__site_local.csv"]

    def read(name):
        with open(out / name) as fh:
            lines = [line for line in fh.read().splitlines() if not line.startswith("#")]
        return list(csv.DictReader(lines))

    labels = [str(i % 3) for i in range(N_CASES)]
    for name, kind, round_tag in ((f"{SITE}__val__site_local.csv", "site_local", "1"),
                                  (f"{SITE}__val__final_global.csv", "final_global", "2"),
                                  (f"{SITE}__test__final_global.csv", "final_global", "2")):
        rows = read(name)
        assert [r["ground_truth"] for r in rows] == labels
        assert {(r["model_kind"], r["round"]) for r in rows} == {(kind, round_tag)}
        assert set(rows[0]) == {"row_index", "ground_truth", "prob_class_0", "prob_class_1", "prob_class_2",
                                "split", "model_kind", "round"}, "no identifier column"

    # The returned final-global rows are the local CSV's rows and reproduce the streamed final_val AUROC.
    returned = read(f"{SITE}__val__final_global.csv")
    with open(run_dir / swarm.ptl.FILENAME_GT_PREDPROB_FINAL_GLOBAL_MODEL_VALIDATION) as fh:
        local = list(csv.DictReader(fh))
    assert [r["row_index"] for r in local] == [r["row_index"] for r in returned]
    assert [float(r["prob_class_2"]) for r in local] == pytest.approx(
        [float(r["prob_class_2"]) for r in returned], abs=1e-6)
    rows = [{"GT": int(r["ground_truth"]), "pred_prob": [float(r[f"prob_class_{k}"]) for k in range(3)]}
            for r in returned]
    assert _auroc_over_scorable_classes(rows) == pytest.approx(
        swarm.runtime.sent[2].metrics["final_val/AUC_ROC"], abs=1e-4)
    with open(run_dir / swarm.ptl.FILENAME_GT_PREDPROB_FINAL_GLOBAL_MODEL_TEST) as fh:
        assert next(csv.reader(fh)) == ["round", "row_index", "ground_truth", "prob_class_0", "prob_class_1",
                                        "prob_class_2"]


def test_export_train_predictions_off_skips_every_train_pass(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_EXPORT_TRAIN_PREDICTIONS", "0")
    swarm.run_main()
    exports = [e for e in swarm.events if e[0].startswith("export:")]
    assert exports and not any(with_train for _, _, with_train in exports)
    run_dir = swarm.objects.run_dir
    assert not (run_dir / swarm.ptl.FILENAME_GT_PREDPROB_SITE_MODEL_TRAIN).exists()
    assert not (run_dir / swarm.ptl.FILENAME_GT_PREDPROB_AGGREGATED_MODEL_TRAIN).exists()
    assert (run_dir / swarm.ptl.FILENAME_GT_PREDPROB_SITE_MODEL_VALIDATION).exists()
    assert (run_dir / swarm.ptl.FILENAME_GT_PREDPROB_AGGREGATED_MODEL_VALIDATION).exists()


def test_without_the_switch_every_round_trains_as_before(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_FINAL_EVAL_ROUND", "0")
    swarm.run_main()
    assert _rounds(swarm.events, "fit") == [0, 1, 2]
    assert _rounds(swarm.events, "test") == []
    assert _rounds(swarm.events, "finalize") == [2]
    assert [e[1] for e in swarm.events if e[0] == "export:aggregated"] == [2]


def test_a_single_evaluation_round_scores_the_initial_global(swarm):
    """num_rounds = 1 with the switch: an evaluation-only job (e.g. a warm-start baseline)."""
    swarm.runtime.total_rounds = 1
    swarm.run_main()
    assert _rounds(swarm.events, "fit") == []
    assert _rounds(swarm.events, "send") == [0]
    assert swarm.runtime.sent[0].params is swarm.runtime.handed_out[0].params


def test_a_failing_test_split_or_finalize_still_returns_the_global(swarm, monkeypatch, caplog):
    """The evaluation round's outputs are best effort: whatever fails, the received global goes back, so
    one site's unreadable test image cannot hold the consortium's last round until the learn-task timeout."""
    def broken_test(*args, **kwargs):
        swarm.events.append(("test", swarm.runtime.round))
        raise RuntimeError("unreadable test image")

    def broken_finalize(*args, **kwargs):
        swarm.events.append(("finalize", swarm.runtime.round))
        raise RuntimeError("cannot reload the best local checkpoint")

    monkeypatch.setattr(swarm.objects.trainer, "test", broken_test)
    monkeypatch.setattr(swarm.ptl, "finalize_training", broken_finalize)
    with caplog.at_level(logging.ERROR):
        swarm.run_main()

    assert _rounds(swarm.events, "test") == [2] and _rounds(swarm.events, "finalize") == [2]
    assert _rounds(swarm.events, "send") == [0, 1, 2]
    received, result = swarm.runtime.handed_out[2], swarm.runtime.sent[2]
    assert result.params is received.params and result.metrics["final_val/round"] == 2.0
    run_dir = swarm.objects.run_dir
    assert (run_dir / swarm.ptl.FILENAME_GT_PREDPROB_FINAL_GLOBAL_MODEL_VALIDATION).exists()
    assert not (run_dir / swarm.ptl.FILENAME_GT_PREDPROB_FINAL_GLOBAL_MODEL_TEST).exists()
    returned = {tuple(path.split("/")[1:3]) for path, _ in swarm.objects.writer.texts if "/round_2/" in path}
    assert returned == {("val", "final_global")}, "no test rows from a test pass that did not finish"
    assert not any(k.startswith("final_test/") for k in swarm.objects.streamed.keys())
    errors = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    assert any(m.startswith("Final global test failed") for m in errors)
    assert any(m.startswith("Local checkpoint test export (finalize) failed") for m in errors)


def test_a_failing_validation_of_the_received_global_fails_the_round(swarm, monkeypatch):
    """The one step that is not optional: it loads the global, and the result must carry its metrics."""
    validate = swarm.ptl.validate_global_model

    def validate_fails_in_the_last_round(*args, **kwargs):
        if swarm.runtime.round == 2:
            raise RuntimeError("Failed to load model state dict")
        return validate(*args, **kwargs)

    monkeypatch.setattr(swarm.ptl, "validate_global_model", validate_fails_in_the_last_round)
    with pytest.raises(RuntimeError, match="Failed to load model state dict"):
        swarm.run_main()
    assert _rounds(swarm.events, "send") == [0, 1]


RECORDING_HOOK = """
from round_hooks import RoundHook

EVENTS = []


class _Hook(RoundHook):
    name = "recording"

    def after_receive(self, ctx):
        EVENTS.append(("after_receive", ctx.current_round, ctx.is_final_round, ctx.is_eval_only_round))

    def before_fit(self, ctx):
        EVENTS.append(("before_fit", ctx.current_round))

    def after_round(self, ctx):
        EVENTS.append(("after_round", ctx.current_round))

    def on_end(self, ctx):
        EVENTS.append(("on_end", ctx.current_round, ctx.input_model.params is not None))


def create_hook(env, logger):
    return _Hook()
"""


def test_hooks_in_the_evaluation_round(swarm, monkeypatch):
    swarm.write_hook("recording_hook", RECORDING_HOOK)
    hook = importlib.import_module("recording_hook")
    monkeypatch.setenv("ODELIA_ROUND_HOOKS", "recording_hook")
    swarm.run_main()
    assert hook.EVENTS == [
        ("after_receive", 0, False, False), ("before_fit", 0), ("after_round", 0),
        ("after_receive", 1, False, False), ("before_fit", 1), ("after_round", 1),
        # evaluation only: no fit, so neither before_fit nor after_round; the received params stay readable
        ("after_receive", 2, True, True), ("on_end", 2, True),
    ]


# --------------------------------------------------------------------------------------------------
# NVFlare: zero update -> identical global; best-global selection reads val/AUC_ROC
# --------------------------------------------------------------------------------------------------


def _global_weights():
    """A global as the evaluation round receives it. Integer buffers are float64 by then: NVFlare's weighted
    average promotes them in every aggregation (v * weight), which is not specific to this round."""
    import numpy as np

    rng = np.random.default_rng(7)
    return {
        "layer.weight": rng.standard_normal((3, 4)).astype(np.float32),
        "layer.bias": rng.standard_normal(3).astype(np.float32),
        "norm.num_batches_tracked": np.array(1234.0, dtype=np.float64),
    }


def _result_shareable(params_np, metrics, site, round_num):
    """The eval round's answer as it leaves the client: diff in the subprocess, numpy on the wire."""
    from nvflare.apis.fl_constant import ReservedKey
    from nvflare.app_common.app_constant import AppConstants
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils
    from nvflare.app_opt.pt.numpy_params_converter import PTToNumpyParamsConverter
    from nvflare.client.config import ClientConfig, TransferType
    from nvflare.client.model_registry import ModelRegistry
    from nvflare.client.flare_agent_with_fl_model import _ConverterContext

    received = FLModel(params={k: torch.as_tensor(v) for k, v in params_np.items()}, current_round=round_num)
    registry = ModelRegistry(ClientConfig({ConfigKey.TASK_EXCHANGE: {ConfigKey.EXCHANGE_FORMAT: "pytorch",
                                                                     ConfigKey.TRANSFER_TYPE: TransferType.DIFF}}),
                             rank="0", flare_agent=None)
    registry.received_task = types.SimpleNamespace(data=received)
    answer = FLModel(params_type=ParamsType.FULL, params=received.params, metrics=metrics,
                     meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1})
    diff = registry._prepare_param_diff(answer)
    assert diff.params_type == ParamsType.DIFF
    shareable = FLModelUtils.to_shareable(diff)
    shareable = PTToNumpyParamsConverter(["train"]).process("train", shareable, _ConverterContext())
    shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, round_num)
    shareable.set_peer_props({ReservedKey.IDENTITY_NAME: site})
    return shareable


def test_unchanged_weights_are_a_zero_update_and_the_final_global_is_identical():
    import numpy as np
    from nvflare.apis.dxo import DataKind, from_shareable
    from nvflare.apis.event_type import EventType
    from nvflare.apis.fl_context import FLContext
    from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
    from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
    from nvflare.app_common.app_constant import AppConstants
    from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator

    global_np = _global_weights()
    fl_ctx = FLContext()
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 2, private=True, sticky=True)
    aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind="WEIGHT_DIFF")  # as config_fed_client.conf
    aggregator.handle_event(EventType.START_RUN, fl_ctx)  # builds its per-key aggregators, as at job start
    for site in ("SITE_A", "SITE_B"):
        shareable = _result_shareable(global_np, {"val/AUC_ROC": 0.8}, site, 2)
        dxo = from_shareable(shareable)
        assert dxo.data_kind == DataKind.WEIGHT_DIFF and dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND) == 1
        assert all(not np.any(v) for v in dxo.data.values()), "an exact zero update for every key"
        assert aggregator.accept(shareable, fl_ctx)
    aggregated = aggregator.aggregate(fl_ctx)

    base = make_model_learnable({k: v.copy() for k, v in global_np.items()}, {})
    fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, base, private=True, sticky=True)
    final = SimpleModelShareableGenerator().shareable_to_learnable(aggregated, fl_ctx)[ModelLearnableKey.WEIGHTS]
    assert set(final) == set(global_np)
    for key, value in global_np.items():
        assert np.asarray(final[key]).tobytes() == value.tobytes(), f"{key} changed"


def _select(metrics_per_site, round_num=2):
    from nvflare.apis.fl_constant import FLContextKey, ReservedKey
    from nvflare.apis.fl_context import FLContext
    from nvflare.app_common.app_constant import AppConstants
    from nvflare.app_common.app_event_type import AppEventType
    from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector

    # key_metric as configured in every ODELIA job (config_fed_client.conf, test_key_metric_consistency.py)
    selector = IntimeModelSelector(key_metric="val/AUC_ROC")
    best = []
    selector.fire_event = lambda event_type, fl_ctx: best.append(fl_ctx.get_prop(AppConstants.VALIDATION_RESULT)) \
        if event_type == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE else None
    accepted = []
    fl_ctx = FLContext()
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, round_num, private=True, sticky=True)
    for site, metrics in metrics_per_site.items():
        peer = FLContext()
        peer.put(ReservedKey.IDENTITY_NAME, site, private=False, sticky=False)
        peer.set_prop(FLContextKey.SHAREABLE, _result_shareable(_global_weights(), metrics, site, round_num),
                      private=True, sticky=False)
        fl_ctx.set_peer_context(peer)
        accepted.append(selector._before_accept(fl_ctx))
    selector._before_aggregate(fl_ctx)
    return accepted, best


def test_best_global_selection_sees_the_final_global():
    """The evaluation round's result is a candidate for CCWF's best global, with the global's val/AUC_ROC."""
    accepted, best = _select({
        "SITE_A": {"val/AUC_ROC": 0.80, "final_val/AUC_ROC": 0.80, "final_val/round": 2.0},
        "SITE_B": {"val/AUC_ROC": 0.70, "final_val/AUC_ROC": 0.70, "final_val/round": 2.0},
    })
    assert accepted == [True, True]
    assert best == [pytest.approx(0.75)]


def test_renaming_the_selection_key_would_silently_disable_best_global_selection():
    """Why set_metric_labels keeps val/* in callback_metrics: without it the selector drops every result."""
    accepted, best = _select({"SITE_A": {"final_val/AUC_ROC": 0.80}, "SITE_B": {"final_val/AUC_ROC": 0.70}})
    assert accepted == [False, False] and best == []
