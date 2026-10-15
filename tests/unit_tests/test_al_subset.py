"""Tests of the al_subset round hook (D3.2 selection, ODELIA_TRAIN_INCLUDE_LIST).

Three layers, all on CPU:

* configuration and the round-hook safety net (pure Python);
* the dataset/DataModule plumbing against the real ODELIA_Dataset3D (built with __new__, as in
  test_odelia_degenerate_input_guard.py) and the real DataModule: subset views, the eval-transform pool
  view, scoring with an unreadable input, the cached-loader reset and worker cap, class weights;
* main.py end to end with a real Lightning Trainer and NVFlare's real flare.patch()/FLCallback (only the
  client transport is faked, as in test_main_round_loop.py): the global is loaded before before_fit, the
  selected cases are the only ones trained on, the epochs per round follow the subset in this and every
  later round (FLCallback's carry-over), class weights, records without UIDs in the run directory,
  streamed aggregates; and, through the implicit reproducibility hook load_hooks adds (unit tests in
  test_reproducibility.py), the round-0 digest check and ODELIA_SEED reproducibility of a whole run.
"""

import importlib
import json
import logging
import sys
import types
from pathlib import Path

import pytest

from conftest import SHARED_CUSTOM_DIR, import_module_from_path

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

H_MODULES = ("al_subset", "al_selection", "reproducibility", "state_digest", "loader_workers", "round_hooks")


@pytest.fixture
def al(monkeypatch):
    """al_subset imported the way main.py's load_hooks imports it (custom/ on sys.path)."""
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    for name in H_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("al_subset")
    yield module
    for name in H_MODULES:
        sys.modules.pop(name, None)


SWARM = {"TRAINING_MODE": "swarm"}
DIGEST = "ab" * 32


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_nothing_set_means_not_active(al):
    assert al.create_hook({}, logging.getLogger("t")) is None


def test_full_selection_config(al):
    config = al.parse_config({
        "ODELIA_AL_STRATEGY": "Class_Matched_Random", "ODELIA_AL_BUDGET": "20", "ODELIA_AL_SEED": "3",
        "ODELIA_AL_POOL_SPLIT": "test", "ODELIA_AL_SELECT_ROUND": "1", "ODELIA_AL_CLASS_WEIGHTS": "keep",
        "ODELIA_SEED": "11", "ODELIA_WARM_START_SHA256": DIGEST, "ODELIA_FINAL_EVAL_ROUND": "1",
    })
    assert (config.strategy, config.budget, config.al_seed) == ("class_matched_random", 20, 3)
    assert config.pool_split == "test"
    assert (config.select_round, config.class_weights) == (1, "keep")
    assert config.final_eval_round and config.selects


def test_defaults(al):
    config = al.parse_config({"ODELIA_AL_STRATEGY": "entropy", "ODELIA_AL_BUDGET": "40"})
    assert (config.pool_split, config.al_seed, config.select_round, config.class_weights) == ("train", 0, 0, "subset")
    assert not config.final_eval_round and config.include_uids is None


@pytest.mark.parametrize(
    "env,message",
    [
        ({"ODELIA_AL_STRATEGY": "entropy"}, "needs ODELIA_AL_BUDGET"),
        ({"ODELIA_AL_STRATEGY": "uncertainty", "ODELIA_AL_BUDGET": "5"}, "is not one of"),
        ({"ODELIA_AL_BUDGET": "20"}, "neither ODELIA_AL_STRATEGY nor ODELIA_TRAIN_INCLUDE_LIST"),
        ({"ODELIA_AL_SEED": "1", "ODELIA_SEED": "1"}, "neither ODELIA_AL_STRATEGY"),
        ({"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "0"}, ">= 1"),
        ({"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "twenty"}, "must be an integer"),
        ({"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "5", "ODELIA_AL_POOL_SPLIT": "val"}, "'val' is refused"),
        ({"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "5", "ODELIA_AL_CLASS_WEIGHTS": "full"}, "is not one of"),
        ({"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGTE": "5"}, "Unknown variable"),
        ({"ODELIA_AL_STRATEGY": "include_list"}, "needs ODELIA_TRAIN_INCLUDE_LIST"),
        ({"ODELIA_AL_STRATEGY": " ", "ODELIA_AL_BUDGET": "5"}, "set but empty"),
        ({"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "5", "ODELIA_AL_SEED": str(2**32)}, "must be in"),
    ],
)
def test_inconsistent_configuration_fails_at_load(al, env, message):
    with pytest.raises(al.ALConfigError, match=message):
        al.parse_config(env)


def test_include_list(al, tmp_path):
    path = tmp_path / "d32_uids.txt"
    path.write_text("# selected offline at TUD\nCAM_003_left\n\nCAM_001_right\n")
    config = al.parse_config({"ODELIA_TRAIN_INCLUDE_LIST": str(path)})
    assert config.strategy == "include_list" and config.budget == 2
    assert config.include_uids == ["CAM_003_left", "CAM_001_right"]
    assert len(config.include_list_sha256) == 64

    assert al.parse_config({"ODELIA_TRAIN_INCLUDE_LIST": str(path), "ODELIA_AL_BUDGET": "2"}).budget == 2
    with pytest.raises(al.ALConfigError, match="lists 2 UIDs"):
        al.parse_config({"ODELIA_TRAIN_INCLUDE_LIST": str(path), "ODELIA_AL_BUDGET": "3"})
    with pytest.raises(al.ALConfigError, match="both choose"):
        al.parse_config(
            {"ODELIA_TRAIN_INCLUDE_LIST": str(path), "ODELIA_AL_STRATEGY": "entropy", "ODELIA_AL_BUDGET": "2"}
        )
    with pytest.raises(al.ALConfigError, match="no such file"):
        al.parse_config({"ODELIA_TRAIN_INCLUDE_LIST": str(tmp_path / "missing.txt")})

    duplicate = tmp_path / "dup.txt"
    duplicate.write_text("A\nB\nA\n")
    with pytest.raises(al.ALConfigError, match="repeats 1 UID") as excinfo:
        al.parse_config({"ODELIA_TRAIN_INCLUDE_LIST": str(duplicate)})
    assert "A\n" not in str(excinfo.value)
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    with pytest.raises(al.ALConfigError, match="lists no UIDs"):
        al.parse_config({"ODELIA_TRAIN_INCLUDE_LIST": str(empty)})


@pytest.mark.parametrize("variable,value", [("ODELIA_AL_STRATEGY", "entropy"), ("ODELIA_TRAIN_INCLUDE_LIST", "/x"),
                                            ("ODELIA_AL_CLASS_WEIGHTS", "keep")])
def test_safety_net_requires_al_subset_to_be_listed(al, variable, value):
    import round_hooks

    with pytest.raises(round_hooks.RoundHookError, match="al_subset") as excinfo:
        round_hooks.load_hooks({**SWARM, variable: value})
    assert variable in str(excinfo.value)


def test_listed_al_subset_loads_through_load_hooks(al):
    import round_hooks

    env = {**SWARM, "ODELIA_ROUND_HOOKS": "al_subset", "ODELIA_AL_STRATEGY": "entropy", "ODELIA_AL_BUDGET": "4"}
    hooks = round_hooks.load_hooks(env, logging.getLogger("t"))
    assert [type(h).__name__ for h in hooks] == ["ALSubsetHook"]
    assert hooks[0].config.budget == 4
    # Listed but nothing configured: inactive, not an error.
    assert round_hooks.load_hooks({**SWARM, "ODELIA_ROUND_HOOKS": "al_subset"}) == []
    # The pins of a warm-started selection run are applied by the implicit reproducibility hook, first.
    hooks = round_hooks.load_hooks({**env, "ODELIA_SEED": "7", "ODELIA_WARM_START_SHA256": DIGEST})
    assert [h.name for h in hooks] == ["reproducibility", "al_subset"]
    # Outside swarm mode the variables are ignored (with a warning), like every hook feature.
    assert round_hooks.load_hooks({"TRAINING_MODE": "local_training", "ODELIA_AL_STRATEGY": "entropy"}) == []


# --------------------------------------------------------------------------------------------------
# Round plan
# --------------------------------------------------------------------------------------------------


def _ctx(al, round_num, params, total=5):
    import round_hooks

    ctx = round_hooks.RoundContext(logger=logging.getLogger("t"), site_name="SITE_A", job_id="job-1")
    ctx.current_round, ctx.total_rounds = round_num, total
    ctx.input_model = types.SimpleNamespace(params=params)
    return ctx


def _params():
    generator = torch.Generator().manual_seed(3)
    return {"layer.weight": torch.randn(3, 4, generator=generator), "layer.bias": torch.randn(3, generator=generator)}


def test_selection_round_must_exist(al):
    hook = al.ALSubsetHook(al.parse_config({"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "2",
                                            "ODELIA_AL_SELECT_ROUND": "4", "ODELIA_FINAL_EVAL_ROUND": "1"}))
    with pytest.raises(RuntimeError, match="would never run"):
        hook.after_receive(_ctx(al, 0, _params(), total=5))  # 5 rounds, the last evaluation-only


def test_a_process_that_starts_after_the_selection_round_refuses(al):
    hook = al.ALSubsetHook(al.parse_config({"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "2"}))
    with pytest.raises(RuntimeError, match="after the selection round"):
        hook.after_receive(_ctx(al, 2, _params()))


def test_no_selection_in_the_evaluation_only_round(al):
    hook = al.ALSubsetHook(al.parse_config({"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "2"}))
    ctx = _ctx(al, 0, _params(), total=1)
    ctx.is_eval_only_round = True
    hook.before_fit(ctx)  # would need a DataModule if it tried to select
    assert hook.summary is None


# --------------------------------------------------------------------------------------------------
# Real ODELIA_Dataset3D and DataModule
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def odelia(monkeypatch):
    pytest.importorskip("torchio")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pytorch_lightning")
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    from data.datamodules import DataModule
    from data.datasets.dataset_3d_odelia import ODELIA_Dataset3D

    def make(uids, labels, transform=None, unreadable=(), split="train"):
        ds = ODELIA_Dataset3D.__new__(ODELIA_Dataset3D)
        ds.df = pd.DataFrame({"UID": list(uids), "Institution": ["CAM_1"] * len(uids), "Lesion": list(labels)})
        ds.item_pointers = list(ds.df.index)
        ds.labels = ["Lesion"]
        ds.class_labels_num = [3]
        ds.split = split
        ds.config = "unilateral"
        ds._bad_input_count = 0
        ds._bad_input_logged = set()
        ds._bad_input_reasons = {}
        ds.transform = transform or (lambda uid: torch.full((2,), float(int(uid.split("_")[1]))))
        ds.get_image_path = lambda uid, institution: f"/data/{uid}"

        def load(path, institution, uid):
            if uid in unreadable:
                raise OSError("unreadable")
            return uid

        ds._load_source_image = load
        ds._bad_input_log_path = lambda: Path("/dev/null")
        return ds

    return types.SimpleNamespace(make=make, DataModule=DataModule, pd=pd)


def test_restrict_dataset_keeps_exactly_the_selection(al, odelia):
    ds = odelia.make([f"u_{i}" for i in range(10)], [i % 3 for i in range(10)])
    ds._bad_input_logged.add(("x", "y"))
    view = al.restrict_dataset(ds, ["u_7", "u_2", "u_5"], transform=lambda uid: "train-transform")
    assert len(view) == 3 and sorted(view.df["UID"]) == ["u_2", "u_5", "u_7"]
    assert view.item_pointers == [0, 1, 2]
    assert {view[i]["uid"] for i in range(3)} == {"u_2", "u_5", "u_7"}
    assert view[0]["source"] == "train-transform"
    assert view._bad_input_logged == set() and ds._bad_input_logged == {("x", "y")}  # not shared
    assert len(ds) == 10 and ds.transform is not view.transform  # the original is untouched
    with pytest.raises(RuntimeError, match="1 UID"):
        al.restrict_dataset(ds, ["u_1", "not_there"])


def test_labels_and_duplicates(al, odelia):
    ds = odelia.make(["u_0", "u_1", "u_2"], [2, 0, 1])
    assert al.labels_by_uid(ds) == {"u_0": 2, "u_1": 0, "u_2": 1}
    assert al.num_classes_of(ds) == 3
    with pytest.raises(RuntimeError, match="duplicate UID"):
        al.labels_by_uid(odelia.make(["u_0", "u_0"], [1, 1]))


class _Scorer(torch.nn.Module):
    """logits = feature value spread over the classes; Dropout makes train vs eval mode observable."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1.0, 0.0, -1.0]))
        self.dropout = torch.nn.Dropout(p=0.9)
        self.seen_training = []

    def forward(self, x):
        self.seen_training.append(self.training)
        return self.dropout(x[:, :1] * self.scale.unsqueeze(0) / 10.0)

    def logits2probabilities(self, logits):
        return torch.softmax(logits, dim=1)


def test_score_pool_uses_eval_mode_and_leaves_out_unreadable_cases(al, odelia):
    uids = [f"u_{i}" for i in range(8)]
    labels = {u: i % 3 for i, u in enumerate(uids)}
    ds = odelia.make(uids, [labels[u] for u in uids], unreadable={"u_3"})
    model = _Scorer()
    model.train()
    model.dropout.train(False)  # a submodule deliberately in eval mode: its own flag must come back
    rows, unusable = al.score_pool(model, al.eval_view(ds, ds.transform), labels, device="cpu")
    assert unusable == ["u_3"]
    assert [r[0] for r in rows] == [u for u in uids if u != "u_3"]
    assert set(model.seen_training) == {False}
    assert model.training is True and model.dropout.training is False
    expected = torch.softmax(torch.tensor([[5.0, 0.0, -5.0]]) / 10.0, dim=1)[0].tolist()
    assert rows[4][1] == pytest.approx(expected)  # u_5
    assert rows[4][2] == labels["u_5"]


def test_replace_training_set_resets_the_cached_loader_and_caps_workers(al, odelia):
    train = odelia.make([f"u_{i}" for i in range(40)], [i % 3 for i in range(40)])
    dm = odelia.DataModule(ds_train=train, ds_val=odelia.make(["v_0", "v_1"], [0, 1], split="val"),
                           ds_test=odelia.make(["t_0", "t_1"], [0, 1], split="test"), batch_size=1, num_workers=8)
    full_loader = dm.train_dataloader()
    assert len(full_loader) == 40
    subset = al.restrict_dataset(train, [f"u_{i}" for i in range(6)])
    before, after = al.replace_training_set(dm, subset)
    assert (before, after) == (8, 1)  # 6 cases, at least 4 per worker (#574)
    assert dm.ds_train is subset and len(dm.train_dataloader()) == 6
    assert dm.train_dataloader() is not full_loader
    assert dm.train_dataloader().num_workers == 1
    assert dm.val_dataloader().num_workers == 8 and dm.test_dataloader().num_workers == 8


def test_replace_training_set_refuses_weighted_sampling(al, odelia):
    train = odelia.make(["u_0", "u_1"], [0, 1])
    dm = odelia.DataModule(ds_train=train, ds_val=train, batch_size=1, num_workers=0, weights=[0.5, 0.5])
    with pytest.raises(RuntimeError, match="per-case weights"):
        al.replace_training_set(dm, train)


def test_class_weights(al):
    assert al.class_weights_from_labels([0, 0, 0, 1, 2, 2], 3) == pytest.approx([6 / 9, 6 / 3, 6 / 6])
    assert al.class_weights_from_labels([0, 0], 3) == pytest.approx([2 / 6, 1.0, 1.0])  # absent class -> 1.0

    model = torch.nn.Module()
    model.register_buffer("_class_weight", torch.tensor([1.0, 2.0, 3.0]))
    buffer = model._class_weight
    weights = al.apply_class_weights(model, [0, 1, 1, 2], 3, "subset")
    assert weights == pytest.approx([4 / 3, 4 / 6, 4 / 3])
    assert model._class_weight is buffer and model._class_weight.tolist() == pytest.approx(weights)
    assert "_class_weight" in model.state_dict()
    assert al.apply_class_weights(model, [0, 0, 0], 3, "keep") == pytest.approx(weights)  # untouched


# --------------------------------------------------------------------------------------------------
# main.py end to end
# --------------------------------------------------------------------------------------------------

pl = pytest.importorskip("pytorch_lightning")


class TinyClassifier(pl.LightningModule):
    """3-class model with the per-site _class_weight buffer the ODELIA models carry (BasicClassifier)."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 3)
        self.register_buffer("_class_weight", torch.ones(3))

    def forward(self, x):
        return self.layer(x)

    def logits2probabilities(self, logits):
        return torch.softmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        return torch.nn.functional.cross_entropy(self(batch["source"]), batch["target"].squeeze(1).long(),
                                                 weight=self._class_weight)

    def validation_step(self, batch, batch_idx):
        loss = torch.nn.functional.cross_entropy(self(batch["source"]), batch["target"].squeeze(1).long())
        self.log("val/loss", loss, batch_size=batch["source"].shape[0])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05)


class FakeODELIADataset(torch.utils.data.Dataset):
    """The ODELIA_Dataset3D surface al_subset uses: df (UID, Lesion), item_pointers, labels, class_labels_num,
    split, and a transform applied in __getitem__ (here it also records which UIDs it served)."""

    def __init__(self, uids, labels, features, transform, split):
        import pandas as pd

        self.df = pd.DataFrame({"UID": uids, "Lesion": labels, "Institution": "SITE_A"})
        self.item_pointers = self.df.index.tolist()
        self.labels = ["Lesion"]
        self.class_labels_num = [3]
        self.split = split
        self.transform = transform
        self.features = features

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        row = self.df.loc[self.item_pointers[index]]
        uid = str(row["UID"])
        return {"uid": uid, "source": self.transform(uid, self.features[uid]),
                "target": np.array([int(row["Lesion"])])}


class RecordingLogger(pl.loggers.Logger):
    def __init__(self):
        super().__init__()
        self.records = []

    @property
    def name(self):
        return "recording"

    @property
    def version(self):
        return 0

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def log_metrics(self, metrics, step=None):
        self.records.append((dict(metrics), step))


class FakeClientRuntime:
    """One task per round; what FLCallback sends becomes the next global (see test_main_round_loop.py)."""

    def __init__(self, total_rounds, initial_params):
        self.total_rounds = total_rounds
        self.global_params = {k: v.detach().clone() for k, v in initial_params.items()}
        self.sent, self.handed_out = [], []
        self.current, self.round, self.next_round = None, None, 0

    def is_running(self):
        if self.current is None:
            if self.next_round >= self.total_rounds:
                return False
            from nvflare.app_common.abstract.fl_model import FLModel

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
        self.sent.append(model)
        self.global_params = {k: v.detach().clone() for k, v in model.params.items()}
        if clear_cache:
            self.current = None

    def clear(self):
        self.current = None


N_POOL = 24


@pytest.fixture
def swarm(monkeypatch, tmp_path):
    pytest.importorskip("pandas")
    nvflare_api = pytest.importorskip("nvflare.app_opt.lightning.api")
    flare_lightning = pytest.importorskip("nvflare.client.lightning")
    import nvflare.client as flare_client
    from nvflare.client.config import ConfigKey
    from pytorch_lightning.callbacks import ModelCheckpoint

    anomaly_mode_before = torch.is_anomaly_enabled()
    for variable in [v for v in list(__import__("os").environ) if v.startswith(("ODELIA_", "EPOCHS_"))]:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TRAINING_MODE", "swarm")
    monkeypatch.setenv("ODELIA_ROUND_HOOKS", "al_subset")
    # Full split: 1 epoch per round (the Trainer below). Subset of 6: 2 * 6 / 6 = 2 epochs per round.
    monkeypatch.setenv("EPOCHS_PER_ROUND", "2")
    monkeypatch.setenv("EPOCHS_REFERENCE_DATASET_SIZE", "6")
    monkeypatch.setenv("EPOCHS_MAX_CAP", "10")

    # threedcnn_ptl imports the ODELIA data/model stack at module level; stub it like test_main_round_loop.
    stubs = {name: types.ModuleType(name) for name in ("data", "data.datasets", "data.datamodules", "env_config",
                                                       "models", "models.models_config")}
    stubs["data.datasets"].ODELIA_Dataset3D = object
    stubs["data.datamodules"].DataModule = object
    for attr in ("build_odelia_manifests", "generate_run_directory", "load_environment_variables",
                 "prepare_odelia_dataset", "prepare_odelia_dataset_without_augmentation"):
        setattr(stubs["env_config"], attr, lambda *args, **kwargs: None)
    stubs["models.models_config"].create_model = lambda *args, **kwargs: None
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    for name in ("threedcnn_ptl", "odelia_main_al_under_test") + H_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    threedcnn_ptl = import_module_from_path("threedcnn_ptl", SHARED_CUSTOM_DIR / "threedcnn_ptl.py")
    datamodule_module = import_module_from_path("odelia_datamodule_real",
                                                SHARED_CUSTOM_DIR / "data" / "datamodules" / "datamodule.py")

    served = {"train": [], "eval": []}

    def train_transform(uid, x):
        served["train"].append(uid)
        return x + 0.05 * torch.randn_like(x)  # stands in for augmentation: draws from the torch RNG

    def eval_transform(uid, x):
        served["eval"].append(uid)
        return x

    generator = torch.Generator().manual_seed(42)
    uids = [f"SITE_A_{i:03d}_{'left' if i % 2 else 'right'}" for i in range(N_POOL)]
    labels = [0 if i % 4 else (1 if i % 8 == 0 else 2) for i in range(N_POOL)]
    features = {uid: torch.randn(4, generator=generator) for uid in uids}
    for uid, label in zip(uids, labels):
        features[uid][label] += 1.5  # learnable signal
    val_uids = [f"VAL_{i}" for i in range(6)]
    test_uids = [f"TEST_{i}" for i in range(4)]
    for uid in val_uids + test_uids:
        features[uid] = torch.randn(4, generator=generator)

    # max_epochs: epochs per round on the full split; a test varies it to model sites with different weighted epochs.
    state = types.SimpleNamespace(events=[], fits=[], served=served, uids=uids, labels=dict(zip(uids, labels)),
                                  total_rounds=3, scratch=tmp_path / "scratch", max_epochs=1)

    def build_run():
        torch.manual_seed(0)
        model = TinyClassifier()
        runtime = FakeClientRuntime(state.total_rounds, model.state_dict())
        runtime.global_params.pop("_class_weight")  # the persistor's model has no per-site buffer
        dm = datamodule_module.DataModule(
            ds_train=FakeODELIADataset(uids, labels, features, train_transform, "train"),
            ds_val=FakeODELIADataset(val_uids, [0, 1, 2, 0, 1, 2], features, eval_transform, "val"),
            ds_test=FakeODELIADataset(test_uids, [0, 1, 2, 0], features, eval_transform, "test"),
            batch_size=2, num_workers=0, pin_memory=False,
        )
        recorder = RecordingLogger()
        trainer = pl.Trainer(
            accelerator="cpu", devices=1, max_epochs=state.max_epochs, logger=[recorder], enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0, default_root_dir=str(tmp_path / "pl"),
            callbacks=[ModelCheckpoint(dirpath=str(tmp_path / "ckpt"), save_last=True)],
        )
        run_dir = tmp_path / "runs" / "SITE_A" / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return types.SimpleNamespace(model=model, runtime=runtime, dm=dm, trainer=trainer, recorder=recorder,
                                     run_dir=run_dir,
                                     env_vars={"model_name": "Tiny", "scratch_dir": str(state.scratch)})

    real_train = threedcnn_ptl.train_local_model  # captured once: run_main() may run several times per test

    def run_main():
        run = build_run()
        state.run = run
        state.fits.clear()
        served["train"].clear()
        served["eval"].clear()
        for name, value in {
            "init": lambda *args, **kwargs: None,
            "get_config": lambda: {ConfigKey.TASK_EXCHANGE: {ConfigKey.TRAIN_WITH_EVAL: True}},
            "receive": run.runtime.receive, "send": run.runtime.send, "clear": run.runtime.clear,
            "is_train": lambda: True, "is_evaluate": lambda: False, "is_submit_model": lambda: False,
        }.items():
            monkeypatch.setattr(nvflare_api, name, value)
        monkeypatch.setattr(flare_lightning, "receive", run.runtime.receive)
        monkeypatch.setattr(flare_lightning, "is_running", run.runtime.is_running)
        monkeypatch.setattr(flare_lightning, "get_site_name", lambda: "SITE_A")
        monkeypatch.setattr(flare_lightning, "get_job_id", lambda: "job-0001")
        monkeypatch.setattr(flare_client, "init", lambda *args, **kwargs: None)

        def prepare_training(logger, num_epochs, site_name, log_dataset_details, weighted_epochs=False):
            return run.dm, run.model, run.trainer.checkpoint_callback, run.trainer, run.run_dir, run.env_vars

        def train_local_model(logger, data_module, model, trainer, *args, **kwargs):
            start_epoch, start_served = trainer.current_epoch, len(served["train"])
            real_train(logger, data_module, model, trainer, *args, **kwargs)
            state.fits.append(types.SimpleNamespace(
                round=run.runtime.round, epochs=trainer.current_epoch - start_epoch,
                uids=list(served["train"][start_served:]), class_weight=model._class_weight.tolist()))

        monkeypatch.setattr(threedcnn_ptl, "set_up_logging", lambda: logging.getLogger("test_al_subset"))
        monkeypatch.setattr(threedcnn_ptl, "prepare_training", prepare_training)
        monkeypatch.setattr(threedcnn_ptl, "train_local_model", train_local_model)
        monkeypatch.setattr(threedcnn_ptl, "finalize_training", lambda *args, **kwargs: None)
        monkeypatch.setattr(threedcnn_ptl, "output_GT_and_classprobs_csv_train_val", lambda *args, **kwargs: None)
        for name in ("odelia_main_al_under_test",) + H_MODULES:
            sys.modules.pop(name, None)
        main = import_module_from_path("odelia_main_al_under_test", SHARED_CUSTOM_DIR / "main.py")
        main.main()
        return run

    state.run_main = run_main
    yield state
    torch.autograd.set_detect_anomaly(anomaly_mode_before)
    for name in ("odelia_main_al_under_test", "threedcnn_ptl") + H_MODULES:
        sys.modules.pop(name, None)


def _record(state):
    path = state.scratch / "odelia_al" / "job-0001" / "SITE_A" / "selection_record.json"
    return json.loads(path.read_text())


def test_selection_at_round0_restricts_training_and_epochs_for_every_round(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_AL_STRATEGY", "entropy")
    monkeypatch.setenv("ODELIA_AL_BUDGET", "6")
    run = swarm.run_main()

    record = _record(swarm)
    selected = record["selected_uids"]
    assert len(selected) == 6 and record["strategy"] == "entropy"
    # Scored once, in dataset order, with the evaluation transform, before the first fit.
    assert swarm.served["eval"][:N_POOL] != [] and set(selected) <= set(swarm.uids)

    assert [f.round for f in swarm.fits] == [0, 1, 2]
    for fit in swarm.fits:
        assert fit.epochs == 2, "every round after the selection trains the subset's epoch count"
        assert set(fit.uids) == set(selected) and len(fit.uids) == 2 * 6
    fl_callback = next(cb for cb in run.trainer.callbacks if type(cb).__name__ == "FLCallback")
    assert fl_callback.max_epochs_per_round == 2

    import al_subset

    expected_weights = al_subset.class_weights_from_labels([swarm.labels[u] for u in selected], 3)
    assert swarm.fits[0].class_weight == pytest.approx(expected_weights)
    assert len(run.runtime.sent) == 3


def test_scores_come_from_the_received_global_and_the_ranking_is_entropy(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_AL_STRATEGY", "entropy")
    monkeypatch.setenv("ODELIA_AL_BUDGET", "6")
    run = swarm.run_main()
    record = _record(swarm)

    # Recompute the pool probabilities with the round-0 global, independently of the hook.
    global0 = run.runtime.handed_out[0].params
    probe = TinyClassifier()
    probe.load_state_dict(global0, strict=False)
    probe.eval()
    import al_selection

    rows = []
    with torch.no_grad():
        for uid in swarm.uids:
            x = run.dm.ds_val.features[uid].unsqueeze(0) if hasattr(run.dm.ds_val, "features") else None
            rows.append((uid, torch.softmax(probe(x), dim=1)[0].tolist(), swarm.labels[uid]))
    assert record["selected_uids"] == al_selection.select_uids(rows, 6, "entropy")
    by_uid = {entry["uid"]: entry for entry in record["pool"]}
    for uid, probs, _ in rows:
        assert by_uid[uid]["probs"] == pytest.approx(probs, abs=1e-6)
    import state_digest

    assert record["global_digest"] == state_digest.state_digest(global0)


def test_selection_in_a_later_round_switches_epochs_from_then_on(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_AL_STRATEGY", "random")
    monkeypatch.setenv("ODELIA_AL_BUDGET", "6")
    monkeypatch.setenv("ODELIA_AL_SELECT_ROUND", "1")
    run = swarm.run_main()
    selected = _record(swarm)["selected_uids"]

    first, second, third = swarm.fits
    assert first.epochs == 1 and len(first.uids) == N_POOL and first.class_weight == [1.0, 1.0, 1.0]
    assert second.epochs == 2 and set(second.uids) == set(selected)
    assert third.epochs == 2 and set(third.uids) == set(selected)
    assert _record(swarm)["round"] == 1


def _sent_num_steps(run):
    from nvflare.app_common.abstract.fl_model import MetaKey

    return [model.meta.get(MetaKey.NUM_STEPS_CURRENT_ROUND) for model in run.runtime.sent]


def test_aggregation_weight_is_the_steps_trained_in_each_round(swarm, monkeypatch):
    """SP-1: sites that train the same subset send the same weight, whatever they trained before the selection.

    Lightning's estimate (FLCallback's default) multiplies the subset's steps per epoch by the cumulative epochs,
    which after a round-1 selection gave [12, 9, 15] and [36, 15, 21] for these two sites.
    """
    monkeypatch.setenv("ODELIA_AL_STRATEGY", "random")
    monkeypatch.setenv("ODELIA_AL_BUDGET", "6")
    monkeypatch.setenv("ODELIA_AL_SELECT_ROUND", "1")
    weights = {}
    for epochs in (1, 3):  # two sites whose weighted epochs on the full split differ
        swarm.max_epochs = epochs
        run = swarm.run_main()
        weights[epochs] = _sent_num_steps(run)
        hook_callbacks = [cb for cb in run.trainer.callbacks if type(cb).__name__ == "ALRoundStepsCallback"]
        assert len(hook_callbacks) == 1 and hook_callbacks[0].reported == weights[epochs]
    # Round 0: 24 cases at batch 2 = 12 steps per epoch. Rounds 1 and 2: the 6 selected cases, 2 epochs = 6 steps.
    assert weights[1] == [12, 6, 6]
    assert weights[3] == [36, 6, 6]

    monkeypatch.setenv("ODELIA_AL_SELECT_ROUND", "0")
    swarm.max_epochs = 1
    assert _sent_num_steps(swarm.run_main()) == [6, 6, 6]


def test_round_steps_callback_reports_the_optimizer_steps_of_the_fit(al, caplog):
    pytest.importorskip("pytorch_lightning")

    class Module:
        pass

    module = Module()
    setattr(module, "__fl_meta__", {"CUSTOM": "kept"})
    trainer = types.SimpleNamespace(global_step=10)
    callback = al.round_steps_callback(types.SimpleNamespace(current_round=4), logging.getLogger("t"))
    callback.on_train_start(trainer, module)
    trainer.global_step = 16
    callback.on_train_end(trainer, module)
    assert getattr(module, "__fl_meta__") == {"CUSTOM": "kept", "NUM_STEPS_CURRENT_ROUND": 6}
    with caplog.at_level(logging.WARNING):
        callback.on_train_start(trainer, module)
        callback.on_train_end(trainer, module)  # no optimizer step: still a positive weight
    assert getattr(module, "__fl_meta__")["NUM_STEPS_CURRENT_ROUND"] == 1 and "reporting weight 1" in caplog.text
    assert callback.reported == [6, 1]


def test_dp_equal_weights_leave_the_aggregation_weight_to_dp_noise(al):
    pytest.importorskip("pytorch_lightning")
    ctx = types.SimpleNamespace(logger=None, current_round=0)
    selection = {"ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "3"}
    hook = al.create_hook({**selection, "ODELIA_DP_EQUAL_WEIGHTS": "1"}, logging.getLogger("t"))
    assert hook.config.dp_equal_weights is True and hook.lightning_callbacks(ctx) == []
    hook = al.create_hook(selection, logging.getLogger("t"))
    assert [type(cb).__name__ for cb in hook.lightning_callbacks(ctx)] == ["ALRoundStepsCallback"]


def test_a_test_split_pool_still_leaves_unreadable_cases_out(al, odelia, monkeypatch):
    """DE-2: with ODELIA_STRICT_EVAL_INPUTS=1 the test split itself stops at an unreadable case; the pool view
    keeps the neighbour + UID check either way."""
    monkeypatch.setenv("ODELIA_STRICT_EVAL_INPUTS", "1")
    uids = [f"u_{i}" for i in range(6)]
    labels = {u: i % 3 for i, u in enumerate(uids)}
    ds = odelia.make(uids, [labels[u] for u in uids], unreadable={"u_2"}, split="test")
    with pytest.raises(RuntimeError, match="never replaced by its neighbour"):
        ds[2]
    rows, unusable = al.score_pool(_Scorer(), al.eval_view(ds, ds.transform), labels, device="cpu")
    assert unusable == ["u_2"] and [r[0] for r in rows] == [u for u in uids if u != "u_2"]


def test_a_training_subset_drawn_from_the_test_split_keeps_resampling(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_AL_STRATEGY", "random")
    monkeypatch.setenv("ODELIA_AL_BUDGET", "2")
    monkeypatch.setenv("ODELIA_AL_POOL_SPLIT", "test")
    run = swarm.run_main()
    assert run.dm.ds_train.split == "test" and run.dm.ds_train.resample_bad_inputs is True
    assert getattr(run.dm.ds_test, "resample_bad_inputs", None) is None  # the evaluation split is untouched


def test_records_keep_uids_at_the_site_and_stream_only_aggregates(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_AL_STRATEGY", "class_matched_random")
    monkeypatch.setenv("ODELIA_AL_BUDGET", "6")
    monkeypatch.setenv("ODELIA_AL_SEED", "9")
    run = swarm.run_main()
    record = _record(swarm)
    selected = record["selected_uids"]

    replay = (swarm.scratch / "odelia_al" / "job-0001" / "SITE_A" / "selected_uids.txt").read_text()
    assert [line for line in replay.splitlines() if not line.startswith("#")] == selected

    summary_text = (run.run_dir / "al_selection_summary.json").read_text()
    assert not any(uid in summary_text for uid in swarm.uids), "the run directory is uploaded by live_sync"
    summary = json.loads(summary_text)
    assert summary["n_selected"] == 6 and summary["seed"] == 9
    assert summary["selection_digest"] == record["selection_digest"]

    streamed = [metrics for metrics, _ in run.recorder.records if any(k.startswith("al/") for k in metrics)]
    assert len(streamed) == 1
    metrics = streamed[0]
    assert set(metrics) == {"al/n_selected", "al/selection_digest", "al/class_count_0", "al/class_count_1",
                            "al/class_count_2", "al/mean_entropy"}
    assert all(isinstance(v, (int, float)) for v in metrics.values())
    assert metrics["al/n_selected"] == 6
    assert metrics["al/selection_digest"] == int(record["selection_digest"][:12], 16)
    counts = {k: sum(1 for u in selected if swarm.labels[u] == k) for k in range(3)}
    assert [metrics[f"al/class_count_{k}"] for k in range(3)] == [counts[k] for k in range(3)]


def test_include_list_trains_exactly_the_listed_cases(swarm, monkeypatch, tmp_path):
    listed = swarm.uids[3:7]
    path = tmp_path / "list.txt"
    path.write_text("\n".join(listed) + "\n")
    monkeypatch.setenv("ODELIA_TRAIN_INCLUDE_LIST", str(path))
    swarm.run_main()
    assert all(set(fit.uids) == set(listed) for fit in swarm.fits)
    assert swarm.served["eval"] == [u for u in swarm.served["eval"] if u.startswith(("VAL_", "TEST_"))]  # no scoring
    record = _record(swarm)
    assert record["strategy"] == "include_list" and record["selected_uids"] == listed
    assert record["mean_entropy"] is None


def test_include_list_with_uids_outside_the_pool_stops_before_training(swarm, monkeypatch, tmp_path):
    path = tmp_path / "list.txt"
    path.write_text(f"{swarm.uids[0]}\nUID_FROM_ANOTHER_FOLD\n")
    monkeypatch.setenv("ODELIA_TRAIN_INCLUDE_LIST", str(path))
    with pytest.raises(RuntimeError, match="1 of 2 UIDs are not in") as excinfo:
        swarm.run_main()
    assert "UID_FROM_ANOTHER_FOLD" not in str(excinfo.value)
    assert swarm.fits == []
    errors = json.loads((swarm.scratch / "odelia_al" / "job-0001" / "SITE_A" / "include_list_errors.json").read_text())
    assert errors["not_in_pool"] == ["UID_FROM_ANOTHER_FOLD"]


def test_warm_start_digest_mismatch_stops_every_site_before_training(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_WARM_START_SHA256", DIGEST)
    import al_subset  # noqa: F401  (imported by main through load_hooks)

    with pytest.raises(RuntimeError, match="ODELIA_WARM_START_SHA256 mismatch"):
        swarm.run_main()
    assert swarm.fits == [] and swarm.run.runtime.sent == []


def _initial_digest():
    torch.manual_seed(0)  # build_run() seeds the same way before building the model the runtime starts from
    initial = TinyClassifier().state_dict()
    initial.pop("_class_weight")
    state_digest = import_module_from_path("state_digest_probe", SHARED_CUSTOM_DIR / "state_digest.py")
    return state_digest.state_digest(initial)


def test_warm_start_digest_match_lets_the_run_proceed(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_WARM_START_SHA256", _initial_digest())
    run = swarm.run_main()
    assert len(run.runtime.sent) == 3


def test_d32_configuration_selects_from_the_verified_warm_start(swarm, monkeypatch):
    """Warm-start digest + seed + entropy selection at round 0: the configuration the D3.2 jobs use."""
    digest = _initial_digest()
    monkeypatch.setenv("ODELIA_WARM_START_SHA256", digest)
    monkeypatch.setenv("ODELIA_SEED", "3")
    monkeypatch.setenv("ODELIA_AL_STRATEGY", "entropy")
    monkeypatch.setenv("ODELIA_AL_BUDGET", "6")
    run = swarm.run_main()
    summary = json.loads((run.run_dir / "al_selection_summary.json").read_text())
    assert summary["global_digest"] == digest, "the selection records the global the reproducibility hook verified"
    assert [f.round for f in swarm.fits] == [0, 1, 2] and all(len(set(f.uids)) == 6 for f in swarm.fits)


def _final_params(swarm, seed):
    import os

    if seed is None:
        os.environ.pop("ODELIA_SEED", None)
    else:
        os.environ["ODELIA_SEED"] = str(seed)
    run = swarm.run_main()
    return run.runtime.sent[-1].params, [list(f.uids) for f in swarm.fits]


def test_odelia_seed_makes_training_reproducible(swarm, monkeypatch):
    monkeypatch.setenv("ODELIA_SEED", "5")  # registered so monkeypatch restores the environment
    params_a, order_a = _final_params(swarm, 5)
    params_b, order_b = _final_params(swarm, 5)
    params_c, order_c = _final_params(swarm, 6)
    assert order_a == order_b and all(torch.equal(params_a[k], params_b[k]) for k in params_a)
    assert order_a != order_c, "another seed must visit the training cases in another order"
    assert any(not torch.equal(params_a[k], params_c[k]) for k in params_a)
