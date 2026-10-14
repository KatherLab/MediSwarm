"""Reloading trained models from their Lightning checkpoints (#627).

finalize_training() exports test-set predictions of the last and the best local model by calling
``model.load_checkpoint_from_file(path, strict=False)`` and ``model.load_best_checkpoint(...)`` on
the trained model. Those used to rebuild the model through Lightning's load_from_checkpoint, which
fails for every ModelWrapper model (challenge models 2-5: the backbone is not in the checkpoint),
so their export was switched off by model name. The loaders now copy the model in memory and load
the checkpoint's weights into the copy.

Each challenge model is trained for one step on CPU with a tiny input, checkpointed by Lightning
and reloaded through exactly those calls. Pretrained initialisation plays no part in a checkpoint
round trip and the files live only in the image, so the models are built without it (4LME_ABMIL's
timm backbone would otherwise download its weights). A model whose package is not installed (the
GitHub runner has no timm, torchvision or dynamic_network_architectures) is skipped with the reason.
"""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")
for _module in ("torchmetrics", "monai", "einops", "x_transformers"):  # the models package imports them
    pytest.importorskip(_module)

from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
MODELS_DIR = CUSTOM_DIR / "models"
if str(CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_DIR))

from models.base_model import ModelWrapper, NON_PARAMETER_STATE_KEYS, state_dict_mismatch  # noqa: E402

CLASS_WEIGHTS = {"weight": [1.0, 2.0, 3.0]}  # as the trainer passes them: adds _class_weight and loss.weight

# Every challenge model in models_config.CHALLENGE_MODELS (test_registry_is_covered keeps this in step).
# factory: persistor_path from the registry; kwargs: its persistor_args minus pretrained weights;
# shape: the smallest input the architecture accepts (instance norm needs > 1 voxel per channel at the
# deepest 1DivideAndConquer stage, SwinUNETR needs multiples of 32, timm's Swin needs 224 x 224 slices).
CHALLENGE_CASES = {
    "1DivideAndConquer": dict(
        requires=("dynamic_network_architectures",),
        factory="challenge.1DivideAndConquer.model.create_model",
        kwargs=dict(num_classes=3, n_input_channels=1, spatial_dims=3, pretrained_path=None),
        shape=(1, 1, 32, 64, 64),
    ),
    "2BCN_AIM": dict(
        requires=(),
        factory="challenge.2bcnaim.swinunetr.create_model",
        kwargs=dict(img_size=224, num_classes=3, n_input_channels=1, spatial_dims=3),
        shape=(1, 1, 32, 64, 64),
    ),
    "3agaldran": dict(
        requires=("torchvision",),
        factory="challenge.3agaldran.model_factory.model_factory",
        kwargs=dict(arch="mvit_v2_s", pretrained_path=None, num_classes=3, in_ch=1, seed=123),
        shape=(1, 1, 8, 32, 32),
    ),
    "4LME_ABMIL": dict(
        requires=("timm",),
        factory="challenge.4abmil.model.create_model",
        kwargs=dict(model_type="swin", n_input_channels=3, num_classes=3),
        shape=(1, 1, 2, 224, 224),
    ),
    "5Pimed": dict(
        requires=(),
        factory="challenge.5pimed.model.create_model",
        kwargs=dict(model_name="resnet18", num_classes=3, n_input_channels=1, spatial_dims=3, norm="batch"),
        shape=(1, 1, 8, 32, 32),
    ),
}


def _load_factory(persistor_path):
    """Import a challenge factory by file path, as models_config.create_model does (names start with digits)."""
    module_path, func_name = persistor_path.rsplit(".", 1)
    file_path = MODELS_DIR.joinpath(*module_path.split(".")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(module_path, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def _build_challenge_model(name, monkeypatch):
    case = CHALLENGE_CASES[name]
    for package in case["requires"]:
        pytest.importorskip(package, reason=f"{name} needs {package}")
    if name == "4LME_ABMIL":
        import timm

        create = timm.create_model
        monkeypatch.setattr(timm, "create_model", lambda *a, **k: create(*a, **{**k, "pretrained": False}))
    torch.manual_seed(0)
    return _load_factory(case["factory"])(**case["kwargs"], loss_kwargs=dict(CLASS_WEIGHTS))


class _Cases(torch.utils.data.Dataset):
    def __init__(self, shape, n=1):
        generator = torch.Generator().manual_seed(0)
        self.items = [
            {"source": torch.randn(shape[1:], generator=generator), "target": torch.tensor([i % 3])} for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _train_one_step(model, shape, run_dir):
    """One optimiser step on CPU, checkpointed the way threedcnn_ptl does (last + top-1)."""
    checkpointing = ModelCheckpoint(dirpath=str(run_dir), save_last=True, save_top_k=1)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, limit_train_batches=1, limit_val_batches=0,
        num_sanity_val_steps=0, logger=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[checkpointing], default_root_dir=str(run_dir),
    )
    trainer.fit(model, train_dataloaders=torch.utils.data.DataLoader(_Cases(shape), batch_size=1))
    return checkpointing


def _saved_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def _assert_holds(model, state):
    loaded = model.state_dict()
    assert set(loaded) == set(state)
    for key, value in state.items():
        assert torch.equal(loaded[key], value), key


def test_registry_is_covered():
    """A challenge model added to models_config.CHALLENGE_MODELS must be added here too."""
    tree = ast.parse((MODELS_DIR / "models_config.py").read_text())
    registry = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CHALLENGE_MODELS" for t in node.targets)
    )
    assert set(registry) == set(CHALLENGE_CASES)
    for name, case in CHALLENGE_CASES.items():
        assert case["factory"] == registry[name]["persistor_path"], name


@pytest.mark.parametrize("name", sorted(CHALLENGE_CASES))
def test_challenge_model_reloads_from_its_lightning_checkpoints(name, tmp_path, monkeypatch):
    model = _build_challenge_model(name, monkeypatch)
    shape = CHALLENGE_CASES[name]["shape"]
    checkpointing = _train_one_step(model, shape, tmp_path)
    last, best = Path(checkpointing.last_model_path), Path(checkpointing.best_model_path)
    assert {"_class_weight", "loss.weight"} <= set(_saved_state(last))  # the buffers the reload must accept

    # Move the in-memory weights away from the files, so a pass proves the weights came from disk.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    moved = {key: value.clone() for key, value in model.state_dict().items()}

    # Exactly the calls in threedcnn_ptl (_save_best_checkpoint, _output_GT_and_classprobs_csv_test).
    last_model = model.load_checkpoint_from_file(tmp_path / last.name, strict=False)
    (tmp_path / "logs").mkdir()
    model.save_best_checkpoint(tmp_path / "logs", best)
    best_model = model.load_best_checkpoint(tmp_path / "logs", tmp_path, strict=False)

    for reloaded, path in ((last_model, last), (best_model, best)):
        assert reloaded is not model and type(reloaded) is type(model)
        assert not reloaded.training  # ready for the prediction export
        _assert_holds(reloaded, _saved_state(path))
        with torch.no_grad():
            logits = reloaded(torch.randn(shape))
        logits = logits[0] if isinstance(logits, tuple) else logits
        assert logits.shape == (1, 3) and torch.isfinite(logits).all()
    _assert_holds(model, moved)  # the trained model itself is untouched
    assert model.training


def test_class_level_call_still_rebuilds_through_lightning(tmp_path, monkeypatch):
    """Called on the class there is nothing to copy: Lightning's rebuild, and its #627 failure, remain."""
    model = _build_challenge_model("5Pimed", monkeypatch)
    checkpointing = _train_one_step(model, CHALLENGE_CASES["5Pimed"]["shape"], tmp_path)
    with pytest.raises(TypeError, match="backbone"):
        type(model).load_checkpoint_from_file(checkpointing.last_model_path)


# --- the copy helper itself, on a model small enough to build in milliseconds -----------------------

def _tiny(out_features=3, class_weights=True):
    torch.manual_seed(0)
    backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, out_features))
    return ModelWrapper(backbone=backbone, in_ch=1, num_classes=3,
                        loss_kwargs=dict(CLASS_WEIGHTS) if class_weights else {})


def _trained_state(model):
    """Train `model` as far as its buffers go (loss.weight appears at the first compute_loss()) and
    return a state dict like the one it would write, with every float value moved."""
    model.compute_loss(torch.randn(1, 3), torch.tensor([[1]]))
    return {key: value + 1 if value.is_floating_point() else value for key, value in model.state_dict().items()}


@pytest.mark.parametrize("wrapper", ["state_dict", "model", None], ids=["lightning", "persistor", "bare"])
def test_copy_accepts_lightning_persistor_and_bare_files(tmp_path, wrapper):
    model = _tiny()
    state = _trained_state(model)
    path = tmp_path / "weights.ckpt"
    torch.save({wrapper: state, "epoch": 3} if wrapper else state, path)
    _assert_holds(model.load_checkpoint_copy(path), state)


def test_non_parameter_buffers_may_differ_unless_strict(tmp_path):
    """A freshly built model has no loss.weight; a checkpoint written after training has one."""
    fresh = _tiny()
    assert "loss.weight" not in fresh.state_dict()
    path = tmp_path / "last.ckpt"
    torch.save({"state_dict": _trained_state(_tiny())}, path)
    fresh.load_checkpoint_copy(path)  # strict=False: accepted
    with pytest.raises(RuntimeError, match=r"unexpected keys \(loss.weight\)"):
        fresh.load_checkpoint_copy(path, strict=True)
    no_weights = _tiny(class_weights=False)  # e.g. a persistor model built without class weights
    _ = no_weights.load_checkpoint_copy(path)
    assert NON_PARAMETER_STATE_KEYS == {"_class_weight", "loss.weight"}


def test_other_architecture_is_refused_not_half_loaded(tmp_path):
    model = _tiny()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    other = _tiny()
    other.backbone.add_module("extra", torch.nn.Linear(3, 3))
    path = tmp_path / "other.ckpt"
    torch.save({"state_dict": other.state_dict()}, path)
    with pytest.raises(RuntimeError, match=r"does not fit this ModelWrapper: 2 unexpected keys \(backbone.extra"):
        model.load_checkpoint_copy(path)
    torch.save({"state_dict": _tiny(out_features=4).state_dict()}, path)  # 4 outputs: shape mismatch
    with pytest.raises(RuntimeError, match=r"backbone.1.weight: checkpoint \(4, 8\), model \(3, 8\)"):
        model.load_checkpoint_copy(path)
    _assert_holds(model, before)


def test_file_without_a_state_dict_is_refused(tmp_path):
    path = tmp_path / "junk.ckpt"
    torch.save({"epoch": 1, "note": "no weights"}, path)
    with pytest.raises(RuntimeError, match="holds no state dict"):
        _tiny().load_checkpoint_copy(path)


def test_state_dict_mismatch_reports_every_kind_of_difference():
    model = {"a": torch.zeros(2), "b": torch.zeros(3), "c": torch.zeros(1), "_class_weight": torch.zeros(3)}
    loaded = {"a": torch.zeros(2), "b": torch.zeros(4), "d": torch.zeros(1), "loss.weight": torch.zeros(3)}
    message = state_dict_mismatch(model, loaded)
    assert "1 of the model's 4 keys missing (c)" in message
    assert "1 unexpected keys (d)" in message
    assert "b: checkpoint (4,), model (3,)" in message
    assert state_dict_mismatch(model, dict(model)) is None
    assert "_class_weight" in state_dict_mismatch(model, {k: v for k, v in model.items() if k != "_class_weight"},
                                                  ignore=frozenset())
