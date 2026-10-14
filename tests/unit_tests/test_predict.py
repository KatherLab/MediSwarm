"""scripts/evaluation/predict.py: architecture guard, MST_SAMMed2D, FOLD/CONFIG pass-through.

predict.py runs in the image for --external_validation and for central evaluation. It used to load
every checkpoint with strict=False and report nothing, so a checkpoint evaluated under the wrong
--model-name was scored with mostly random weights (docs/EVALUATION_PITFALLS.md, E2). These tests
build real models from the model zoo on CPU (challenge_5Pimed, and MST_SAMMed2D with a tiny
SAM-Med2D encoder in place of the 267 M-parameter one) and replace only the dataset.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
for _module in ("sklearn", "pytorch_lightning", "torchmetrics", "monai", "einops", "x_transformers"):
    pytest.importorskip(_module)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICT_PY = REPO_ROOT / "scripts" / "evaluation" / "predict.py"


@pytest.fixture(scope="module")
def predict():
    spec = importlib.util.spec_from_file_location("predict_under_test", PREDICT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # puts the repo's job code on sys.path, as in the image
    # Import the model registry (and with it env_config) now, against the real dataset module,
    # before any test swaps a fake dataset in.
    import models_config  # noqa: F401
    return module


@pytest.fixture
def env(monkeypatch, tmp_path):
    """What docker.sh sets for --external_validation, minus the GPU; no SAM-Med2D weights anywhere."""
    for key, value in {
        "SITE_NAME": "TEST_1", "DATA_DIR": str(tmp_path / "data"), "SCRATCH_DIR": str(tmp_path / "scratch"),
        "MEDISWARM_ALLOW_CPU_MODEL": "1", "MEDISWARM_VERSION": "test",
    }.items():
        monkeypatch.setenv(key, value)
    for key in ("INSTITUTION", "CONFIG", "FOLD", "MODEL_NAME", "SAM_MED2D_CHECKPOINT", "SAM_MED2D_ALLOW_RANDOM_INIT"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


@pytest.fixture
def tiny_sam(monkeypatch, predict):
    """MST_SAMMed2D with a small encoder of the SAM-Med2D class (256 px input, 256 output channels)."""
    from models import mst
    from models.segment_anything.modeling import ImageEncoderViT

    def tiny_encoder():
        torch.manual_seed(0)
        return ImageEncoderViT(img_size=256, patch_size=64, embed_dim=8, depth=1, num_heads=1, out_chans=256,
                               use_rel_pos=True, window_size=2, global_attn_indexes=(), adapter_train=True)

    monkeypatch.setattr(mst, "_new_sam_med2d_encoder", tiny_encoder)
    monkeypatch.setattr(mst, "SAM_MED2D_CHECKPOINT_CANDIDATES", ())
    return mst


def _build(model_name, monkeypatch=None, allow_random=False):
    from models_config import create_model

    if monkeypatch is not None and allow_random:
        monkeypatch.setenv("SAM_MED2D_ALLOW_RANDOM_INIT", "1")
    torch.manual_seed(1)
    model = create_model(model_name=model_name, num_classes=3)
    if monkeypatch is not None:
        monkeypatch.delenv("SAM_MED2D_ALLOW_RANDOM_INIT", raising=False)
    return model


def _save_global(model, path, extra=None):
    """A persistor-format global (FL_global_model.pt) whose values differ from any fresh build."""
    state = {key: value + 0.5 if value.is_floating_point() else value for key, value in model.state_dict().items()}
    state.update(extra or {})
    torch.save({"model": state, "train_conf": {"train": {"model": type(model).__name__}}}, path)
    return state


# --- the architecture guard -------------------------------------------------------------------------

def test_mismatch_description_names_every_kind_of_difference(predict):
    model = {"a": torch.zeros(2), "b": torch.zeros(3), "c": torch.zeros(1), "_class_weight": torch.zeros(3)}
    loaded = {"a": torch.zeros(2), "b": torch.zeros(4), "d": torch.zeros(1), "loss.weight": torch.zeros(3)}
    message = predict.describe_architecture_mismatch(model, loaded)
    assert "1 of the model's 4 keys missing (c)" in message
    assert "1 unexpected keys (d)" in message
    assert "b: checkpoint (4,), model (3,)" in message
    assert predict.describe_architecture_mismatch(model, dict(model)) is None
    assert predict.NON_PARAMETER_STATE_KEYS == {"_class_weight", "loss.weight"}


@pytest.mark.parametrize("layout", ["model", "state_dict", "bare"])
def test_state_dict_found_in_every_checkpoint_layout(predict, layout):
    state = {"w": torch.ones(1)}
    ckpt = state if layout == "bare" else {layout: state, "epoch": 1}
    for checkpoint_type in ("state_dict", "lightning"):
        assert predict.extract_state_dict(ckpt, checkpoint_type, "x.pt") is state


def test_matching_checkpoint_loads_every_tensor(predict, env, tmp_path):
    saved = _save_global(_build("challenge_5Pimed"), tmp_path / "FL_global_model.pt")
    model = predict.load_model("challenge_5Pimed", str(tmp_path / "FL_global_model.pt"), "state_dict", 3,
                               torch.device("cpu"))
    assert not model.training
    loaded = model.state_dict()
    for key, value in saved.items():
        assert torch.equal(loaded[key], value), key


def test_trained_non_parameter_buffers_are_accepted(predict, env, tmp_path):
    """A trainer's model carries _class_weight and loss.weight; the model built here has neither."""
    weights = torch.tensor([1.0, 2.0, 3.0])
    _save_global(_build("challenge_5Pimed"), tmp_path / "g.pt", extra={"_class_weight": weights, "loss.weight": weights})
    predict.load_model("challenge_5Pimed", str(tmp_path / "g.pt"), "state_dict", 3, torch.device("cpu"))


def test_wrong_architecture_is_refused(predict, env, tiny_sam, tmp_path):
    """E2: a 5Pimed global evaluated as MST_SAMMed2D used to be scored with random weights."""
    _save_global(_build("challenge_5Pimed"), tmp_path / "g.pt")
    with pytest.raises(predict.ArchitectureMismatchError) as excinfo:
        predict.load_model("MST_SAMMed2D", str(tmp_path / "g.pt"), "state_dict", 3, torch.device("cpu"))
    message = str(excinfo.value)
    assert "does not hold MST_SAMMed2D weights" in message
    assert "keys missing" in message and "unexpected keys" in message
    assert "records model class 'ModelWrapper'" in message and "--model-name" in message


def test_different_class_count_is_refused(predict, env, tmp_path):
    from models_config import create_model

    torch.manual_seed(1)
    _save_global(create_model(model_name="challenge_5Pimed", num_classes=4), tmp_path / "g.pt")
    with pytest.raises(predict.ArchitectureMismatchError, match=r"another shape .*checkpoint \(4"):
        predict.load_model("challenge_5Pimed", str(tmp_path / "g.pt"), "state_dict", 3, torch.device("cpu"))


def test_lightning_checkpoint_loads(predict, env, tmp_path):
    model = _build("challenge_5Pimed")
    state = {key: value + 1 if value.is_floating_point() else value for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, "hyper_parameters": {}, "epoch": 3}, tmp_path / "last.ckpt")
    loaded = predict.load_model("challenge_5Pimed", str(tmp_path / "last.ckpt"), "lightning", 3, torch.device("cpu"))
    assert torch.equal(loaded.state_dict()["backbone.net.fc.weight"], state["backbone.net.fc.weight"])


# --- MST_SAMMed2D -----------------------------------------------------------------------------------

def test_mst_sammed2d_evaluates_without_the_sam_med2d_file(predict, env, tiny_sam, monkeypatch, tmp_path, caplog):
    """Every weight comes from the evaluated checkpoint, so the 2.56 GB pretrained file is not needed --
    while building the model for training without it still refuses."""
    saved = _save_global(_build("MST_SAMMed2D", monkeypatch, allow_random=True), tmp_path / "g.pt")
    with caplog.at_level("INFO"):
        model = predict.load_model("MST_SAMMed2D", str(tmp_path / "g.pt"), "state_dict", 3, torch.device("cpu"))
    assert any("SAM-Med2D pretrained weights not read" in r.getMessage() for r in caplog.records)
    loaded = model.state_dict()
    assert all(torch.equal(loaded[key], value) for key, value in saved.items())
    with pytest.raises(RuntimeError, match="MST_SAMMed2D refuses to start"):
        _build("MST_SAMMed2D")


# --- model name, FOLD, CONFIG, DATA_DIR ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["MST_SAMMed2D", "MST", "ResNet18", "MedicalNet", "1DivideAndConquer",
                                  "challenge_4LME_ABMIL", "challenge"])
def test_every_create_model_name_passes_the_name_check(predict, name):
    predict.check_model_name(name)


@pytest.mark.parametrize("name", ["Swin3D", "challenge_4abmil", "mst_sammed2d"])
def test_unknown_names_fail_before_any_data_is_read(predict, name):
    with pytest.raises(ValueError, match="Unsupported model name"):
        predict.check_model_name(name)


class _FakeDataset(torch.utils.data.Dataset):
    """Stands in for ODELIA_Dataset3D: records its arguments, serves three tiny cases."""

    calls = []

    def __init__(self, **kwargs):
        _FakeDataset.calls.append(kwargs)
        generator = torch.Generator().manual_seed(0)
        self.items = [
            {"source": torch.randn(1, 2, 32, 32, generator=generator), "target": torch.tensor([i]), "uid": f"case_{i}"}
            for i in range(3)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


@pytest.fixture
def fake_dataset(monkeypatch):
    import torch.multiprocessing

    _FakeDataset.calls = []
    module = types.ModuleType("data.datasets")
    module.ODELIA_Dataset3D = _FakeDataset
    monkeypatch.setitem(sys.modules, "data.datasets", module)
    # predict.py starts one loader worker per CPU; three cases load fine in-process.
    monkeypatch.setattr(torch.multiprocessing, "cpu_count", lambda: 0)
    return _FakeDataset


def test_fold_config_and_data_dir_reach_the_dataset(predict, env, fake_dataset, monkeypatch):
    monkeypatch.setenv("INSTITUTION", "UMCU")
    settings = predict.resolve_data_settings("unilateral_BD", "2")
    predict.setup_dataloader("test", 1, settings)
    assert fake_dataset.calls[-1] == dict(institutions="UMCU", split="test", config="unilateral_BD", fold=2,
                                          path_root=str(env / "data"))
    monkeypatch.setenv("CONFIG", "original")
    monkeypatch.setenv("FOLD", "1")
    predict.setup_dataloader("val", 1)  # defaults come from the environment
    assert fake_dataset.calls[-1]["config"] == "original" and fake_dataset.calls[-1]["fold"] == 1


@pytest.mark.parametrize("fold", ["-1", "x", "1.5"])
def test_invalid_fold_is_refused(predict, env, fold):
    with pytest.raises(ValueError, match="FOLD must be"):
        predict.resolve_data_settings("unilateral", fold)


def _run_main(predict, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["predict.py", *args])
    try:
        predict.main()
    except SystemExit as exc:
        return exc.code
    return 0


def test_main_reports_what_it_evaluated(predict, env, tiny_sam, fake_dataset, monkeypatch, tmp_path, capsys):
    """End to end on CPU: MST_SAMMed2D global, CONFIG/FOLD from the command line, provenance in the JSON."""
    _save_global(_build("MST_SAMMed2D", monkeypatch, allow_random=True), tmp_path / "FL_global_model.pt")
    code = _run_main(predict, monkeypatch, "--checkpoint", str(tmp_path / "FL_global_model.pt"),
                     "--model-name", "MST_SAMMed2D", "--device", "cpu", "--output-dir", str(tmp_path / "out"),
                     "--config", "unilateral_BD", "--fold", "3")
    assert code == 0
    printed = capsys.readouterr().out
    assert f"Checkpoint:   {tmp_path / 'FL_global_model.pt'}" in printed
    assert "Architecture: MST_SAMMed2D (model class MST" in printed
    assert "CONFIG=unilateral_BD, FOLD=3" in printed
    result = json.loads((tmp_path / "out" / "prediction_results.json").read_text())[0]
    assert result["config"] == "unilateral_BD" and result["fold"] == 3 and result["institution"] == "TEST_1"
    assert result["model_class"] == "MST" and len(result["checkpoint_sha256"]) == 64
    assert result["metrics"]["num_samples"] == 3
    assert fake_dataset.calls[-1]["fold"] == 3


def test_main_fails_on_a_wrong_architecture(predict, env, fake_dataset, monkeypatch, tmp_path):
    _save_global(_build("challenge_5Pimed"), tmp_path / "g.pt")
    code = _run_main(predict, monkeypatch, "--checkpoint", str(tmp_path / "g.pt"), "--model-name", "challenge_2BCN_AIM",
                     "--device", "cpu", "--output-dir", str(tmp_path / "out"))
    assert code == 1
    result = json.loads((tmp_path / "out" / "prediction_results.json").read_text())[0]
    assert result["status"] == "error" and "does not hold challenge_2BCN_AIM weights" in result["error"]
