"""BasicClassifier must not mutate its kwargs dicts (#430).

`__init__` takes mutable default arguments and then pops/updates them. Without
defensive copies, `loss_kwargs.pop('weight')` empties the *caller's* dict, so a
second model built from the same config trains silently unweighted -- no
exception, no warning, just a wrong loss on an imbalanced task.
"""

import inspect
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchmetrics")
pytest.importorskip("pytorch_lightning")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from conftest import MODELS_DIR, import_module_from_path  # noqa: E402

BASE_MODEL_PATH = MODELS_DIR / "base_model.py"


@pytest.fixture(scope="module")
def base_model():
    return import_module_from_path("_test_base_model_kwargs", BASE_MODEL_PATH)


def _weights():
    return torch.tensor([1.0, 2.0, 3.0])


def test_caller_loss_kwargs_is_not_mutated(base_model):
    cfg = {"weight": _weights()}
    base_model.BasicClassifier(in_ch=1, out_ch=3, spatial_dims=3, loss_kwargs=cfg)
    assert "weight" in cfg, "construction popped 'weight' out of the caller's dict"


def test_second_model_from_the_same_config_keeps_class_weights(base_model):
    """The silent-failure case: reusing one config dict for two models."""
    cfg = {"weight": _weights()}
    first = base_model.BasicClassifier(in_ch=1, out_ch=3, spatial_dims=3, loss_kwargs=cfg)
    second = base_model.BasicClassifier(in_ch=1, out_ch=3, spatial_dims=3, loss_kwargs=cfg)
    assert first._class_weight is not None
    assert second._class_weight is not None, "second model silently lost its class weights"
    assert torch.equal(second._class_weight, _weights())


def test_metric_kwargs_defaults_stay_empty(base_model):
    base_model.BasicClassifier(in_ch=1, out_ch=3, spatial_dims=3)
    params = inspect.signature(base_model.BasicClassifier.__init__).parameters
    assert params["aucroc_kwargs"].default == {}, "shared default dict was polluted"
    assert params["acc_kwargs"].default == {}, "shared default dict was polluted"


def test_caller_metric_kwargs_are_not_mutated(base_model):
    aucroc, acc = {}, {}
    base_model.BasicClassifier(in_ch=1, out_ch=3, spatial_dims=3, aucroc_kwargs=aucroc, acc_kwargs=acc)
    assert aucroc == {} and acc == {}


def test_models_with_different_class_counts_get_their_own_metrics(base_model):
    three = base_model.BasicClassifier(in_ch=1, out_ch=3, spatial_dims=3)
    two = base_model.BasicClassifier(in_ch=1, out_ch=2, spatial_dims=3)
    assert three.acc["val_"].num_classes == 3
    assert two.acc["val_"].num_classes == 2
