"""Unit tests for models/medicalnet.py (the VHIO model, MedicalNet 3D-ResNet34).

No GPU, no dataset and no download: the backbone is built without weights for the forward
test, and the checkpoint loader is exercised with a checkpoint written from a fresh backbone
in the original MedicalNet format (`state_dict` under `module.` prefixes plus a `conv_seg`
head that must be ignored).
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("torchmetrics")
pytest.importorskip("monai")
pytest.importorskip("einops")
pytest.importorskip("x_transformers")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from conftest import REPO_ROOT, SHARED_CUSTOM_DIR  # noqa: E402

if str(SHARED_CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_CUSTOM_DIR))

from models.medicalnet import MedicalNet, _MedicalNetResNet34  # noqa: E402


def test_forward_shape_without_weights():
    model = MedicalNet(n_input_channels=1, num_classes=3, spatial_dims=3)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, 32, 32, 16))
    assert out.shape == (2, 3)


def test_rejects_multi_channel_or_2d():
    with pytest.raises(ValueError):
        MedicalNet(n_input_channels=3, num_classes=3, spatial_dims=3)
    with pytest.raises(ValueError):
        MedicalNet(n_input_channels=1, num_classes=3, spatial_dims=2)


def test_loads_original_medicalnet_checkpoint_format(tmp_path):
    source = _MedicalNetResNet34(num_classes=3)
    state = {"module." + k: v for k, v in source.backbone.state_dict().items()}
    state["module.conv_seg.0.weight"] = torch.zeros(1)  # segmentation head, must be skipped
    ckpt = tmp_path / "resnet_34_23dataset.pth"
    torch.save({"state_dict": state}, ckpt)

    loaded = _MedicalNetResNet34(num_classes=3, pretrained_path=str(ckpt))
    for k, v in source.backbone.state_dict().items():
        if k.startswith("conv_seg."):
            continue  # segmentation head: skipped by the loader and unused by the classifier
        assert torch.equal(loaded.backbone.state_dict()[k], v), k


def test_checkpoint_with_foreign_keys_is_rejected(tmp_path):
    ckpt = tmp_path / "wrong.pth"
    torch.save({"state_dict": {"module.not_a_layer.weight": torch.zeros(1)}}, ckpt)
    with pytest.raises(RuntimeError):
        _MedicalNetResNet34(num_classes=3, pretrained_path=str(ckpt))


def test_weight_file_name_matches_build_script():
    build_script = REPO_ROOT / "scripts" / "build" / "_cacheAndCopyPretrainedModelWeights.sh"
    from models.models_config import MEDICALNET_WEIGHTS_FILE
    assert MEDICALNET_WEIGHTS_FILE in build_script.read_text()


def test_frozen_batchnorm_ignores_training_flag():
    """Lightning sets module.training directly; the output must not depend on it."""
    from models.medicalnet import FrozenBatchNorm3d
    torch.manual_seed(0)
    model = _MedicalNetResNet34(num_classes=3, freeze_bn=True)
    bns = [m for m in model.modules() if isinstance(m, FrozenBatchNorm3d)]
    n_plain = sum(isinstance(m, torch.nn.BatchNorm3d) for m in _MedicalNetResNet34(num_classes=3).modules())
    assert len(bns) == n_plain > 0 and not any(isinstance(m, torch.nn.BatchNorm3d) for m in model.modules())
    x = torch.randn(2, 1, 16, 32, 32)
    stats = bns[5].running_mean.clone()
    for m in model.modules():
        m.training = True  # what Lightning's mode restore does
    with torch.no_grad():
        y_train = model(x)
    model.eval()
    with torch.no_grad():
        y_eval = model(x)
    assert torch.allclose(y_train, y_eval, atol=1e-6)
    assert torch.equal(bns[5].running_mean, stats)
    assert bns[5].weight.requires_grad


def test_frozen_batchnorm_keeps_state_dict_keys():
    frozen = _MedicalNetResNet34(num_classes=3, freeze_bn=True).state_dict().keys()
    plain = _MedicalNetResNet34(num_classes=3, freeze_bn=False).state_dict().keys()
    assert set(frozen) == set(plain)


def test_learning_rate_env_override(monkeypatch):
    monkeypatch.setenv("MEDICALNET_LR", "5e-5")
    model = MedicalNet(n_input_channels=1, num_classes=3, spatial_dims=3)
    assert model.optimizer_kwargs["lr"] == 5e-5
