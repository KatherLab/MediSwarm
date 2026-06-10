import json
import importlib.machinery
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
sys.path.insert(0, str(SHARED_CUSTOM_DIR))

fake_pandas = types.ModuleType("pandas")
fake_pandas.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)
sys.modules.setdefault("pandas", fake_pandas)


class _FakeTorchioZNormalization:
    def __init__(self, masking_method=None, **_kwargs):
        self.masking_method = masking_method

    def znorm(self, image_data, mask):
        values = image_data.masked_select(mask).float()
        if values.numel() == 0:
            return None
        std = values.std(unbiased=False)
        if std.item() == 0:
            return None
        return (image_data - values.mean().to(image_data.dtype)) / std.to(image_data.dtype)


class _FakeScalarImage:
    def __init__(self, tensor=None, affine=None, path=None):
        self.data = tensor
        self.affine = affine
        self.path = path


class _FakeSubject(dict):
    def check_consistent_space(self):
        return None


class _FakeCropOrPad:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


fake_torchio = types.ModuleType("torchio")
fake_torchio.__spec__ = importlib.machinery.ModuleSpec("torchio", loader=None)
fake_torchio.ZNormalization = _FakeTorchioZNormalization
fake_torchio.ScalarImage = _FakeScalarImage
fake_torchio.LabelMap = _FakeScalarImage
fake_torchio.Subject = _FakeSubject
fake_torchio.Image = _FakeScalarImage
fake_torchio.CropOrPad = _FakeCropOrPad
fake_torchio.INTENSITY = "intensity"
fake_torchio.Compose = MagicMock()
fake_torchio.Lambda = MagicMock()
fake_torchio.ToCanonical = MagicMock()
fake_torchio.Resample = MagicMock()
fake_torchio.Flip = MagicMock()
fake_torchio.OneOf = MagicMock()
fake_torchio.RandomAffine = MagicMock()
fake_torchio.RandomFlip = MagicMock()
fake_torchio.RandomNoise = MagicMock()
fake_torchio.Pad = MagicMock()
fake_torchio.Crop = MagicMock()

fake_torchio_typing = types.ModuleType("torchio.typing")
fake_torchio_typing.__spec__ = importlib.machinery.ModuleSpec("torchio.typing", loader=None)
fake_torchio_typing.TypeRangeFloat = tuple
fake_torchio_typing.TypeTripletInt = tuple

fake_torchio_transform = types.ModuleType("torchio.transforms.transform")
fake_torchio_transform.__spec__ = importlib.machinery.ModuleSpec("torchio.transforms.transform", loader=None)
fake_torchio_transform.TypeMaskingMethod = object

sys.modules.setdefault("torchio", fake_torchio)
sys.modules.setdefault("torchio.typing", fake_torchio_typing)
sys.modules.setdefault("torchio.transforms.transform", fake_torchio_transform)

from data.augmentation.augmentations_3d import ZNormalization  # noqa: E402
from data.datasets.dataset_3d_odelia import ODELIA_Dataset3D  # noqa: E402


def _read_guard_records(tmp_path):
    log_path = tmp_path / "odelia_zero_std_guard.jsonl"
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def test_zero_std_masked_image_returns_zeros_and_logs_locally(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    image_data = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    mask = torch.ones_like(image_data, dtype=torch.bool)

    normalized = ZNormalization(percentiles=(0, 100))._znorm(
        image_data=image_data,
        mask=mask,
        image_name="img",
        image_path="/sensitive/site/UID/Sub_1.nii.gz",
    )

    assert torch.equal(normalized, torch.zeros_like(image_data))
    record = _read_guard_records(tmp_path)[0]
    assert record["reason"] == "zero_std_after_masking_or_clipping"
    assert record["masked_voxels"] == 8
    assert record["image_path"].endswith("Sub_1.nii.gz")

    stdout = capsys.readouterr().out
    assert "[ODELIA_ZERO_STD_GUARD]" in stdout
    assert "/sensitive/site/UID" not in stdout


def test_empty_or_tiny_mask_returns_zeros_and_logs_locally(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    image_data = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    mask = torch.zeros_like(image_data, dtype=torch.bool)
    mask[..., 0, 0, 0] = True

    normalized = ZNormalization(percentiles=(0, 100))._znorm(
        image_data=image_data,
        mask=mask,
        image_name="img",
        image_path="/sensitive/site/UID/Sub_1.nii.gz",
    )

    assert torch.equal(normalized, torch.zeros_like(image_data))
    record = _read_guard_records(tmp_path)[0]
    assert record["reason"] == "empty_or_too_small_mask"
    assert record["masked_voxels"] == 1


def test_normal_image_still_normalizes_without_guard_log(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    image_data = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    mask = torch.ones_like(image_data, dtype=torch.bool)

    normalized = ZNormalization(percentiles=(0, 100))._znorm(
        image_data=image_data,
        mask=mask,
        image_name="img",
        image_path="/sensitive/site/UID/Sub_1.nii.gz",
    )

    assert torch.isfinite(normalized).all()
    assert not torch.equal(normalized, torch.zeros_like(image_data))
    assert normalized.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert not (tmp_path / "odelia_zero_std_guard.jsonl").exists()


def test_odelia_dataset_attaches_source_path_before_transform(tmp_path):
    source_path = tmp_path / "USZ_1" / "data_unilateral" / "uid-1" / "Sub_1.nii.gz"
    captured = {}

    class _Values:
        def __init__(self, values):
            self.values = values

    class _Row(dict):
        def __getitem__(self, key):
            if isinstance(key, list):
                return _Values([dict.__getitem__(self, item) for item in key])
            return dict.__getitem__(self, key)

    class _Loc:
        def __init__(self, row):
            self.row = row

        def __getitem__(self, _idx):
            return self.row

    class _Frame:
        def __init__(self, row):
            self.loc = _Loc(row)

    dataset = object.__new__(ODELIA_Dataset3D)
    dataset.item_pointers = [0]
    dataset.df = _Frame(_Row({"UID": "uid-1", "Institution": "USZ_1", "Lesion": 2}))
    dataset.labels = ["Lesion"]
    dataset.get_image_path = lambda uid, institution: source_path
    dataset._load_source_image = lambda path_img, institution, uid: _FakeScalarImage(
        tensor=torch.ones((1, 2, 2, 2), dtype=torch.float32),
        affine=np.eye(4),
    )

    def _capture_transform(image):
        captured["path"] = image.path
        return image.data

    dataset.transform = _capture_transform

    sample = ODELIA_Dataset3D.__getitem__(dataset, 0)

    assert captured["path"] == str(source_path)
    assert sample["uid"] == "uid-1"
    assert sample["target"].tolist() == [2]
