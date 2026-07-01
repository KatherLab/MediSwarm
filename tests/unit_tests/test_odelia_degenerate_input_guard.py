"""Tests for the degenerate/corrupt-input guard (skip-resample, not zero-fill)."""
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
CUSTOM = REPO / "application" / "jobs" / "_shared" / "custom"
sys.path.insert(0, str(CUSTOM))

# The ODELIA data pipeline requires torchio/pandas; the lightweight unit-test CI env
# doesn't install them, so skip this module there (it runs fully in the odelia image).
pytest.importorskip("torchio")
pytest.importorskip("pandas")

from data.augmentation.augmentations_3d import ZNormalization, DegenerateImageError  # noqa: E402
from data.datasets.dataset_3d_odelia import ODELIA_Dataset3D  # noqa: E402


def test_znorm_raises_degenerate_on_constant_image():
    z = ZNormalization(percentiles=(0.5, 99.5), masking_method=lambda x: (x > x.min()) & (x < x.max()))
    x = torch.ones((1, 4, 4, 4))                  # constant volume -> empty intensity mask
    mask = (x > x.min()) & (x < x.max())
    with pytest.raises(DegenerateImageError) as exc:
        z._znorm(x.clone(), mask, "img", "/path")
    assert exc.value.reason == "empty_or_too_small_mask"


def _make_dataset(transform, n=5):
    import pandas as pd
    ds = ODELIA_Dataset3D.__new__(ODELIA_Dataset3D)
    ds.df = pd.DataFrame({"UID": [f"u{i}" for i in range(n)], "Institution": ["X"] * n, "Lesion": [0] * n})
    ds.item_pointers = list(ds.df.index)
    ds.labels = ["Lesion"]
    ds.split = "train"
    ds.config = "unilateral"
    ds._bad_input_count = 0
    ds._bad_input_logged = set()
    ds.transform = transform
    ds.get_image_path = lambda uid, inst: f"/{uid}"
    ds._load_source_image = lambda path, inst, uid: uid
    return ds


def test_getitem_skips_degenerate_and_resamples():
    def transform(uid):
        if uid in ("u0", "u1"):
            raise DegenerateImageError(uid, "/p", "zero_std_after_masking_or_clipping", 0)
        return f"img:{uid}"
    ds = _make_dataset(transform, n=5)
    out = ds[0]                       # u0, u1 degenerate -> resample to u2 (NOT a zero image)
    assert out["uid"] == "u2"
    assert out["source"] == "img:u2"
    assert ds._bad_input_count == 2


def test_getitem_handles_corrupt_load():
    ds = _make_dataset(lambda uid: f"img:{uid}", n=5)

    def loader(path, inst, uid):
        if uid == "u0":
            raise EOFError("Compressed file ended before the end-of-stream marker was reached")
        return uid
    ds._load_source_image = loader
    out = ds[0]
    assert out["uid"] == "u1"
    assert ds._bad_input_count == 1


def test_getitem_aborts_when_too_many_bad():
    def transform(uid):
        raise DegenerateImageError(uid, "/p", "zero_std", 0)
    ds = _make_dataset(transform, n=40)
    with pytest.raises(RuntimeError, match="too many degenerate/corrupt inputs"):
        _ = ds[0]


def test_getitem_propagates_real_transform_bugs():
    def transform(uid):
        raise ValueError("a real bug, not a degenerate image")
    ds = _make_dataset(transform, n=5)
    with pytest.raises(ValueError, match="a real bug"):
        _ = ds[0]


def test_find_degenerate_inputs_flags_constant_and_corrupt():
    ds = _make_dataset(lambda uid: uid, n=3)

    def loader(path, inst, uid):
        if uid == "u0":
            raise OSError("corrupt")
        if uid == "u1":
            return types.SimpleNamespace(data=torch.ones((1, 4, 4, 4)))   # constant
        return types.SimpleNamespace(data=torch.randn((1, 4, 4, 4)))      # ok
    ds._load_source_image = loader
    reasons = {uid: reason for uid, inst, reason in ds.find_degenerate_inputs()}
    assert "load_error" in reasons["u0"]
    assert reasons["u1"] in ("empty_or_too_small_mask", "zero_std")
    assert "u2" not in reasons
