"""Tests for the degenerate/corrupt-input guard (skip-resample in training splits, stop in val/test, never zero-fill)."""
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


def _make_dataset(transform, n=5, split="train"):
    import pandas as pd
    ds = ODELIA_Dataset3D.__new__(ODELIA_Dataset3D)
    ds.df = pd.DataFrame({"UID": [f"u{i}" for i in range(n)], "Institution": ["X"] * n, "Lesion": [0] * n})
    ds.item_pointers = list(ds.df.index)
    ds.labels = ["Lesion"]
    ds.split = split
    ds.config = "unilateral"
    ds._bad_input_count = 0
    ds._bad_input_logged = set()
    ds._bad_input_reasons = {}
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
    with pytest.raises(RuntimeError) as exc:
        _ = ds[0]

    message = str(exc.value)
    assert "too many unusable inputs" in message
    assert "9x zero_std" in message, "must break the failures down by reason (#416)"
    assert "re-export or exclude" in message, "degenerate content -> excluding is the right remedy"


def test_abort_on_unreadable_files_says_fix_permissions_not_delete_data():
    """The #416 regression: a site deleted valid data because the guard said 'exclude'.

    When every failure is a file-access error the data is fine and the PERMISSIONS
    are wrong, so the message must say so and must NOT advise excluding anything.
    """
    def loader(path):
        raise PermissionError(13, "Permission denied", str(path))

    ds = _make_dataset(transform=lambda uid: uid, n=40)
    ds.load_img = loader
    ds._load_source_image = lambda path_img, institution, uid: loader(path_img)

    with pytest.raises(RuntimeError) as exc:
        _ = ds[0]

    message = str(exc.value)
    assert "PERMISSION/OWNERSHIP" in message
    assert "do NOT exclude the data" in message
    assert "load_error:PermissionError" in message


def test_getitem_propagates_real_transform_bugs():
    def transform(uid):
        raise ValueError("a real bug, not a degenerate image")
    ds = _make_dataset(transform, n=5)
    with pytest.raises(ValueError, match="a real bug"):
        _ = ds[0]


def _degenerate_u1(uid):
    if uid == "u1":
        raise DegenerateImageError(uid, "/p", "empty_or_too_small_mask", 0)
    return f"img:{uid}"


@pytest.fixture
def scratch(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def strict_eval(monkeypatch):
    monkeypatch.setenv("ODELIA_STRICT_EVAL_INPUTS", "1")


@pytest.mark.parametrize("split", ["val", "test"])
def test_evaluation_splits_substitute_by_default_and_say_so(split, scratch, caplog):
    """DE-2 default: a strict run must not end at round 0 over one bad case, so the neighbour is served, loudly."""
    ds = _make_dataset(_degenerate_u1, n=5, split=split)
    with caplog.at_level("WARNING"):
        assert ds[1]["uid"] == "u2"
    assert "SUBSTITUTED by a neighbouring case" in caplog.text and "ODELIA_STRICT_EVAL_INPUTS=1" in caplog.text


@pytest.mark.parametrize("split", ["val", "test"])
def test_strict_evaluation_splits_never_replace_a_bad_case_by_its_neighbour(split, scratch, strict_eval):
    """DE-2 with ODELIA_STRICT_EVAL_INPUTS=1: the neighbour would be scored twice and the bad case would vanish."""
    ds = _make_dataset(_degenerate_u1, n=5, split=split)
    assert ds[0]["uid"] == "u0"
    with pytest.raises(RuntimeError) as exc:
        ds[1]
    message = str(exc.value)
    assert f"unusable input in the {split} split" in message and "never replaced by its neighbour" in message
    assert "ODELIA_EXCLUDE_UIDS_FILE" in message
    assert "u1" not in message.replace(str(scratch), "")  # the UID stays in the local jsonl
    assert ds._bad_input_count == 1
    assert '"uid": "u1"' in (scratch / "odelia_bad_inputs.jsonl").read_text()
    assert ds[2]["uid"] == "u2"  # every other position still serves its own case


def test_an_unreadable_evaluation_case_says_fix_permissions(scratch, strict_eval):
    ds = _make_dataset(lambda uid: uid, n=3, split="test")

    def loader(path, inst, uid):
        raise PermissionError(13, "Permission denied", str(path))
    ds._load_source_image = loader
    with pytest.raises(RuntimeError) as exc:
        ds[0]
    assert "load_error:PermissionError" in str(exc.value) and "do NOT exclude the data" in str(exc.value)


def test_resampling_follows_the_split_unless_set_per_instance(scratch, strict_eval):
    val = _make_dataset(_degenerate_u1, n=5, split="val")
    val.resample_bad_inputs = True  # e.g. al_subset's pool view, which detects the substitution by UID
    assert val[1]["uid"] == "u2"
    train = _make_dataset(_degenerate_u1, n=5, split="train")
    assert train[1]["uid"] == "u2"
    train.resample_bad_inputs = False
    with pytest.raises(RuntimeError, match="never replaced by its neighbour"):
        train[1]


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
