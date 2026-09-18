"""A fold that does not exist in a site's data must fail fast and say so (#411).

Before this guard, a missing fold surfaced only deep inside PyTorch, minutes into
a run, as `num_samples should be a positive integer value, but got num_samples=0`
-- which names neither the fold, nor the split, nor split.csv. Worse, an empty
`val`/`test` raised nothing at all (SequentialSampler happily yields zero
batches), so checkpoint selection silently monitored nothing.

The number of folds is a property of each site's split.csv (a generator setting,
not an invariant), so these tests assert validation against the DATA, never
against a hard-coded count.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
sys.path.insert(0, str(CUSTOM))

pytest.importorskip("torchio")
pd = pytest.importorskip("pandas")

from data.datasets.dataset_3d_odelia import (  # noqa: E402
    ODELIA_Dataset3D,
    ODELIAInstitutionManifest,
)

INSTITUTION = "CAM_1"
FOLDS_PRESENT = [0, 1, 2]
SPLITS = ["train", "val", "test"]


def _manifest(tmp_path, folds=FOLDS_PRESENT, empty_split=None):
    """Build a real manifest: 3 folds x train/val/test, 2 UIDs each."""
    split_rows, uids = [], []
    for fold in folds:
        for split in SPLITS:
            for i in range(2):
                uid = f"UID_{fold}_{split}_{i}"
                if not (empty_split is not None and split == empty_split):
                    split_rows.append({"UID": uid, "Fold": fold, "Split": split})
                    uids.append(uid)

    split_df = pd.DataFrame(split_rows)
    annotation_df = pd.DataFrame({"UID": uids, "Lesion": [0] * len(uids)})

    meta_dir = tmp_path / INSTITUTION / "metadata_unilateral"
    meta_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(meta_dir / "split.csv", index=False)

    return ODELIAInstitutionManifest(
        path_root=tmp_path,
        institution=INSTITUTION,
        config="unilateral",
        meta_dir="metadata_unilateral",
        data_dir="data_unilateral",
        annotation_df=annotation_df,
        split_df=split_df,
        image_uids=uids,
    )


def _build(manifest, fold, split):
    return ODELIA_Dataset3D(
        path_root=manifest.path_root,
        institutions=INSTITUTION,
        config="unilateral",
        split=split,
        fold=fold,
        manifests={INSTITUTION: manifest},
    )


def test_existing_fold_builds_every_split(tmp_path):
    manifest = _manifest(tmp_path)
    for split in SPLITS:
        assert len(_build(manifest, fold=1, split=split)) == 2


def test_missing_fold_names_the_folds_that_do_exist(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        _build(manifest, fold=3, split="train")

    message = str(exc.value)
    assert "FOLD=3" in message
    assert INSTITUTION in message
    assert "[0, 1, 2]" in message, "must list the folds actually present"
    assert "split.csv" in message, "must point at the file to fix"


@pytest.mark.parametrize("split", SPLITS)
def test_empty_split_is_fatal(tmp_path, split):
    """Covers the silent case: an empty val/test used to raise nothing at all."""
    manifest = _manifest(tmp_path, empty_split=split)
    with pytest.raises(RuntimeError, match=f"no samples for FOLD=1 / split='{split}'"):
        _build(manifest, fold=1, split=split)


def test_empty_split_can_be_allowed_explicitly(tmp_path, monkeypatch):
    """A site that genuinely holds no test data must have an escape hatch."""
    monkeypatch.setenv("ODELIA_ALLOW_EMPTY_SPLITS", "test")
    manifest = _manifest(tmp_path, empty_split="test")
    assert len(_build(manifest, fold=1, split="test")) == 0
    # ...but only for the split named
    monkeypatch.setenv("ODELIA_ALLOW_EMPTY_SPLITS", "test")
    empty_train = _manifest(tmp_path, empty_split="train")
    with pytest.raises(RuntimeError):
        _build(empty_train, fold=1, split="train")


def test_fold_present_but_no_images_on_disk_is_caught(tmp_path):
    """split.csv lists the fold, but no image UIDs survive -- a different failure."""
    manifest = _manifest(tmp_path)
    manifest.image_uids_set = set()  # site has the CSV rows but none of the images
    manifest._dataframe_cache.clear()

    with pytest.raises(RuntimeError) as exc:
        _build(manifest, fold=1, split="train")

    message = str(exc.value)
    assert "exists in split.csv" in message, "must distinguish this from a missing fold"
    assert "images actually present on disk" in message
