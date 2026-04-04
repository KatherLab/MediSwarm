"""
Unit tests for the DataModule (PyTorch Lightning LightningDataModule).

Tests verify:
  - Initialization with/without optional parameters
  - train_dataloader, val_dataloader, test_dataloader creation
  - Weighted vs random sampling
  - Error paths when datasets are None
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler

# ---------------------------------------------------------------------------
# Direct import of datamodule.py by file path to avoid the "data" package
# name collision (our conftest mocks the "data" package for env_config).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATAMODULE_PATH = (
    REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
    / "data" / "datamodules" / "datamodule.py"
)

# Import MockDataset from conftest
from conftest import MockDataset, import_module_from_path


@pytest.fixture(autouse=True)
def _import_dm():
    """Make the DataModule class available to all tests via import helper."""
    # Nothing to mock for the datamodule itself — it only imports
    # pytorch_lightning and torch, which are available.
    pass


def _get_datamodule_class():
    """Import the DataModule class from the source file."""
    mod = import_module_from_path("_test_datamodule", DATAMODULE_PATH)
    return mod.DataModule


# ===================================================================
# Initialization tests
# ===================================================================

class TestDataModuleInit:

    def test_basic_init(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        assert dm.ds_train is ds_train
        assert dm.ds_val is ds_val
        assert dm.ds_test is ds_test

    def test_default_batch_size(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        assert dm.batch_size == 1

    def test_custom_batch_size(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(
            ds_train=ds_train, ds_val=ds_val, ds_test=ds_test,
            batch_size=4,
        )
        assert dm.batch_size == 4

    def test_separate_val_test_batch_sizes(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(
            ds_train=ds_train, ds_val=ds_val, ds_test=ds_test,
            batch_size=2, batch_size_val=3, batch_size_test=5,
        )
        assert dm.batch_size == 2
        assert dm.batch_size_val == 3
        assert dm.batch_size_test == 5

    def test_val_test_batch_default_to_train(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(
            ds_train=ds_train, ds_val=ds_val, ds_test=ds_test,
            batch_size=7,
        )
        assert dm.batch_size_val == 7
        assert dm.batch_size_test == 7

    def test_seed_stored(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(
            ds_train=ds_train, ds_val=ds_val, ds_test=ds_test,
            seed=42,
        )
        assert dm.seed == 42

    def test_weights_stored(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        weights = [1.0] * len(ds_train)
        dm = DataModule(
            ds_train=ds_train, ds_val=ds_val, ds_test=ds_test,
            weights=weights,
        )
        assert dm.weights is weights

    def test_hyperparameters_dict(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        assert isinstance(dm.hyperparameters, dict)
        assert "self" not in dm.hyperparameters
        assert "__class__" not in dm.hyperparameters


# ===================================================================
# Dataloader tests
# ===================================================================

class TestTrainDataloader:

    def test_returns_dataloader(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        dl = dm.train_dataloader()
        assert dl is not None

    def test_random_sampler_by_default(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        dl = dm.train_dataloader()
        assert isinstance(dl.sampler, RandomSampler)

    def test_weighted_sampler_when_weights_given(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        weights = [1.0] * len(ds_train)
        dm = DataModule(
            ds_train=ds_train, ds_val=ds_val, ds_test=ds_test,
            weights=weights,
        )
        dl = dm.train_dataloader()
        assert isinstance(dl.sampler, WeightedRandomSampler)

    def test_raises_without_train_set(self):
        DataModule = _get_datamodule_class()
        dm = DataModule(ds_train=None, ds_val=MockDataset(5), ds_test=MockDataset(5))
        with pytest.raises(AssertionError, match="training set"):
            dm.train_dataloader()

    def test_drop_last_true(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        dl = dm.train_dataloader()
        assert dl.drop_last is True

    def test_iterates_batches(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(
            ds_train=ds_train, ds_val=ds_val, ds_test=ds_test,
            batch_size=2, num_workers=0,
        )
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "source" in batch
        assert batch["source"].shape[0] == 2


class TestValDataloader:

    def test_returns_dataloader(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        dl = dm.val_dataloader()
        assert dl is not None

    def test_drop_last_false(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        dl = dm.val_dataloader()
        assert dl.drop_last is False

    def test_raises_without_val_set(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, _, _ = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=None, ds_test=MockDataset(5))
        with pytest.raises(AssertionError, match="validation set"):
            dm.val_dataloader()


class TestTestDataloader:

    def test_returns_dataloader(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        dl = dm.test_dataloader()
        assert dl is not None

    def test_drop_last_false(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, ds_test = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        dl = dm.test_dataloader()
        assert dl.drop_last is False

    def test_raises_without_test_set(self, mock_datasets):
        DataModule = _get_datamodule_class()
        ds_train, ds_val, _ = mock_datasets
        dm = DataModule(ds_train=ds_train, ds_val=ds_val, ds_test=None)
        with pytest.raises(AssertionError, match="test set"):
            dm.test_dataloader()
