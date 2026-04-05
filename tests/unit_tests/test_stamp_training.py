"""
Unit tests for stamp_training.py — environment loading, weighted epochs,
and validation metric callback.

stamp_training.py imports lightning, torch, and STAMP (at function level).
We can test load_stamp_environment() and compute_weighted_epochs() without
any STAMP dependencies since they are pure Python / os.environ logic.
ValidationMetricCallback only needs lightning (available in CI).
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Ensure the STAMP custom directory is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_CUSTOM_DIR = (
    REPO_ROOT / "application" / "jobs" / "STAMP_classification" / "app" / "custom"
)
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"


def _clear_stamp_modules():
    """Remove cached stamp_training module so reimport works cleanly."""
    for mod_name in list(sys.modules):
        if mod_name in ("stamp_training",):
            del sys.modules[mod_name]


@pytest.fixture(autouse=True)
def _importable_stamp_training():
    """
    Add the STAMP custom dir to sys.path, mock STAMP-specific imports,
    and clear the cached module so each test gets a fresh import.
    """
    paths_to_add = [str(STAMP_CUSTOM_DIR), str(SHARED_CUSTOM_DIR)]
    original_path = sys.path[:]

    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Mock the stamp package and its submodules — they're not installed in CI
    stamp_mocks = {}
    stamp_modules = [
        "stamp", "stamp.modeling", "stamp.modeling.data",
        "stamp.modeling.config", "stamp.modeling.train",
        "stamp.modeling.transforms", "stamp.modeling.registry",
    ]
    for mod in stamp_modules:
        stamp_mocks[mod] = MagicMock()

    with patch.dict(sys.modules, stamp_mocks):
        _clear_stamp_modules()
        yield

    _clear_stamp_modules()
    sys.path[:] = original_path


def _import_stamp_training():
    _clear_stamp_modules()
    import stamp_training
    return stamp_training


# ===================================================================
# Tests for load_stamp_environment()
# ===================================================================

class TestLoadStampEnvironment:
    """Test load_stamp_environment() with mocked os.environ."""

    @pytest.fixture
    def stamp_env_vars(self, tmp_path):
        """Minimal required env vars for load_stamp_environment()."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        return {
            "SITE_NAME": "TEST_SITE_A",
            "SCRATCH_DIR": str(scratch),
            "STAMP_CLINI_TABLE": str(tmp_path / "clini.csv"),
            "STAMP_FEATURE_DIR": str(tmp_path / "features"),
        }

    def test_returns_dict(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert isinstance(result, dict)

    def test_reads_site_name(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["site_name"] == "TEST_SITE_A"

    def test_reads_scratch_dir(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["scratch_dir"] == stamp_env_vars["SCRATCH_DIR"]

    def test_reads_clini_table(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["clini_table"] == stamp_env_vars["STAMP_CLINI_TABLE"]

    def test_reads_feature_dir(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["feature_dir"] == stamp_env_vars["STAMP_FEATURE_DIR"]

    def test_default_task_is_classification(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["task"] == "classification"

    def test_default_model_name_is_vit(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["model_name"] == "vit"

    def test_default_dim_input(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["dim_input"] == 1024

    def test_default_num_classes(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["num_classes"] == 3

    def test_default_bag_size(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["bag_size"] == 512

    def test_default_batch_size(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["batch_size"] == 64

    def test_default_max_epochs(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["max_epochs"] == 32

    def test_default_patience(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["patience"] == 16

    def test_custom_overrides(self, stamp_env_vars):
        env = {
            **stamp_env_vars,
            "STAMP_TASK": "survival",
            "STAMP_MODEL_NAME": "attmil",
            "STAMP_DIM_INPUT": "768",
            "STAMP_NUM_CLASSES": "2",
            "STAMP_BAG_SIZE": "256",
            "STAMP_BATCH_SIZE": "32",
            "STAMP_MAX_EPOCHS": "50",
            "STAMP_PATIENCE": "10",
            "STAMP_MAX_LR": "5e-5",
            "STAMP_DIV_FACTOR": "10.0",
            "STAMP_SEED": "123",
        }
        with patch.dict(os.environ, env, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["task"] == "survival"
            assert result["model_name"] == "attmil"
            assert result["dim_input"] == 768
            assert result["num_classes"] == 2
            assert result["bag_size"] == 256
            assert result["batch_size"] == 32
            assert result["max_epochs"] == 50
            assert result["patience"] == 10
            assert result["max_lr"] == 5e-5
            assert result["div_factor"] == 10.0
            assert result["seed"] == 123

    def test_output_dir_auto_generated(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["output_dir"]
            assert "TEST_SITE_A" in result["output_dir"]
            assert "STAMP_vit" in result["output_dir"]

    def test_output_dir_explicit(self, stamp_env_vars, tmp_path):
        env = {**stamp_env_vars, "STAMP_OUTPUT_DIR": str(tmp_path / "custom_out")}
        with patch.dict(os.environ, env, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["output_dir"] == str(tmp_path / "custom_out")

    def test_raises_without_site_name(self, stamp_env_vars):
        env = {k: v for k, v in stamp_env_vars.items() if k != "SITE_NAME"}
        with patch.dict(os.environ, env, clear=True):
            st = _import_stamp_training()
            with pytest.raises(KeyError):
                st.load_stamp_environment()

    def test_raises_without_scratch_dir(self, stamp_env_vars):
        env = {k: v for k, v in stamp_env_vars.items() if k != "SCRATCH_DIR"}
        with patch.dict(os.environ, env, clear=True):
            st = _import_stamp_training()
            with pytest.raises(KeyError):
                st.load_stamp_environment()

    def test_raises_without_clini_table(self, stamp_env_vars):
        env = {k: v for k, v in stamp_env_vars.items() if k != "STAMP_CLINI_TABLE"}
        with patch.dict(os.environ, env, clear=True):
            st = _import_stamp_training()
            with pytest.raises(KeyError):
                st.load_stamp_environment()

    def test_raises_without_feature_dir(self, stamp_env_vars):
        env = {k: v for k, v in stamp_env_vars.items() if k != "STAMP_FEATURE_DIR"}
        with patch.dict(os.environ, env, clear=True):
            st = _import_stamp_training()
            with pytest.raises(KeyError):
                st.load_stamp_environment()

    def test_mediswarm_version_default(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["mediswarm_version"] == "unset"

    def test_mediswarm_version_custom(self, stamp_env_vars):
        env = {**stamp_env_vars, "MEDISWARM_VERSION": "1.3.0"}
        with patch.dict(os.environ, env, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            assert result["mediswarm_version"] == "1.3.0"

    def test_returns_all_expected_keys(self, stamp_env_vars):
        with patch.dict(os.environ, stamp_env_vars, clear=False):
            st = _import_stamp_training()
            result = st.load_stamp_environment()
            expected_keys = {
                "site_name", "scratch_dir", "mediswarm_version",
                "clini_table", "feature_dir", "slide_table", "output_dir",
                "task", "ground_truth_label", "patient_label",
                "filename_label", "time_label", "status_label",
                "model_name", "feature_type", "dim_input", "num_classes",
                "bag_size", "batch_size", "max_epochs", "patience",
                "max_lr", "div_factor", "num_workers", "seed",
            }
            assert set(result.keys()) == expected_keys


# ===================================================================
# Tests for compute_weighted_epochs()
# ===================================================================

class TestComputeWeightedEpochs:
    """Test compute_weighted_epochs() logic."""

    def test_reference_size_returns_base(self):
        """Site with exactly reference_size samples gets base_epochs."""
        with patch.dict(os.environ, {
            "STAMP_EPOCHS_PER_ROUND": "5",
            "STAMP_EPOCHS_REFERENCE_DATASET_SIZE": "200",
            "STAMP_EPOCHS_MAX_CAP": "20",
        }, clear=False):
            st = _import_stamp_training()
            result = st.compute_weighted_epochs(200, "test_site")
            assert result == 5

    def test_half_size_doubles_epochs(self):
        """Site with half the reference size gets double epochs."""
        with patch.dict(os.environ, {
            "STAMP_EPOCHS_PER_ROUND": "5",
            "STAMP_EPOCHS_REFERENCE_DATASET_SIZE": "200",
            "STAMP_EPOCHS_MAX_CAP": "20",
        }, clear=False):
            st = _import_stamp_training()
            result = st.compute_weighted_epochs(100, "test_site")
            assert result == 10

    def test_double_size_halves_epochs(self):
        """Site with double the reference size gets half epochs (min 1)."""
        with patch.dict(os.environ, {
            "STAMP_EPOCHS_PER_ROUND": "4",
            "STAMP_EPOCHS_REFERENCE_DATASET_SIZE": "200",
            "STAMP_EPOCHS_MAX_CAP": "20",
        }, clear=False):
            st = _import_stamp_training()
            result = st.compute_weighted_epochs(400, "test_site")
            assert result == 2

    def test_very_large_site_gets_minimum_1(self):
        """Very large site is clamped to minimum of 1 epoch."""
        with patch.dict(os.environ, {
            "STAMP_EPOCHS_PER_ROUND": "5",
            "STAMP_EPOCHS_REFERENCE_DATASET_SIZE": "200",
            "STAMP_EPOCHS_MAX_CAP": "20",
        }, clear=False):
            st = _import_stamp_training()
            result = st.compute_weighted_epochs(100000, "test_site")
            assert result == 1

    def test_very_small_site_capped_at_max(self):
        """Very small site is clamped to max_cap."""
        with patch.dict(os.environ, {
            "STAMP_EPOCHS_PER_ROUND": "5",
            "STAMP_EPOCHS_REFERENCE_DATASET_SIZE": "200",
            "STAMP_EPOCHS_MAX_CAP": "20",
        }, clear=False):
            st = _import_stamp_training()
            result = st.compute_weighted_epochs(5, "test_site")
            assert result == 20

    def test_zero_samples_returns_base(self):
        """Zero samples falls back to base_epochs."""
        with patch.dict(os.environ, {
            "STAMP_EPOCHS_PER_ROUND": "5",
            "STAMP_EPOCHS_REFERENCE_DATASET_SIZE": "200",
            "STAMP_EPOCHS_MAX_CAP": "20",
        }, clear=False):
            st = _import_stamp_training()
            result = st.compute_weighted_epochs(0, "test_site")
            assert result == 5

    def test_negative_samples_returns_base(self):
        """Negative samples falls back to base_epochs."""
        with patch.dict(os.environ, {
            "STAMP_EPOCHS_PER_ROUND": "5",
            "STAMP_EPOCHS_REFERENCE_DATASET_SIZE": "200",
            "STAMP_EPOCHS_MAX_CAP": "20",
        }, clear=False):
            st = _import_stamp_training()
            result = st.compute_weighted_epochs(-10, "test_site")
            assert result == 5

    def test_returns_integer(self):
        """Result is always an integer."""
        with patch.dict(os.environ, {
            "STAMP_EPOCHS_PER_ROUND": "5",
            "STAMP_EPOCHS_REFERENCE_DATASET_SIZE": "200",
            "STAMP_EPOCHS_MAX_CAP": "20",
        }, clear=False):
            st = _import_stamp_training()
            result = st.compute_weighted_epochs(150, "test_site")
            assert isinstance(result, int)

    def test_defaults_from_env(self):
        """Test default env var values (no STAMP_EPOCHS_* set)."""
        # Clear any existing STAMP_EPOCHS_ vars
        env_clean = {k: v for k, v in os.environ.items()
                     if not k.startswith("STAMP_EPOCHS_")}
        with patch.dict(os.environ, env_clean, clear=True):
            st = _import_stamp_training()
            result = st.compute_weighted_epochs(200, "test_site")
            # Default: base=5, reference=200, so 200/200*5 = 5
            assert result == 5


# ===================================================================
# Tests for ValidationMetricCallback
# ===================================================================

class TestValidationMetricCallback:
    """Test ValidationMetricCallback attributes."""

    def test_init_defaults(self):
        st = _import_stamp_training()
        cb = st.ValidationMetricCallback()
        assert cb.last_val_loss is None
        assert cb.last_val_auroc is None

    def test_on_validation_epoch_end_captures_loss(self):
        st = _import_stamp_training()
        cb = st.ValidationMetricCallback()

        # Mock trainer with callback_metrics
        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {
            "validation_loss": torch.tensor(0.42),
        }
        mock_module = MagicMock()

        cb.on_validation_epoch_end(mock_trainer, mock_module)
        assert cb.last_val_loss is not None
        assert abs(cb.last_val_loss - 0.42) < 1e-6

    def test_on_validation_epoch_end_captures_auroc(self):
        st = _import_stamp_training()
        cb = st.ValidationMetricCallback()

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {
            "validation_loss": torch.tensor(0.5),
            "val_MulticlassAUROC": torch.tensor(0.85),
        }
        mock_module = MagicMock()

        cb.on_validation_epoch_end(mock_trainer, mock_module)
        assert cb.last_val_auroc is not None
        assert abs(cb.last_val_auroc - 0.85) < 1e-6

    def test_on_validation_epoch_end_no_loss(self):
        st = _import_stamp_training()
        cb = st.ValidationMetricCallback()

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {}
        mock_module = MagicMock()

        cb.on_validation_epoch_end(mock_trainer, mock_module)
        assert cb.last_val_loss is None

    def test_on_validation_epoch_end_val_auroc_key(self):
        """Test that val_auroc key also works (alternative to val_MulticlassAUROC)."""
        st = _import_stamp_training()
        cb = st.ValidationMetricCallback()

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {
            "validation_loss": torch.tensor(0.3),
            "val_auroc": torch.tensor(0.91),
        }
        mock_module = MagicMock()

        cb.on_validation_epoch_end(mock_trainer, mock_module)
        assert cb.last_val_auroc is not None
        assert abs(cb.last_val_auroc - 0.91) < 1e-6

    def test_on_validation_epoch_end_prefers_val_auroc(self):
        """When both val_auroc and val_MulticlassAUROC exist, val_auroc comes first."""
        st = _import_stamp_training()
        cb = st.ValidationMetricCallback()

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {
            "validation_loss": torch.tensor(0.3),
            "val_auroc": torch.tensor(0.80),
            "val_MulticlassAUROC": torch.tensor(0.90),
        }
        mock_module = MagicMock()

        cb.on_validation_epoch_end(mock_trainer, mock_module)
        # val_auroc is checked first in the for loop
        assert abs(cb.last_val_auroc - 0.80) < 1e-6
