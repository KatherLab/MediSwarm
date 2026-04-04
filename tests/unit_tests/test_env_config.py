"""
Unit tests for env_config.py — environment variable loading and
run-directory generation.

env_config.py has a module-level ``from data.datasets import ODELIA_Dataset3D``
which is unavailable in CI, so we mock it via the _patch_heavy_imports fixture.
"""

import os
import sys
import re
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the shared custom directory is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"


@pytest.fixture(autouse=True)
def _importable_env_config(_patch_heavy_imports):
    """
    Add the custom dir to sys.path and clear the cached module so each
    test gets a fresh import.
    """
    if str(SHARED_CUSTOM_DIR) not in sys.path:
        sys.path.insert(0, str(SHARED_CUSTOM_DIR))

    for mod_name in list(sys.modules):
        if mod_name == "env_config":
            del sys.modules[mod_name]

    yield

    # Cleanup: remove from path (optional, keeps test isolation)
    if str(SHARED_CUSTOM_DIR) in sys.path:
        sys.path.remove(str(SHARED_CUSTOM_DIR))


def _import_env_config():
    sys.modules.pop("env_config", None)
    import env_config
    return env_config


# ===================================================================
# Tests for load_environment_variables()
# ===================================================================

class TestLoadEnvironmentVariables:
    """Test load_environment_variables() with mocked os.environ."""

    def test_returns_all_expected_keys(self, mock_env_vars):
        ec = _import_env_config()
        result = ec.load_environment_variables()
        expected_keys = {
            "site_name", "task_data_name", "scratch_dir", "data_dir",
            "max_epochs", "min_peers", "max_peers", "local_compare_flag",
            "use_adaptive_sync", "sync_frequency", "model_name",
            "prediction_flag", "mediswarm_version",
        }
        assert set(result.keys()) == expected_keys

    def test_reads_site_name(self, mock_env_vars):
        ec = _import_env_config()
        result = ec.load_environment_variables()
        assert result["site_name"] == "TEST_SITE"

    def test_reads_scratch_dir(self, mock_env_vars):
        ec = _import_env_config()
        result = ec.load_environment_variables()
        assert result["scratch_dir"] == mock_env_vars["SCRATCH_DIR"]

    def test_reads_data_dir(self, mock_env_vars):
        ec = _import_env_config()
        result = ec.load_environment_variables()
        assert result["data_dir"] == mock_env_vars["DATA_DIR"]

    def test_model_name(self, mock_env_vars):
        ec = _import_env_config()
        result = ec.load_environment_variables()
        assert result["model_name"] == "MST"

    def test_max_epochs_is_int(self, mock_env_vars):
        ec = _import_env_config()
        result = ec.load_environment_variables()
        assert isinstance(result["max_epochs"], int)
        assert result["max_epochs"] == 5

    def test_local_compare_flag_is_bool(self, mock_env_vars):
        ec = _import_env_config()
        result = ec.load_environment_variables()
        assert isinstance(result["local_compare_flag"], bool)
        assert result["local_compare_flag"] is False

    def test_local_compare_flag_true(self, tmp_scratch_dir, tmp_data_dir):
        env = {
            "SITE_NAME": "S1",
            "SCRATCH_DIR": tmp_scratch_dir,
            "DATA_DIR": tmp_data_dir,
            "LOCAL_COMPARE_FLAG": "True",
        }
        with patch.dict(os.environ, env, clear=False):
            ec = _import_env_config()
            result = ec.load_environment_variables()
            assert result["local_compare_flag"] is True

    def test_defaults_when_optional_vars_missing(self, tmp_scratch_dir, tmp_data_dir):
        """Only the three required vars are set; everything else should default."""
        env = {
            "SITE_NAME": "MINIMAL",
            "SCRATCH_DIR": tmp_scratch_dir,
            "DATA_DIR": tmp_data_dir,
        }
        with patch.dict(os.environ, env, clear=False):
            ec = _import_env_config()
            result = ec.load_environment_variables()
            assert result["model_name"] == "ResNet101"  # default
            assert result["max_epochs"] == 100  # default
            assert result["mediswarm_version"] == "unset"  # default

    def test_raises_without_site_name(self, tmp_scratch_dir, tmp_data_dir):
        """SITE_NAME is mandatory — KeyError expected."""
        env = {
            "SCRATCH_DIR": tmp_scratch_dir,
            "DATA_DIR": tmp_data_dir,
        }
        # Clear SITE_NAME if it happens to be set
        with patch.dict(os.environ, env, clear=True):
            ec = _import_env_config()
            with pytest.raises(KeyError):
                ec.load_environment_variables()

    def test_raises_without_scratch_dir(self, tmp_data_dir):
        """SCRATCH_DIR is mandatory."""
        env = {
            "SITE_NAME": "X",
            "DATA_DIR": tmp_data_dir,
        }
        with patch.dict(os.environ, env, clear=True):
            ec = _import_env_config()
            with pytest.raises(KeyError):
                ec.load_environment_variables()

    def test_raises_without_data_dir(self, tmp_scratch_dir):
        """DATA_DIR is mandatory."""
        env = {
            "SITE_NAME": "X",
            "SCRATCH_DIR": tmp_scratch_dir,
        }
        with patch.dict(os.environ, env, clear=True):
            ec = _import_env_config()
            with pytest.raises(KeyError):
                ec.load_environment_variables()


# ===================================================================
# Tests for generate_run_directory()
# ===================================================================

class TestGenerateRunDirectory:

    def test_returns_string_path(self, tmp_scratch_dir):
        ec = _import_env_config()
        result = ec.generate_run_directory(
            scratch_dir=tmp_scratch_dir,
            task_data_name="Odelia",
            model_name="MST",
            local_compare_flag=False,
        )
        assert isinstance(result, str)

    def test_path_contains_model_name(self, tmp_scratch_dir):
        ec = _import_env_config()
        result = ec.generate_run_directory(
            scratch_dir=tmp_scratch_dir,
            task_data_name="Odelia",
            model_name="ResNet50",
            local_compare_flag=False,
        )
        assert "ResNet50" in result

    def test_path_contains_task_data_name(self, tmp_scratch_dir):
        ec = _import_env_config()
        result = ec.generate_run_directory(
            scratch_dir=tmp_scratch_dir,
            task_data_name="MyData",
            model_name="MST",
            local_compare_flag=False,
        )
        assert "MyData" in result

    def test_swarm_mode_label(self, tmp_scratch_dir):
        ec = _import_env_config()
        result = ec.generate_run_directory(
            scratch_dir=tmp_scratch_dir,
            task_data_name="Odelia",
            model_name="MST",
            local_compare_flag=False,
        )
        assert "swarm_learning" in result

    def test_local_compare_mode_label(self, tmp_scratch_dir):
        ec = _import_env_config()
        result = ec.generate_run_directory(
            scratch_dir=tmp_scratch_dir,
            task_data_name="Odelia",
            model_name="MST",
            local_compare_flag=True,
        )
        assert "local_compare" in result

    def test_path_starts_with_scratch_dir(self, tmp_scratch_dir):
        ec = _import_env_config()
        result = ec.generate_run_directory(
            scratch_dir=tmp_scratch_dir,
            task_data_name="Odelia",
            model_name="MST",
            local_compare_flag=False,
        )
        assert result.startswith(tmp_scratch_dir)

    def test_creates_scratch_dir_if_missing(self, tmp_path):
        new_dir = str(tmp_path / "nonexistent" / "scratch")
        ec = _import_env_config()
        result = ec.generate_run_directory(
            scratch_dir=new_dir,
            task_data_name="Odelia",
            model_name="MST",
            local_compare_flag=False,
        )
        assert os.path.isdir(new_dir)

    def test_timestamp_format(self, tmp_scratch_dir):
        ec = _import_env_config()
        result = ec.generate_run_directory(
            scratch_dir=tmp_scratch_dir,
            task_data_name="Odelia",
            model_name="MST",
            local_compare_flag=False,
        )
        basename = os.path.basename(result)
        # Expected format: YYYY_MM_DD_HHMMSS_Odelia_MST_swarm_learning
        assert re.match(r"\d{4}_\d{2}_\d{2}_\d{6}_", basename)
