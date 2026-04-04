"""
Unit tests for the centralized model configuration module.

Tests cover:
  - CHALLENGE_MODELS / DEFAULT_MODEL dictionaries
  - get_all_model_names()
  - get_model_config()
  - get_persistor_config()
  - get_unified_model_name()
  - create_model() error paths (GPU guard)

models_config.py imports ``from models import ResNet, MST, Swin3D`` which
triggers imports of monai, einops, x_transformers, pytorch_lightning, etc.
We mock the entire ``models`` package before importing models_config so that
the tests run in CPU-only CI without any of those heavy dependencies.
"""

import os
import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import SHARED_CUSTOM_DIR, MODELS_DIR, import_module_from_path

MODELS_CONFIG_PATH = MODELS_DIR / "models_config.py"


# ---------------------------------------------------------------------------
# We need env_config importable (models_config does
#   ``from env_config import load_environment_variables``).
# And env_config does ``from data.datasets import ODELIA_Dataset3D``
# which is mocked by _patch_heavy_imports.
# ---------------------------------------------------------------------------


def _clear_custom_modules():
    """Remove cached modules from the custom tree so reimport works cleanly."""
    to_remove = [k for k in list(sys.modules) if k in (
        "models_config", "env_config",
    )]
    for k in to_remove:
        del sys.modules[k]


def _import_models_config():
    """
    Import models_config.py by file path.

    We need:
      1. ``env_config`` importable (models_config does ``from env_config import …``)
      2. ``models`` package mocked (models_config does ``from models import ResNet, MST, Swin3D``)
    Both are set up by the autouse fixture below.
    """
    _clear_custom_modules()
    return import_module_from_path("models_config", MODELS_CONFIG_PATH)


@pytest.fixture(autouse=True)
def _setup_imports(_patch_heavy_imports):
    """
    Prepare sys.path and sys.modules so that models_config.py can be imported
    without triggering heavy model dependencies.
    """
    paths_to_add = [str(SHARED_CUSTOM_DIR), str(MODELS_DIR)]
    original_path = sys.path[:]

    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Mock the ``models`` package so that
    # ``from models import ResNet, MST, Swin3D`` resolves to MagicMock objects
    # instead of triggering the real __init__.py → resnet → monai/einops chain.
    mock_models_pkg = MagicMock()
    mock_models_pkg.ResNet = MagicMock()
    mock_models_pkg.MST = MagicMock()
    mock_models_pkg.Swin3D = MagicMock()

    # Also mock sub-modules that the package __init__.py would import
    saved = {}
    keys_to_mock = [
        "models", "models.base_model", "models.resnet",
        "models.mst", "models.swin3D",
    ]
    for k in keys_to_mock:
        saved[k] = sys.modules.get(k)
        sys.modules[k] = mock_models_pkg

    _clear_custom_modules()

    yield

    # Restore original state
    for k in keys_to_mock:
        if saved[k] is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = saved[k]

    _clear_custom_modules()
    sys.path[:] = original_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logger():
    return logging.getLogger("test_models_config")


# ===================================================================
# Tests for CHALLENGE_MODELS dictionary
# ===================================================================

class TestChallengeModelsDict:
    """Validate the structure of the CHALLENGE_MODELS constant."""

    def test_contains_five_models(self):
        mc = _import_models_config()
        assert len(mc.CHALLENGE_MODELS) == 5

    def test_expected_model_keys(self):
        mc = _import_models_config()
        expected = {
            "1DivideAndConquer",
            "2BCN_AIM",
            "3agaldran",
            "4LME_ABMIL",
            "5Pimed",
        }
        assert set(mc.CHALLENGE_MODELS.keys()) == expected

    @pytest.mark.parametrize("model_name", [
        "1DivideAndConquer", "2BCN_AIM", "3agaldran", "4LME_ABMIL", "5Pimed",
    ])
    def test_each_model_has_required_keys(self, model_name):
        mc = _import_models_config()
        config = mc.CHALLENGE_MODELS[model_name]
        assert "team_name" in config
        assert "persistor_path" in config
        assert "persistor_args" in config
        assert isinstance(config["persistor_args"], dict)

    @pytest.mark.parametrize("model_name", [
        "1DivideAndConquer", "2BCN_AIM", "3agaldran", "4LME_ABMIL", "5Pimed",
    ])
    def test_persistor_args_has_num_classes(self, model_name):
        mc = _import_models_config()
        args = mc.CHALLENGE_MODELS[model_name]["persistor_args"]
        assert "num_classes" in args
        assert args["num_classes"] == 3


class TestDefaultModel:
    """Validate the DEFAULT_MODEL constant."""

    def test_default_is_mst(self):
        mc = _import_models_config()
        assert "MST" in mc.DEFAULT_MODEL

    def test_mst_has_persistor_path(self):
        mc = _import_models_config()
        assert mc.DEFAULT_MODEL["MST"]["persistor_path"] == "mst.MST"


# ===================================================================
# Tests for get_all_model_names()
# ===================================================================

class TestGetAllModelNames:
    def test_returns_list(self):
        mc = _import_models_config()
        names = mc.get_all_model_names()
        assert isinstance(names, list)

    def test_returns_five_entries(self):
        mc = _import_models_config()
        assert len(mc.get_all_model_names()) == 5

    def test_contains_expected_names(self):
        mc = _import_models_config()
        names = mc.get_all_model_names()
        for name in ["1DivideAndConquer", "2BCN_AIM", "3agaldran", "4LME_ABMIL", "5Pimed"]:
            assert name in names


# ===================================================================
# Tests for get_model_config()
# ===================================================================

class TestGetModelConfig:
    def test_known_challenge_model(self):
        mc = _import_models_config()
        config = mc.get_model_config(_logger(), "3agaldran")
        assert config is not None
        assert config["team_name"] == "3agaldran"

    def test_unknown_model_falls_back_to_mst(self):
        mc = _import_models_config()
        config = mc.get_model_config(_logger(), "NonExistentModel")
        assert config is not None
        assert config["persistor_path"] == "mst.MST"

    @pytest.mark.parametrize("model_name", [
        "1DivideAndConquer", "2BCN_AIM", "3agaldran", "4LME_ABMIL", "5Pimed",
    ])
    def test_each_challenge_model_returns_config(self, model_name):
        mc = _import_models_config()
        config = mc.get_model_config(_logger(), model_name)
        assert config is not None
        assert "persistor_path" in config


# ===================================================================
# Tests for get_persistor_config()
# ===================================================================

class TestGetPersistorConfig:
    def test_returns_path_and_args(self):
        mc = _import_models_config()
        pc = mc.get_persistor_config(_logger(), "5Pimed")
        assert pc is not None
        assert "persistor_path" in pc
        assert "persistor_args" in pc

    def test_unknown_model_returns_mst_persistor(self):
        mc = _import_models_config()
        pc = mc.get_persistor_config(_logger(), "BogusModel")
        assert pc is not None
        assert pc["persistor_path"] == "mst.MST"


# ===================================================================
# Tests for get_unified_model_name()
# ===================================================================

class TestGetUnifiedModelName:
    def test_explicit_challenge_variant(self):
        mc = _import_models_config()
        env = {"model_name": "ResNet101"}
        name = mc.get_unified_model_name(_logger(), "3agaldran", env)
        assert name == "challenge_3agaldran"

    def test_variant_none_reads_env(self):
        mc = _import_models_config()
        env = {"model_name": "ResNet50"}
        name = mc.get_unified_model_name(_logger(), None, env)
        assert name == "ResNet50"

    def test_variant_none_default_mst(self):
        mc = _import_models_config()
        env = {}
        name = mc.get_unified_model_name(_logger(), None, env)
        assert name == "MST"

    def test_variant_challenge_selects_first(self):
        mc = _import_models_config()
        env = {"model_name": "anything"}
        name = mc.get_unified_model_name(_logger(), "challenge", env)
        first = mc.get_all_model_names()[0]
        assert name == f"challenge_{first}"

    def test_variant_non_challenge_passthrough(self):
        mc = _import_models_config()
        env = {"model_name": "whatever"}
        name = mc.get_unified_model_name(_logger(), "Swin3D", env)
        assert name == "Swin3D"


# ===================================================================
# Tests for create_model() — error paths (no GPU in CI)
# ===================================================================

class TestCreateModelErrorPaths:
    """Test create_model() behaviour when no GPU is available."""

    def test_raises_without_gpu(self, mock_env_dict):
        mc = _import_models_config()
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="requires a GPU"):
                mc.create_model(
                    logger=_logger(),
                    model_name="MST",
                    env_vars=mock_env_dict,
                )

    def test_raises_for_unknown_model_with_gpu(self, mock_env_dict):
        mc = _import_models_config()
        with patch("torch.cuda.is_available", return_value=True):
            with pytest.raises(ValueError, match="Unsupported model name"):
                mc.create_model(
                    logger=_logger(),
                    model_name="TotallyBogusModel",
                    env_vars=mock_env_dict,
                )

    @pytest.mark.gpu
    def test_create_mst_with_gpu(self, mock_env_dict):
        """Only runs when a CUDA GPU is available."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available")
        mc = _import_models_config()
        model = mc.create_model(
            logger=_logger(),
            model_name="MST",
            env_vars=mock_env_dict,
        )
        assert model is not None
