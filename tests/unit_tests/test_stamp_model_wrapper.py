"""
Unit tests for stamp_model_wrapper.py — environment reading and model creation.

stamp_model_wrapper.py imports STAMP at function level (inside _build_stamp_model).
_get_stamp_env() is pure os.environ logic and needs no STAMP dependencies.
create_stamp_model() and _build_stamp_model() require STAMP and are tested
with mocked imports.
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


def _clear_wrapper_modules():
    """Remove cached stamp_model_wrapper module."""
    for mod_name in list(sys.modules):
        if mod_name in ("stamp_model_wrapper",):
            del sys.modules[mod_name]


@pytest.fixture(autouse=True)
def _importable_wrapper():
    """Add STAMP custom dir to sys.path and clean up after."""
    original_path = sys.path[:]
    if str(STAMP_CUSTOM_DIR) not in sys.path:
        sys.path.insert(0, str(STAMP_CUSTOM_DIR))

    _clear_wrapper_modules()
    yield
    _clear_wrapper_modules()
    sys.path[:] = original_path


def _import_wrapper():
    _clear_wrapper_modules()
    import stamp_model_wrapper
    return stamp_model_wrapper


# ===================================================================
# Tests for _get_stamp_env()
# ===================================================================

class TestGetStampEnv:
    """Test _get_stamp_env() environment variable reading."""

    def test_returns_dict(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert isinstance(result, dict)

    def test_default_task(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert result["task"] == "classification"

    def test_default_model_name(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert result["model_name"] == "vit"

    def test_default_feature_type(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert result["feature_type"] == "tile"

    def test_default_dim_input(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert result["dim_input"] == 1024

    def test_default_num_classes(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert result["num_classes"] == 3

    def test_default_max_lr(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert result["max_lr"] == 1e-4

    def test_default_div_factor(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert result["div_factor"] == 25.0

    def test_custom_overrides(self):
        env = {
            "STAMP_TASK": "survival",
            "STAMP_MODEL_NAME": "attmil",
            "STAMP_FEATURE_TYPE": "conch",
            "STAMP_DIM_INPUT": "768",
            "STAMP_NUM_CLASSES": "5",
            "STAMP_MAX_LR": "2e-3",
            "STAMP_DIV_FACTOR": "50.0",
        }
        with patch.dict(os.environ, env, clear=False):
            smw = _import_wrapper()
            result = smw._get_stamp_env()
            assert result["task"] == "survival"
            assert result["model_name"] == "attmil"
            assert result["feature_type"] == "conch"
            assert result["dim_input"] == 768
            assert result["num_classes"] == 5
            assert result["max_lr"] == 2e-3
            assert result["div_factor"] == 50.0

    def test_returns_expected_keys(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        expected_keys = {
            "task", "model_name", "feature_type",
            "dim_input", "num_classes", "max_lr", "div_factor",
        }
        assert set(result.keys()) == expected_keys

    def test_dim_input_is_int(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert isinstance(result["dim_input"], int)

    def test_num_classes_is_int(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert isinstance(result["num_classes"], int)

    def test_max_lr_is_float(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert isinstance(result["max_lr"], float)

    def test_div_factor_is_float(self):
        smw = _import_wrapper()
        result = smw._get_stamp_env()
        assert isinstance(result["div_factor"], float)


# ===================================================================
# Tests for _build_stamp_model() — with mocked STAMP
# ===================================================================

class TestBuildStampModel:
    """Test _build_stamp_model() with mocked STAMP imports."""

    def _setup_stamp_mocks(self):
        """Create mocks for STAMP registry and config modules."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.randn(10, 10)]

        mock_lit_class = MagicMock(return_value=mock_model)
        mock_model_class = MagicMock()
        mock_model_name = MagicMock()
        mock_model_params = MagicMock()
        mock_model_params.model_dump.return_value = {"vit": {"use_alibi": True}}

        return {
            "lit_class": mock_lit_class,
            "model_class": mock_model_class,
            "model_name": mock_model_name,
            "model_params": mock_model_params,
            "model": mock_model,
        }

    def test_build_returns_model(self):
        mocks = self._setup_stamp_mocks()

        # Mock the STAMP imports
        mock_registry = MagicMock()
        mock_registry.ModelName.return_value = mocks["model_name"]
        mock_registry.load_model_class.return_value = (
            mocks["lit_class"], mocks["model_class"]
        )

        mock_config = MagicMock()
        mock_config.ModelParams.return_value = mocks["model_params"]

        stamp_modules = {
            "stamp": MagicMock(),
            "stamp.modeling": MagicMock(),
            "stamp.modeling.registry": mock_registry,
            "stamp.modeling.config": mock_config,
        }

        with patch.dict(sys.modules, stamp_modules):
            smw = _import_wrapper()
            env = smw._get_stamp_env()
            model = smw._build_stamp_model(env)
            assert model is mocks["model"]

    def test_build_raises_without_stamp(self):
        """Without STAMP installed, _build_stamp_model should raise ImportError."""
        # Ensure stamp is NOT in sys.modules
        saved = {}
        for key in list(sys.modules):
            if key.startswith("stamp"):
                saved[key] = sys.modules.pop(key)

        try:
            smw = _import_wrapper()
            env = smw._get_stamp_env()
            with pytest.raises(ImportError, match="STAMP is not installed"):
                smw._build_stamp_model(env)
        finally:
            sys.modules.update(saved)


# ===================================================================
# Tests for create_stamp_model() — kwargs override
# ===================================================================

class TestCreateStampModel:
    """Test create_stamp_model() entry point with kwargs."""

    def test_kwargs_override_env(self):
        """create_stamp_model() should merge kwargs with env vars."""
        smw = _import_wrapper()

        # Mock _build_stamp_model to capture the env dict
        captured_env = {}

        def fake_build(env):
            captured_env.update(env)
            return MagicMock()

        with patch.object(smw, "_build_stamp_model", side_effect=fake_build):
            smw.create_stamp_model(num_classes=5, dim_input=768)
            assert captured_env["num_classes"] == 5
            assert captured_env["dim_input"] == 768

    def test_none_kwargs_ignored(self):
        """None kwargs should not override env vars."""
        smw = _import_wrapper()

        captured_env = {}

        def fake_build(env):
            captured_env.update(env)
            return MagicMock()

        with patch.object(smw, "_build_stamp_model", side_effect=fake_build):
            smw.create_stamp_model(num_classes=None)
            # Should use default from env, not None
            assert captured_env["num_classes"] == 3  # default
