"""
Shared test fixtures for MediSwarm unit tests.

Provides mock objects for CUDA, environment variables, datasets, and
other dependencies that are unavailable in CI (no GPU, no data files).
"""

import os
import sys
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
MODELS_DIR = SHARED_CUSTOM_DIR / "models"


# ---------------------------------------------------------------------------
# Environment variable fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_scratch_dir(tmp_path):
    """Create a temporary scratch directory."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return str(scratch)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data = tmp_path / "data"
    data.mkdir()
    return str(data)


@pytest.fixture
def mock_env_vars(tmp_scratch_dir, tmp_data_dir):
    """Set minimal required environment variables for testing."""
    env = {
        "SITE_NAME": "TEST_SITE",
        "SCRATCH_DIR": tmp_scratch_dir,
        "DATA_DIR": tmp_data_dir,
        "MODEL_NAME": "MST",
        "MAX_EPOCHS": "5",
        "MIN_PEERS": "2",
        "MAX_PEERS": "10",
        "LOCAL_COMPARE_FLAG": "False",
        "USE_ADAPTIVE_SYNC": "False",
        "SYNC_FREQUENCY": "1024",
        "PREDICT_FLAG": "ext",
        "MEDISWARM_VERSION": "test-0.0.0",
    }
    with patch.dict(os.environ, env, clear=False):
        yield env


@pytest.fixture
def mock_env_dict(tmp_scratch_dir, tmp_data_dir):
    """Return an env_vars dict as produced by load_environment_variables()."""
    return {
        "site_name": "TEST_SITE",
        "task_data_name": "Odelia",
        "scratch_dir": tmp_scratch_dir,
        "data_dir": tmp_data_dir,
        "max_epochs": 5,
        "min_peers": 2,
        "max_peers": 10,
        "local_compare_flag": False,
        "use_adaptive_sync": False,
        "sync_frequency": 1024,
        "model_name": "MST",
        "prediction_flag": "ext",
        "mediswarm_version": "test-0.0.0",
    }


# ---------------------------------------------------------------------------
# Module-import helpers
# ---------------------------------------------------------------------------

# Packages that are heavy / unavailable in CI but imported at the module level
# by the production code.  Each key is the top-level module name; nested
# sub-modules that are explicitly imported (``from X.Y import Z``) are listed
# separately so that attribute access on the mock works correctly.
_HEAVY_PACKAGES = [
    # Medical imaging
    "monai", "monai.networks", "monai.networks.nets",
    "monai.transforms",
    # ODELIA dataset package (lives inside the custom tree, but its deps —
    # e.g. tcia_utils — are heavy)
    "data", "data.datasets", "data.datamodules",
    # timm (used by some challenge models)
    "timm", "timm.models",
]


@pytest.fixture
def _patch_heavy_imports():
    """
    Patch heavy / unavailable imports so that modules like env_config and
    models_config can be imported in CI without the real dependency stack.

    This fixture mocks out entire package trees (monai, data.datasets, etc.)
    in ``sys.modules`` for the duration of the test.
    """
    mocks = {}
    for pkg in _HEAVY_PACKAGES:
        mocks[pkg] = MagicMock()

    # Make sure data.datasets.ODELIA_Dataset3D is accessible
    mock_dataset_class = MagicMock()
    mocks["data.datasets"].ODELIA_Dataset3D = mock_dataset_class

    with patch.dict(sys.modules, mocks):
        yield mock_dataset_class


# ---------------------------------------------------------------------------
# Helper: import a .py file by absolute path
# ---------------------------------------------------------------------------

def import_module_from_path(module_name: str, file_path: Path):
    """Import a Python module given its file path, bypassing normal lookup."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Simple mock dataset for DataModule tests
# ---------------------------------------------------------------------------

class MockDataset:
    """Minimal dataset that returns dummy tensors."""

    def __init__(self, size: int = 10, num_classes: int = 3):
        import torch
        self.size = size
        self.num_classes = num_classes
        self.data = torch.randn(size, 1, 32, 32, 32)
        self.targets = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        import torch
        return {
            "source": self.data[idx],
            "target": self.targets[idx].clone().detach(),
            "uid": f"mock_uid_{idx}",
        }


@pytest.fixture
def mock_datasets():
    """Create mock train/val/test datasets."""
    return MockDataset(20), MockDataset(5), MockDataset(5)
