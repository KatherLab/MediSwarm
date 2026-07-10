"""FOLD env var selects the cross-validation fold end-to-end (#411).

`fold` is a *run-level* parameter: NVFlare ships one job to every client, so all
sites train the same fold and K-fold CV is K sequential swarm runs. Default is 0,
which reproduces the behavior from before the fold was configurable.
"""

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
LOGGER = logging.getLogger("test_fold_selection")


@pytest.fixture(autouse=True)
def _importable_env_config(_patch_heavy_imports):
    if str(SHARED_CUSTOM_DIR) not in sys.path:
        sys.path.insert(0, str(SHARED_CUSTOM_DIR))
    sys.modules.pop("env_config", None)
    yield
    sys.modules.pop("env_config", None)


def _import_env_config():
    import env_config
    return env_config


def test_fold_defaults_to_zero(mock_env_vars, monkeypatch):
    monkeypatch.delenv("FOLD", raising=False)
    assert _import_env_config().load_environment_variables()["fold"] == 0


def test_fold_is_read_from_the_environment(mock_env_vars, monkeypatch):
    monkeypatch.setenv("FOLD", "3")
    assert _import_env_config().load_environment_variables()["fold"] == 3


def test_blank_fold_falls_back_to_zero(mock_env_vars, monkeypatch):
    monkeypatch.setenv("FOLD", "  ")
    assert _import_env_config().load_environment_variables()["fold"] == 0


@pytest.mark.parametrize("bad", ["-1", "5", "99"])
def test_out_of_range_fold_is_rejected(mock_env_vars, monkeypatch, bad):
    monkeypatch.setenv("FOLD", bad)
    with pytest.raises(ValueError, match="FOLD must be in"):
        _import_env_config().load_environment_variables()


def test_non_integer_fold_is_rejected(mock_env_vars, monkeypatch):
    monkeypatch.setenv("FOLD", "two")
    with pytest.raises(ValueError, match="FOLD must be an integer"):
        _import_env_config().load_environment_variables()


def test_runtime_settings_carry_the_fold(mock_env_vars, monkeypatch):
    monkeypatch.setenv("FOLD", "4")
    ec = _import_env_config()
    assert ec.resolve_odelia_runtime_settings()["fold"] == 4


def test_every_dataset_is_built_for_the_selected_fold(mock_env_vars, monkeypatch, _patch_heavy_imports):
    monkeypatch.setenv("FOLD", "3")
    ec = _import_env_config()
    mock_dataset = _patch_heavy_imports

    ec.prepare_odelia_dataset(LOGGER, manifests={})

    assert mock_dataset.call_count == 3, "train/val/test datasets should be built"
    for call in mock_dataset.call_args_list:
        assert call.kwargs["fold"] == 3, f"dataset built for wrong fold: {call.kwargs.get('fold')}"


def test_run_directory_is_namespaced_by_fold(mock_env_vars, monkeypatch, _patch_heavy_imports):
    monkeypatch.setenv("FOLD", "3")
    ec = _import_env_config()
    *_, path_run_dir, run_name = ec.prepare_odelia_dataset(LOGGER, manifests={})
    assert "fold3" in run_name
    assert "fold3" in str(path_run_dir)


def test_unaugmented_datasets_use_the_same_fold(mock_env_vars, monkeypatch, _patch_heavy_imports):
    monkeypatch.setenv("FOLD", "2")
    ec = _import_env_config()
    mock_dataset = _patch_heavy_imports

    ec.prepare_odelia_dataset_without_augmentation(manifests={})

    assert mock_dataset.call_count == 3
    for call in mock_dataset.call_args_list:
        assert call.kwargs["fold"] == 2
