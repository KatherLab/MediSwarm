import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHER_PATH = REPO_ROOT / "scripts" / "admin" / "patch_warm_start_job.py"


def _import_patcher():
    spec = importlib.util.spec_from_file_location("patch_warm_start_job_under_test", str(PATCHER_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG_TEMPLATE = """components = [
  {
    id = "swarm_client"
    args {
      min_responses_required = 5
    }
  }
  {
    id = "persistor"
    path = "warm_continue.WarmStartablePTFileModelPersistor"
    args {
      # WARM-CONTINUE
      source_ckpt_file_full_name = "/scratch/mediswarm_latest_global.pt"
      latest_global_path = "/scratch/mediswarm_latest_global.pt"
    }
  }
]
"""

SERVER_CONFIG_TEMPLATE = """workflows = [
  {
    args {
      num_rounds = 20
      min_clients = 5  # tolerate drops
    }
  }
]
"""


def test_patch_config_inserts_fresh_mode_before_source_checkpoint():
    patcher = _import_patcher()

    patched = patcher.patch_config_text(CONFIG_TEMPLATE, "fresh")

    assert 'warm_start_mode = "fresh"' in patched
    assert patched.index('warm_start_mode = "fresh"') < patched.index("source_ckpt_file_full_name")


def test_patch_config_maps_continue_to_require():
    patcher = _import_patcher()

    patched = patcher.patch_config_text(CONFIG_TEMPLATE, "continue")

    assert 'warm_start_mode = "require"' in patched


def test_patch_config_replaces_existing_mode():
    patcher = _import_patcher()
    existing = CONFIG_TEMPLATE.replace(
        'source_ckpt_file_full_name = "/scratch/mediswarm_latest_global.pt"',
        'warm_start_mode = "auto"\n      source_ckpt_file_full_name = "/scratch/mediswarm_latest_global.pt"',
    )

    patched = patcher.patch_config_text(existing, "continue")

    assert 'warm_start_mode = "require"' in patched
    assert 'warm_start_mode = "auto"' not in patched


def test_patch_config_can_patch_min_responses_required():
    patcher = _import_patcher()

    patched = patcher.patch_config_text(CONFIG_TEMPLATE, "fresh", min_responses_required=2)

    assert "min_responses_required = 2" in patched
    assert "min_responses_required = 5" not in patched


def test_patch_server_config_can_patch_rounds_and_min_clients():
    patcher = _import_patcher()

    patched = patcher.patch_server_config_text(SERVER_CONFIG_TEMPLATE, num_rounds=2, min_clients=2)

    assert "num_rounds = 2" in patched
    assert "min_clients = 2  # tolerate drops" in patched


def test_patch_numeric_assignment_rejects_missing_key():
    patcher = _import_patcher()

    with pytest.raises(ValueError, match="numeric assignment"):
        patcher.patch_numeric_assignment(CONFIG_TEMPLATE, "num_rounds", 2)


def test_patch_config_rejects_invalid_mode():
    patcher = _import_patcher()

    with pytest.raises(ValueError, match="Invalid warm-start mode"):
        patcher.patch_config_text(CONFIG_TEMPLATE, "resume")


def test_patch_job_dir_smoke_prepares_admin_local_continue_job(tmp_path):
    patcher = _import_patcher()
    job_dir = tmp_path / "mediswarm_jobs" / "ODELIA_ternary_classification_continue"
    config_dir = job_dir / "app" / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config_fed_client.conf"
    server_config_path = config_dir / "config_fed_server.conf"
    config_path.write_text(CONFIG_TEMPLATE)
    server_config_path.write_text(SERVER_CONFIG_TEMPLATE)

    patched_config = patcher.patch_job_dir(
        job_dir,
        "continue",
        num_rounds=2,
        min_clients=2,
        min_responses_required=2,
    )

    assert patched_config == config_path
    patched_client = config_path.read_text()
    patched_server = server_config_path.read_text()
    assert 'warm_start_mode = "require"' in patched_client
    assert "min_responses_required = 2" in patched_client
    assert "num_rounds = 2" in patched_server
    assert "min_clients = 2" in patched_server
