import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHER_PATH = REPO_ROOT / "scripts" / "admin" / "patch_warm_start_job.py"
PREPARE_HELPER_PATH = REPO_ROOT / "kit_admin_tools" / "prepare_odelia_job.sh"
KIT_INJECTOR_PATH = REPO_ROOT / "scripts" / "build" / "_injectLiveSyncIntoStartupKits.sh"


def _import_patcher():
    spec = importlib.util.spec_from_file_location("patch_warm_start_job_under_test", str(PATCHER_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_helper_defaults_to_exact_odelia_eight_unless_counts_are_explicit():
    helper = PREPARE_HELPER_PATH.read_text()

    assert (
        'DEFAULT_STRICT_CLIENTS="CAM_1,VHIO_1,USZ_1,RUMC_1,MHA_1,RSH_1,UMCU_1,UKA_1"'
        in helper
    )
    assert 'STRICT_CLIENTS="$DEFAULT_STRICT_CLIENTS"' in helper
    assert 'CUSTOM_CLIENT_COUNTS_SET=true' in helper
    assert '[ "$STRICT_CLIENTS_EXPLICIT" != true ] && [ "$CUSTOM_CLIENT_COUNTS_SET" = true ]' in helper


def test_prepare_helper_stages_before_replacing_and_uses_kit_paired_patcher():
    helper = PREPARE_HELPER_PATH.read_text()
    injector = KIT_INJECTOR_PATH.read_text()

    assert 'PATCH_ARGS=(--job-dir "/job_out/$TEMP_NAME"' in helper
    assert 'mv -- "$TEMP_HOST" "$DEST_HOST"' in helper
    assert 'rm -rf "$DEST_HOST"' not in helper
    assert '-v "$PATCHER_HOST":/mediswarm_tools/patch_warm_start_job.py:ro' in helper
    assert 'python3 /mediswarm_tools/patch_warm_start_job.py' in helper
    assert 'cp "$ADMIN_PATCHER_SOURCE" "$STARTUP_DIR/patch_warm_start_job.py"' in injector


CONFIG_TEMPLATE = """components = [
  {
    id = "swarm_client"
    args {
      final_result_ack_timeout = 86400
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

META_CONFIG_TEMPLATE = """name = "challenge_1DivideAndConquer"
resource_spec {}
deploy_map {
  app = [
    "@ALL"
  ]
}
min_clients = 2
mandatory_clients = []
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


def test_patch_config_can_insert_broadcast_last_result():
    patcher = _import_patcher()

    patched = patcher.patch_config_text(CONFIG_TEMPLATE, "fresh", broadcast_last_result=False)

    assert "broadcast_last_result = false" in patched
    assert patched.index("final_result_ack_timeout = 86400") < patched.index("broadcast_last_result = false")


def test_patch_config_can_replace_broadcast_last_result():
    patcher = _import_patcher()
    existing = CONFIG_TEMPLATE.replace(
        "final_result_ack_timeout = 86400",
        "final_result_ack_timeout = 86400\n      broadcast_last_result = true",
    )

    patched = patcher.patch_config_text(existing, "fresh", broadcast_last_result=False)

    assert "broadcast_last_result = false" in patched
    assert "broadcast_last_result = true" not in patched


def test_patch_server_config_can_patch_rounds_and_min_clients():
    patcher = _import_patcher()

    patched = patcher.patch_server_config_text(
        SERVER_CONFIG_TEMPLATE,
        num_rounds=2,
        min_clients=2,
        configure_min_clients=3,
    )

    assert "num_rounds = 2" in patched
    assert "min_clients = 2  # tolerate drops" in patched
    assert "configure_min_clients = 3" in patched
    assert patched.index("min_clients = 2") < patched.index("configure_min_clients = 3")


def test_patch_server_config_replaces_existing_configure_min_clients():
    patcher = _import_patcher()
    existing = SERVER_CONFIG_TEMPLATE.replace(
        "min_clients = 5  # tolerate drops",
        "min_clients = 5  # tolerate drops\n      configure_min_clients = 5",
    )

    patched = patcher.patch_server_config_text(existing, configure_min_clients=3)

    assert "configure_min_clients = 3" in patched
    assert "configure_min_clients = 5" not in patched


def test_parse_strict_clients_trims_and_preserves_order():
    patcher = _import_patcher()

    assert patcher.parse_strict_clients("CAM_1, MHA_1,UKA_1") == ("CAM_1", "MHA_1", "UKA_1")


@pytest.mark.parametrize("value", ["", "CAM_1,,UKA_1", "CAM_1,  ,UKA_1"])
def test_parse_strict_clients_rejects_empty_names(value):
    patcher = _import_patcher()

    with pytest.raises(ValueError, match="non-empty client names"):
        patcher.parse_strict_clients(value)


def test_parse_strict_clients_rejects_duplicate_names():
    patcher = _import_patcher()

    with pytest.raises(ValueError, match=r"duplicate client name\(s\): CAM_1"):
        patcher.parse_strict_clients("CAM_1,MHA_1,CAM_1")


def test_strict_profile_patches_all_client_membership_and_count_fields():
    patcher = _import_patcher()
    clients = ("CAM_1", "MHA_1", "UKA_1")

    patched_server = patcher.patch_server_config_text(
        SERVER_CONFIG_TEMPLATE,
        min_clients=len(clients),
        configure_min_clients=len(clients),
        participating_clients=clients,
    )
    patched_meta = patcher.patch_meta_config_text(META_CONFIG_TEMPLATE, clients)

    assert "min_clients = 3  # tolerate drops" in patched_server
    assert "configure_min_clients = 3" in patched_server
    assert 'participating_clients = ["CAM_1", "MHA_1", "UKA_1"]' in patched_server
    assert "min_clients = 3" in patched_meta
    assert 'mandatory_clients = ["CAM_1", "MHA_1", "UKA_1"]' in patched_meta


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
        configure_min_clients=3,
        min_responses_required=2,
        broadcast_last_result=False,
    )

    assert patched_config == config_path
    patched_client = config_path.read_text()
    patched_server = server_config_path.read_text()
    assert 'warm_start_mode = "require"' in patched_client
    assert "min_responses_required = 2" in patched_client
    assert "broadcast_last_result = false" in patched_client
    assert "num_rounds = 2" in patched_server
    assert "min_clients = 2" in patched_server
    assert "configure_min_clients = 3" in patched_server


def test_patch_job_dir_strict_clients_sets_one_consistent_exact_profile(tmp_path):
    patcher = _import_patcher()
    job_dir = tmp_path / "strict_job"
    config_dir = job_dir / "app" / "config"
    config_dir.mkdir(parents=True)
    client_path = config_dir / "config_fed_client.conf"
    server_path = config_dir / "config_fed_server.conf"
    meta_path = job_dir / "meta.conf"
    client_path.write_text(CONFIG_TEMPLATE)
    server_path.write_text(SERVER_CONFIG_TEMPLATE)
    meta_path.write_text(META_CONFIG_TEMPLATE)

    patcher.patch_job_dir(
        job_dir,
        "continue",
        strict_clients="CAM_1,MHA_1,UKA_1",
    )

    assert "min_responses_required = 3" in client_path.read_text()
    assert "min_clients = 3  # tolerate drops" in server_path.read_text()
    assert "configure_min_clients = 3" in server_path.read_text()
    assert 'participating_clients = ["CAM_1", "MHA_1", "UKA_1"]' in server_path.read_text()
    assert "min_clients = 3" in meta_path.read_text()
    assert 'mandatory_clients = ["CAM_1", "MHA_1", "UKA_1"]' in meta_path.read_text()


@pytest.mark.parametrize(
    ("argument", "value", "error_name"),
    [
        ("min_clients", 2, "min_clients"),
        ("configure_min_clients", 2, "configure_min_clients"),
        ("min_responses_required", 2, "min_responses_required"),
    ],
)
def test_patch_job_dir_rejects_counts_that_conflict_with_strict_clients_without_writing(
    tmp_path, argument, value, error_name
):
    patcher = _import_patcher()
    job_dir = tmp_path / "strict_job"
    config_dir = job_dir / "app" / "config"
    config_dir.mkdir(parents=True)
    client_path = config_dir / "config_fed_client.conf"
    server_path = config_dir / "config_fed_server.conf"
    meta_path = job_dir / "meta.conf"
    originals = {
        client_path: CONFIG_TEMPLATE,
        server_path: SERVER_CONFIG_TEMPLATE,
        meta_path: META_CONFIG_TEMPLATE,
    }
    for path, text in originals.items():
        path.write_text(text)

    with pytest.raises(ValueError, match=rf"{error_name}=2 conflicts with strict_clients count 3"):
        patcher.patch_job_dir(
            job_dir,
            "continue",
            strict_clients="CAM_1,MHA_1,UKA_1",
            **{argument: value},
        )

    assert {path: path.read_text() for path in originals} == originals


def test_patch_job_dir_strict_profile_validates_every_file_before_writing(tmp_path):
    patcher = _import_patcher()
    job_dir = tmp_path / "strict_job"
    config_dir = job_dir / "app" / "config"
    config_dir.mkdir(parents=True)
    client_path = config_dir / "config_fed_client.conf"
    server_path = config_dir / "config_fed_server.conf"
    client_path.write_text(CONFIG_TEMPLATE)
    server_path.write_text(SERVER_CONFIG_TEMPLATE)

    with pytest.raises(FileNotFoundError, match="Missing job metadata config"):
        patcher.patch_job_dir(job_dir, "continue", strict_clients="CAM_1,MHA_1,UKA_1")

    assert client_path.read_text() == CONFIG_TEMPLATE
    assert server_path.read_text() == SERVER_CONFIG_TEMPLATE


# ---------------------------------------------------------------------------
# --fold (#411): launcher env prefix + per-fold warm-start mirror
# ---------------------------------------------------------------------------

LAUNCHER_CONFIG_TEMPLATE = """
  {
    id = "launcher"
    path = "nvflare.app_common.launchers.subprocess_launcher.SubprocessLauncher"
    args {
      script = "KEY_METRIC=val/AUC_ROC MODEL_NAME=1DivideAndConquer python3 custom/{app_script}  {app_config} "
      launch_once = true
    }
  }
"""


def _fold_config():
    return CONFIG_TEMPLATE + LAUNCHER_CONFIG_TEMPLATE


def test_patch_launcher_env_prepends_fold_token():
    patched = _import_patcher().patch_launcher_env(_fold_config(), "FOLD", 2)
    assert 'script = "FOLD=2 KEY_METRIC=val/AUC_ROC MODEL_NAME=1DivideAndConquer python3' in patched


def test_patch_launcher_env_is_idempotent():
    once = _import_patcher().patch_launcher_env(_fold_config(), "FOLD", 2)
    twice = _import_patcher().patch_launcher_env(once, "FOLD", 4)
    assert twice.count("FOLD=") == 1
    assert 'script = "FOLD=4 KEY_METRIC=val/AUC_ROC' in twice


def test_patch_launcher_env_preserves_other_tokens():
    patched = _import_patcher().patch_launcher_env(_fold_config(), "FOLD", 3)
    assert "KEY_METRIC=val/AUC_ROC" in patched
    assert "MODEL_NAME=1DivideAndConquer" in patched
    assert "python3 custom/{app_script}" in patched


def test_fold_zero_keeps_the_legacy_mirror_path():
    patched = _import_patcher().patch_fold_global_paths(_fold_config(), 0)
    assert '"/scratch/mediswarm_latest_global.pt"' in patched
    assert "_fold" not in patched


def test_nonzero_fold_gets_its_own_mirror_path():
    patched = _import_patcher().patch_fold_global_paths(_fold_config(), 2)
    assert patched.count('"/scratch/mediswarm_latest_global_fold2.pt"') == 2
    assert '"/scratch/mediswarm_latest_global.pt"' not in patched


def test_fold_mirror_path_patch_is_idempotent():
    once = _import_patcher().patch_fold_global_paths(_fold_config(), 2)
    twice = _import_patcher().patch_fold_global_paths(once, 3)
    assert twice.count('"/scratch/mediswarm_latest_global_fold3.pt"') == 2
    assert "fold2" not in twice


def test_patch_config_text_without_fold_leaves_launcher_and_mirror_alone():
    patched = _import_patcher().patch_config_text(_fold_config(), "fresh")
    assert "FOLD=" not in patched
    assert '"/scratch/mediswarm_latest_global.pt"' in patched


def test_patch_config_text_with_fold_sets_both():
    patched = _import_patcher().patch_config_text(_fold_config(), "fresh", fold=2)
    assert 'script = "FOLD=2 KEY_METRIC=val/AUC_ROC' in patched
    assert patched.count('"/scratch/mediswarm_latest_global_fold2.pt"') == 2
