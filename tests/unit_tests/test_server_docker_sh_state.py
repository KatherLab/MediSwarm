"""The server kit keeps NVFlare's job store, and the MS6 archive, on the host (#607, #612).

NVFlare's server stores every submitted job (meta, job zip, the run's workspace zip) under
/tmp/nvflare inside the container. The coordinator ran from a container whose only mount was the
kit, so recreating it on a new image would have deleted the whole run record. docker.sh now
bind-mounts a host directory there, plus /archive for the per-round global models of MS6.

The rendered server docker.sh is run against a fake `docker` that records its arguments.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"
PROVISIONED = "jefftud/odelia:1.9.0-provisioned"

FAKE_DOCKER = """#!/bin/bash
printf '%s\\n' "$@" > "$DOCKER_ARGS_LOG"
printf -- '---\\n' >> "$DOCKER_ARGS_LOG"
"""


@pytest.fixture
def server_kit(tmp_path):
    script = yaml.safe_load(TEMPLATE.read_text())["docker_svr_sh"]
    script = re.sub(r"\{~~docker_image~~\}", PROVISIONED, script)
    script = re.sub(r"\{~~[a-z_]+~~\}", "8002", script)
    script = script.replace("__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__", "abc1234")
    kit = tmp_path / "kit"
    startup = kit / "startup"
    startup.mkdir(parents=True)
    (startup / "docker.sh").write_text(script)
    (startup / "docker.sh").chmod(0o755)
    # Every kit carries the client-style release channel; the server must not follow it.
    (startup / "image.conf").write_text("MEDISWARM_IMAGE=jefftud/odelia:current\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "docker").write_text(FAKE_DOCKER)
    (bindir / "docker").chmod(0o755)
    return kit, bindir, tmp_path / "docker_args.log"


def run_docker_sh(server_kit, *args, env_extra=None):
    kit, bindir, log = server_kit
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MEDISWARM_") and k != "SVR_NAME"}
    env.update({"PATH": f"{bindir}:{env.get('PATH', '/usr/bin:/bin')}", "DOCKER_ARGS_LOG": str(log)})
    env.update(env_extra or {})
    result = subprocess.run(["./docker.sh", "--no_pull", *args], cwd=kit / "startup", env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return log.read_text().split("---\n")[0].splitlines()


def mounts(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]


def test_start_server_mounts_job_store_and_archive_inside_the_kit_by_default(server_kit):
    kit = server_kit[0]
    argv = run_docker_sh(server_kit, "--start_server")
    assert argv[0] == "run"
    startup = str(kit / "startup")
    assert f"{startup}/../server_state:/tmp/nvflare" in mounts(argv)
    assert f"{startup}/../archive:/archive" in mounts(argv)
    assert (kit / "server_state").is_dir() and (kit / "archive").is_dir()
    assert PROVISIONED in argv, "image.conf (:current) must not move the server"
    # The container command is unchanged: it still clears its own lock (#630).
    assert argv[-1].startswith("rm -f ../daemon_pid.fl ../pid.fl; nohup ./start.sh")


def test_server_conf_persists_paths_and_image(server_kit, tmp_path):
    kit = server_kit[0]
    (kit / "startup" / "server.conf").write_text(
        f"MEDISWARM_SERVER_STATE_DIR={tmp_path}/coordinator/nvflare\n"
        f"MEDISWARM_ARCHIVE_DIR={tmp_path}/archive_root\n"
        "MEDISWARM_IMAGE=jefftud/odelia:1.9.1\n")
    argv = run_docker_sh(server_kit, "--start_server")
    assert f"{tmp_path}/coordinator/nvflare:/tmp/nvflare" in mounts(argv)
    assert f"{tmp_path}/archive_root:/archive" in mounts(argv)
    assert "jefftud/odelia:1.9.1" in argv


def test_exported_paths_win_over_server_conf_and_image_flag_wins_over_all(server_kit, tmp_path):
    kit = server_kit[0]
    (kit / "startup" / "server.conf").write_text(
        f"MEDISWARM_SERVER_STATE_DIR={tmp_path}/from_conf\nMEDISWARM_IMAGE=jefftud/odelia:1.9.1\n")
    argv = run_docker_sh(server_kit, "--image", "jefftud/odelia:1.9.2", "--start_server",
                         env_extra={"MEDISWARM_SERVER_STATE_DIR": f"{tmp_path}/from_env",
                                    "MEDISWARM_IMAGE": "jefftud/odelia:from-env"})
    assert f"{tmp_path}/from_env:/tmp/nvflare" in mounts(argv)
    assert "jefftud/odelia:1.9.2" in argv
    assert "jefftud/odelia:from-env" not in argv, "an exported MEDISWARM_IMAGE must not move the server"


def test_interactive_sees_the_same_store(server_kit):
    kit = server_kit[0]
    argv = run_docker_sh(server_kit, "--interactive")
    assert f"{kit / 'startup'}/../server_state:/tmp/nvflare" in mounts(argv)


def test_list_licenses_mounts_nothing(server_kit):
    argv = run_docker_sh(server_kit, "--list_licenses")
    assert mounts(argv) == []
