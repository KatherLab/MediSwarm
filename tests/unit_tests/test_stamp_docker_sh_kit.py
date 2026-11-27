"""Client docker.sh of the DECADE (STAMP) 1.7.1 kit: the ODELIA kit fixes, ported.

Before 1.7.1 the STAMP docker.sh deleted the kit's lock files on every call (F17: a preflight
next to the running client started a second NVFlare client of the site), had no --status and no
guard against a second client of the site, and let an old kit's live_sync daemon keep uploading
for the site. These tests run the generated STAMP client docker.sh on the stub host of
test_docker_sh_status_and_migrate.py (stub docker, nvidia-smi, dpkg-query; real fake daemons).
"""

import importlib.util
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_spec = importlib.util.spec_from_file_location(
    "docker_sh_status_harness", Path(__file__).resolve().parent / "test_docker_sh_status_and_migrate.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_TEMPLATE = REPO_ROOT / "docker_config" / "master_template_STAMP.yml"
SITE = harness.SITE
BUILD_ID = "abc1234"
CLIENT = f"stamp_swarm_client_{SITE}_{BUILD_ID}"
OLD_CLIENT = f"stamp_swarm_client_{SITE}_7e56dd4"
STAMP_ENV = {"STAMP_CLINI_TABLE": "/data/TESTSITE/clini_table.csv", "STAMP_FEATURE_DIR": "/data/TESTSITE/features",
             "STAMP_GROUND_TRUTH_LABEL": "MSI-High", "STAMP_NUM_CLASSES": "2"}

# Records how docker.sh started it and the environment it got; behaves like a daemon.
FAKE_LIVE_SYNC = r'''#!/usr/bin/env bash
here="$(cd "$(dirname "$0")" && pwd)"
printf '%s\n' "$@" > "$here/.fake_live_sync_args.$$"
env | grep -E '^MEDISWARM_(KIT_VERSION|SYNC_GLOBAL_MODELS_DEFAULT)=' > "$here/.fake_live_sync_env.$$" || true
trap 'exit 0' TERM
while true; do sleep 0.1; done
'''


def _stamp_kit(host, root, version="1.7.1", build_id=BUILD_ID, site=SITE):
    """A DECADE client kit: the STAMP template rendered as the build and the lighter write it."""
    script = yaml.safe_load(STAMP_TEMPLATE.read_text())["docker_cln_sh"]
    script = script.replace("__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_DOCKER_IMAGE__", version)
    script = script.replace("__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__", build_id)
    script = script.replace("{~~client_name~~}", site).replace("{~~docker_image~~}", f"jefftud/decade:{version}")
    script = re.sub(r"\{~~[a-z_]+~~\}", "x", script)
    startup = Path(root) / "startup"
    harness._write(startup / "docker.sh", script, 0o755)
    harness._write(startup / "image.conf", "MEDISWARM_IMAGE=jefftud/decade:1.7.1\n")
    harness._write(startup / "fed_client.json", json.dumps({
        "format_version": 2, "client": {"fqsn": site},
        "overseer_agent": {"path": "x", "args": {"sp_end_point": f"127.0.0.1:{host.fl_port}:{host.admin_port}"}},
    }))
    harness._write(startup / "live_sync.sh", FAKE_LIVE_SYNC, 0o755)
    return Path(root)


def _site_data(data_dir, site="TESTSITE"):
    """What STAMP_ENV points to, under the host folder given as --data_dir (mounted at /data)."""
    harness._write(Path(data_dir) / site / "clini_table.csv", "PATIENT,MSI-High\nP1,1\n")
    (Path(data_dir) / site / "features").mkdir(parents=True, exist_ok=True)


@pytest.fixture
def host(tmp_path):
    h = harness.Host(tmp_path)
    h.new_kit = _stamp_kit(h, tmp_path / "kits" / "TESTSITE_171")
    _site_data(h.data)
    yield h
    for pid in _daemon_pids(h.new_kit):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    h.close()


def _daemon_pids(kit):
    return [int(p.name.rsplit(".", 1)[1]) for p in (Path(kit) / "startup").glob(".fake_live_sync_args.*")]


def _started_daemon_pids(kit, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline and not _daemon_pids(kit):
        time.sleep(0.1)
    return _daemon_pids(kit)


def _run(host, *args, kit=None, env=STAMP_ENV, data_dir=None):
    return host.run("--data_dir", str(data_dir or host.data), "--scratch_dir", str(host.scratch), "--GPU", "device=0",
                    "--no_pull", *args, kit=kit, extra_env=env)


def _client(host, name=CLIENT, kit=None, **extra):
    host.add_container(name, kit or host.new_kit, image="jefftud/decade:1.7.1",
                       env=[f"SITE_NAME={SITE}", "TRAINING_MODE=swarm", "STAMP_NUM_CLASSES=2",
                            "STAMP_GROUND_TRUTH_LABEL=MSI-High"], **extra)


def _runs(host, detached=None):
    runs = [c for c in host.calls() if c and c[0] == "run" and "nvidia-smi" not in c]
    if detached is not None:
        runs = [c for c in runs if ("-d" in c) is detached]
    return runs


def _name_of(run_call):
    return next(a.split("=", 1)[1] for a in run_call if a.startswith("--name="))


# ── generated script ──────────────────────────────────────────────────────────


def test_generated_script_is_valid_bash(host):
    result = subprocess.run(["bash", "-n", str(host.new_kit / "startup" / "docker.sh")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_container_name_line_is_what_the_deploy_tests_parse(host):
    """run_stamp_deploy_test.sh and container_scope.sh scope their cleanup by this exact line (#472)."""
    text = (host.new_kit / "startup" / "docker.sh").read_text()
    assert re.findall(r"(?m)^CONTAINER_NAME=stamp_swarm_.*_[0-9a-f]{7,40}$", text) == [f"CONTAINER_NAME={CLIENT}"]


def test_heartbeat_reads_the_kit_version_from_the_first_mediswarm_version(host):
    text = (host.new_kit / "startup" / "docker.sh").read_text()
    assert re.search(r"(?<=MEDISWARM_VERSION=)\S+", text).group(0) == "1.7.1"


# ── F17/F19: one-off modes next to the running client ─────────────────────────


def test_preflight_next_to_the_running_client_keeps_its_locks_and_gets_its_own_name(host):
    _client(host)
    pid_fl = harness._write(host.new_kit / "pid.fl", "4242\n")
    daemon_fl = harness._write(host.new_kit / "daemon_pid.fl", "4243\n")
    result = _run(host, "--preflight_check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert pid_fl.exists() and daemon_fl.exists(), "docker.sh must never delete the kit's locks (F17)"
    run = _runs(host, detached=False)[-1]
    assert "TRAINING_MODE=preflight_check" in run
    assert _name_of(run).startswith(CLIENT + "_") and _name_of(run) != CLIENT
    assert f"This kit's client container {CLIENT} exists" in result.stdout
    assert host.state["containers"][CLIENT]["state"] == "running"


def test_container_name_names_a_one_off_run(host):
    _client(host)
    result = _run(host, "--container_name", "my_preflight", "--preflight_check")
    assert result.returncode == 0, result.stdout
    assert _name_of(_runs(host, detached=False)[-1]) == "my_preflight"


def test_without_a_client_a_preflight_uses_the_kit_name(host):
    result = _run(host, "--preflight_check")
    assert result.returncode == 0, result.stdout
    assert _name_of(_runs(host, detached=False)[-1]) == CLIENT


# ── --start_client: one client per site ───────────────────────────────────────


def test_start_client_refuses_while_another_kits_client_of_the_site_runs(host, tmp_path):
    old_kit = _stamp_kit(host, tmp_path / "kits" / "TESTSITE_160", version="1.6.0", build_id="7e56dd4")
    _client(host, OLD_CLIENT, kit=old_kit)
    result = _run(host, "--start_client")
    assert result.returncode == 2, result.stdout
    assert f"The client container {OLD_CLIENT} of another kit" in result.stdout
    assert f"docker rm -f {OLD_CLIENT}" in result.stdout
    assert not _runs(host, detached=True), "no client was started"


def test_start_client_refuses_while_an_old_client_runs_on_this_kit_folder(host):
    """A new kit unzipped over the old folder: the old client (old name) still mounts this folder."""
    _client(host, OLD_CLIENT)
    result = _run(host, "--start_client")
    assert result.returncode == 2, result.stdout
    assert f"A container of this kit is running: {OLD_CLIENT}" in result.stdout
    assert not _runs(host, detached=True)


def test_start_client_next_to_its_own_running_client_only_starts_live_sync(host):
    _client(host)
    result = _run(host, "--start_client")
    assert result.returncode == 0, result.stdout
    assert "is already running; it is left as it is" in result.stdout
    assert not _runs(host, detached=True)
    assert len(_started_daemon_pids(host.new_kit)) == 1


def test_start_client_refuses_on_driver_library_mismatch(host):
    host.set_nvidia_smi(harness.NVIDIA_SMI_MISMATCH, 18)
    result = _run(host, "--start_client")
    assert result.returncode == 2, result.stdout
    assert "[FAIL] NVIDIA driver/library version mismatch" in result.stdout and "F18" in result.stdout
    assert not _runs(host, detached=True)


def test_start_client_needs_the_stamp_settings(host):
    result = _run(host, "--start_client", env={})
    assert result.returncode == 1
    assert "Missing required STAMP environment variable(s): STAMP_CLINI_TABLE STAMP_FEATURE_DIR" in result.stdout


def test_start_client_starts_the_client_and_its_daemon(host):
    lock = harness._write(host.new_kit / "daemon_pid.fl", "4242\n")
    result = _run(host, "--start_client")
    assert result.returncode == 0, result.stdout + result.stderr
    assert lock.exists()
    start = _runs(host, detached=True)[-1]
    assert _name_of(start) == CLIENT
    assert start[-1].startswith("rm -f ../daemon_pid.fl ../pid.fl; nohup ./start.sh")
    assert "STAMP_NUM_CLASSES" in start and "TRAINING_MODE=swarm" in start
    pids = _started_daemon_pids(host.new_kit)
    assert len(pids) == 1
    startup = host.new_kit / "startup"
    args = (startup / f".fake_live_sync_args.{pids[0]}").read_text().splitlines()
    assert args == ["--mode", "swarm", "--site-name", SITE, "--kit-root", str(host.new_kit),
                    "--startup-dir", str(startup), "--scratch-dir", str(host.scratch), "--container-name", CLIENT]
    env = (startup / f".fake_live_sync_env.{pids[0]}").read_text()
    assert "MEDISWARM_KIT_VERSION=1.7.1" in env and "MEDISWARM_SYNC_GLOBAL_MODELS_DEFAULT=1" in env


# ── STAMP_* data paths exist under --data_dir (26 Nov 2026) ───────────────────
# A client started with another --data_dir than its preflight check died inside the swarm job
# (FileNotFoundError: STAMP_SLIDE_TABLE ...) and the strict run aborted 600 s later for every site.

RUN_MODES = ["--start_client", "--preflight_check", "--local_training"]


def _nothing_started(host):
    assert harness._mutating_calls(host.calls()) == [], "no container may start"
    assert not _daemon_pids(host.new_kit), "no live_sync may start"


@pytest.mark.parametrize("mode", RUN_MODES)
def test_a_missing_slide_table_stops_the_start_on_the_host(host, mode):
    env = {**STAMP_ENV, "STAMP_SLIDE_TABLE": "/data/TESTSITE/slide_table.csv"}
    result = _run(host, mode, env=env)
    out = result.stdout
    assert result.returncode == 1, out
    assert "[FAIL] STAMP_SLIDE_TABLE: file not found." in out
    assert "Path in the container:  /data/TESTSITE/slide_table.csv" in out
    assert f"Path on this computer:  {host.data}/TESTSITE/slide_table.csv" in out
    assert out.count(f"--data_dir used:  {host.data}   (mounted at /data in the container)") == 1
    assert "STAMP_CLINI_TABLE:" not in out and "STAMP_FEATURE_DIR:" not in out, "those exist"
    assert "[ABORT] Nothing was started." in out
    _nothing_started(host)


@pytest.mark.parametrize("mode", ["--start_client", "--local_training"])
def test_another_data_dir_than_the_preflight_names_every_missing_path(host, tmp_path, mode):
    other = tmp_path / "other_data"
    other.mkdir()
    result = _run(host, mode, data_dir=other)
    out = result.stdout
    assert result.returncode == 1, out
    assert "[FAIL] STAMP_CLINI_TABLE: file not found." in out
    assert "[FAIL] STAMP_FEATURE_DIR: folder not found." in out
    assert f"Path on this computer:  {other}/TESTSITE/clini_table.csv" in out
    assert f"Path on this computer:  {other}/TESTSITE/features" in out
    assert out.count(f"--data_dir used:  {other}   (mounted at /data in the container)") == 1, "named once"
    assert "Start with the same --data_dir as your preflight check" in out
    assert "does not exist on this computer" not in out
    _nothing_started(host)


def test_a_data_dir_that_does_not_exist_is_named(host, tmp_path):
    gone = tmp_path / "no_such_folder"
    result = _run(host, "--start_client", data_dir=gone)
    assert result.returncode == 1, result.stdout
    assert f"The --data_dir folder {gone} does not exist on this computer." in result.stdout
    _nothing_started(host)


def test_a_preflight_with_a_wrong_data_dir_says_which_folder_to_give(host, tmp_path):
    other = tmp_path / "other_data"
    other.mkdir()
    result = _run(host, "--preflight_check", data_dir=other)
    assert result.returncode == 1, result.stdout
    assert "Give the --data_dir that holds your data" in result.stdout
    assert "same --data_dir as your preflight" not in result.stdout
    _nothing_started(host)


def test_a_relative_data_dir_is_checked_where_docker_sh_mounts_it(host):
    """docker.sh makes a relative --data_dir absolute from the folder it is called in (the kit's startup/)."""
    rel = os.path.relpath(host.data, host.new_kit / "startup")
    result = _run(host, "--preflight_check", data_dir=rel)
    assert result.returncode == 0, result.stdout
    assert "[FAIL]" not in result.stdout


def test_a_table_that_is_a_folder_or_features_that_are_a_file_fail(host):
    (host.data / "TESTSITE" / "slides").mkdir()
    harness._write(host.data / "TESTSITE" / "features.h5", "x")
    env = {**STAMP_ENV, "STAMP_SLIDE_TABLE": "/data/TESTSITE/slides",
           "STAMP_FEATURE_DIR": "/data/TESTSITE/features.h5"}
    result = _run(host, "--start_client", env=env)
    assert result.returncode == 1, result.stdout
    assert "[FAIL] STAMP_SLIDE_TABLE: this is a folder, not a file." in result.stdout
    assert "[FAIL] STAMP_FEATURE_DIR: this is a file, not a folder." in result.stdout
    _nothing_started(host)


@pytest.mark.parametrize("mode", RUN_MODES)
def test_all_paths_present_start_as_before(host, mode):
    harness._write(host.data / "TESTSITE" / "slide_table.csv", "PATIENT,FILENAME\nP1,s1.h5\n")
    env = {**STAMP_ENV, "STAMP_SLIDE_TABLE": "/data/TESTSITE/slide_table.csv"}
    result = _run(host, mode, env=env)
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "[FAIL]" not in out and "[WARN] STAMP_" not in out
    run = _runs(host, detached=(mode == "--start_client"))[-1]
    assert "STAMP_SLIDE_TABLE" in run, "the path is still forwarded into the container"


def test_a_path_not_under_data_warns_and_starts(host):
    env = {**STAMP_ENV, "STAMP_FEATURE_DIR": "/mnt/features"}
    result = _run(host, "--preflight_check", env=env)
    out = result.stdout
    assert result.returncode == 0, out
    assert "[WARN] STAMP_FEATURE_DIR=/mnt/features is not under /data." in out
    assert "use a path that starts with /data/" in out.lower()
    assert "[FAIL]" not in out
    assert "TRAINING_MODE=preflight_check" in _runs(host, detached=False)[-1]


# ── --start_client: other kits' live_sync daemons (#614, F16) ─────────────────


def _wait_dead(proc, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.1)
    return False


def test_start_client_retires_other_decade_kits_daemons_only(host, tmp_path):
    old = host.start_fake_daemon(_stamp_kit(host, tmp_path / "kits" / "TESTSITE_160", "1.6.0", "7e56dd4"))
    gone_root = _stamp_kit(host, tmp_path / "kits" / "TESTSITE_150", "1.5.0", "1234567")
    gone = host.start_fake_daemon(gone_root)
    odelia_root = tmp_path / "kits" / "ODELIA_TESTSITE"
    host.make_kit(odelia_root, "1.9.3", "fedcba9", harness.TEMPLATE.read_text())
    odelia = host.start_fake_daemon(odelia_root)
    other_site = host.start_fake_daemon(_stamp_kit(host, tmp_path / "kits" / "OTHER_160", site="OTHER_1"),
                                        site="OTHER_1")
    subprocess.run(["rm", "-rf", str(gone_root)], check=True)

    result = _run(host, "--start_client")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _wait_dead(old) and _wait_dead(gone), result.stdout
    assert odelia.poll() is None, "another consortium's daemon of the same site name keeps running"
    assert other_site.poll() is None
    assert len(_started_daemon_pids(host.new_kit)) == 1


def test_a_daemon_of_the_old_code_in_this_folder_is_replaced(host):
    """Unzipped over the old folder: the daemon that ran before live_sync.sh was replaced is old code."""
    old = host.start_fake_daemon(host.new_kit, write_pid_file=True)
    time.sleep(0.5)   # let bash read its loop before the file changes under it
    assert old.poll() is None
    new_file = harness._write(host.new_kit / "startup" / "live_sync.sh", FAKE_LIVE_SYNC, 0o755)   # the new kit's
    later = time.time() + 300   # ... unzipped five minutes after the old daemon started
    os.utime(new_file, (later, later))
    result = _run(host, "--start_client")
    assert result.returncode == 0, result.stdout
    assert "runs an older live_sync.sh of this kit folder" in result.stdout
    assert _wait_dead(old)
    pids = _started_daemon_pids(host.new_kit)
    assert len(pids) == 1 and pids[0] != old.pid


def test_this_kits_current_daemon_is_kept(host):
    _client(host)
    current = host.start_fake_daemon(host.new_kit, write_pid_file=True)
    result = _run(host, "--start_client")
    assert result.returncode == 0, result.stdout
    assert f"Live sync daemon already running (PID {current.pid})" in result.stdout
    assert current.poll() is None


# ── --status ───────────────────────────────────────────────────────────────────


def test_status_reports_and_changes_nothing(host, tmp_path):
    _client(host, health="healthy")
    host.state["images"] = {"jefftud/decade:1.7.1": {"id": harness.OLD_IMAGE_ID},
                            harness.OLD_IMAGE_ID: {"repo_digests": [f"jefftud/decade@{harness.OLD_DIGEST}"]}}
    host.start_fake_daemon(host.new_kit, write_pid_file=True)
    old_kit = _stamp_kit(host, tmp_path / "kits" / "TESTSITE_160", "1.6.0", "7e56dd4")
    _client(host, OLD_CLIENT, kit=old_kit, state="exited")
    old_daemon = host.start_fake_daemon(old_kit)
    harness._write(host.new_kit / "daemon_pid.fl", "1\n")
    before = harness._snapshot(host.tmp / "kits")
    result = host.run("--status", extra_env={"STAMP_NUM_CLASSES": "3"})
    out = result.stdout
    assert result.returncode == 0, out
    assert "kit 1.7.1 at" in out
    assert "configured image jefftud/decade:1.7.1 (from startup/image.conf)" in out
    assert f"Client container {CLIENT}" in out and "[OK]   running, health healthy" in out
    assert "STAMP_NUM_CLASSES=2" in out and "STAMP_GROUND_TRUTH_LABEL=MSI-High" in out
    assert "differ from the running client's" in out
    assert f"remove it with: docker rm -f {OLD_CLIENT}" in out
    assert f"live_sync PID {old_daemon.pid} (swarm) of the kit {old_kit} still uploads" in out
    assert "daemon_pid.fl present (normal while the client runs)" in out
    assert harness._mutating_calls(host.calls()) == []
    assert harness._snapshot(host.tmp / "kits") == before


def test_status_reads_the_driver_version_of_620_series_drivers(host):
    _client(host, health="healthy")
    host.set_nvidia_smi(harness.NVIDIA_SMI_KMD, 0)
    out = host.run("--status").stdout
    assert "[OK]   NVIDIA driver 620.70.06 answers on the host" in out
    assert "nvidia-smi fails" not in out


def test_status_does_not_combine_with_a_run_mode(host):
    result = host.run("--status", "--start_client")
    assert result.returncode == 1 and "--status does not combine with a run mode" in result.stdout
