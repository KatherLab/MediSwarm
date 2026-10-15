"""Client docker.sh for the 1.9.0 kits: --status, --migrate_from, the F18 pre-check and #522.

Every site moves from a 1.8.1 (some still from a 1.6.0) kit to the 1.9.0 kit. Done by hand
this went wrong in the field: an old kit's live_sync daemon kept running and clobbered the
new kit's uploads (#614, F16), stale lock files kept clients offline (F3), and an unattended
NVIDIA update left a node unable to start any new container until a reboot (F18). These
tests run the generated client docker.sh against a stub `docker`, `nvidia-smi` and
`dpkg-query` on PATH and real (fake) live_sync processes, and check:

- `--status` reports kit, image, container, live sync, driver, cgroup driver, coordinator
  and lock files, and changes nothing;
- `--migrate_from <old kit>` is a dry run by default, and with `--yes` retires the old
  container(s) and daemon(s), clears the old kit's locks, carries image.conf/sync.conf site
  settings over and rebuilds the exact --start_client command (`--start` runs it), never
  deleting data, scratch or kit folders;
- `--start_client` refuses on a driver/library mismatch and warns on the systemd cgroup driver;
- the /data double-mount guard is a path-prefix test (#522);
- every existing flag keeps working.
"""

import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"
SITE = "TESTSITE"
BUILD_ID = "abc1234"
NEW_CONTAINER = f"odelia_swarm_client_{SITE}_{BUILD_ID}"
OLD_CONTAINER = f"odelia_swarm_client_{SITE}_old1234"
OLD_IMAGE_ID = "sha256:" + "a" * 64
OLD_DIGEST = "sha256:" + "b" * 64
NEW_DIGEST = "sha256:" + "c" * 64

NVIDIA_SMI_OK = """Tue Oct 14 10:00:00 2026
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
"""
NVIDIA_SMI_MISMATCH = """Failed to initialize NVML: Driver/library version mismatch
NVML library version: 550.127
"""

# A Python stand-in for the docker CLI. It answers from a JSON state file (containers,
# images, cgroup driver, ...), applies stop/rm to that state, and logs every call.
DOCKER_STUB = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["DOCKER_STUB_LOG"], "a") as log:
    log.write(json.dumps(args) + "\n")
path = os.environ["DOCKER_STUB_STATE"]
with open(path) as f:
    state = json.load(f)
C = state.setdefault("containers", {})
I = state.setdefault("images", {})

def save():
    with open(path, "w") as f:
        json.dump(state, f)

def fmt():
    return args[args.index("--format") + 1] if "--format" in args else ""

cmd = args[0] if args else ""
if cmd == "ps":
    sys.stdout.write("".join(n + "\n" for n, c in C.items() if "-a" in args or c["state"] == "running"))
elif cmd == "inspect":
    c = C.get(args[-1])
    if c is None:
        sys.exit(1)
    f = fmt()
    if ".State.Status" in f:
        kit = next((m["src"] for m in c.get("mounts", []) if m["dst"] == "/startupkit"), "")
        sys.stdout.write("|".join([c["state"], c.get("health", "none"), str(c.get("restarts", 0)),
                                   c.get("started", "2026-10-12T08:00:00.123456789Z"), c["image"],
                                   c["image_id"], c.get("user", "1000:1000"), str(c.get("exit_code", 0)), kit]) + "\n")
    elif ".Config.Env" in f:
        sys.stdout.write("".join(e + "\n" for e in c.get("env", [])))
    elif ".Mounts" in f:
        sys.stdout.write("".join("%s|%s|%s\n" % (m["src"], m["dst"], str(m.get("rw", True)).lower())
                                 for m in c.get("mounts", [])))
    elif "DeviceRequests" in f:
        sys.stdout.write(c.get("gpu_request", ""))
elif cmd == "image" and args[1:2] == ["inspect"]:
    img = I.get(args[-1])
    if img is None:
        # A bare existence check (no --format) finds the image unless a test removes it on purpose.
        sys.exit(0 if not fmt() and state.get("images_present_by_default", True) else 1)
    f = fmt()
    if f == "{{.Id}}":
        sys.stdout.write(img["id"] + "\n")
    elif "RepoDigests" in f:
        sys.stdout.write(" ".join(img.get("repo_digests", [])) + "\n")
    elif ".Config.Env" in f:
        sys.stdout.write("".join(e + "\n" for e in img.get("env", [])))
elif cmd == "exec":
    c = C.get(args[1], {})
    sys.stdout.write(c.get("exec_out", ""))
    sys.exit(c.get("exec_rc", 0))
elif cmd == "top":
    sys.stdout.write(C.get(args[1], {}).get("top", "UID PID PPID C STIME TTY TIME CMD\n"))
elif cmd == "info":
    sys.stdout.write(state.get("cgroup_driver", "cgroupfs") + "\n")
elif cmd == "buildx":
    digest = state.get("registry_digest")
    if not digest:
        sys.exit(1)
    if args[1:2] == ["imagetools"]:
        sys.stdout.write("Name:      docker.io/%s\nMediaType: application/vnd.oci.image.index.v1+json\nDigest:    %s\n"
                         % (args[-1], digest))
elif cmd == "stop":
    if args[-1] not in C:
        sys.exit(1)
    C[args[-1]]["state"] = "exited"
    save()
elif cmd == "rm":
    if C.pop(args[-1], None) is None:
        sys.exit(1)
    save()
elif cmd == "run" and "nvidia-smi" in args and "-d" not in args:
    sys.exit(state.get("gpu_run_rc", 0))
'''


def _render(template_text, version, build_id=BUILD_ID, image="jefftud/odelia:1.9.0", site=SITE):
    """The client docker.sh as the build (version/build id) and the lighter (names) write it."""
    script = yaml.safe_load(template_text)["docker_cln_sh"]
    script = script.replace("__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_DOCKER_IMAGE__", version)
    script = script.replace("__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__", build_id)
    script = script.replace("{~~client_name~~}", site).replace("{~~docker_image~~}", image)
    return re.sub(r"\{~~[a-z_]+~~\}", "x", script)


def _old_template(tag):
    """The template of a released kit when the tag is available locally, else the current one.

    --migrate_from only reads an old kit's docker.sh for its `--env SITE_NAME=` and
    `--env MEDISWARM_VERSION=` lines, identical in v1.6.0, v1.8.1 and main.
    """
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{tag}:docker_config/master_template.yml"],
                              capture_output=True, text=True, check=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return TEMPLATE.read_text()


def _write(path, text, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mode:
        path.chmod(mode)
    return path


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Host:
    """A tmp host: stub tools on PATH, a new 1.9.0 kit, optional old kit(s) and fake daemons."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.stubs = tmp_path / "stubs"
        self.state_file = tmp_path / "docker_state.json"
        self.log_file = tmp_path / "docker_calls.log"
        self.daemons = []
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.fl_port = self.listener.getsockname()[1]
        self.admin_port = _free_port()  # nothing listens there
        self.state = {"containers": {}, "images": {}, "cgroup_driver": "cgroupfs"}
        _write(self.stubs / "docker", DOCKER_STUB, 0o755)
        _write(self.stubs / "curl", "#!/usr/bin/env bash\nexit 7\n", 0o755)
        _write(self.stubs / "dpkg-query", "#!/usr/bin/env bash\nprintf 'ii  nvidia-driver-550\\nii  "
               "libnvidia-compute-550\\nii  nvidia-container-toolkit\\nun  nvidia-driver-535\\n'\n", 0o755)
        self.set_nvidia_smi(NVIDIA_SMI_OK, 0)
        self.log_file.write_text("")
        self.new_kit = self.make_kit(tmp_path / "kits" / "TESTSITE_190", "1.9.0", BUILD_ID, TEMPLATE.read_text(),
                                     image_conf="# Which image this node runs.\nMEDISWARM_IMAGE=jefftud/odelia:current\n")
        self.data = tmp_path / "data"
        self.scratch = tmp_path / "scratch"
        self.data.mkdir()
        self.scratch.mkdir()

    def set_nvidia_smi(self, text, rc):
        _write(self.stubs / "nvidia-smi", f"#!/usr/bin/env bash\ncat <<'EOF'\n{text}EOF\nexit {rc}\n", 0o755)

    def make_kit(self, root, version, build_id, template_text, image_conf=None, sync_conf=None, site=SITE):
        startup = root / "startup"
        script = _write(startup / "docker.sh", _render(template_text, version, build_id, site=site), 0o755)
        assert script.exists()
        if image_conf is not None:
            _write(startup / "image.conf", image_conf)
        _write(startup / "sync.conf", sync_conf if sync_conf is not None else
               'REMOTE_HOST="upload.example.org"\nREMOTE_USER="mediswarm-upload"\nREMOTE_BASE="/srv/live"\n'
               "LOG_SYNC_INTERVAL=30\nCKPT_SYNC_INTERVAL=180\n")
        _write(startup / "fed_client.json", json.dumps({
            "format_version": 2,
            "servers": [{"name": "odelia", "service": {"scheme": "http"}, "identity": "server"}],
            "client": {"fqsn": site, "is_leaf": True},
            "overseer_agent": {"path": "x", "args": {"sp_end_point": f"127.0.0.1:{self.fl_port}:{self.admin_port}"}},
        }, indent=2))
        return root

    def start_fake_daemon(self, kit_root, site=SITE, write_pid_file=False):
        """A process whose command line is that of a live_sync daemon started by docker.sh."""
        startup = Path(kit_root) / "startup"
        script = _write(startup / "live_sync.sh", "#!/usr/bin/env bash\nwhile true; do sleep 0.2; done\n", 0o755)
        proc = subprocess.Popen(["bash", str(script), "--mode", "swarm", "--site-name", site, "--kit-root",
                                 str(kit_root), "--startup-dir", str(startup), "--scratch-dir", str(self.scratch)],
                                start_new_session=True)
        self.daemons.append(proc)
        if write_pid_file:
            _write(startup / ".mediswarm_sync" / "swarm_sync.pid", f"{proc.pid}\n")
        return proc

    def add_container(self, name, kit_root, state="running", image="jefftud/odelia:current", image_id=OLD_IMAGE_ID,
                      env=None, mounts=None, **extra):
        self.state["containers"][name] = {
            "state": state, "image": image, "image_id": image_id, "user": f"{os.getuid()}:{os.getgid()}",
            "env": env if env is not None else [f"SITE_NAME={SITE}", "GPU_DEVICE=device=0", "MODEL_NAME=MST",
                                                "TRAINING_MODE=swarm"],
            "mounts": mounts if mounts is not None else [
                {"src": str(kit_root), "dst": "/startupkit"}, {"src": str(self.scratch), "dst": "/scratch"},
                {"src": str(self.data), "dst": "/data", "rw": False}],
            "exec_out": "GPU 0: NVIDIA RTX A6000 (UUID: GPU-1234)\n", **extra}

    def run(self, *args, kit=None, extra_env=None, timeout=180):
        kit = Path(kit or self.new_kit)
        self.state_file.write_text(json.dumps(self.state))
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("MEDISWARM_", "ODELIA_")) and k not in ("MODEL_NAME", "INSTITUTION", "GPU2USE")}
        env.update({"PATH": f"{self.stubs}:{os.environ['PATH']}", "DOCKER_STUB_STATE": str(self.state_file),
                    "DOCKER_STUB_LOG": str(self.log_file), "MEDISWARM_STATUS_REGISTRY_CHECK": "0",
                    "TMPDIR": str(self.tmp), "HOME": str(self.tmp)})
        env.update(extra_env or {})
        result = subprocess.run(["bash", str(kit / "startup" / "docker.sh"), *args], cwd=kit / "startup", env=env,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        self.state = json.loads(self.state_file.read_text())
        return result

    def calls(self):
        return [json.loads(line) for line in self.log_file.read_text().splitlines() if line.strip()]

    def close(self):
        self.listener.close()
        for proc in self.daemons:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


@pytest.fixture
def host(tmp_path):
    h = Host(tmp_path)
    yield h
    h.close()


def _snapshot(root):
    """Path -> (mode, size, mtime, content) of every file under root, to prove nothing changed."""
    snap = {}
    for p in sorted(Path(root).rglob("*")):
        st = p.lstat()
        content = p.read_bytes() if p.is_file() and not p.is_symlink() else b""
        snap[str(p)] = (stat.S_IMODE(st.st_mode), st.st_size, st.st_mtime_ns, content)
    return snap


MUTATING = {"stop", "rm", "run", "pull", "kill", "restart", "start", "create", "update", "rename"}


def _mutating_calls(calls):
    return [c for c in calls if c and c[0] in MUTATING]


# ── generated script basics ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def docker_sh():
    return _render(TEMPLATE.read_text(), "1.9.0")


def test_generated_script_is_valid_bash(docker_sh, tmp_path):
    path = _write(tmp_path / "docker.sh", docker_sh)
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# Every flag of the 1.8.1 client docker.sh (#5 of the brief: they all keep working).
EXISTING_FLAGS = ["--data_dir", "--scratch_dir", "--GPU", "--no_pull", "--dummy_training", "--preflight_check",
                  "--local_training", "--start_client", "--list_licenses", "--interactive", "--run_script", "--job",
                  "--log_dataset_details", "--institution", "--model_name", "--num_epochs", "--fold", "--image",
                  "--config", "--container_name", "--docker_options", "--external_validation", "--checkpoint",
                  "--split"]


@pytest.mark.parametrize("flag", EXISTING_FLAGS + ["--status", "--migrate_from", "--yes", "--start"])
def test_every_flag_is_still_parsed(docker_sh, flag):
    assert re.search(rf"^\s+{re.escape(flag)}\)", docker_sh, re.M), f"docker.sh no longer accepts {flag}"


def test_heartbeat_still_reads_the_kit_version(docker_sh):
    """build_heartbeat.sh scrapes the FIRST 'MEDISWARM_VERSION=' of docker.sh as kit_version."""
    first = re.search(r"(?<=MEDISWARM_VERSION=)\S+", docker_sh).group(0).strip('"')
    assert first == "1.9.0"


def test_ci_container_name_rewrite_still_matches():
    """runIntegrationTests.sh rewrites the line 'CONTAINER_NAME=odelia_swarm_client_<site>_<id>'."""
    script = _render(TEMPLATE.read_text(), "1.9.0", build_id="095c1b7", site="client_A")
    assert script.count("CONTAINER_NAME=odelia_swarm_client_client_A_095c1b7") == 1


@pytest.mark.parametrize("args, marker", [
    (["--dummy_training"], "TRAINING_MODE=local_training"),
    (["--preflight_check"], "TRAINING_MODE=preflight_check"),
    (["--local_training"], "TRAINING_MODE=local_training"),
    (["--external_validation"], "EV_SPLIT=test"),
    (["--run_script", "echo hi"], "echo hi"),
    (["--interactive"], "--detach-keys=ctrl-x"),
    (["--start_client"], "TRAINING_MODE=swarm"),
])
def test_existing_run_modes_still_reach_docker_run(host, args, marker):
    result = host.run("--data_dir", str(host.data), "--scratch_dir", str(host.scratch), "--GPU", "device=0",
                      "--no_pull", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    runs = [c for c in host.calls() if c[0] == "run" and "-L" not in c]
    assert runs and any(marker in " ".join(c) for c in runs), (runs, result.stdout)


def test_list_licenses_needs_no_prompt_and_prints_no_extra_output(host):
    result = host.run("--no_pull", "--list_licenses")
    assert result.returncode == 0
    assert result.stdout.strip() == "[INFO] No interactive terminal detected, disabling TTY."


# ── #522: the /data double mount is guarded by a path-prefix test ─────────────


@pytest.mark.parametrize("data_dir, self_mount", [
    ("/data", False),
    ("/data/sub", False),
    ("/mnt/database/x", True),        # contains "/data": skipped by the old substring match
    ("/home/u/data_backup", True),    # ditto
    ("/mnt/sda1/ODELIA", True),
])
def test_data_self_mount_uses_a_path_prefix_test(host, data_dir, self_mount):
    result = host.run("--data_dir", data_dir, "--scratch_dir", str(host.scratch), "--GPU", "device=0", "--no_pull",
                      "--run_script", "true")
    assert result.returncode == 0, result.stderr
    run = next(c for c in host.calls() if c[0] == "run")
    assert f"{data_dir}:/data/:ro" in run
    assert (f"{data_dir}:{data_dir}:ro" in run) is self_mount, run


# ── --start_client pre-checks: F18 refusal, F2 warning, F3 lock clearing kept ──


def test_start_client_refuses_on_driver_library_mismatch(host):
    host.set_nvidia_smi(NVIDIA_SMI_MISMATCH, 18)
    result = host.run("--data_dir", str(host.data), "--scratch_dir", str(host.scratch), "--GPU", "device=0",
                      "--no_pull", "--start_client")
    assert result.returncode == 2, result.stdout
    out = result.stdout
    assert "[FAIL] NVIDIA driver/library version mismatch" in out
    assert "reboot this host" in out and "F18" in out
    assert "sudo apt-mark hold nvidia-driver-550 libnvidia-compute-550" in out
    assert "nvidia-container-toolkit" not in out.split("apt-mark hold", 1)[1].splitlines()[0]
    assert "(now 550.127)" in out
    assert not [c for c in host.calls() if c[0] == "run"], "no container may be started on a mismatch"
    assert "fix_docker_cgroupfs" not in out, "the F2 advice does not apply to a driver mismatch"


def test_start_client_warns_on_systemd_cgroup_driver_and_still_starts(host):
    host.state["cgroup_driver"] = "systemd"
    result = host.run("--data_dir", str(host.data), "--scratch_dir", str(host.scratch), "--GPU", "device=0",
                      "--no_pull", "--start_client")
    assert result.returncode == 0, result.stdout
    assert "[WARN] Docker cgroup driver is 'systemd'" in result.stdout
    assert "fix_docker_cgroupfs.sh" in result.stdout
    assert any(c[0] == "run" and "-d" in c for c in host.calls())


def test_start_client_checks_the_coordinator_from_fed_client_json(host):
    result = host.run("--data_dir", str(host.data), "--scratch_dir", str(host.scratch), "--GPU", "device=0",
                      "--no_pull", "--start_client")
    assert f"[PASS] Swarm server 127.0.0.1:{host.fl_port} reachable." in result.stdout


def test_start_client_leaves_a_stale_lock_to_the_new_container(host):
    """SO-2: docker.sh never deletes the kit's locks on the host (a running daemon may own them); the
    client container clears them in its own command (#630), when no NVFlare process of the kit can run."""
    lock = _write(host.new_kit / "daemon_pid.fl", "4242\n")
    result = host.run("--data_dir", str(host.data), "--scratch_dir", str(host.scratch), "--GPU", "device=0",
                      "--no_pull", "--start_client")
    assert result.returncode == 0
    assert lock.exists(), "the host side never removes a lock"
    assert "the client container clears them when it starts" in result.stdout
    start = next(c for c in host.calls() if c[0] == "run" and "-d" in c)
    assert start[-1].startswith("rm -f ../daemon_pid.fl ../pid.fl; nohup ./start.sh")


# ── --status ───────────────────────────────────────────────────────────────────


def _healthy(host):
    host.add_container(NEW_CONTAINER, host.new_kit, health="healthy")
    host.state["images"] = {"jefftud/odelia:current": {"id": OLD_IMAGE_ID},
                            OLD_IMAGE_ID: {"repo_digests": [f"jefftud/odelia@{OLD_DIGEST}"]}}
    host.start_fake_daemon(host.new_kit, write_pid_file=True)
    hb = _write(host.new_kit / "startup" / ".mediswarm_sync" / "swarm_heartbeat.json", json.dumps(
        {"status": "running", "sync_status": "ok", "job_id": "7a678d9a", "ip_address": "10.0.0.1"}, indent=2))
    assert hb.exists()


def test_status_reports_a_healthy_client_and_changes_nothing(host):
    _healthy(host)
    _write(host.new_kit / "daemon_pid.fl", "1\n")
    before = _snapshot(host.tmp / "kits")
    result = host.run("--status")
    out = result.stdout
    assert result.returncode == 0, out
    assert "kit 1.9.0 at" in out
    assert "configured image jefftud/odelia:current (from startup/image.conf)" in out
    assert "[OK]   running, health healthy, 0 restart(s)" in out
    assert "[OK]   the container runs the image that jefftud/odelia:current points to" in out
    assert "GPU visible inside the container: GPU 0: NVIDIA RTX A6000" in out
    assert "[OK]   daemon running (PID" in out
    assert "status running, upload ok, job 7a678d9a" in out
    assert "[OK]   NVIDIA driver 550.120 answers on the host" in out
    assert "[OK]   Docker cgroup driver is cgroupfs" in out
    assert f"[OK]   127.0.0.1:{host.fl_port} (FL port) reachable" in out
    assert f"[WARN] 127.0.0.1:{host.admin_port} (admin port) not reachable" in out
    assert "daemon_pid.fl present (normal while the client runs)" in out
    assert "Summary: 0 problem(s), 1 warning(s)." in out
    # read-only: no container/image change, no prompt, no file touched (the lock stays)
    assert _mutating_calls(host.calls()) == []
    assert _snapshot(host.tmp / "kits") == before
    assert "10.0.0.1" not in out


def test_status_flags_a_driver_library_mismatch_with_the_fix(host):
    _healthy(host)
    host.set_nvidia_smi(NVIDIA_SMI_MISMATCH, 18)
    result = host.run("--status")
    assert result.returncode == 1
    assert "[FAIL] NVIDIA driver/library version mismatch" in result.stdout
    assert "reboot this host" in result.stdout and "sudo apt-mark hold nvidia-driver-550" in result.stdout


def test_status_warns_on_the_systemd_cgroup_driver(host):
    _healthy(host)
    host.state["cgroup_driver"] = "systemd"
    result = host.run("--status")
    assert "[WARN] Docker cgroup driver is systemd" in result.stdout
    assert "fix_docker_cgroupfs.sh" in result.stdout and "F2" in result.stdout


def test_status_tells_when_current_moved_since_the_container_started(host):
    _healthy(host)
    # pulled again locally since the start, and the registry has moved on as well
    host.state["images"]["jefftud/odelia:current"] = {"id": "sha256:" + "d" * 64}
    host.state["registry_digest"] = NEW_DIGEST
    result = host.run("--status", extra_env={"MEDISWARM_STATUS_REGISTRY_CHECK": "1"})
    out = result.stdout
    assert "jefftud/odelia:current was pulled again after the container started" in out
    assert "jefftud/odelia:current moved on the registry (now cccccccccccc)" in out
    assert "registry digest bbbbbbbbbbbb" in out


def test_status_confirms_current_did_not_move(host):
    _healthy(host)
    host.state["registry_digest"] = OLD_DIGEST
    result = host.run("--status", extra_env={"MEDISWARM_STATUS_REGISTRY_CHECK": "1"})
    assert "has not moved on the registry since the running image was pulled" in result.stdout


def test_status_reports_no_gpu_inside_the_container(host):
    _healthy(host)
    host.state["containers"][NEW_CONTAINER].update(exec_out="Failed to initialize NVML: Unknown Error\n", exec_rc=255)
    result = host.run("--status")
    assert result.returncode == 1
    assert "[FAIL] no GPU inside the running container (Failed to initialize NVML: Unknown Error)" in result.stdout


def test_status_reports_stale_locks_and_what_old_kits_left_behind(host, tmp_path):
    old_kit = host.make_kit(tmp_path / "kits" / "TESTSITE_181", "1.8.1", "old1234", _old_template("v1.8.1"))
    host.add_container(NEW_CONTAINER, host.new_kit, state="exited", exit_code=137)
    host.add_container(OLD_CONTAINER, old_kit)
    host.start_fake_daemon(old_kit)
    _write(host.new_kit / "daemon_pid.fl", "4242\n")
    result = host.run("--status")
    out = result.stdout
    assert "the container is exited (exit code 137)" in out
    assert f"{OLD_CONTAINER} is running" in out and f"--migrate_from {old_kit}" in out
    assert f"of the kit {old_kit} still uploads for {SITE}" in out and "F16" in out
    assert "stale daemon_pid.fl" in out and "F3" in out
    assert (host.new_kit / "daemon_pid.fl").exists(), "--status must not clear the lock"


def test_status_fails_when_the_coordinator_fl_port_is_closed(host):
    _healthy(host)
    host.listener.close()
    result = host.run("--status")
    assert result.returncode == 1
    assert f"[FAIL] 127.0.0.1:{host.fl_port} (FL port) not reachable" in result.stdout


def test_status_without_a_client_container(host):
    result = host.run("--status")
    assert "this kit has no client container yet" in result.stdout
    assert _mutating_calls(host.calls()) == []


def test_status_and_migrate_reject_run_modes_and_stray_flags(host):
    assert host.run("--status", "--start_client").returncode == 1
    result = host.run("--yes")
    assert result.returncode == 1 and "only apply to --migrate_from" in result.stdout
    assert host.run("--status", "--migrate_from", str(host.tmp)).returncode == 1
    assert _mutating_calls(host.calls()) == []


# ── --migrate_from ────────────────────────────────────────────────────────────

OLD_IMAGE_CONF = """# Which image this node runs. Read by docker.sh on every run.
MEDISWARM_IMAGE=jefftud/odelia:1.8.1
MEDISWARM_DOCKER_OPTIONS="--cpus 8 -v /mnt/links:/mnt/links:ro -e ODELIA_RETURN_PER_CASE=1"
ODELIA_NUM_WORKERS=4
"""
OLD_SYNC_CONF = ('REMOTE_HOST="172.24.4.65"\nREMOTE_USER="mediswarm-upload"\nREMOTE_BASE="/srv/live"\n'
                 "LOG_SYNC_INTERVAL=30\nCKPT_SYNC_INTERVAL=180\nexport MEDISWARM_SITE_IP=10.1.2.3\n")


def _old_181(host, tmp_path, tag="v1.8.1", version="1.8.1", running_job=False):
    old_kit = host.make_kit(tmp_path / "kits" / f"TESTSITE_{version}", version, "old1234", _old_template(tag),
                            image_conf=OLD_IMAGE_CONF, sync_conf=OLD_SYNC_CONF)
    _write(old_kit / "daemon_pid.fl", "4242\n")
    _write(old_kit / "pid.fl", "4243\n")
    _write(old_kit / "local" / "resources.json", "{}")
    env = [f"SITE_NAME={SITE}", "DATA_DIR=/data", "SCRATCH_DIR=/scratch", "GPU_DEVICE=device=1",
           "MODEL_NAME=MST_SAMMed2D", "NUM_EPOCHS=100", "FOLD=0", "CONFIG=unilateral", f"INSTITUTION={SITE}_2",
           "MEDISWARM_VERSION=1.8.1", "TRAINING_MODE=swarm", "ODELIA_NUM_WORKERS=4", "ODELIA_HASH_NUM_WORKERS=2",
           "ODELIA_RETURN_PER_CASE=1", "WANDB_API_KEY=secret-value-123", "PATH=/usr/bin", "KMP_DUPLICATE_LIB_OK=TRUE"]
    mounts = [{"src": "/etc/passwd", "dst": "/etc/passwd"}, {"src": "/etc/group", "dst": "/etc/group"},
              {"src": str(old_kit), "dst": "/startupkit"}, {"src": str(host.scratch), "dst": "/scratch"},
              {"src": str(host.data), "dst": "/data", "rw": False},
              {"src": str(host.data), "dst": str(host.data), "rw": False},
              {"src": "/mnt/links", "dst": "/mnt/links", "rw": False}]
    top = "UID PID PPID C STIME TTY TIME CMD\n1000 11 1 0 08:00 ? 00:00:01 python3 -u -m nvflare.private.fed.app.client.client_train\n"
    if running_job:
        top += "1000 99 11 0 09:00 ? 00:10:00 python3 -u -m nvflare.private.fed.app.client.worker_process -m x\n"
    host.add_container(OLD_CONTAINER, old_kit, env=env, mounts=mounts, top=top)
    host.state["images"][OLD_IMAGE_ID] = {"env": ["PATH=/usr/bin", "KMP_DUPLICATE_LIB_OK=TRUE"]}
    daemon = host.start_fake_daemon(old_kit, write_pid_file=True)
    return old_kit, daemon


def test_migrate_dry_run_shows_the_plan_and_changes_nothing(host, tmp_path):
    old_kit, daemon = _old_181(host, tmp_path)
    before = _snapshot(host.tmp / "kits")
    result = host.run("--migrate_from", str(old_kit))
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert f"from the old kit {old_kit} (kit 1.8.1)" in out
    assert f"- {OLD_CONTAINER}: running, image jefftud/odelia:current, kit {old_kit}, TRAINING_MODE=swarm" in out
    assert f"- PID {daemon.pid} (swarm) of the kit {old_kit}" in out
    assert f"- {old_kit}/daemon_pid.fl" in out and f"- {old_kit}/pid.fl" in out
    # image.conf: the site's docker options and knob carried, the old image pin is not
    assert '+MEDISWARM_DOCKER_OPTIONS="--cpus 8 -v /mnt/links:/mnt/links:ro -e ODELIA_RETURN_PER_CASE=1"' in out
    assert "+ODELIA_NUM_WORKERS=4" in out
    assert "+MEDISWARM_IMAGE" not in out
    assert "Not carried: the old kit's MEDISWARM_IMAGE=jefftud/odelia:1.8.1" in out
    # sync.conf: the site's own values carried, the identical defaults are not repeated
    assert '+REMOTE_HOST="172.24.4.65"' in out and "+export MEDISWARM_SITE_IP=10.1.2.3" in out
    assert "+LOG_SYNC_INTERVAL" not in out
    # the exact start command, rebuilt from the container (mounts, GPU, env)
    command = next(line.strip() for line in out.splitlines() if "./docker.sh --data_dir" in line)
    assert command == (f"cd {host.new_kit / 'startup'} && env ODELIA_HASH_NUM_WORKERS=2 ./docker.sh "
                       f"--data_dir {host.data} --scratch_dir {host.scratch} --GPU device=1 "
                       f"--institution {SITE}_2 --model_name MST_SAMMed2D --start_client")
    assert "WANDB_API_KEY" in out and "secret-value-123" not in out
    assert "Dry run: nothing was changed." in out
    assert _mutating_calls(host.calls()) == []
    assert daemon.poll() is None, "a dry run must not stop the old daemon"
    assert _snapshot(host.tmp / "kits") == before


def test_migrate_yes_retires_the_old_kit_and_carries_the_settings(host, tmp_path):
    old_kit, daemon = _old_181(host, tmp_path)
    result = host.run("--migrate_from", str(old_kit), "--yes")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    calls = host.calls()
    assert ["stop", "-t", "10", OLD_CONTAINER] in calls and ["rm", OLD_CONTAINER] in calls
    assert OLD_CONTAINER not in host.state["containers"]
    daemon.wait(timeout=10)
    assert f"stopped live_sync PID {daemon.pid}" in out
    assert not (old_kit / "daemon_pid.fl").exists() and not (old_kit / "pid.fl").exists()
    assert not (old_kit / "startup" / ".mediswarm_sync" / "swarm_sync.pid").exists()
    # the old kit folder, data and scratch stay
    assert (old_kit / "startup" / "docker.sh").exists() and (old_kit / "local" / "resources.json").exists()
    assert host.data.is_dir() and host.scratch.is_dir()
    image_conf = (host.new_kit / "startup" / "image.conf").read_text()
    assert image_conf.startswith("# Which image this node runs.\nMEDISWARM_IMAGE=jefftud/odelia:current\n")
    assert 'MEDISWARM_DOCKER_OPTIONS="--cpus 8' in image_conf and "jefftud/odelia:1.8.1" not in image_conf
    sync_conf = (host.new_kit / "startup" / "sync.conf").read_text()
    assert sync_conf.rstrip().endswith('REMOTE_HOST="172.24.4.65"\nexport MEDISWARM_SITE_IP=10.1.2.3')
    assert list((host.new_kit / "startup").glob("image.conf.before_migration.*"))
    # the merged files still source cleanly, and the site value wins
    probe = subprocess.run(["bash", "-c", f'. "{host.new_kit}/startup/sync.conf"; . "{host.new_kit}/startup/image.conf";'
                            ' echo "$REMOTE_HOST|$MEDISWARM_IMAGE|$ODELIA_NUM_WORKERS"'], capture_output=True, text=True)
    assert probe.stdout.strip() == "172.24.4.65|jefftud/odelia:current|4"
    assert "start command: cd" in (host.new_kit / "startup" / ".migrated_from_old_kit").read_text()
    # The only container run is the throwaway GPU check of the new image before the old client is stopped (SO-3).
    started = [c for c in calls if c[0] == "run" and "nvidia-smi" not in c]
    assert not started, "without --start nothing is started"
    assert [c for c in calls if c[0] == "run" and "nvidia-smi" in c], "the new image's GPU check ran first"

    # idempotent: a second run finds nothing left to do and carries nothing twice
    again = host.run("--migrate_from", str(old_kit), "--yes")
    assert again.returncode == 0, again.stdout
    assert "none (no container named" in again.stdout and "none running" in again.stdout
    assert again.stdout.count("no site settings to carry") == 2
    assert "The previous migration recorded: cd" in again.stdout
    assert (host.new_kit / "startup" / "image.conf").read_text() == image_conf


def test_migrate_yes_start_starts_the_new_client_with_the_old_settings(host, tmp_path):
    old_kit, daemon = _old_181(host, tmp_path)
    result = host.run("--migrate_from", str(old_kit), "--yes", "--start")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    daemon.wait(timeout=10)
    calls = host.calls()
    assert ["pull", "jefftud/odelia:current"] in calls, "the new kit's image line wins over the old pin"
    start = next(c for c in calls if c[0] == "run" and "-d" in c)
    joined = " ".join(start)
    assert f"--name={NEW_CONTAINER}" in start and "--gpus=device=1" in start
    assert f"{host.data}:/data/:ro" in start and f"{host.scratch}:/scratch/" in start
    for env in ["MODEL_NAME=MST_SAMMed2D", f"INSTITUTION={SITE}_2", "ODELIA_NUM_WORKERS=4",
                "ODELIA_HASH_NUM_WORKERS=2", "TRAINING_MODE=swarm"]:
        assert env in start, (env, joined)
    # image.conf's carried docker options reach the new container
    assert "--cpus 8 -v /mnt/links:/mnt/links:ro -e ODELIA_RETURN_PER_CASE=1" in joined
    assert calls.index(["stop", "-t", "10", OLD_CONTAINER]) < calls.index(start)


def test_migrate_yes_refuses_while_a_job_runs_in_the_old_client(host, tmp_path):
    old_kit, daemon = _old_181(host, tmp_path, running_job=True)
    before = _snapshot(host.tmp / "kits")
    result = host.run("--migrate_from", str(old_kit), "--yes", "--start")
    assert result.returncode == 3, result.stdout
    assert "[!] a job is running in it" in result.stdout and "Nothing was changed." in result.stdout
    assert _mutating_calls(host.calls()) == []
    assert daemon.poll() is None
    assert _snapshot(host.tmp / "kits") == before


def test_migrate_from_a_1_6_0_kit(host, tmp_path):
    old_kit, daemon = _old_181(host, tmp_path, tag="v1.6.0", version="1.6.0")
    result = host.run("--migrate_from", str(old_kit / "startup"))   # the startup/ folder works too
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"from the old kit {old_kit} (kit 1.6.0)" in result.stdout
    assert f"- PID {daemon.pid} (swarm) of the kit {old_kit}" in result.stdout


def test_migrate_stops_daemons_of_every_old_kit_of_the_site_but_not_others(host, tmp_path):
    old_kit, daemon = _old_181(host, tmp_path)
    older = host.make_kit(tmp_path / "kits" / "TESTSITE_160", "1.6.0", "older16", _old_template("v1.6.0"))
    older_daemon = host.start_fake_daemon(older)                       # CAM in #614: a 1.6.0 daemon still running
    other_site = host.make_kit(tmp_path / "kits" / "OTHER_1", "1.8.1", "old1234", TEMPLATE.read_text(), site="OTHER_1")
    other_daemon = host.start_fake_daemon(other_site, site="OTHER_1")
    decade = tmp_path / "kits" / "decade_TESTSITE"
    _write(decade / "startup" / "docker.sh", "CONTAINER_NAME=stamp_swarm_client_TESTSITE_x\n", 0o755)
    decade_daemon = host.start_fake_daemon(decade)                     # same site name, other consortium
    own_daemon = host.start_fake_daemon(host.new_kit)                   # the new kit's own daemon stays
    result = host.run("--migrate_from", str(old_kit), "--yes")
    assert result.returncode == 0, result.stdout
    daemon.wait(timeout=10)
    older_daemon.wait(timeout=10)
    for keep in (other_daemon, decade_daemon, own_daemon):
        assert keep.poll() is None
    assert f"stopped live_sync PID {older_daemon.pid} ({older})" in result.stdout


def test_migrate_passes_unexplained_mounts_again(host, tmp_path):
    old_kit = host.make_kit(tmp_path / "kits" / "TESTSITE_181", "1.8.1", "old1234", _old_template("v1.8.1"),
                            image_conf="MEDISWARM_IMAGE=jefftud/odelia:current\n")
    host.add_container(OLD_CONTAINER, old_kit, state="exited", mounts=[
        {"src": str(old_kit), "dst": "/startupkit"}, {"src": str(host.scratch), "dst": "/scratch"},
        {"src": str(host.data), "dst": "/data", "rw": False}, {"src": "/opt/extra", "dst": "/extra", "rw": False}])
    result = host.run("--migrate_from", str(old_kit))
    assert "--docker_options -v\\ /opt/extra:/extra:ro --start_client" in result.stdout
    assert "was started with --docker_options" in result.stdout
    assert "set outside docker.sh" not in result.stdout, "no image env to compare with: no guessing"


def test_migrate_without_an_old_container_asks_for_the_folders(host, tmp_path):
    old_kit = host.make_kit(tmp_path / "kits" / "TESTSITE_181", "1.8.1", "old1234", _old_template("v1.8.1"))
    result = host.run("--migrate_from", str(old_kit), "--yes", "--start")
    assert result.returncode == 1
    assert "./docker.sh --data_dir <your data folder> --scratch_dir <your scratch folder>" in result.stdout
    assert "there is no old client container to read them from" in result.stdout
    assert "Nothing was changed." in result.stdout and not (host.new_kit / "startup" / ".migrated_from_old_kit").exists()
    # with the folders given on the command line it can start
    result = host.run("--migrate_from", str(old_kit), "--yes", "--start", "--data_dir", str(host.data),
                      "--scratch_dir", str(host.scratch), "--GPU", "device=0")
    assert result.returncode == 0, result.stdout
    assert any(c[0] == "run" and "-d" in c for c in host.calls())


def test_migrate_into_a_kit_without_image_conf_keeps_the_old_pin_behind(host, tmp_path):
    old_kit, daemon = _old_181(host, tmp_path)
    (host.new_kit / "startup" / "image.conf").unlink()
    result = host.run("--migrate_from", str(old_kit), "--yes")
    assert result.returncode == 0, result.stdout
    assert "is missing in this kit and is created with the old kit's settings" in result.stdout
    created = (host.new_kit / "startup" / "image.conf").read_text()
    assert "MEDISWARM_IMAGE" not in created, "the old image pin must not come back through a missing file"
    assert 'MEDISWARM_DOCKER_OPTIONS="--cpus 8' in created and "ODELIA_NUM_WORKERS=4" in created


def test_migrate_refuses_another_sites_kit_and_this_kit_itself(host, tmp_path):
    other = host.make_kit(tmp_path / "kits" / "OTHER_1", "1.8.1", "old1234", TEMPLATE.read_text(), site="OTHER_1")
    result = host.run("--migrate_from", str(other), "--yes")
    assert result.returncode == 1 and "is the kit of 'OTHER_1'" in result.stdout
    result = host.run("--migrate_from", str(host.new_kit), "--yes")
    assert result.returncode == 1 and "is this kit itself" in result.stdout
    result = host.run("--migrate_from", str(tmp_path / "nowhere"))
    assert result.returncode == 1 and "is not a startup kit folder" in result.stdout
    assert _mutating_calls(host.calls()) == []


def test_migrate_does_not_start_for_another_user(host, tmp_path):
    old_kit, daemon = _old_181(host, tmp_path)
    host.state["containers"][OLD_CONTAINER]["user"] = "0:0"
    result = host.run("--migrate_from", str(old_kit))
    assert " && sudo env ODELIA_HASH_NUM_WORKERS=2 ./docker.sh" in result.stdout or os.getuid() == 0
    if os.getuid() != 0:
        result = host.run("--migrate_from", str(old_kit), "--yes", "--start")
        assert result.returncode == 1 and "ran as another user" in result.stdout
        assert _mutating_calls(host.calls()) == [], "a refused --start must not change anything"
        assert daemon.poll() is None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))


# ── 1.9.0 review fixes: SO-3 and SO-4 ─────────────────────────────────────────


def test_migrate_yes_keeps_the_old_client_when_the_new_image_cannot_use_the_gpu(host, tmp_path):
    """SO-3: the new kit's checks run before anything is stopped; a failure leaves the old client running."""
    old_kit, daemon = _old_181(host, tmp_path)
    host.state["gpu_run_rc"] = 1
    result = host.run("--migrate_from", str(old_kit), "--yes")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Nothing was changed: the old client keeps running" in result.stdout
    calls = host.calls()
    assert not [c for c in calls if c[0] in ("stop", "rm")]
    assert host.state["containers"][OLD_CONTAINER]["state"] == "running"
    daemon.kill()


def test_start_client_refuses_while_an_old_kits_client_runs(host, tmp_path):
    """SO-4: two clients of one site push each other out of the swarm; --migrate_from is the way."""
    old_kit, daemon = _old_181(host, tmp_path)
    result = host.run("--data_dir", str(host.data), "--scratch_dir", str(host.scratch), "--GPU", "device=0",
                      "--no_pull", "--start_client")
    assert result.returncode == 2, result.stdout + result.stderr
    assert f"The client container {OLD_CONTAINER} of another kit" in result.stdout
    assert "--migrate_from" in result.stdout
    assert not [c for c in host.calls() if c[0] == "run" and "-d" in c], "no client was started"
    daemon.kill()
