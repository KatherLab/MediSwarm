"""kit_live_sync/live_sync.sh of the 1.9.0 kits, run for real against a fake ssh, rsync and docker.

The daemon runs on each site host and uploads a run to the TUD monitor. These tests start the
real script (and the real build_heartbeat.sh) in a temporary kit with stub `ssh`, `rsync` and
`docker` first on PATH: the stubs write "uploads" into a local directory, so the tests can check
what would have left the site. They cover:

- what leaves the site by default (review finding DE-4): logs, heartbeats, TensorBoard event files
  and summaries yes; model files only with MEDISWARM_SYNC_MODELS=1; per-case prediction CSVs only
  when the client runs with ODELIA_RETURN_PER_CASE=1 or with MEDISWARM_SYNC_PREDICTIONS=1, never
  when the job forces ODELIA_RETURN_PER_CASE=0;
- the kit's own upload key (#615): startup/upload_key + upload_known_hosts are used with
  IdentitiesOnly and a pinned host key, and a refused kit key falls back to sync.conf;
- the client container watchdog (#614): the swarm daemon ends a few minutes after its client
  container stopped, and says "stopped" instead of writing a final heartbeat;
- the remote swarm/_active folder is removed only by the daemon that wrote it (#614);
- the heartbeat's sync status is cleared after a clean cycle and reports the daemon's version.
"""

import json
import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_LIVE_SYNC = REPO_ROOT / "kit_live_sync"
SITE = "TESTSITE_1"
JOB = "644a5ee5-c998-4ced-9d78-66a898d98c26"
RUN = "2026_10_15_101500_TESTSITE_1_MST_swarm_learning"
CONTAINER = f"odelia_swarm_client_{SITE}_abc1234"
REMOTE_HOST = "upload.example.org"
HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJevh9BVJBrEVJKLX5BnrdW7Y0COGjTKxSdzsvUygeu5"
SYNC_CONF = (f'REMOTE_HOST="{REMOTE_HOST}"\nREMOTE_USER="mediswarm-upload"\nREMOTE_BASE="/srv/live"\n'
             "LOG_SYNC_INTERVAL=0.2\nCKPT_SYNC_INTERVAL=0\n"
             'SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes '
             '-o ConnectTimeout=5"\n')

# Decides, for the ssh options of one connection, whether the fake upload host lets it in.
# FAKE_REFUSE=kit refuses the kit key (-i), =legacy refuses connections without -i; the file
# FAKE_DOWN makes the host unreachable.
AUTH = r'''
def refused(words):
    if os.path.exists(os.environ["FAKE_DOWN"]):
        return True
    with_key = "-i" in words
    refuse = os.environ.get("FAKE_REFUSE", "")
    return (refuse == "kit" and with_key) or (refuse == "legacy" and not with_key)
'''

FAKE_SSH = r'''#!/usr/bin/env python3
import json, os, shutil, sys
''' + AUTH + r'''
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(["ssh"] + args) + "\n")
WITH_VALUE = {"-o", "-i", "-p", "-l", "-F", "-E", "-S", "-J", "-c", "-m", "-b"}
i = 0
while i < len(args) and args[i].startswith("-"):
    i += 2 if args[i] in WITH_VALUE else 1
if refused(args[:i]):
    sys.stderr.write("Permission denied (publickey).\n")
    sys.exit(255)
command = " ".join(args[i + 1:])
root = os.environ["FAKE_REMOTE"]
if command == "echo ok":
    print("ok")
elif command.startswith("mkdir -p '"):
    os.makedirs(root + command[len("mkdir -p '"):-1], exist_ok=True)
elif command.startswith("rm -rf '"):
    shutil.rmtree(root + command[len("rm -rf '"):-1], ignore_errors=True)
else:
    sys.exit(1)
'''

# Applies rsync's include/exclude rules the way live_sync uses them (basename patterns, first
# match wins, '*/' = every directory) and copies into FAKE_REMOTE. A file named in FAKE_FAIL_ON
# makes every transfer whose arguments contain that text fail; FAKE_VANISHED_ON exits 24.
FAKE_RSYNC = r'''#!/usr/bin/env python3
import fnmatch, json, os, shutil, sys
''' + AUTH + r'''
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(["rsync"] + args) + "\n")
rules, pos, ssh_words, i = [], [], [], 0
while i < len(args):
    a = args[i]
    if a == "-e":
        ssh_words = args[i + 1].split()
        i += 2
        continue
    if a.startswith("--include="):
        rules.append((True, a.split("=", 1)[1]))
    elif a.startswith("--exclude="):
        rules.append((False, a.split("=", 1)[1]))
    elif not a.startswith("-"):
        pos.append(a)
    i += 1
if refused(ssh_words):
    sys.exit(255)
for name, rc in (("FAKE_FAIL_ON", 23), ("FAKE_VANISHED_ON", 24)):
    path = os.environ.get(name, "")
    if path and os.path.exists(path) and open(path).read().strip() in " ".join(args):
        sys.exit(rc)
src, dst = pos[-2], pos[-1]
target = os.environ["FAKE_REMOTE"] + dst.split(":", 1)[1]

def allowed(name, is_dir):
    for include, pat in rules:
        if pat.endswith("/"):
            if is_dir and fnmatch.fnmatchcase(name, pat[:-1]):
                return include
        elif fnmatch.fnmatchcase(name, pat):
            return include
    return True

if src.endswith("/"):
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if allowed(d, True)]
        for fn in filenames:
            if allowed(fn, False):
                rel = os.path.relpath(os.path.join(dirpath, fn), src)
                os.makedirs(os.path.dirname(os.path.join(target, rel)), exist_ok=True)
                shutil.copy(os.path.join(dirpath, fn), os.path.join(target, rel))
else:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy(src, target)
'''

# docker ps / inspect answered from a JSON state file the test may rewrite while the daemon runs.
FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
try:
    with open(os.environ["FAKE_DOCKER_STATE"]) as f:
        containers = json.load(f)
except (OSError, ValueError):
    containers = {}
fmt = args[args.index("--format") + 1] if "--format" in args else ""
if args[:1] == ["ps"]:
    wanted = [a.split("=", 1)[1] for a in args if a.startswith("name=")]
    for name, c in containers.items():
        if ("-a" in args or c["state"] == "running") and all(w in name for w in wanted):
            print(name)
elif args[:1] == ["inspect"]:
    c = containers.get(args[-1])
    if c is None:
        sys.exit(1)
    if ".Config.Env" in fmt:
        sys.stdout.write("".join(e + "\n" for e in c.get("env", [])))
    elif ".Mounts" in fmt:
        sys.stdout.write("".join("%s|%s\n" % (m["dst"], m["src"]) for m in c.get("mounts", [])))
    elif ".Config.Image" in fmt:
        print(c.get("image", "jefftud/odelia:1.9.0"))
    elif ".Image" in fmt:
        print(c.get("image_id", "sha256:" + "c" * 64))
    elif ".State.Status" in fmt:
        print(c["state"])
else:
    sys.exit(1)
'''


def _write(path, text, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mode is not None:
        path.chmod(mode)
    return path


class Kit:
    """A temporary site host: one client kit, its scratch run directory, stubs and the fake remote."""

    def __init__(self, tmp_path, with_job=True):
        self.tmp = tmp_path
        self.kit = tmp_path / "kits" / "TESTSITE_190"
        self.startup = self.kit / "startup"
        self.scratch = tmp_path / "scratch"
        self.remote = tmp_path / "remote"
        self.stubs = tmp_path / "stubs"
        self.log = tmp_path / "calls.log"
        self.down = tmp_path / "upload_host_down"
        self.docker_state = tmp_path / "docker_state.json"
        self.procs = []
        for name in ("live_sync.sh", "build_heartbeat.sh"):
            _write(self.startup / name, (KIT_LIVE_SYNC / name).read_text(), 0o755)
        _write(self.startup / "sync.conf", SYNC_CONF)
        _write(self.startup / "docker.sh", "#!/usr/bin/env bash\n  --env MEDISWARM_VERSION=1.9.0 \\\n", 0o755)
        _write(self.startup / "nohup.out", f"Run name: {RUN}\n")
        _write(self.kit / "log.txt", "client log\n")
        _write(self.stubs / "ssh", FAKE_SSH, 0o755)
        _write(self.stubs / "rsync", FAKE_RSYNC, 0o755)
        _write(self.stubs / "docker", FAKE_DOCKER, 0o755)
        self.remote.mkdir()
        self.log.write_text("")
        (tmp_path / "t").mkdir()
        run = self.scratch / "runs" / SITE / RUN
        for name in ("events.out.tfevents.1760000000.host", "data_layout.json", "last.ckpt", "epoch=3-step=40.ckpt",
                     "last_global_model.ckpt", "site_model_gt_and_classprob_validation.csv",
                     "aggregated_model_gt_and_classprob_validation.csv",
                     "final_global_model_gt_and_classprob_test.csv", "best_site_model_gt_and_classprob_test.csv",
                     "lightning_logs/version_0/hparams.yaml", "al_selection_summary.json"):
            _write(run / name, f"{name}\n")
        if with_job:
            self.add_job()
        self.set_containers({CONTAINER: {"state": "running", "env": [f"SITE_NAME={SITE}"],
                                         "mounts": [{"src": str(self.kit), "dst": "/startupkit"}]}})

    def add_job(self, launcher_env=""):
        app = self.kit / JOB / f"app_{SITE}"
        _write(app / "FL_global_model.pt", "global\n")
        _write(app / "best_FL_global_model.pt", "best global\n")
        _write(app / "config" / "config_fed_client.conf",
               f'{{ executors = [{{ script = "{launcher_env} python3 custom/main.py" }}] }}\n')

    def set_containers(self, containers):
        tmp = self.docker_state.with_suffix(".tmp")
        tmp.write_text(json.dumps(containers))
        tmp.replace(self.docker_state)

    def env(self, **extra):
        env = {"PATH": f"{self.stubs}:/usr/local/bin:/usr/bin:/bin", "HOME": str(self.tmp),
               "TMPDIR": str(self.tmp / "t"), "FAKE_LOG": str(self.log), "FAKE_REMOTE": str(self.remote),
               "FAKE_DOWN": str(self.down), "FAKE_DOCKER_STATE": str(self.docker_state), "LANG": "C.UTF-8"}
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def start(self, mode="swarm", container=CONTAINER, **env):
        args = ["bash", str(self.startup / "live_sync.sh"), "--mode", mode, "--site-name", SITE,
                "--kit-root", str(self.kit), "--startup-dir", str(self.startup), "--scratch-dir", str(self.scratch)]
        if container:
            args += ["--container-name", container]
        out = open(self.tmp / f"daemon_{len(self.procs)}.out", "w")
        proc = subprocess.Popen(args, env=self.env(**env), stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
        self.procs.append(proc)
        return proc

    def stop(self, proc, timeout=60):
        """SIGTERM, as docker.sh and --migrate_from do: the daemon runs its final upload and exits."""
        proc.send_signal(signal.SIGTERM)
        return proc.wait(timeout=timeout)

    def close(self):
        for proc in self.procs:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)

    def remote_dir(self, run_id=JOB, mode="swarm"):
        return self.remote / "srv" / "live" / SITE / mode / run_id

    def uploaded(self):
        return sorted(str(p.relative_to(self.remote)) for p in self.remote.rglob("*") if p.is_file())

    def calls(self, tool):
        return [c[1:] for c in map(json.loads, self.log.read_text().splitlines()) if c[0] == tool]

    def daemon_log(self):
        log = self.startup / ".mediswarm_sync" / "live_sync_daemon.log"
        return log.read_text() if log.exists() else ""  # polled before the daemon's first line

    def heartbeat(self, name="heartbeat.json", run_id=JOB):
        return json.loads((self.remote_dir(run_id) / name).read_text())


def _wait_for(condition, timeout=30, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _cycles(kit, n):
    """Wait until the daemon has uploaded the job's heartbeat in n cycles."""
    _wait_for(lambda: sum(1 for c in kit.calls("rsync") if c[-1].endswith(f"{JOB}/heartbeat.json")) >= n,
              what=f"{n} sync cycles")


@pytest.fixture
def kit(tmp_path):
    k = Kit(tmp_path)
    yield k
    k.close()


def _names(kit):
    return [Path(p).name for p in kit.uploaded()]


# ── what leaves the site (DE-4) ───────────────────────────────────────────────


def test_by_default_models_and_per_case_rows_stay_at_the_site(kit):
    proc = kit.start()
    _cycles(kit, 2)
    assert kit.stop(proc) == 0
    names = _names(kit)
    for expected in ("heartbeat.json", "heartbeat_final.json", "nohup.out", "log.txt", "live_sync_daemon.log",
                     "events.out.tfevents.1760000000.host", "data_layout.json", "hparams.yaml",
                     "al_selection_summary.json"):
        assert expected in names, (expected, names)
    leaked = [n for n in names if n.endswith((".ckpt", ".pt")) or "_gt_and_classprob_" in n]
    assert leaked == [], leaked
    beat = kit.heartbeat()
    assert beat["sync_models"] is False and beat["sync_predictions"] is False
    assert "Upload policy: models: stay at the site; per-case predictions: stay at the site (no per-case opt-in)" \
        in kit.daemon_log()


def test_an_opted_in_site_uploads_its_per_case_rows_but_no_models(kit):
    kit.set_containers({CONTAINER: {"state": "running", "env": [f"SITE_NAME={SITE}", "ODELIA_RETURN_PER_CASE=1"]}})
    proc = kit.start()
    _cycles(kit, 2)
    assert kit.stop(proc) == 0
    names = _names(kit)
    for csv in ("site_model_gt_and_classprob_validation.csv", "aggregated_model_gt_and_classprob_validation.csv",
                "final_global_model_gt_and_classprob_test.csv", "best_site_model_gt_and_classprob_test.csv"):
        assert csv in names
    assert not [n for n in names if n.endswith((".ckpt", ".pt"))]
    assert kit.heartbeat()["sync_predictions"] is True


def test_model_files_only_with_sync_models(kit):
    _write(kit.startup / "sync.conf", SYNC_CONF + "MEDISWARM_SYNC_MODELS=1\n")
    proc = kit.start()
    _cycles(kit, 2)
    assert kit.stop(proc) == 0
    names = _names(kit)
    for model in ("FL_global_model.pt", "best_FL_global_model.pt", "last.ckpt", "epoch=3-step=40.ckpt",
                  "last_global_model.ckpt"):
        assert model in names
    assert not [n for n in names if "_gt_and_classprob_" in n]
    assert kit.heartbeat()["sync_models"] is True


@pytest.mark.parametrize("opted_in, switch, launcher, uploaded", [
    (False, "1", "", True),                               # the site allows it without the coordinator opt-in
    (True, "0", "", False),                               # the site forbids it although opted in
    (True, "", "ODELIA_RETURN_PER_CASE=0", False),        # a DP job forces per-case return off
    (True, "1", "ODELIA_RETURN_PER_CASE=0", False),       # ... and that wins over the site's switch
])
def test_per_case_switches(kit, opted_in, switch, launcher, uploaded):
    kit.add_job(launcher_env=launcher)
    env = [f"SITE_NAME={SITE}"] + (["ODELIA_RETURN_PER_CASE=1"] if opted_in else [])
    kit.set_containers({CONTAINER: {"state": "running", "env": env}})
    proc = kit.start(**({"MEDISWARM_SYNC_PREDICTIONS": switch} if switch else {}))
    _cycles(kit, 2)
    assert kit.stop(proc) == 0
    assert any("_gt_and_classprob_" in n for n in _names(kit)) is uploaded


@pytest.mark.parametrize("image_conf, docker_options, uploaded", [
    ('MEDISWARM_DOCKER_OPTIONS="--env ODELIA_RETURN_PER_CASE=1"\n', None, True),
    ('# MEDISWARM_DOCKER_OPTIONS="--env ODELIA_RETURN_PER_CASE=1"\n', None, False),
    ('MEDISWARM_DOCKER_OPTIONS="--cpus 2" # --env ODELIA_RETURN_PER_CASE=1\n', None, False),
    # docker.sh tells the daemon which options it passed (a one-off --docker_options beats image.conf)
    ('MEDISWARM_DOCKER_OPTIONS="--env ODELIA_RETURN_PER_CASE=1"\n', "", False),
    ("", "--cpus 2 -e ODELIA_RETURN_PER_CASE=1", True),
])
def test_without_a_container_to_ask_the_docker_options_decide(kit, image_conf, docker_options, uploaded):
    """--local_training: the --rm container is gone by the final upload, so the options decide."""
    kit.set_containers({})
    _write(kit.startup / "image.conf", image_conf)
    env = {} if docker_options is None else {"MEDISWARM_SYNC_DOCKER_OPTIONS": docker_options}
    proc = kit.start(mode="local", **env)
    _wait_for(lambda: (kit.remote_dir(RUN, "local") / "heartbeat.json").exists(), what="local heartbeat")
    assert kit.stop(proc) == 0
    assert (kit.remote_dir(RUN, "local") / "heartbeat_final.json").exists()
    assert any("_gt_and_classprob_" in n for n in _names(kit)) is uploaded
    assert not [n for n in _names(kit) if n.endswith(".ckpt")]


# ── upload key (#615) ─────────────────────────────────────────────────────────


def _ssh_words(kit):
    """The ssh options of every upload: ssh's own arguments and rsync's -e command."""
    words = [c[:c.index(f"mediswarm-upload@{REMOTE_HOST}")] for c in kit.calls("ssh")]
    words += [c[c.index("-e") + 1].split()[1:] for c in kit.calls("rsync")]
    return words


def _option(words, name):
    return [words[i + 1] for i, w in enumerate(words[:-1]) if w == "-o" and words[i + 1].startswith(name + "=")]


def test_the_kits_upload_key_and_pinned_host_key_are_used(kit):
    key = _write(kit.startup / "upload_key", "-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n", 0o644)
    known = _write(kit.startup / "upload_known_hosts", f"{REMOTE_HOST},172.24.4.65 {HOST_KEY}\n")
    proc = kit.start()
    _cycles(kit, 1)
    assert kit.stop(proc) == 0
    for words in _ssh_words(kit):
        assert words[words.index("-i") + 1] == str(key)
        assert _option(words, "IdentitiesOnly") == ["IdentitiesOnly=yes"]
        assert _option(words, "StrictHostKeyChecking") == ["StrictHostKeyChecking=yes"]
        assert _option(words, "UserKnownHostsFile") == [f"UserKnownHostsFile={known}"]
        assert _option(words, "BatchMode") == ["BatchMode=yes"]          # the rest of sync.conf stays
    assert stat.S_IMODE(key.stat().st_mode) == 0o600, "ssh ignores a private key others can read"
    assert kit.heartbeat()["upload_auth"] == "kit-key"


def test_a_known_hosts_file_for_another_host_keeps_the_sync_conf_host_key_settings(kit):
    key = _write(kit.startup / "upload_key", "fake key\n", 0o600)
    _write(kit.startup / "upload_known_hosts", f"other.example.org {HOST_KEY}\n")
    proc = kit.start()
    _cycles(kit, 1)
    assert kit.stop(proc) == 0
    for words in _ssh_words(kit):
        assert words[words.index("-i") + 1] == str(key)
        assert _option(words, "StrictHostKeyChecking") == ["StrictHostKeyChecking=no"]


def test_a_refused_kit_key_falls_back_to_the_sync_conf_settings(kit):
    _write(kit.startup / "upload_key", "fake key\n", 0o600)
    _write(kit.startup / "upload_known_hosts", f"{REMOTE_HOST} {HOST_KEY}\n")
    proc = kit.start(FAKE_REFUSE="kit")
    _cycles(kit, 1)
    assert kit.stop(proc) == 0
    assert kit.heartbeat()["upload_auth"] == "legacy-fallback"
    assert "does not accept this kit's upload key" in kit.daemon_log()
    rsync_words = [c[c.index("-e") + 1].split() for c in kit.calls("rsync")]
    assert rsync_words and all("-i" not in w for w in rsync_words)


def test_a_kit_without_upload_key_uses_sync_conf(kit):
    proc = kit.start()
    _cycles(kit, 1)
    assert kit.stop(proc) == 0
    for words in _ssh_words(kit):
        assert "-i" not in words
        assert _option(words, "StrictHostKeyChecking") == ["StrictHostKeyChecking=no"]
        assert _option(words, "ControlMaster") == ["ControlMaster=auto"]       # #602 multiplexing kept
    assert kit.heartbeat()["upload_auth"] == "legacy"


def test_the_swarm_daemon_waits_for_an_unreachable_upload_host(kit):
    kit.down.write_text("")
    proc = kit.start(MEDISWARM_SYNC_CONNECT_RETRY_SECONDS="0.3")
    _wait_for(lambda: "still cannot connect" in kit.daemon_log(), what="a retry")
    assert proc.poll() is None, "a swarm daemon must not give up for the whole run"
    kit.down.unlink()
    _cycles(kit, 1)
    assert kit.stop(proc) == 0


# ── client container watchdog (#614) ──────────────────────────────────────────


def test_the_swarm_daemon_ends_when_its_client_container_is_gone(kit):
    proc = kit.start(MEDISWARM_SYNC_CONTAINER_GONE_MINUTES="0.02")    # 1.2 s, rounded up to 2 s
    _cycles(kit, 2)
    time.sleep(2.5)
    assert proc.poll() is None, "the daemon must keep running while its container runs"
    assert kit.heartbeat()["client_state"] == "running" and kit.heartbeat()["client_container"] == CONTAINER
    kit.set_containers({CONTAINER: {"state": "exited"}})
    assert proc.wait(timeout=30) == 0
    beat = kit.heartbeat()
    assert beat["status"] == "stopped"
    assert not (kit.remote_dir() / "heartbeat_final.json").exists(), "a finished job's logs must keep the say"
    assert "has not been running for" in kit.daemon_log()
    local = json.loads((kit.startup / ".mediswarm_sync" / "swarm_heartbeat.json").read_text())
    assert local["status"] == "stopped", "docker.sh --status reads this file"


def test_a_waiting_daemon_marks_its_active_folder_stopped(tmp_path):
    kit = Kit(tmp_path, with_job=False)
    try:
        proc = kit.start(MEDISWARM_SYNC_CONTAINER_GONE_MINUTES="0.02")
        active = kit.remote_dir("_active")
        _wait_for(lambda: (active / "heartbeat.json").exists(), what="the waiting heartbeat")
        kit.set_containers({})
        assert proc.wait(timeout=30) == 0
        assert json.loads((active / "heartbeat.json").read_text())["status"] == "stopped"
    finally:
        kit.close()


def test_a_decade_kit_container_is_found_by_its_kit_mount(kit):
    """The DECADE docker.sh does not pass --container-name: the container mounting the kit counts."""
    stamp = f"stamp_swarm_client_{SITE}_def5678"
    kit.set_containers({stamp: {"state": "running", "mounts": [{"src": str(kit.startup) + "/..", "dst": "/startupkit/"}]},
                        "unrelated": {"state": "running", "mounts": [{"src": "/elsewhere", "dst": "/startupkit"}]}})
    proc = kit.start(container="", MEDISWARM_SYNC_CONTAINER_GONE_MINUTES="0.02")
    _cycles(kit, 2)
    time.sleep(2.5)
    assert proc.poll() is None
    assert kit.heartbeat()["client_container"] == stamp
    kit.set_containers({"unrelated": {"state": "running", "mounts": [{"src": "/elsewhere", "dst": "/startupkit"}]}})
    assert proc.wait(timeout=30) == 0


def test_a_container_never_seen_running_gets_the_start_grace(kit):
    kit.set_containers({})
    start = time.time()
    proc = kit.start(MEDISWARM_SYNC_CONTAINER_GONE_MINUTES="0.01", MEDISWARM_SYNC_CONTAINER_START_GRACE_MINUTES="0.06")
    assert proc.wait(timeout=30) == 0
    assert time.time() - start >= 3.5, "docker run may still be pulling the image"


def test_without_docker_access_the_daemon_keeps_running(kit):
    _write(kit.stubs / "docker", "#!/usr/bin/env bash\necho 'permission denied' >&2\nexit 1\n", 0o755)
    proc = kit.start(MEDISWARM_SYNC_CONTAINER_GONE_MINUTES="0.01")
    _cycles(kit, 2)
    time.sleep(1.5)
    assert proc.poll() is None
    assert kit.heartbeat()["client_state"] == "unknown"
    assert kit.stop(proc) == 0


# ── the shared swarm/_active folder (#614) ────────────────────────────────────


def test_only_the_daemon_that_wrote_active_removes_it(tmp_path):
    # Another kit's daemon (a job in its kit, never waiting) must leave the folder alone.
    other = Kit(tmp_path / "other")
    active = other.remote_dir("_active")
    (active).mkdir(parents=True)
    _write(active / "heartbeat.json", "{}")
    try:
        proc = other.start()
        _cycles(other, 2)
        assert other.stop(proc) == 0
        assert not [c for c in other.calls("ssh") if c[-1].startswith("rm -rf")]
        assert (active / "heartbeat.json").exists()
    finally:
        other.close()

    # The daemon that waited removes exactly its own folder once its job starts.
    kit = Kit(tmp_path / "own", with_job=False)
    active = kit.remote_dir("_active")
    try:
        proc = kit.start()
        _wait_for(lambda: (active / "heartbeat.json").exists(), what="the waiting heartbeat")
        kit.add_job()
        _cycles(kit, 1)
        assert kit.stop(proc) == 0
        removed = [c[-1] for c in kit.calls("ssh") if c[-1].startswith("rm -rf")]
        assert removed == [f"rm -rf '/srv/live/{SITE}/swarm/_active'"]
        assert not active.exists()
    finally:
        kit.close()


# ── heartbeat ──────────────────────────────────────────────────────────────────


def test_sync_status_is_cleared_after_a_clean_cycle(kit):
    fail = kit.tmp / "fail_on"
    fail.write_text("nohup.out")
    local = kit.startup / ".mediswarm_sync" / "swarm_heartbeat.json"
    proc = kit.start(FAKE_FAIL_ON=fail)
    _wait_for(lambda: local.exists() and json.loads(local.read_text())["sync_status"] == "error",
              what="an error heartbeat")
    assert "nohup.out" in json.loads(local.read_text())["sync_error"]
    fail.unlink()
    _wait_for(lambda: json.loads(local.read_text())["sync_status"] == "ok", what="an ok heartbeat")
    assert json.loads(local.read_text())["sync_error"] == ""
    assert kit.stop(proc) == 0


def test_files_vanishing_during_an_upload_are_no_error(kit):
    vanished = kit.tmp / "vanished_on"
    vanished.write_text("run_dir")
    proc = kit.start(FAKE_VANISHED_ON=vanished)
    _cycles(kit, 3)
    assert kit.stop(proc) == 0
    assert kit.heartbeat()["sync_status"] == "ok"


def test_heartbeat_reports_versions(kit):
    proc = kit.start()
    _cycles(kit, 1)
    assert kit.stop(proc) == 0
    beat = kit.heartbeat("heartbeat_final.json")
    assert beat["live_sync_version"] == beat["sync_version"] == "3"
    assert beat["kit_version"] == "1.9.0"
    assert beat["status"] == "finished"
    assert beat["image_ref"] == "jefftud/odelia:1.9.0"


def test_the_kit_version_docker_sh_passes_wins(kit):
    proc = kit.start(MEDISWARM_KIT_VERSION="1.9.1")
    _cycles(kit, 1)
    assert kit.stop(proc) == 0
    assert kit.heartbeat()["kit_version"] == "1.9.1"
