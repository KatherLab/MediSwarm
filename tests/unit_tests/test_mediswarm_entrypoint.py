"""The image ENTRYPOINT clears a stale NVFlare lock only when the container starts the kit daemon.

Sites keep their 1.8.1 kits, whose container command (`bash -c "nohup ./start.sh ..."`, started
with --restart=unless-stopped) does not clear the lock that #630 added to newer kits. After a power
loss sub_start.sh then finds <kit>/daemon_pid.fl naming a PID that exists again in the restarted
container and refuses to start, and the node stays offline. docker_config/mediswarm_entrypoint.sh
gives those kits the #630 behaviour from inside the image.

The danger is the opposite mistake: removing pid.fl next to a LIVE daemon makes sub_start.sh start
a second NVFlare process. So the script must act only for the kit launcher, only in an NVFlare kit
in swarm mode (client) or a server kit, and only in a fresh PID namespace, and it must leave every
other command (preflight, local training, external validation, --run_script, the admin console,
CI's python3 calls) byte-for-byte unchanged. These tests run the script with bash against
simulated kit directories.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # only the template-drift cases need it; they are skipped without it
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "docker_config" / "mediswarm_entrypoint.sh"
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"
DOCKERFILE = REPO_ROOT / "docker_config" / "Dockerfile_ODELIA"

# Container commands of the kits the sites run today (v1.8.1 master_template.yml, docker_cln_sh
# --start_client and docker_svr_sh --start_server); they can no longer change.
CLIENT_CMD_181 = "nohup ./start.sh >> nohup.out 2>&1 && /bin/bash"
SERVER_CMD_181 = "nohup ./start.sh >> nohup.out 2>&1 && chmod a+r nohup.out && /bin/bash"
# The same commands after #630 (kits built from main clear the lock themselves).
CLIENT_CMD_630 = "rm -f ../daemon_pid.fl ../pid.fl; " + CLIENT_CMD_181
SERVER_CMD_630 = "rm -f ../daemon_pid.fl ../pid.fl; " + SERVER_CMD_181

# Stand-in for the kit's start.sh: records whether the lock was still there when it ran.
START_SH_STUB = """#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -e "$DIR/../daemon_pid.fl" ]; then echo "lock=present"; else echo "lock=absent"; fi > "$DIR/../launch_report"
echo kit-start-ran
"""


def make_kit(root, role, stale_pid="10"):
    """A startup kit as NVFlare provisions it, with the locks a crash leaves behind."""
    startup = root / "kit" / "startup"
    startup.mkdir(parents=True)
    (startup / "start.sh").write_text(START_SH_STUB)
    (startup / "start.sh").chmod(0o755)
    (startup / "sub_start.sh").write_text("#!/bin/bash\necho sub_start must not run in these tests\nexit 9\n")
    (startup / "sub_start.sh").chmod(0o755)
    (startup / f"fed_{role}.json").write_text("{}\n")
    (root / "kit" / "daemon_pid.fl").write_text(f"{stale_pid}\n")
    (root / "kit" / "pid.fl").write_text(f"{int(stale_pid) + 2}\n")
    return startup


def locks(kit_startup):
    kit = kit_startup.parent
    return sorted(p.name for p in kit.iterdir() if p.name.endswith(".fl"))


def base_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("TRAINING_MODE", "MEDISWARM_ENTRYPOINT_CLEAR_LOCKS")}
    env.update(extra)
    return env


def run_entrypoint(args, cwd, env, fresh_namespace=False):
    """Run the entrypoint like Docker does (as the program, args = container command).

    fresh_namespace=True stands in for "this is PID 1 of a new container": the test process can
    never be that, so the namespace probe is replaced after sourcing the script. Everything else
    (matching, kit checks, the removal, the exec) is the real code.
    """
    if fresh_namespace:
        cmd = ["bash", "-c",
               'source "$1"; shift; _pid_namespace_is_fresh() { return 0; }; main "$@"',
               "entrypoint-under-test", str(ENTRYPOINT), *args]
    else:
        cmd = ["bash", str(ENTRYPOINT), *args]
    return subprocess.run(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)


# ----------------------------------------------------------------------- the restart it fixes

@pytest.mark.parametrize("command", [CLIENT_CMD_181, CLIENT_CMD_630], ids=["kit-1.8.1", "kit-main"])
def test_client_restart_clears_stale_locks_before_start_sh(tmp_path, command):
    startup = make_kit(tmp_path, "client")
    result = run_entrypoint(["/bin/bash", "-c", command], startup, base_env(TRAINING_MODE="swarm"),
                            fresh_namespace=True)
    assert result.returncode == 0, result.stderr
    assert locks(startup) == []
    assert (startup.parent / "launch_report").read_text().strip() == "lock=absent"
    assert "kit-start-ran" in (startup / "nohup.out").read_text()
    assert "removed stale" in result.stderr and "daemon_pid.fl (pid 10)" in result.stderr


@pytest.mark.parametrize("command", [SERVER_CMD_181, SERVER_CMD_630], ids=["kit-1.8.1", "kit-main"])
def test_server_restart_clears_stale_locks_without_training_mode(tmp_path, command):
    """The server container gets no TRAINING_MODE; fed_server.json identifies the kit."""
    startup = make_kit(tmp_path, "server")
    result = run_entrypoint(["/bin/bash", "-c", command], startup, base_env(), fresh_namespace=True)
    assert result.returncode == 0, result.stderr
    assert locks(startup) == []
    assert (startup.parent / "launch_report").read_text().strip() == "lock=absent"


def test_absolute_launcher_path_resolves_its_own_kit(tmp_path):
    startup = make_kit(tmp_path, "server")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = run_entrypoint([str(startup / "start.sh")], elsewhere, base_env(), fresh_namespace=True)
    assert result.returncode == 0, result.stderr
    assert locks(startup) == []
    assert "kit-start-ran" in result.stdout


# ----------------------------------------------------------------------- when it must not act

def test_locks_stay_when_other_processes_share_the_namespace(tmp_path):
    """The real probe: with an unrelated process alive (as a daemon would be), nothing is removed.

    The sibling is started explicitly because a test container can otherwise hold nothing but
    the test's own ancestors, which the probe rightly treats as a fresh namespace.
    """
    startup = make_kit(tmp_path, "client")
    sibling = subprocess.Popen(["sleep", "60"])
    try:
        result = run_entrypoint(["/bin/bash", "-c", CLIENT_CMD_181], startup, base_env(TRAINING_MODE="swarm"))
    finally:
        sibling.kill()
        sibling.wait()
    assert locks(startup) == ["daemon_pid.fl", "pid.fl"]
    assert (startup.parent / "launch_report").read_text().strip() == "lock=present"
    assert "other processes are visible" in result.stderr


@pytest.mark.parametrize("training_mode", [None, "preflight_check", "local_training"])
def test_client_kit_outside_swarm_mode_keeps_locks(tmp_path, training_mode):
    """--run_script and the site-local modes never set TRAINING_MODE=swarm; only --start_client does."""
    startup = make_kit(tmp_path, "client")
    env = base_env() if training_mode is None else base_env(TRAINING_MODE=training_mode)
    result = run_entrypoint(["/bin/bash", "-c", CLIENT_CMD_181], startup, env, fresh_namespace=True)
    assert result.returncode == 0, result.stderr
    assert locks(startup) == ["daemon_pid.fl", "pid.fl"]
    assert result.stderr == ""


def test_kill_switch_keeps_locks(tmp_path):
    startup = make_kit(tmp_path, "client")
    env = base_env(TRAINING_MODE="swarm", MEDISWARM_ENTRYPOINT_CLEAR_LOCKS="0")
    result = run_entrypoint(["/bin/bash", "-c", CLIENT_CMD_181], startup, env, fresh_namespace=True)
    assert result.returncode == 0, result.stderr
    assert locks(startup) == ["daemon_pid.fl", "pid.fl"]


def test_start_sh_outside_a_kit_is_left_alone(tmp_path):
    """A start.sh without sub_start.sh and fed_*.json next to it is not an NVFlare kit."""
    startup = make_kit(tmp_path, "client")
    (startup / "fed_client.json").unlink()
    result = run_entrypoint(["/bin/bash", "-c", CLIENT_CMD_181], startup, base_env(TRAINING_MODE="swarm"),
                            fresh_namespace=True)
    assert result.returncode == 0, result.stderr
    assert locks(startup) == ["daemon_pid.fl", "pid.fl"]


# Commands docker.sh and CI run in a kit's startup dir that must never touch the locks, even in a
# fresh container with TRAINING_MODE=swarm (the worst case). Real paths are replaced by harmless
# commands of the same shape.
OTHER_COMMANDS = {
    "preflight/local-training": ["/bin/bash", "-c", "echo /MediSwarm/application/jobs/x/app/custom/main.py"],
    "external-validation": ["/bin/bash", "-c", 'echo python3 /MediSwarm/scripts/evaluation/predict.py --split "$EV"'],
    "run-script": ["/bin/bash", "-c", "ls -1 . > /dev/null && echo script-ran"],
    "admin-console": ["/bin/bash", "-c", "./fl_admin.sh || true"],
    "reads-start-sh": ["/bin/bash", "-c", "cat ./start.sh > /dev/null; echo done"],
    "mentions-start-sh": ["/bin/bash", "-c", "echo nohup ./start.sh"],
    "ci-python": [sys.executable, "-c", "print('ci-python')"],
    "interactive-shell": ["/bin/bash"],
}


@pytest.mark.parametrize("name", sorted(OTHER_COMMANDS))
def test_other_commands_run_unchanged_and_keep_locks(tmp_path, name):
    startup = make_kit(tmp_path, "client")
    command = OTHER_COMMANDS[name]
    env = base_env(TRAINING_MODE="swarm")
    via_entrypoint = run_entrypoint(command, startup, env, fresh_namespace=True)
    direct = subprocess.run(command, cwd=startup, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)
    assert locks(startup) == ["daemon_pid.fl", "pid.fl"]
    assert (via_entrypoint.returncode, via_entrypoint.stdout, via_entrypoint.stderr) == \
        (direct.returncode, direct.stdout, direct.stderr)


def test_command_sees_exactly_the_container_environment_and_exit_code(tmp_path):
    """bash would otherwise add PWD and SHLVL to a `python3 ...` command's environment."""
    probe = "import json, os, sys; print(json.dumps(dict(os.environ), sort_keys=True)); sys.exit(3)"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "ODELIA_X": "a b\nc"}
    via_entrypoint = run_entrypoint([sys.executable, "-c", probe], tmp_path, env)
    direct = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert via_entrypoint.returncode == direct.returncode == 3
    seen = json.loads(via_entrypoint.stdout)
    # Python itself may add LC_CTYPE (PEP 538) -- in both runs alike, so compare run against run.
    assert seen == json.loads(direct.stdout)
    assert seen["ODELIA_X"] == "a b\nc"
    assert "PWD" not in seen and "SHLVL" not in seen


def test_no_command_starts_the_default_shell(tmp_path):
    result = subprocess.run(["bash", str(ENTRYPOINT)], cwd=tmp_path, env=base_env(),
                            input="echo $((6 * 7))\n", capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert result.stdout.strip() == "42"


# ----------------------------------------------------------------------- the pieces

def _bash_with_entrypoint(snippet, *args):
    return subprocess.run(["bash", "-c", 'source "$1"; shift; ' + snippet, "entrypoint-under-test",
                           str(ENTRYPOINT), *args], capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("extra_pid, own_entry, expected", [
    (None, True, "fresh"),        # only this shell and its parent: a just-started container
    ("4242", True, "busy"),      # anything else running: e.g. --pid=host, or a docker exec
    (None, False, "busy"),       # /proc without our own entry is not trusted
])
def test_namespace_probe_on_a_fake_proc(tmp_path, extra_pid, own_entry, expected):
    snippet = """
        _PROC_ROOT=$1
        if [ "$2" = yes ]; then
            mkdir -p "$_PROC_ROOT/$$"
            printf 'Name:\\tbash\\nPPid:\\t1\\n' > "$_PROC_ROOT/$$/status"
        fi
        mkdir -p "$_PROC_ROOT/1"
        printf 'Name:\\tdocker-init\\nPPid:\\t0\\n' > "$_PROC_ROOT/1/status"
        if [ -n "$3" ]; then mkdir -p "$_PROC_ROOT/$3"; fi
        if _pid_namespace_is_fresh; then echo fresh; else echo busy; fi
    """
    result = _bash_with_entrypoint(snippet, str(tmp_path / "proc"), "yes" if own_entry else "no", extra_pid or "")
    assert result.stdout.strip() == expected, result.stderr


def _kit_commands_in_template(key):
    if yaml is None:
        return []
    script = yaml.safe_load(TEMPLATE.read_text())[key]
    commands = []
    for line in script.splitlines():
        line = line.strip()
        if "./start.sh" in line and line.startswith('/bin/bash -c "'):
            commands.append(line[len('/bin/bash -c "'):-1])
    return commands


@pytest.mark.parametrize("key", ["docker_cln_sh", "docker_svr_sh"])
def test_template_start_commands_are_found(key):
    if yaml is None:
        pytest.skip("PyYAML not installed")
    assert len(_kit_commands_in_template(key)) == 1, key


@pytest.mark.parametrize("command", [
    CLIENT_CMD_181, SERVER_CMD_181, CLIENT_CMD_630, SERVER_CMD_630,
    *_kit_commands_in_template("docker_cln_sh"), *_kit_commands_in_template("docker_svr_sh"),
])
def test_every_kit_start_command_is_recognised(command):
    """Guards the matcher against drifting away from the commands the kits really run."""
    result = _bash_with_entrypoint('_find_kit_launcher /bin/bash -c "$1"; echo "[$_KIT_SCRIPT]"', command)
    assert result.stdout.strip() == "[./start.sh]", (command, result.stderr)


@pytest.mark.parametrize("args", [
    ["/bin/bash", "-c", "./fl_admin.sh"],
    ["/bin/bash", "-c", "/MediSwarm/scripts/_list_licenses.sh"],
    ["/bin/bash", "-c", "nvidia-smi -L"],
    ["/bin/bash", "-c", "cat ./start.sh"],
    ["/bin/bash", "-c", "./start.shx"],
    ["python3", "./start.sh"],
    ["./restart.sh"],
])
def test_non_launchers_are_not_recognised(args):
    result = _bash_with_entrypoint('_find_kit_launcher "$@"; echo "[$_KIT_SCRIPT]"', *args)
    assert result.stdout.strip() == "[]", (args, result.stderr)


def test_dockerfile_installs_the_entrypoint_and_keeps_bash_as_default_command():
    """An ENTRYPOINT resets an inherited CMD, so CMD ["/bin/bash"] must be declared again."""
    lines = [line.strip() for line in DOCKERFILE.read_text().splitlines()]
    assert 'ENTRYPOINT ["/usr/local/bin/mediswarm_entrypoint.sh"]' in lines
    assert lines.index('CMD ["/bin/bash"]') > lines.index('ENTRYPOINT ["/usr/local/bin/mediswarm_entrypoint.sh"]')
    assert any(line.startswith("COPY ") and line.endswith(" /usr/local/bin/mediswarm_entrypoint.sh")
               and "docker_config/mediswarm_entrypoint.sh" in line for line in lines)
    assert os.access(ENTRYPOINT, os.X_OK), "git must keep the entrypoint executable (mode 100755)"
