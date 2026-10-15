"""Deploy-test harness: container scoping, credentials, image override, launcher, example conf.

The test hosts (dl0, dl2, dl3) also run another project's containers
(swarm-<hash>-dd2-node-*), and every ODELIA kit uses the odelia_swarm_* naming of the
production coordinator and the partner sites. These tests pin what the harness may remove,
that passwords never reach a command line, and that a candidate image can be run with
existing kits (server and admin included).
"""

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = REPO_ROOT / "scripts" / "deploy"
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"
EXAMPLE_CONF = DEPLOY / "deploy_sites_4node_local.dl3.conf.example"
KIT_IMAGE = "jefftud/odelia:1.8.1-dev.260913.f7c66bb"
SUFFIX = "f7c66bb"


def bash(script, env=None, cwd=None):
    full_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp")}
    full_env.update(env or {})
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + script],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=cwd or REPO_ROOT,
    )


def sourced(script, env=None):
    return bash(f'source "{DEPLOY}/deploy_common.sh"\nsource "{DEPLOY}/container_scope.sh"\n{script}', env=env)


def render_template_block(key, client="TEST_A_1"):
    """The docker.sh a kit gets from master_template.yml's ``key`` block."""
    lines = TEMPLATE.read_text().splitlines()
    start = lines.index(f"{key}: |") + 1
    body = []
    for line in lines[start:]:
        if line and not line.startswith(" "):
            break
        body.append(line[2:])
    text = "\n".join(body) + "\n"
    return (
        text.replace("{~~docker_image~~}", KIT_IMAGE)
        .replace("{~~client_name~~}", client)
        .replace("__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__", SUFFIX)
        .replace("__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_DOCKER_IMAGE__", "1.8.1-dev.260913.f7c66bb")
    )


def make_prod(tmp_path, kits=("TEST_A_1", "dl3.tud.de")):
    prod = tmp_path / "workspace" / "odelia_deploy_test_x_model_test" / "prod_01"
    for kit in kits:
        startup = prod / kit / "startup"
        startup.mkdir(parents=True)
        key = "docker_svr_sh" if "." in kit else "docker_cln_sh"
        (startup / "docker.sh").write_text(render_template_block(key, client=kit))
    return prod


# ── container_scope.sh ────────────────────────────────────────────────────


def test_suffix_is_read_from_client_and_server_kits(tmp_path):
    prod = make_prod(tmp_path)
    r = sourced(f'kit_container_suffix "{prod}" odelia_swarm')
    assert r.returncode == 0 and r.stdout.strip() == SUFFIX
    server_only = make_prod(tmp_path / "s", kits=("dl3.tud.de",))
    r = sourced(f'kit_container_suffix "{server_only}" odelia_swarm')
    assert r.returncode == 0 and r.stdout.strip() == SUFFIX, r.stderr


def test_no_suffix_means_no_filter(tmp_path):
    (tmp_path / "empty" / "X" / "startup").mkdir(parents=True)
    (tmp_path / "empty" / "X" / "startup" / "docker.sh").write_text("CONTAINER_NAME=whatever\n")
    for prod in (tmp_path / "empty", tmp_path / "missing", ""):
        r = sourced(f'kit_container_suffix "{prod}" odelia_swarm')
        assert r.returncode != 0 and r.stdout == ""
    assert sourced("kit_container_filter odelia_swarm notahash TEST_A_1").returncode != 0
    assert sourced("kit_container_filter odelia_swarm 'f7c66bb|.*' TEST_A_1").returncode != 0


MATCH = [
    "odelia_swarm_client_TEST_A_1_f7c66bb",
    "odelia_swarm_client_TEST_B_1_f7c66bb",
    "odelia_swarm_server_flserver_f7c66bb",
    "odelia_swarm_admin_f7c66bb",
]
NO_MATCH = [
    "swarm-1a7f19db-dd2-node-A",  # the other project on dl0/dl2/dl3
    "swarm-1a7f19db-dd2-node-B",
    "odelia_swarm_server_flserver_a19be57",  # production coordinator
    "odelia_swarm_client_CAM_1_a76c691",  # a partner's 1.8.1 client
    "odelia_swarm_client_TEST_C_1_f7c66bb",  # same kit, site not in this conf
    "xodelia_swarm_client_TEST_A_1_f7c66bb",
    "odelia_swarm_client_TEST_A_1_f7c66bb_old",
    "stamp_swarm_client_TEST_A_1_f7c66bb",
    "nvflare_simulator",
    "train_matched_1",
]


def test_filter_matches_only_this_kit_and_these_sites():
    r = sourced("kit_container_filter odelia_swarm f7c66bb TEST_A_1 TEST_B_1")
    assert r.returncode == 0, r.stderr
    regex = r.stdout.strip()
    for name in MATCH:
        assert re.search(regex, name), name
    for name in NO_MATCH:
        assert not re.search(regex, name), name
    # The same regex through grep -E, as the harness uses it (remote hosts included).
    listing = "\\n".join(MATCH + NO_MATCH) + "\\n"
    r = bash(f"printf '%b' '{listing}' | grep -E '{regex}'")
    assert sorted(r.stdout.split()) == sorted(MATCH)


def test_site_names_are_literal_in_the_filter():
    regex = sourced("kit_container_filter odelia_swarm f7c66bb A.1").stdout.strip()
    assert re.search(regex, "odelia_swarm_client_A.1_f7c66bb")
    assert not re.search(regex, "odelia_swarm_client_AX1_f7c66bb")


def test_server_filter_ignores_clients():
    regex = sourced("kit_server_container_filter odelia_swarm f7c66bb").stdout.strip()
    assert re.search(regex, "odelia_swarm_server_flserver_f7c66bb")
    assert not re.search(regex, "odelia_swarm_client_TEST_A_1_f7c66bb")
    assert not re.search(regex, "odelia_swarm_server_flserver_a19be57")


def test_no_harness_removes_containers_by_a_broad_pattern():
    """'odelia_swarm|nvflare|^swarm-' matched the other project's node containers."""
    for path in DEPLOY.glob("*.sh"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
            assert "^swarm-" not in code, f"{path.name}:{number}"
            assert "odelia_swarm|nvflare" not in code, f"{path.name}:{number}"


def _stub(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def _harness_tree(tmp_path, kit_lines=None):
    """A repo-shaped copy of the harness with a built 2-client test kit, stub tools and a conf."""
    repo = tmp_path / "repo"
    for rel in ("scripts/deploy/run_deploy_test.sh", "scripts/deploy/deploy_common.sh",
                "scripts/deploy/container_scope.sh", "scripts/build/getVersionNumber.sh",
                "application/provision/project_deploy_test_4site.yml"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, repo / rel)
    prod = make_prod(tmp_path / "kits", kits=("TEST_A_1", "TEST_B_1", "dl3.tud.de"))
    workspace = repo / "workspace" / "odelia_deploy_test_1.9.0-test_model_test"
    shutil.copytree(prod, workspace / "prod_01")
    if kit_lines is not None:
        for docker_sh in (workspace / "prod_01").glob("*/startup/docker.sh"):
            docker_sh.write_text(kit_lines)
    deploy_base = tmp_path / "deploy"
    (deploy_base / "dl3.tud.de").mkdir(parents=True)
    (deploy_base / "dl3.tud.de" / "job_store.txt").write_text("keep me")

    fake = tmp_path / "bin"
    fake.mkdir()
    _stub(fake / "docker", f'''case "$1" in
  ps) printf '%s\\n' {" ".join(MATCH + NO_MATCH)} ;;
  rm) shift; [[ "$1" == -f ]] && shift; printf '%s\\n' "$@" >> "$DOCKER_RM_LOG" ;;
esac
''')
    # ssh runs the remote command here, as if every client host were this one.
    _stub(fake / "ssh", 'for last; do :; done\necho "$*" >> "$SSH_LOG"\nbash -c "$last"\n')
    for tool in ("sshpass", "expect", "jq", "sleep"):
        _stub(fake / tool, "exit 0\n")
    _stub(fake / "sudo", "echo sudo called >&2; exit 1\n")

    conf = tmp_path / "deploy_sites_test.conf"
    conf.write_text(f"""CLIENT_SITES=(A B)
A_HOST=host-a
A_USER=swarm
A_SITE_NAME=TEST_A_1
B_HOST=host-b
B_USER=swarm
B_SITE_NAME=TEST_B_1
COSMOS_DEPLOY_DIR={deploy_base}
PROJECT_FILE=application/provision/project_deploy_test_4site.yml
SERVER_NAME=dl3.tud.de
""")
    env = {
        "PATH": f"{fake}:/usr/bin:/bin",
        "MEDISWARM_IMAGE_VERSION": "1.9.0-test",
        "DOCKER_RM_LOG": str(tmp_path / "rm.log"),
        "SSH_LOG": str(tmp_path / "ssh.log"),
    }
    return repo, conf, env, deploy_base


def test_cleanup_only_removes_exactly_this_kits_containers_everywhere(tmp_path):
    """The path the deploy-test workflow's always() cleanup step runs (it used 'odelia|stamp|nvflare')."""
    repo, conf, env, deploy_base = _harness_tree(tmp_path)
    r = bash(f'"{repo}/scripts/deploy/run_deploy_test.sh" --conf "{conf}" --cleanup-only', env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    removed = (tmp_path / "rm.log").read_text().split()
    # Once here (server/admin host) and once per client host, nothing else.
    assert sorted(removed) == sorted(MATCH * 3)
    hosts = (tmp_path / "ssh.log").read_text()
    assert "swarm@host-a" in hosts and "swarm@host-b" in hosts
    # The server's job store survives a cleanup-only call.
    assert (deploy_base / "dl3.tud.de" / "job_store.txt").read_text() == "keep me"
    assert "sudo called" not in r.stderr


def test_cleanup_only_refuses_without_a_kit_suffix(tmp_path):
    repo, conf, env, _ = _harness_tree(tmp_path, kit_lines="CONTAINER_NAME=odelia_swarm_client_TEST_A_1\n")
    r = bash(f'"{repo}/scripts/deploy/run_deploy_test.sh" --conf "{conf}" --cleanup-only', env=env)
    assert r.returncode != 0
    assert not (tmp_path / "rm.log").exists()
    assert "refusing" in (r.stdout + r.stderr)


# ── deploy_common.sh: variables and credentials ──────────────────────────


def test_optional_site_variables_do_not_abort_under_set_u():
    r = sourced('v=$(site_var DL3A INSTITUTION); echo "[$v]"')
    assert r.returncode == 0 and r.stdout.strip() == "[]", r.stderr


def test_site_password_sources_and_precedence(tmp_path):
    secret_file = tmp_path / "pass"
    secret_file.write_text("from-file\nsecond line\n")
    assert sourced("site_pass A", env={"A_PASS": "plain"}).stdout == "plain"
    assert sourced("site_pass A", env={"A_PASS_ENV": "MY_PW", "MY_PW": "from-env"}).stdout == "from-env"
    assert sourced("site_pass A", env={"A_PASS_FILE": str(secret_file)}).stdout == "from-file"
    both = {"A_PASS_ENV": "MY_PW", "MY_PW": "from-env", "A_PASS_FILE": str(secret_file)}
    assert sourced("site_pass A", env=both).stdout == "from-env"
    assert sourced("site_pass A").stdout == ""


def test_password_never_reaches_argv(tmp_path):
    """`sshpass -p <secret>` is readable by every user in `ps`; the harness uses -e."""
    fake = tmp_path / "bin"
    fake.mkdir()
    for tool in ("sshpass", "ssh", "scp"):
        script = fake / tool
        script.write_text(f'#!/bin/sh\necho "{tool} argv: $*"\necho "{tool} SSHPASS=${{SSHPASS:-}}"\n')
        script.chmod(0o755)
    env = {"PATH": f"{fake}:/usr/bin:/bin", "A_HOST": "h", "A_USER": "u", "A_PASS_ENV": "PW", "PW": "s3cret"}
    r = sourced('SSH_OPTS="-o X=1"; remote_exec A "echo hi"; remote_copy A /tmp/x /tmp/y; remote_fetch A /r /l', env)
    assert r.returncode == 0, r.stderr
    argv_lines = [ln for ln in r.stdout.splitlines() if "argv:" in ln]
    assert len(argv_lines) == 3 and all("s3cret" not in ln for ln in argv_lines)
    assert all(ln.startswith("sshpass argv: -e ") for ln in argv_lines)
    assert "sshpass SSHPASS=s3cret" in r.stdout
    # Without any password: plain ssh with keys only.
    r = sourced('SSH_OPTS=""; remote_exec B "echo hi"', {"PATH": f"{fake}:/usr/bin:/bin", "B_HOST": "h", "B_USER": "u"})
    assert r.stdout.startswith("ssh argv: -o BatchMode=yes")


@pytest.mark.parametrize("path", ["run_deploy_test.sh", "deploy_common.sh"])
def test_no_password_on_command_lines(path):
    text = (DEPLOY / path).read_text()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "sshpass -p" not in code
    assert "echo '$(site_var" not in code  # sudo password spliced into a remote command


# ── Image under test with existing kits ───────────────────────────────────


@pytest.mark.parametrize("key, client", [("docker_svr_sh", "dl3.tud.de"), ("docker_adm_sh", "admin"),
                                         ("docker_cln_sh", "TEST_A_1")])
def test_kit_script_with_image_rewrites_exactly_the_image_line(tmp_path, key, client):
    startup = tmp_path / "startup"
    startup.mkdir()
    original = render_template_block(key, client=client)
    (startup / "docker.sh").write_text(original)
    r = sourced(f'kit_script_with_image "{startup}" jefftud/odelia:1.9.0-rc1')
    assert r.returncode == 0, r.stderr
    out = Path(r.stdout.strip())
    assert out.parent == startup and os.access(out, os.X_OK)  # next to the kit: DIR resolves the same
    new = out.read_text().splitlines()
    old = original.splitlines()
    assert len(new) == len(old)
    changed = [(a, b) for a, b in zip(old, new) if a != b]
    assert changed == [(f"DOCKER_IMAGE={KIT_IMAGE}", "DOCKER_IMAGE=jefftud/odelia:1.9.0-rc1")]
    assert (startup / "docker.sh").read_text() == original


def test_kit_script_with_image_refuses_bad_input(tmp_path):
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "docker.sh").write_text('echo no image here\nDOCKER_IMAGE="${CLI_IMAGE:-x}"\n')
    assert sourced(f'kit_script_with_image "{startup}" jefftud/odelia:1.9.0').returncode != 0
    assert not (startup / ".docker_image_override.sh").exists()
    (startup / "docker.sh").write_text("DOCKER_IMAGE=a/b:1\n")
    assert sourced(f'kit_script_with_image "{startup}" "a/b:1; rm -rf /"').returncode != 0


def test_harness_reads_its_retry_count_from_the_conf():
    text = (DEPLOY / "run_deploy_test.sh").read_text()
    assert 'MAX_RETRIES="${MAX_RETRIES:-2}"' in text
    assert not re.search(r"^MAX_RETRIES=2\b", text, re.MULTILINE)


def test_every_training_step_failure_fails_the_attempt():
    text = (DEPLOY / "run_deploy_test.sh").read_text()
    body = text.split("_run_training_attempt() {", 1)[1].split("\n}\n", 1)[0]
    for step in ("stop_all", "start_server", 'start_clients "$model_name"', 'verify_client_model_env "$model_name"',
                 "wait_for_client_registration", 'submit_job "$job_name"'):
        assert f"{step} || return 1" in body, step


# ── launch_testbed.sh refusals ────────────────────────────────────────────


def _conf(tmp_path, body, mode=0o600):
    conf = tmp_path / "t.conf"
    conf.write_text(body + "\nPROJECT_FILE=application/provision/project_deploy_test_4site.yml\n")
    conf.chmod(mode)
    return conf


def launch(conf, *extra, env=None):
    return bash(f'"{DEPLOY}/launch_testbed.sh" --conf "{conf}" --kit-version 1.8.1-dev.260913.f7c66bb '
                + " ".join(extra), env=env)


def test_launcher_refuses_readable_plaintext_passwords(tmp_path):
    r = launch(_conf(tmp_path, "CLIENT_SITES=(A)\nA_PASS=secret", mode=0o644))
    assert r.returncode != 0 and "chmod 600" in r.stderr and "secret" not in r.stderr


def test_launcher_refuses_a_readable_password_file(tmp_path):
    pw = tmp_path / "pw"
    pw.write_text("secret\n")
    pw.chmod(0o644)
    r = launch(_conf(tmp_path, f"CLIENT_SITES=(A)\nA_PASS_FILE={pw}"))
    assert r.returncode != 0 and "readable by other users" in r.stderr


def test_launcher_refuses_an_unset_password_variable(tmp_path):
    r = launch(_conf(tmp_path, "CLIENT_SITES=(A)\nA_PASS_ENV=DL_SWARM_PASS_NOT_SET"))
    assert r.returncode != 0 and "DL_SWARM_PASS_NOT_SET" in r.stderr


def test_launcher_needs_a_kit_version(tmp_path):
    conf = _conf(tmp_path, "CLIENT_SITES=(A)")
    r = bash(f'"{DEPLOY}/launch_testbed.sh" --conf "{conf}"')
    assert r.returncode != 0 and "--kit-version" in r.stderr


@pytest.mark.skipif(shutil.which("jq") is None, reason="launch_testbed.sh needs jq (present on the test hosts and CI runners)")
def test_launcher_reports_missing_kits(tmp_path):
    r = launch(_conf(tmp_path, "CLIENT_SITES=(A)\nA_PASS_ENV=PW"), env={"PW": "x"})
    assert r.returncode != 0 and "no kits for 1.8.1-dev.260913.f7c66bb" in r.stderr


# ── The dl3 example conf ──────────────────────────────────────────────────


def test_example_conf_uses_four_distinct_sites_and_no_secrets():
    text = EXAMPLE_CONF.read_text()
    assert not re.search(r"^[A-Z0-9_]+_PASS=\S", text, re.MULTILINE), "no password in a tracked file"
    r = bash(
        f'source "{EXAMPLE_CONF}"\n'
        'for s in "${CLIENT_SITES[@]}"; do v="${s}_INSTITUTION"; p="${s}_PASS_ENV"; n="${s}_SITE_NAME";'
        ' echo "$s ${!n} ${!v} ${!p}"; done\n'
        'echo "EVAL $EVAL_SITE_NAME $PROJECT_FILE $COSMOS_PASS_ENV"'
    )
    assert r.returncode == 0, r.stderr
    rows = [line.split() for line in r.stdout.splitlines()]
    clients = [row for row in rows if row[0] != "EVAL"]
    assert [row[2] for row in clients] == ["CAM_1", "MHA_1", "UKA_1", "UMCU_1"]
    assert [row[1] for row in clients] == ["TEST_A_1", "TEST_B_1", "TEST_C_1", "TEST_D_1"]
    assert all(row[3] == "DL_SWARM_PASS" for row in clients)
    eval_row = [row for row in rows if row[0] == "EVAL"][0]
    assert eval_row[1] == "RUMC_1" and eval_row[1] not in {row[2] for row in clients}
    assert (REPO_ROOT / eval_row[2]).is_file()
    assert eval_row[3] == "DL_SWARM_PASS"
