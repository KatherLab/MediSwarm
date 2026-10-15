"""1.9.0 live-sync upload keys: the guard's areas (PS-1), the key registration and the kit build.

The guard runs behind a fake ``ssh`` that behaves like sshd's forced command (the remote command
arrives in SSH_ORIGINAL_COMMAND, the key line's arguments are passed through), with the upload root
moved into a temporary directory. The uploads use the sites' exact rsync flags.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUARD = REPO / "server_tools" / "mediswarm_upload_guard.sh"
REGISTER = REPO / "scripts" / "admin" / "register_upload_keys.sh"
INJECT = REPO / "scripts" / "build" / "_injectLiveSyncIntoStartupKits.sh"

needs_rrsync = pytest.mark.skipif(
    shutil.which("rsync") is None or not Path("/usr/bin/rrsync").exists(),
    reason="needs rsync and /usr/bin/rrsync, as on the monitor host",
)


def _executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def area(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    guard = _executable(tmp_path / "guard.sh", GUARD.read_text().replace("/srv/mediswarm/live", str(live)))
    src = tmp_path / "src"
    src.mkdir()
    (src / "log.txt").write_text("log\n")

    def ssh_for(site: str = "") -> Path:
        # The key line's command= arguments (the site) reach the guard as $1, as sshd passes them.
        return _executable(tmp_path / f"fakessh_{site or 'legacy'}", (
            "#!/bin/bash\nshift  # host\n"
            f'SSH_ORIGINAL_COMMAND="$*" SSH_CLIENT="127.0.0.1 1 2" exec {guard} {site}\n'))

    def upload(dest: str, site: str = "") -> int:
        return subprocess.run(["rsync", "-az", "--partial", "--mkpath", "-e", str(ssh_for(site)),
                               str(src / "log.txt"), f"host:{dest}"], capture_output=True, text=True).returncode

    def remote(cmd: str, site: str = "") -> int:
        return subprocess.run([str(ssh_for(site)), "host", cmd], capture_output=True, text=True).returncode

    return live, upload, remote, tmp_path


@needs_rrsync
def test_legacy_key_uploads_anywhere_in_the_tree_but_not_outside(area):
    live, upload, _, tmp = area
    assert upload(f"{live}/SITE_A/swarm/job1/log.txt") == 0
    assert (live / "SITE_A" / "swarm" / "job1" / "log.txt").exists()
    home = tmp / "home" / ".ssh"
    home.mkdir(parents=True)
    assert upload(f"{home}/authorized_keys") != 0, "PS-1: no key may write outside the upload tree"
    assert not (home / "authorized_keys").exists()
    assert upload(f"{live}/../escape.txt") != 0
    assert not (tmp / "escape.txt").exists()


@needs_rrsync
def test_site_key_is_confined_to_its_own_folder(area):
    live, upload, remote, _ = area
    assert upload(f"{live}/SITE_A/swarm/job1/log.txt", site="SITE_A") == 0
    assert upload(f"{live}/SITE_B/swarm/job1/log.txt", site="SITE_A") != 0
    assert not (live / "SITE_B").exists()
    assert remote(f"mkdir -p '{live}/SITE_A/swarm/job2'", site="SITE_A") == 0
    assert remote(f"mkdir -p '{live}/SITE_B/x'", site="SITE_A") != 0
    assert remote(f"rm -rf '{live}/SITE_A'", site="SITE_A") != 0, "a key never removes its whole area"
    assert remote(f"rm -rf '{live}/SITE_A/swarm/job2'", site="SITE_A") == 0
    assert remote("echo ok", site="SITE_A") == 0


@needs_rrsync
def test_a_symlink_inside_the_area_cannot_lead_out(area):
    live, upload, _, tmp = area
    outside = tmp / "outside"
    outside.mkdir()
    (live / "SITE_A").mkdir()
    (live / "SITE_A" / "link").symlink_to(outside)
    assert upload(f"{live}/SITE_A/link/log.txt", site="SITE_A") != 0
    assert not (outside / "log.txt").exists()


def test_malformed_key_line_is_refused(area):
    _, _, remote, tmp = area
    guard = tmp / "guard.sh"
    run = subprocess.run([str(guard), "SITE A"], env={**os.environ, "SSH_ORIGINAL_COMMAND": "echo ok"},
                         capture_output=True, text=True)
    assert run.returncode != 0
    run = subprocess.run([str(guard), "SITE_A", "extra"], env={**os.environ, "SSH_ORIGINAL_COMMAND": "echo ok"},
                         capture_output=True, text=True)
    assert run.returncode != 0


# ── register_upload_keys.sh ───────────────────────────────────────────────────────────


def _key_line(site, blob="AAAAC3NzaC1lZDI1NTE5AAAAIA" + "x" * 43, version="1.9.0"):
    return f'restrict,command="/etc/ssh/mediswarm_upload_guard.sh {site}" ssh-ed25519 {blob} {site}@kit-{version}\n'


def _register(tmp_path, fragment_text, *args):
    fragment = tmp_path / "upload_authorized_keys.txt"
    fragment.write_text(fragment_text)
    ak = tmp_path / "authorized_keys"
    if not ak.exists():
        ak.write_text('restrict,command="/etc/ssh/mediswarm_upload_guard.sh" ssh-ed25519 AAAAOLD OLD_SITE\n')
    owner = f"{os.getuid()}:{os.getgid()}"
    run = subprocess.run(["bash", str(REGISTER), str(fragment), "--authorized-keys", str(ak), "--owner", owner, *args],
                         capture_output=True, text=True)
    return run, ak


def test_register_adds_lines_idempotently_and_keeps_the_rest(tmp_path):
    run, ak = _register(tmp_path, _key_line("SITE_A") + _key_line("SITE_B", blob="AAAAC3NzaC1lZDI1NTE5AAAAIB" + "y" * 43))
    assert run.returncode == 0, run.stdout + run.stderr
    text = ak.read_text()
    assert "OLD_SITE" in text and "SITE_A@kit-1.9.0" in text and "SITE_B@kit-1.9.0" in text
    assert oct(ak.stat().st_mode & 0o777) == "0o640"
    again, _ = _register(tmp_path, _key_line("SITE_A") + _key_line("SITE_B", blob="AAAAC3NzaC1lZDI1NTE5AAAAIB" + "y" * 43))
    assert again.returncode == 0
    assert ak.read_text() == text, "a rerun changes nothing"


def test_register_refuses_a_line_that_does_not_force_the_guard(tmp_path):
    run, ak = _register(tmp_path, "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIA" + "z" * 43 + " shell-key\n")
    assert run.returncode != 0
    assert "shell-key" not in ak.read_text()


def test_register_dry_run_and_retire(tmp_path):
    run, ak = _register(tmp_path, _key_line("SITE_A"), "--dry-run")
    assert run.returncode == 0 and "SITE_A@kit" not in ak.read_text()
    _register(tmp_path, _key_line("SITE_A", version="1.8.9"))
    run, ak = _register(tmp_path, _key_line("SITE_A", blob="AAAAC3NzaC1lZDI1NTE5AAAAIC" + "w" * 43),
                        "--retire-comment-prefix", "SITE_A@kit-")
    assert run.returncode == 0, run.stdout + run.stderr
    text = ak.read_text()
    assert "SITE_A@kit-1.8.9" not in text and "SITE_A@kit-1.9.0" in text and "OLD_SITE" in text


# ── kit build: one key per client kit, the fragment outside the kits ───────────────────


def test_kit_build_gives_each_client_kit_a_key_and_writes_the_fragment(tmp_path):
    pytest.importorskip("cryptography")
    repo = tmp_path / "repo"
    for rel in ("kit_live_sync", "kit_admin_tools", "scripts/admin/patch_warm_start_job.py", "odelia_image.version",
                "scripts/build/_injectLiveSyncIntoStartupKits.sh"):
        src = REPO / rel
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    (repo / "proj.yml").write_text("name: testproj\nbuilders:\n  - args:\n      docker_image: jefftud/odelia:1.9.0\n")
    prod = repo / "workspace" / "testproj" / "prod_00"
    for kit, marker in (("SITE_A", "fed_client.json"), ("SITE_B", "fed_client.json"), ("server.example", "fed_server.json"),
                        ("admin@example.org", "fl_admin.sh")):
        startup = prod / kit / "startup"
        startup.mkdir(parents=True)
        (startup / marker).write_text("{}\n")
        (startup / "docker.sh").write_text("#!/bin/bash\n")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    host_key = Ed25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()
    env = {**os.environ, "MEDISWARM_UPLOAD_KEYS": "1", "MEDISWARM_UPLOAD_HOST_KEY": host_key, "VERSION": "1.9.0"}
    run = subprocess.run(["bash", str(repo / "scripts/build/_injectLiveSyncIntoStartupKits.sh"), "proj.yml"],
                         cwd=repo, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    for site in ("SITE_A", "SITE_B"):
        startup = prod / site / "startup"
        assert oct((startup / "upload_key").stat().st_mode & 0o777) == "0o600"
        assert host_key.split()[1] in (startup / "upload_known_hosts").read_text()
        assert "# >>> kit upload key >>>" in (startup / "sync.conf").read_text()
    for kit in ("server.example", "admin@example.org"):
        assert not (prod / kit / "startup" / "upload_key").exists(), "server and admin kits upload nothing"
    fragment = (prod / "upload_authorized_keys.txt").read_text()
    lines = [line for line in fragment.splitlines() if line and not line.startswith("#")]
    assert len(lines) == 2
    assert lines[0].startswith('restrict,command="/etc/ssh/mediswarm_upload_guard.sh SITE_A" ssh-ed25519 ')
    assert lines[0].endswith("SITE_A@kit-1.9.0")
    assert "PRIVATE" not in fragment
