"""The upload guard skips oversized files without failing the sites' live_sync upload.

Runs the real guard behind a fake ``ssh`` that behaves like sshd's forced command
(the remote command arrives in SSH_ORIGINAL_COMMAND), with the upload root moved
into a temporary directory and the cap lowered so the test files stay small.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUARD = REPO / "server_tools" / "mediswarm_upload_guard.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None or not Path("/usr/bin/rrsync").exists(),
    reason="needs rsync and /usr/bin/rrsync, as on the monitor host",
)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def upload_env(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    guard = tmp_path / "guard.sh"
    guard.write_text(GUARD.read_text().replace("/srv/mediswarm/live", str(live)))
    _make_executable(guard)
    fake_ssh = tmp_path / "fakessh"
    fake_ssh.write_text(
        "#!/bin/bash\n"
        "shift  # host\n"
        f'SSH_ORIGINAL_COMMAND="$*" SSH_CLIENT="127.0.0.1 1 2" MEDISWARM_UPLOAD_MAX_SIZE=4M exec {guard}\n'
    )
    _make_executable(fake_ssh)
    src = tmp_path / "src"
    src.mkdir()
    (src / "small.ckpt").write_bytes(os.urandom(2_000_000))
    (src / "big.ckpt").write_bytes(os.urandom(6_000_000))
    (src / "log.txt").write_text("log\n")
    return live, fake_ssh, src


def _rsync(fake_ssh, *args):
    # Same flags as kit_live_sync/live_sync.sh in the 1.8.1 kits.
    return subprocess.run(["rsync", "-az", "--partial", "--mkpath", "-e", str(fake_ssh), *args],
                          capture_output=True, text=True)


def test_directory_upload_skips_only_oversized_files(upload_env):
    live, fake_ssh, src = upload_env
    result = _rsync(fake_ssh, f"{src}/", f"host:{live}/SITE_1/run/")
    assert result.returncode == 0, result.stderr
    uploaded = sorted(p.name for p in (live / "SITE_1" / "run").iterdir())
    assert uploaded == ["log.txt", "small.ckpt"]


def test_single_oversized_file_is_skipped_without_error(upload_env):
    live, fake_ssh, src = upload_env
    result = _rsync(fake_ssh, str(src / "big.ckpt"), f"host:{live}/SITE_1/big.ckpt")
    assert result.returncode == 0, result.stderr
    assert not (live / "SITE_1" / "big.ckpt").exists()
