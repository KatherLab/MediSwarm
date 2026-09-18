"""Acquiring the host lock must not disturb the caller's stderr.

`exec {FD}>file` with a redirection attached applies that redirection to the CURRENT
SHELL, permanently. An `exec {FD}>"$LOCK" 2>/dev/null` in acquire_host_lock therefore
sent the *sourcing script's* stderr to /dev/null for the rest of its run.

Both deploy orchestrators and the image build source this file, and their progress and
error reporting (info/ok/warn/err in deploy_common.sh) all go to stderr. So a failing
2-node deploy test printed nothing at all -- the run just looked silent, which is worse
than a crash. Caught while chasing a DECADE validation failure that was partly this.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_SH = REPO_ROOT / "scripts" / "ci" / "host_gpu_lock.sh"


def _run(script: str, tmp_path):
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "MEDISWARM_HOST_LOCK": str(tmp_path / "t.lock")},
    )


def test_stderr_survives_acquiring_the_lock(tmp_path):
    r = _run(
        f'. "{LOCK_SH}"\n'
        'acquire_host_lock "unit test" 5 >/dev/null\n'
        'echo "STDERR-AFTER-LOCK" >&2\n',
        tmp_path,
    )
    assert "STDERR-AFTER-LOCK" in r.stderr, (
        "acquire_host_lock swallowed the caller's stderr -- the deploy orchestrators "
        f"would report nothing. stderr was: {r.stderr!r}"
    )


def test_stderr_survives_the_wait_helper(tmp_path):
    r = _run(
        f'. "{LOCK_SH}"\n'
        "wait_for_host_lock 5 >/dev/null\n"
        'echo "STDERR-AFTER-WAIT" >&2\n',
        tmp_path,
    )
    assert "STDERR-AFTER-WAIT" in r.stderr


def test_no_redirection_is_attached_to_the_exec(tmp_path):
    """Guard the specific construct, so the bug cannot be reintroduced verbatim."""
    for line in LOCK_SH.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("exec {") and not stripped.startswith("#"):
            assert "2>" not in stripped and "&>" not in stripped, (
                f"redirection attached to a bare `exec`: {stripped!r} -- this "
                "permanently rebinds the sourcing shell's stderr."
            )
