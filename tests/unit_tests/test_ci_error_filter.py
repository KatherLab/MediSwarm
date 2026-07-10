"""The integration-test error filter must ignore NVFlare's shutdown audit race
but still catch every real error (#434).

The old assertion was `! grep -qi "error"`, so any line containing the substring
"error" failed the build -- including a harmless traceback NVFlare emits after
training has already completed. The same commit then passed or failed on timing.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FILTER = REPO_ROOT / "scripts" / "ci" / "_error_filter.sh"

# Verbatim from the failing run of PR #400 (validate-swarm), after training finished.
BENIGN_SHUTDOWN_NOISE = """\
Epoch 9: 100%|##########| 10/10 [00:01<00:00,  8.11it/s]
2026-07-10 13:27:46,507 - root - ERROR - Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/site-packages/nvflare/fuel/hci/server/reg.py", line 105, in process_command
    self._do_command(conn, command)
  File "/opt/conda/lib/python3.10/site-packages/nvflare/fuel/hci/server/audit.py", line 40, in pre_command
    event_id = self.auditor.add_event(
  File "/opt/conda/lib/python3.10/site-packages/nvflare/fuel/sec/audit.py", line 61, in add_event
    self.audit_file.write(line + "\\n")
ValueError: I/O operation on closed file.
Starting shutdown of NVFLARE
"""


def _has_real_error(output: str) -> bool:
    """Exit status of has_real_error() from the shell helper."""
    script = f'source "{FILTER}"; has_real_error "$1"'
    result = subprocess.run(["bash", "-c", script, "_", output], capture_output=True)
    return result.returncode == 0


def test_filter_exists():
    assert FILTER.is_file()


def test_shutdown_audit_race_is_not_a_real_error():
    """The exact output that turned PR #400 red must be treated as clean."""
    assert not _has_real_error(BENIGN_SHUTDOWN_NOISE)


def test_clean_output_has_no_error():
    assert not _has_real_error("Epoch 9: 100%\nTraining completed successfully\n")


@pytest.mark.parametrize("line", [
    "RuntimeError: shape '[1, 3]' is invalid for input of size 4",
    "CUDA error: out of memory",
    "ERROR - failed to connect to the FL server",
    "docker: Error response from daemon: manifest unknown",
    "ValueError: too many degenerate inputs",
])
def test_genuine_errors_are_still_fatal(line):
    """The filter must narrow the check, never disable it."""
    assert _has_real_error(f"Epoch 9: 100%\n{line}\n"), f"missed a real error: {line}"


def test_real_error_alongside_benign_noise_is_still_caught():
    """A genuine failure must not hide behind the whitelisted shutdown traceback."""
    assert _has_real_error(BENIGN_SHUTDOWN_NOISE + "RuntimeError: CUDA out of memory\n")
