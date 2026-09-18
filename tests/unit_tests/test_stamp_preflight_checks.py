"""The STAMP/DECADE kit must run the same pre-run host checks the docs promise (#451).

The ODELIA kit's docker.sh has `_preflight_host_checks()` (GPU-usable-in-container,
cgroup driver, server reachability, stale daemon_pid.fl). The STAMP/DECADE kit shipped
NONE of them, yet three docs told sites `docker.sh --preflight_check` runs checks -- so
the four DECADE sites got a preflight that checked nothing. This asserts the checks are
now present in the STAMP template, adapted for STAMP (no ODELIA-only preprocess-cache
check; the STAMP container name).
"""

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_TEMPLATE = REPO_ROOT / "docker_config" / "master_template_STAMP.yml"


@pytest.fixture(scope="module")
def stamp_sh():
    template = yaml.safe_load(STAMP_TEMPLATE.read_text())
    return re.sub(r"\{~~[a-z_]+~~\}", "TESTSITE", template["docker_cln_sh"])


def test_stamp_kit_has_preflight_host_checks(stamp_sh):
    assert "_preflight_host_checks()" in stamp_sh, "STAMP kit still ships no pre-run checks"


def test_preflight_runs_for_the_gpu_using_modes(stamp_sh):
    assert re.search(
        r'if \[ -n "\$DUMMY_TRAINING" \] \|\| \[ -n "\$PREFLIGHT_CHECK" \] \|\| \[ -n "\$START_CLIENT" \]; then\s*\n\s*_preflight_host_checks',
        stamp_sh,
    ), "preflight is defined but never called before the execution modes"


def test_gpu_check_is_present_and_aborts_on_failure(stamp_sh):
    """The load-bearing check: GPU usable in the container, and a [FAIL] aborts."""
    assert "nvidia-smi -L" in stamp_sh and "GPU usable inside the container" in stamp_sh
    assert "[ABORT] Pre-run checks failed" in stamp_sh
    assert re.search(r'if \[ "\$fail" = "1" \]; then\s*\n\s*echo "\[ABORT\]', stamp_sh)


def test_prints_a_pre_run_checks_block(stamp_sh):
    """Matches the docs' promise of a 'Pre-run checks' block with [PASS]/[FAIL]."""
    assert "Pre-run checks" in stamp_sh
    assert "[PASS]" in stamp_sh and "[FAIL]" in stamp_sh


def test_uses_the_stamp_container_name_not_odelia(stamp_sh):
    assert "stamp_swarm_client" in stamp_sh
    assert "odelia_swarm_client" not in stamp_sh, "copied the ODELIA container name verbatim"


def test_does_not_port_the_odelia_only_preprocess_cache_check(stamp_sh):
    """STAMP reads pre-extracted H5 features -- it has no ODELIA_PREPROCESS_CACHE_DIR."""
    assert "ODELIA_PREPROCESS_CACHE_DIR" not in stamp_sh


def test_generated_stamp_script_is_valid_bash(stamp_sh):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(stamp_sh)
        path = handle.name
    result = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
