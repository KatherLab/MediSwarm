"""The partner handbook must only tell sites to run things that exist.

Partner instructions previously drifted from the code -- e.g. three docs promised a
pre-run check block that one kit never had, and two sites were told to set a variable
in a way that guarantees the run aborts. A handbook that lies is worse than no
handbook, because a site cannot tell which half to trust.

So: every docker.sh flag and every script the handbook names must actually exist.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDBOOK = REPO_ROOT / "docs" / "consortium" / "PARTNER_HANDBOOK.md"
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"

yaml = pytest.importorskip("yaml")

# flags belonging to other tools the handbook legitimately mentions
NOT_DOCKER_SH = {"--install-timer", "--iter", "--data_dir"}


@pytest.fixture(scope="module")
def handbook():
    if not HANDBOOK.is_file():
        pytest.skip("handbook not present")
    return HANDBOOK.read_text()


@pytest.fixture(scope="module")
def docker_sh():
    return yaml.safe_load(TEMPLATE.read_text())["docker_cln_sh"]


def test_every_docker_sh_flag_in_the_handbook_exists(handbook, docker_sh):
    used = set(re.findall(r"docker\.sh[^\n`]*", handbook))
    flags = set()
    for line in used:
        flags |= set(re.findall(r"--[A-Za-z_]{3,}", line))
    flags -= NOT_DOCKER_SH

    missing = sorted(f for f in flags if f"{f})" not in docker_sh)
    assert not missing, f"handbook tells sites to run flags docker.sh does not have: {missing}"


def test_every_client_script_in_the_handbook_exists(handbook):
    scripts = set(re.findall(r"scripts/client_node_setup/\S+?\.sh", handbook))
    missing = sorted(s for s in scripts if not (REPO_ROOT / s).is_file())
    assert not missing, f"handbook references scripts that do not exist: {missing}"


def test_handbook_does_not_tell_sites_to_set_the_cache_to_a_host_path(handbook):
    """The mistake that has already cost two sites a failed run."""
    assert not re.search(r'ODELIA_PREPROCESS_CACHE_DIR\s*=\s*"?\$SCRATCHDIR', handbook), (
        "ODELIA_PREPROCESS_CACHE_DIR is a CONTAINER path; interpolating the host "
        "$SCRATCHDIR lands it under the read-only /data mount and aborts the run"
    )


def test_handbook_does_not_advise_deleting_data_on_a_permission_error(handbook):
    """A site deleted six valid cases because the old wording said 'exclude'."""
    assert "Do not delete or exclude the data" in handbook or "do NOT exclude" in handbook
