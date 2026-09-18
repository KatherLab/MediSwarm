"""docker.sh must expose --fold for the local modes (#411).

Only `--start_client` gets its fold from the submitted job (the admin patches a
FOLD=N prefix onto the launcher, so all sites necessarily train the same fold).
`--local_training` / `--preflight_check` / `--dummy_training` bypass NVFlare
entirely and run main.py directly, so for them FOLD can ONLY arrive through the
container environment. Without this flag those modes could only ever train fold 0
-- which meant a site had no way to verify it actually HAS fold N before a run.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"


@pytest.fixture(scope="module")
def docker_sh():
    """The client docker.sh exactly as the lighter generates it into a startup kit."""
    template = yaml.safe_load(TEMPLATE.read_text())
    script = template["docker_cln_sh"]
    return re.sub(r"\{~~[a-z_]+~~\}", "TESTSITE", script)


def test_fold_flag_is_accepted(docker_sh):
    assert "--fold)" in docker_sh, "docker.sh does not accept --fold"


def test_fold_is_forwarded_into_the_container(docker_sh):
    assert "--env FOLD=" in docker_sh, "FOLD is never passed to the container"


def test_fold_follows_the_cli_over_env_over_default_precedence(docker_sh):
    """Same contract as --num_epochs: CLI > env > default, so it survives sudo."""
    assert "--env FOLD=${CLI_FOLD:-${FOLD:-0}}" in docker_sh


def test_fold_defaults_to_zero(docker_sh):
    """Fold 0 is the behavior from before the fold was configurable."""
    match = re.search(r"--env FOLD=\$\{CLI_FOLD:-\$\{FOLD:-(\d+)\}\}", docker_sh)
    assert match and match.group(1) == "0"


def test_fold_is_documented_as_a_local_mode_flag(docker_sh):
    """Sites must not think they can pick a fold for a swarm run."""
    assert "--fold <n>" in docker_sh, "--fold missing from the usage block"
    usage = docker_sh[docker_sh.index("--fold <n>"):]
    assert "ignored" in usage.lower() and "start_client" in usage, (
        "usage must say the swarm fold comes from the job, not this flag"
    )


def test_generated_script_is_valid_bash(docker_sh):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(docker_sh)
        path = handle.name
    result = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert result.returncode == 0, f"generated docker.sh is not valid bash:\n{result.stderr}"
