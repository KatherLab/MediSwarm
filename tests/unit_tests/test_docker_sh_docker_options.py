"""docker.sh must let a site pass extra `docker run` options without editing it (#218).

Partners reported needing host-specific `docker run` options (e.g. --cpus /
--cpuset-cpus to pin CPUs). Hand-editing the generated docker.sh is error-prone and
gets lost on the next kit, so the script takes them as a `--docker_options` flag (or a
persistent MEDISWARM_DOCKER_OPTIONS env / image.conf line) and appends them to the
`docker run` invocation. Appended LAST, so a site option can override a built-in.
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


def test_docker_options_flag_is_accepted(docker_sh):
    assert "--docker_options)" in docker_sh, "docker.sh does not accept --docker_options"


def test_extra_options_are_appended_to_the_docker_run_options(docker_sh):
    """They must reach `docker run`, appended last so they can override a built-in."""
    assert "${EXTRA_DOCKER_OPTIONS}" in docker_sh
    assert '${DOCKER_OPTIONS_B} ${EXTRA_DOCKER_OPTIONS}"' in docker_sh, (
        "extra options must be appended last so a site can override a built-in option"
    )


def test_precedence_is_cli_over_env(docker_sh):
    assert 'EXTRA_DOCKER_OPTIONS="${CLI_DOCKER_OPTIONS:-${MEDISWARM_DOCKER_OPTIONS:-}}"' in docker_sh


def test_env_is_captured_before_image_conf_is_sourced(docker_sh):
    """A one-off MEDISWARM_DOCKER_OPTIONS must beat image.conf, same as the image var."""
    cap = docker_sh.index('ENV_DOCKER_OPTIONS="${MEDISWARM_DOCKER_OPTIONS:-}"')
    srcconf = docker_sh.index('. "$DIR/image.conf"')
    restore = docker_sh.index('MEDISWARM_DOCKER_OPTIONS="${ENV_DOCKER_OPTIONS:-')
    assert cap < srcconf, "env must be captured BEFORE image.conf is sourced"
    assert srcconf < restore, "the captured env must be re-applied AFTER sourcing"


def test_no_extra_options_by_default(docker_sh):
    """With nothing passed, EXTRA_DOCKER_OPTIONS is empty -- behavior is unchanged."""
    # neither CLI nor env set -> both `:-` fallbacks yield the empty string
    assert "CLI_DOCKER_OPTIONS" in docker_sh
    assert '--docker_options)      CLI_DOCKER_OPTIONS="$2"' in docker_sh


def test_generated_script_is_valid_bash(docker_sh):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(docker_sh)
        path = handle.name
    result = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert result.returncode == 0, f"generated docker.sh is not valid bash:\n{result.stderr}"
