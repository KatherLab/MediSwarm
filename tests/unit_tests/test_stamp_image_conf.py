"""The STAMP/DECADE kit must follow a release channel, with ODELIA's precedence.

DECADE kits previously pinned whatever tag they were provisioned with: every software
fix meant re-issuing four startup kits. ODELIA solved this with startup/image.conf plus
a documented precedence (--image > MEDISWARM_IMAGE env > image.conf > provisioned
default); the STAMP template had none of it.

These tests execute the rendered resolution logic rather than grepping for it, so a
future edit that reorders the chain (e.g. sourcing image.conf after capturing the env)
fails here instead of silently changing which image four hospitals run.
"""

import re
import subprocess
import textwrap

import pytest

yaml = pytest.importorskip("yaml")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "docker_config" / "master_template_STAMP.yml"

PROVISIONED = "jefftud/decade:1.5.0-provisioned"


@pytest.fixture(scope="module")
def resolution_snippet():
    """The image-resolution block from the client docker.sh, ready to execute."""
    script = yaml.safe_load(TEMPLATE.read_text())["docker_cln_sh"]
    script = script.replace("{~~docker_image~~}", PROVISIONED)

    start = script.index("DOCKER_IMAGE=" + PROVISIONED)
    end = script.index('DOCKER_IMAGE="${CLI_IMAGE:-${MEDISWARM_IMAGE:-$DOCKER_IMAGE}}"')
    snippet = script[start : end + len('DOCKER_IMAGE="${CLI_IMAGE:-${MEDISWARM_IMAGE:-$DOCKER_IMAGE}}"')]
    return textwrap.dedent(snippet)


def _resolve(snippet, tmp_path, image_conf=None, env_image=None, cli_image=None):
    """Run the block with a given combination of overrides; return the chosen image."""
    if image_conf is not None:
        (tmp_path / "image.conf").write_text(f"MEDISWARM_IMAGE={image_conf}\n")

    prelude = f'DIR="{tmp_path}"\n'
    prelude += f'CLI_IMAGE="{cli_image}"\n' if cli_image else 'CLI_IMAGE=""\n'
    if env_image:
        prelude += f'export MEDISWARM_IMAGE="{env_image}"\n'

    result = subprocess.run(
        ["bash", "-c", prelude + snippet + '\necho "RESOLVED=$DOCKER_IMAGE"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return re.search(r"RESOLVED=(\S+)", result.stdout).group(1)


def test_defaults_to_provisioned_tag(resolution_snippet, tmp_path):
    assert _resolve(resolution_snippet, tmp_path) == PROVISIONED


def test_image_conf_channel_beats_the_provisioned_default(resolution_snippet, tmp_path):
    """The whole point: re-tagging :current moves a site with no kit re-issue."""
    got = _resolve(resolution_snippet, tmp_path, image_conf="jefftud/decade:current")
    assert got == "jefftud/decade:current"


def test_env_beats_image_conf(resolution_snippet, tmp_path):
    """Regression for the #449 follow-up: sourcing image.conf must not clobber the env."""
    got = _resolve(
        resolution_snippet,
        tmp_path,
        image_conf="jefftud/decade:current",
        env_image="jefftud/decade:pinned-by-env",
    )
    assert got == "jefftud/decade:pinned-by-env"


def test_cli_flag_wins_over_everything(resolution_snippet, tmp_path):
    got = _resolve(
        resolution_snippet,
        tmp_path,
        image_conf="jefftud/decade:current",
        env_image="jefftud/decade:pinned-by-env",
        cli_image="jefftud/decade:one-off",
    )
    assert got == "jefftud/decade:one-off"


def test_image_flag_is_accepted_by_the_arg_parser():
    """--image must be parseable, or the one-off override is unreachable."""
    script = yaml.safe_load(TEMPLATE.read_text())["docker_cln_sh"]
    assert "--image)" in script, "STAMP docker.sh does not accept --image"
    # It takes a value, so it must be guarded like the other value-taking flags (#443):
    # a bare `--image` with an empty variable would otherwise swallow the next flag.
    assert re.search(r"--image\)\s+_need_value", script)
