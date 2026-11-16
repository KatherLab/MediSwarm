"""The release channel in a kit's image.conf must follow that project's own image.

We provision two consortia from one build script: ODELIA (jefftud/odelia) and DECADE
(jefftud/decade). _injectLiveSyncIntoStartupKits.sh used to hardcode
`MEDISWARM_IMAGE=jefftud/odelia:current` into every kit it touched, which meant a DECADE
rebuild shipped all four DECADE sites a channel pointing at the ODELIA image. It was
inert only because the STAMP docker.sh did not yet source image.conf -- the moment it
did, every DECADE node would pull the wrong image.

These tests run the real injector against a throwaway workspace and assert the rendered
image.conf follows the project YAML's docker_image line.
"""

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INJECTOR = REPO_ROOT / "scripts" / "build" / "_injectLiveSyncIntoStartupKits.sh"


def _run_injector(tmp_path, project_name, docker_image, env=None, check=True):
    """Provision a minimal fake kit tree, run the injector, return image.conf text."""
    workspace = REPO_ROOT / "workspace" / project_name
    startup = workspace / "prod_00" / "SITE_1" / "startup"
    startup.mkdir(parents=True, exist_ok=True)
    # The injector only treats a directory as a kit if it holds a docker.sh.
    (startup / "docker.sh").write_text("#!/usr/bin/env bash\n")

    project_yml = tmp_path / "project.yml"
    project_yml.write_text(
        textwrap.dedent(
            f"""\
            name: {project_name}
            participants:
              - name: SITE_1
                type: client
                docker_image: {docker_image}
            """
        )
    )

    run_env = {k: v for k, v in os.environ.items() if not k.startswith("MEDISWARM_") and k != "VERSION"}
    run_env["MEDISWARM_UPLOAD_KEYS"] = "0"
    run_env.update(env or {})
    try:
        result = subprocess.run(
            ["bash", str(INJECTOR), str(project_yml)],
            cwd=REPO_ROOT,
            check=check,
            capture_output=True,
            text=True,
            env=run_env,
        )
        if not check and result.returncode != 0:
            return result
        return (startup / "image.conf").read_text()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.parametrize(
    "project_name,docker_image,expected_repo,foreign_repo",
    [
        ("test_channel_odelia", "jefftud/odelia:1.6.0", "jefftud/odelia", "jefftud/decade"),
        ("test_channel_decade", "jefftud/decade:1.6.0", "jefftud/decade", "jefftud/odelia"),
    ],
)
def test_channel_follows_project_image(
    tmp_path, project_name, docker_image, expected_repo, foreign_repo
):
    conf = _run_injector(tmp_path, project_name, docker_image)

    assert f"MEDISWARM_IMAGE={expected_repo}:current" in conf
    # The regression that motivated this file: the other consortium's repo must not
    # appear anywhere in the rendered file, not even in the "pin a version" example.
    assert foreign_repo not in conf


def test_unversioned_placeholder_image_still_yields_a_channel(tmp_path):
    """Production YAMLs carry a substitution placeholder as the tag, not a real version."""
    conf = _run_injector(
        tmp_path,
        "test_channel_placeholder",
        "jefftud/decade:__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__",
    )
    assert "MEDISWARM_IMAGE=jefftud/decade:current" in conf
    assert "__REPLACED_BY" not in conf


def test_real_project_yamls_map_to_their_own_consortium(tmp_path):
    """Guard the two production rosters themselves, not just synthetic inputs."""
    for yml_name, expected in [
        ("project_Odelia_allsites.yml", "jefftud/odelia:current"),
        ("project_DECADE_allsites.yml", "jefftud/decade:current"),
    ]:
        yml = REPO_ROOT / "application" / "provision" / yml_name
        line = next(
            l for l in yml.read_text().splitlines() if l.strip().startswith("docker_image:")
        )
        value = line.split("docker_image:", 1)[1].strip()
        repo = value.rsplit(":", 1)[0] if "/" not in value.rsplit(":", 1)[-1] else value
        assert f"{repo}:current" == expected, f"{yml_name} would follow the wrong channel"


def test_example_tag_follows_the_kit_version(tmp_path):
    """A DECADE kit's 'pin a version' example named ODELIA's version, a tag jefftud/decade never had."""
    conf = _run_injector(tmp_path, "test_channel_example_tag", "jefftud/decade:1.7.1", env={"VERSION": "1.7.1"})
    assert "MEDISWARM_IMAGE=jefftud/decade:1.7.1" in conf  # the commented example
    assert "MEDISWARM_IMAGE=jefftud/decade:current" in conf.splitlines()


def test_a_pinned_kit_image_replaces_the_channel(tmp_path):
    conf = _run_injector(tmp_path, "test_channel_pinned", "jefftud/decade:1.7.1",
                         env={"VERSION": "1.7.1", "MEDISWARM_KIT_IMAGE": "jefftud/decade:1.7.1"})
    active = [line for line in conf.splitlines() if line.startswith("MEDISWARM_IMAGE=")]
    assert active == ["MEDISWARM_IMAGE=jefftud/decade:1.7.1"]
    assert "pinned to one image" in conf


def test_a_pin_to_another_repository_is_refused(tmp_path):
    result = _run_injector(tmp_path, "test_channel_foreign_pin", "jefftud/decade:1.7.1",
                           env={"MEDISWARM_KIT_IMAGE": "jefftud/odelia:1.9.3"}, check=False)
    assert result.returncode != 0 and "is not an image of jefftud/decade" in result.stderr
