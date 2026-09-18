"""Production startup kits must survive an image bump.

NVFlare keys its provisioning state off the project `name:` (lighter/ctx.py), and
CertBuilder only reuses the root CA and each site's private key if that state dir
already exists. So putting the build version inside the project name gives every
build a fresh PKI -- a new root CA and a new client.key for all sites -- which
silently invalidates every kit already deployed at a hospital.

That is what forced a kit re-issue on every image bump (49 ODELIA provisioning
runs in workspace/). Guard it: the production consortium projects must keep a
STABLE name, while still substituting the image tag per build.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVISION_DIR = REPO_ROOT / "application" / "provision"

PLACEHOLDER = "__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__"

# Kits from these projects are deployed to hospitals and must be long-lived.
PRODUCTION_PROJECTS = [
    "project_Odelia_allsites.yml",
    "project_DECADE_allsites.yml",
]

NAME_RE = re.compile(r"^name:\s*(.+)$", re.M)
IMAGE_RE = re.compile(r"^\s*docker_image:\s*(.+)$", re.M)


def _project(filename):
    path = PROVISION_DIR / filename
    if not path.is_file():
        pytest.skip(f"{filename} not present")
    return path, path.read_text()


@pytest.mark.parametrize("filename", PRODUCTION_PROJECTS)
def test_production_project_name_is_version_free(filename):
    """A version in the name = a new root CA every build = every deployed kit dies."""
    path, text = _project(filename)

    match = NAME_RE.search(text)
    assert match, f"{filename} has no `name:`"
    name = match.group(1).strip()

    assert PLACEHOLDER not in name, (
        f"{filename}: the build version is back in the project name ({name!r}). "
        "That regenerates the PKI on every build and invalidates every deployed kit."
    )
    assert not re.search(r"\d+\.\d+\.\d+", name), (
        f"{filename}: `name: {name}` looks version-pinned; keep it stable."
    )


@pytest.mark.parametrize("filename", PRODUCTION_PROJECTS)
def test_production_project_still_substitutes_the_image_tag(filename):
    """The image tag MUST still be substituted per build -- only the name is stable."""
    path, text = _project(filename)

    match = IMAGE_RE.search(text)
    assert match, f"{filename} has no `docker_image:`"
    assert PLACEHOLDER in match.group(1), (
        f"{filename}: docker_image lost its version placeholder, so kits would ship "
        "pointing at a stale image tag."
    )


def test_build_script_does_not_mutate_the_tracked_project_yml():
    """It used to sed the version in and back out again, corrupting the file if the
    version string appeared anywhere else. It must build from a temp copy."""
    script = (REPO_ROOT / "scripts" / "build" / "_buildStartupKits.sh").read_text()

    assert "BUILD_YML=" in script, "build script must provision from a temp copy"
    assert 'sed -i' not in script, (
        "build script still edits the tracked project yml in place"
    )
    assert "$BUILD_YML" in script, "the temp copy must be the file passed to provisioning"
