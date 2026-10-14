"""Rebuilding startup kits must keep the consortium's root CA, and losing it must stop the build.

Every kit trusts exactly one root CA (startup/rootCA.pem), and the coordinator accepts only
participants whose certificate that CA signed. NVFlare stores the CA, and every participant's
certificate and private key, in <workspace>/<project>/state/cert.json and reuses them when the
file exists. That is what lets a 1.9.0 kit replace a 1.8.1 kit at a site without touching the
running coordinator. Without the file NVFlare silently mints a new CA, and every kit from that
build is rejected (ClientConnectorCertificateError, F9 in docs/SWARM_FAILURE_MODES.md).

These tests provision a real TEST project from application/provision/ with the NVFlare fork,
always into a temporary workspace (never workspace/odelia_allsites), and run the kit archiver
on the result.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("nvflare.lighter")

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_PROJECT = REPO_ROOT / "application" / "provision" / "project_deploy_test_2site.yml"
ARCHIVER = REPO_ROOT / "scripts" / "build" / "_generateStartupKitArchives.sh"
PLACEHOLDER = "__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__"
VERSION = "0.0.0-catest"
PROJECT_NAME = f"odelia_deploy_test_{VERSION}_model_test"
SITES = ["TEST_A_1", "TEST_B_1"]
SERVER = "dl3.tud.de"
ADMIN = "jiefu.zhu@tu-dresden.de"

needs_zip = pytest.mark.skipif(shutil.which("zip") is None, reason="the archiver needs the zip tool")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provision(base: Path) -> Path:
    """Provision the test project into base/workspace like _buildStartupKits.sh does; return the new prod dir."""
    yml = base / "project.build.yml"
    yml.write_text(TEST_PROJECT.read_text().replace(PLACEHOLDER, VERSION))
    result = subprocess.run(
        [sys.executable, "-m", "nvflare.cli", "provision", "-p", str(yml), "-w", str(base / "workspace"), "--force"],
        cwd=base,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return sorted((base / "workspace" / PROJECT_NAME).glob("prod_*"))[-1]


def _archive(base: Path, extra_env=None) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(base)}
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(ARCHIVER), str(base / "project.build.yml"), VERSION],
        cwd=base,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@pytest.fixture(scope="module")
def two_builds(tmp_path_factory):
    """Two packaged builds of one test project in one temporary workspace, as for a kit re-issue."""
    base = tmp_path_factory.mktemp("ca_reuse")
    archived = []
    first = _provision(base)
    if shutil.which("zip"):
        archived.append(_archive(base))
    state_after_first = json.loads((base / "workspace" / PROJECT_NAME / "state" / "cert.json").read_text())
    second = _provision(base)
    if shutil.which("zip"):
        # compared against the packaged first build
        archived.append(_archive(base))
    return base, first, second, state_after_first, archived


def test_rebuild_reuses_the_root_ca_from_state(two_builds):
    base, first, second, state, _ = two_builds
    expected = hashlib.sha256(state["root_cert"].encode("ascii")).hexdigest()
    for prod in (first, second):
        cas = {_sha256(p) for p in prod.glob("*/startup/rootCA.pem")}
        assert cas == {expected}, f"{prod.name}: kits must carry the CA stored in state/cert.json"


def test_rebuild_keeps_every_participant_identity(two_builds):
    """A re-issued kit is a drop-in replacement: same certificate and private key per participant."""
    base, first, second, _, _ = two_builds
    pairs = [(site, "client") for site in SITES] + [(SERVER, "server"), (ADMIN, "client")]
    for participant, stem in pairs:
        for suffix in (".crt", ".key"):
            name = f"startup/{stem}{suffix}"
            assert _sha256(first / participant / name) == _sha256(second / participant / name), (
                f"{participant}: {name} changed between two builds that share state/"
            )


@needs_zip
def test_archiver_packages_a_rebuild_with_the_same_ca(two_builds):
    base, first, second, state, archived = two_builds
    for result in archived:
        assert result.returncode == 0, result.stdout + result.stderr
    assert list(first.glob("*.zip")) and list(second.glob("*.zip"))

    sums = (second / "SHA256SUMS.txt").read_text().splitlines()
    assert sorted(line.split()[1] for line in sums) == sorted(f"{site}_{VERSION}.zip" for site in SITES), (
        "SHA256SUMS.txt is for the shared folder: site kits only, no server or admin kit"
    )
    for line in sums:
        digest, name = line.split()
        assert _sha256(second / name) == digest

    root_ca = hashlib.sha256(state["root_cert"].encode("ascii")).hexdigest()
    with zipfile.ZipFile(second / f"TEST_A_1_{VERSION}.zip") as kit:
        manifest = kit.read("TEST_A_1/kit_manifest.txt").decode()
    assert f"# root_ca_sha256: {root_ca}" in manifest
    assert "# kind: client" in manifest
    assert f"# kit_version: {VERSION}" in manifest
    assert f"{root_ca}  startup/rootCA.pem" in manifest


@needs_zip
def test_lost_state_mints_a_new_ca_and_the_archiver_refuses_it(tmp_path):
    """The F9 hazard itself: without state/cert.json the next build has a different CA."""
    base = tmp_path
    _provision(base)
    first_result = _archive(base)
    assert first_result.returncode == 0, first_result.stdout + first_result.stderr

    (base / "workspace" / PROJECT_NAME / "state" / "cert.json").unlink()
    third = _provision(base)
    old_ca = {_sha256(p) for p in (base / "workspace" / PROJECT_NAME / "prod_00").glob("*/startup/rootCA.pem")}
    new_ca = {_sha256(p) for p in third.glob("*/startup/rootCA.pem")}
    assert len(new_ca) == 1 and new_ca != old_ca, "NVFlare is expected to mint a new CA without state/"

    refused = _archive(base)
    assert refused.returncode != 0
    assert "different root CA" in refused.stderr
    assert not list(third.glob("*.zip")), "a refused build must not leave kits ready to hand out"

    # A plain rerun must still be refused: state/ now holds the new CA, and the unpackaged
    # build must not become the reference.
    fourth = _provision(base)
    assert _archive(base).returncode != 0
    assert not list(fourth.glob("*.zip"))

    allowed = _archive(base, {"MEDISWARM_ALLOW_NEW_ROOT_CA": "1"})
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert "NEW root CA" in allowed.stdout
