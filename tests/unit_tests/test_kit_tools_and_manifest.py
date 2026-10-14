"""What the 1.9.0 kit build adds around NVFlare provisioning.

- Client kits carry scripts/client_node_setup/: the partner instructions and docker.sh's own
  pre-run checks name these scripts, but sites only receive the kit (RUMC, October 2026: the
  1.8.1 kit had no such folder).
- Admin kits carry the consortium job builder and run watcher when the build checkout has them,
  and the build does not fail when it does not.
- Every kit carries kit_manifest.txt (site, kind, version, expected image, root CA, sha256 per
  file), and the site kits get a SHA256SUMS.txt for the shared folder.
- _buildStartupKits.sh refuses to provision a production project whose state/cert.json is
  missing, because NVFlare would mint a new root CA (F9).

The scripts run against throwaway kit trees in tmp_path. A fake `docker` on PATH records the
call instead of provisioning anything.
"""

import os
import re
import shutil
import stat
import subprocess
import textwrap
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "scripts" / "build"
ADD_TOOLS = BUILD_DIR / "_addToolsToStartupKits.sh"
ARCHIVER = BUILD_DIR / "_generateStartupKitArchives.sh"
BUILD_KITS = BUILD_DIR / "_buildStartupKits.sh"
SETUP_DIR = REPO_ROOT / "scripts" / "client_node_setup"
PROVISION_DIR = REPO_ROOT / "application" / "provision"

PROJECT = "kit_tools_test"
needs_zip = pytest.mark.skipif(shutil.which("zip") is None, reason="the archiver needs the zip tool")


def _env(extra=None):
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"}
    env.update(extra or {})
    return env


def _run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env or _env(), timeout=120)


def _fake_prod(work: Path, ca: bytes = b"CA-ONE", prod="prod_00", server_ca=None) -> Path:
    """A provisioned-looking prod dir: one client, one server and one admin kit."""
    prod_dir = work / "workspace" / PROJECT / prod
    kits = {
        "SITE_1": ("fed_client.json", ca),
        "srv.example.org": ("fed_server.json", server_ca or ca),
        "admin@example.org": ("fl_admin.sh", ca),
    }
    for name, (marker, root) in kits.items():
        startup = prod_dir / name / "startup"
        startup.mkdir(parents=True)
        (startup / marker).write_text("{}\n")
        (startup / "rootCA.pem").write_bytes(root)
        (startup / "docker.sh").write_text(
            "#!/usr/bin/env bash\nDOCKER_IMAGE=jefftud/odelia:9.9.9\n"
            + ("  --env MEDISWARM_VERSION=9.9.9 \\\n" if marker == "fed_client.json" else "")
        )
        (startup / "image.conf").write_text(
            "#   MEDISWARM_IMAGE=jefftud/odelia:1.0.0\nMEDISWARM_IMAGE=jefftud/odelia:current\n"
        )
    (prod_dir / "SITE_1" / "readme.txt").write_text("site kit\n")
    return prod_dir


def _project_yml(work: Path) -> Path:
    yml = work / "project.yml"
    yml.write_text(
        textwrap.dedent(
            f"""\
            name: {PROJECT}
            participants:
              - name: SITE_1
                type: client
            builders:
              - path: nvflare.lighter.impl.static_file.StaticFileBuilder
                args:
                  docker_image: jefftud/odelia:__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__
            """
        )
    )
    return yml


# --------------------------------------------------------------------------- tools


def _tracked_setup_files():
    """Names of the files git tracks in scripts/client_node_setup (all files when git cannot tell).

    A copied checkout without its .git, or an image without git (the containerised test run),
    falls back to the folder.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--", "scripts/client_node_setup"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0:
        names = [Path(line).name for line in result.stdout.splitlines() if line.count("/") == 2]
        if names:
            return names
    return [p.name for p in SETUP_DIR.iterdir() if p.is_file()]


def test_client_kit_gets_every_client_node_setup_file(tmp_path):
    """Set equality with the repository folder: a helper of a new file type must not be dropped."""
    prod = _fake_prod(tmp_path)
    result = _run(["bash", str(ADD_TOOLS), str(_project_yml(tmp_path))], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    shipped = prod / "SITE_1" / "scripts" / "client_node_setup"
    expected = sorted(_tracked_setup_files())
    assert sorted(p.name for p in shipped.iterdir()) == expected, (
        "every file in scripts/client_node_setup must reach the client kit; extend SETUP_PATTERNS "
        "in _addToolsToStartupKits.sh for a new kind of helper"
    )
    for script in shipped.glob("*.sh"):
        assert script.stat().st_mode & stat.S_IXUSR, f"{script.name} must stay executable in the kit"
    # The documented command works as written from the kit folder, and setup_vpntunnel.sh's
    # "repository root" (two levels above itself) is the kit folder.
    assert (prod / "SITE_1" / "scripts" / "client_node_setup" / "setup_vpntunnel.sh").is_file()
    assert not (prod / "srv.example.org" / "scripts").exists()
    assert not (prod / "admin@example.org" / "scripts").exists()


def _mini_repo(root: Path, with_admin_tools: bool) -> Path:
    """A checkout with this script, a client_node_setup folder holding stray files, and admin tools."""
    (root / "scripts" / "build").mkdir(parents=True)
    shutil.copy2(ADD_TOOLS, root / "scripts" / "build" / ADD_TOOLS.name)
    setup = root / "scripts" / "client_node_setup"
    setup.mkdir(parents=True)
    (setup / "tool.sh").write_text("#!/usr/bin/env bash\n")
    (setup / "unit.service").write_text("[Unit]\n")
    (setup / "README.md").write_text("helpers\n")
    (setup / "UKA.ovpn").write_text("secret vpn profile\n")
    (setup / "debug.log").write_text("noise\n")
    admin = root / "scripts" / "admin"
    admin.mkdir(parents=True)
    (admin / "patch_warm_start_job.py").write_text("import json\n")
    (admin / "unrelated_server_tool.py").write_text("import os\n")
    if with_admin_tools:
        # Both ways a tool reaches a sibling: an import line, and loading it by file path the
        # way the real builder does (_load_sibling("hocon_lite")).
        (admin / "build_consortium_job.py").write_text(
            '"""Builder.\n\nfrom the image, not a module name\n"""\nimport json\nfrom _job_common import x\n'
            'parser = _load_sibling("_parser")\n'
        )
        (admin / "_job_common.py").write_text("import patch_warm_start_job\nx = 1\n")
        (admin / "_parser.py").write_text("import re\n")
        (admin / "watch_run.py").write_text("import sys\n")
    return root / "scripts" / "build" / ADD_TOOLS.name


def test_admin_tools_and_their_sibling_modules_are_shipped(tmp_path):
    script = _mini_repo(tmp_path / "repo", with_admin_tools=True)
    work = tmp_path / "work"
    prod = _fake_prod(work)
    result = _run(["bash", str(script), str(_project_yml(work))], work)
    assert result.returncode == 0, result.stdout + result.stderr

    admin = prod / "admin@example.org" / "startup"
    for name in ("build_consortium_job.py", "watch_run.py", "_job_common.py", "_parser.py", "patch_warm_start_job.py"):
        assert (admin / name).is_file(), f"{name} missing from the admin kit"
        assert (admin / name).stat().st_mode & stat.S_IXUSR
    assert not (admin / "unrelated_server_tool.py").exists(), "only the named tools and what they import"

    shipped = {p.name for p in (prod / "SITE_1" / "scripts" / "client_node_setup").iterdir()}
    assert shipped == {"tool.sh", "unit.service", "README.md"}
    assert "UKA.ovpn" not in shipped, "a VPN profile with credentials must never reach the kits"


@pytest.mark.skipif(
    not (REPO_ROOT / "scripts" / "admin" / "build_consortium_job.py").is_file(),
    reason="this checkout has no consortium job builder",
)
def test_the_real_admin_tools_run_from_the_admin_kit(tmp_path):
    """The kit has no checkout: each shipped tool must start with only what sits next to it."""
    prod = _fake_prod(tmp_path)
    result = _run(["bash", str(ADD_TOOLS), str(_project_yml(tmp_path))], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    startup = prod / "admin@example.org" / "startup"
    for tool in ("build_consortium_job.py", "watch_run.py"):
        if not (REPO_ROOT / "scripts" / "admin" / tool).is_file():
            continue
        # Run as an operator would, from the kit folder; -E -s keep PYTHONPATH and the user
        # site out, so only the kit folder (the script's own directory) supplies modules.
        run = subprocess.run(
            ["python3", "-E", "-s", tool, "--help"], cwd=startup, capture_output=True, text=True, timeout=60
        )
        assert run.returncode == 0, f"{tool} --help from the admin kit failed:\n{run.stdout}{run.stderr}"


def test_missing_admin_tools_do_not_fail_the_build(tmp_path):
    script = _mini_repo(tmp_path / "repo", with_admin_tools=False)
    work = tmp_path / "work"
    prod = _fake_prod(work)
    result = _run(["bash", str(script), str(_project_yml(work))], work)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "build_consortium_job.py is not in this checkout" in result.stdout
    assert not (prod / "admin@example.org" / "startup" / "build_consortium_job.py").exists()


# --------------------------------------------------------------------------- archives


@needs_zip
def test_archives_carry_a_verifiable_manifest(tmp_path):
    prod = _fake_prod(tmp_path)
    result = _run(["bash", str(ARCHIVER), str(_project_yml(tmp_path)), "9.9.9"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    manifest = (prod / "SITE_1" / "kit_manifest.txt").read_text()
    for line in (
        "# site: SITE_1",
        "# kind: client",
        "# kit_version: 9.9.9",
        "# image: jefftud/odelia:9.9.9",
        "# image_channel: jefftud/odelia:current",
        "# template_version: 9.9.9",
        f"# project: {PROJECT}",
    ):
        assert line in manifest.splitlines(), f"missing {line!r}"
    # The checksum part is plain sha256sum output and verifies from the kit folder.
    check = subprocess.run(
        "grep -v '^#' kit_manifest.txt | sha256sum -c --quiet",
        shell=True, cwd=prod / "SITE_1", capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    sums_lines = [line for line in manifest.splitlines() if not line.startswith("#")]
    assert any(line.endswith("  startup/rootCA.pem") for line in sums_lines)
    assert not any(line.endswith("kit_manifest.txt") for line in sums_lines), "the manifest cannot list itself"

    with zipfile.ZipFile(prod / "SITE_1_9.9.9.zip") as kit:
        assert "SITE_1/kit_manifest.txt" in kit.namelist()

    sums = (prod / "SHA256SUMS.txt").read_text().split()
    assert sums[1::2] == ["SITE_1_9.9.9.zip"], "only site kits go into SHA256SUMS.txt"

    rows = (prod / "kit_manifest.csv").read_text().splitlines()
    assert rows[0] == "site,version,file,sha256,kind,image,root_ca_sha256"
    kinds = {row.split(",")[0]: row.split(",")[4] for row in rows[1:]}
    assert kinds == {"SITE_1": "client", "srv.example.org": "server", "admin@example.org": "admin"}


@needs_zip
def test_kits_of_one_build_must_share_one_root_ca(tmp_path):
    prod = _fake_prod(tmp_path, ca=b"CA-ONE", server_ca=b"CA-TWO")
    result = _run(["bash", str(ARCHIVER), str(_project_yml(tmp_path)), "9.9.9"], tmp_path)
    assert result.returncode != 0
    assert "different root CAs" in result.stderr
    assert not list(prod.glob("*.zip"))


@needs_zip
def test_a_kit_without_docker_sh_is_still_packaged(tmp_path):
    prod = _fake_prod(tmp_path)
    (prod / "SITE_1" / "startup" / "docker.sh").unlink()
    (prod / "SITE_1" / "startup" / "image.conf").unlink()
    result = _run(["bash", str(ARCHIVER), str(_project_yml(tmp_path)), "9.9.9"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = (prod / "SITE_1" / "kit_manifest.txt").read_text()
    assert "# image: jefftud/odelia:9.9.9" in manifest, "falls back to the project's docker_image"
    assert "# image_channel: none" in manifest


@needs_zip
def test_a_kit_from_another_template_version_is_flagged(tmp_path):
    prod = _fake_prod(tmp_path)
    result = _run(["bash", str(ARCHIVER), str(_project_yml(tmp_path)), "9.9.10"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "docker.sh comes from the 9.9.9 template, the kit is labelled 9.9.10" in result.stdout
    assert (prod / "SITE_1_9.9.10.zip").is_file()


# --------------------------------------------------------------------------- provisioning guard


def _fake_docker(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker_calls.log"
    docker = bin_dir / "docker"
    docker.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{log}"\n')
    docker.chmod(0o755)
    return log


def _build(tmp_path, project_file: Path, extra_env=None):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    yml = work / project_file.name
    shutil.copy2(project_file, yml)
    log = _fake_docker(tmp_path)
    env = _env({"PATH": f"{tmp_path / 'bin'}:/usr/local/bin:/usr/bin:/bin"})
    env.update(extra_env or {})
    result = _run(["bash", str(BUILD_KITS), yml.name, "9.9.9", "example/image:9.9.9"], work, env)
    calls = log.read_text() if log.exists() else ""
    return result, calls, work


@pytest.mark.parametrize("project", ["project_Odelia_allsites.yml", "project_DECADE_allsites.yml"])
def test_production_project_without_ca_state_is_refused(tmp_path, project):
    result, calls, work = _build(tmp_path, PROVISION_DIR / project)
    assert result.returncode != 0
    assert calls == "", "docker must not be started: provisioning would mint a new root CA"
    assert "REFUSING to provision the production project" in result.stderr
    assert "MEDISWARM_ALLOW_NEW_ROOT_CA=1" in result.stderr
    assert not list(work.glob("*.build.yml")), "the temporary project copy is cleaned up"


def test_production_project_with_ca_state_is_provisioned(tmp_path):
    work = tmp_path / "work"
    state = work / "workspace" / "odelia_allsites" / "state"
    state.mkdir(parents=True)
    (state / "cert.json").write_text('{"root_cert": "x"}')
    result, calls, _ = _build(tmp_path, PROVISION_DIR / "project_Odelia_allsites.yml")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Reusing the root CA" in result.stdout
    assert "nvflare provision" in calls
    # the tool step runs between the live-sync injection and the archives
    assert calls.index("_injectLiveSyncIntoStartupKits.sh") < calls.index("_addToolsToStartupKits.sh")
    assert calls.index("_addToolsToStartupKits.sh") < calls.index("_generateStartupKitArchives.sh")
    assert "MEDISWARM_ALLOW_NEW_ROOT_CA=0" in calls


def test_explicit_override_allows_a_new_production_ca(tmp_path):
    result, calls, _ = _build(
        tmp_path, PROVISION_DIR / "project_Odelia_allsites.yml", {"MEDISWARM_ALLOW_NEW_ROOT_CA": "1"}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MEDISWARM_ALLOW_NEW_ROOT_CA=1" in calls, "the archiver's CA check must see the override too"


def test_test_projects_still_get_a_fresh_ca(tmp_path):
    """Test projects carry the version in their name and start without state by design."""
    result, calls, _ = _build(tmp_path, PROVISION_DIR / "project_deploy_test_2site.yml")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "nvflare provision" in calls
    assert "creates a new root CA" in result.stdout


# --------------------------------------------------------------------------- partner texts

GUIDE = REPO_ROOT / "docs" / "KIT_1.9.0_SITE_GUIDE.md"
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"
# Options of other commands that the texts show: sha256sum -c, docker ps, and the docker run
# option inside image.conf's MEDISWARM_DOCKER_OPTIONS.
NOT_DOCKER_SH = {"--ignore-missing", "--filter", "--env"}
# The migration options come with the 1.9.0 client docker.sh (#640), built in parallel.
MIGRATION_FLAGS = {"--migrate_from", "--yes", "--start", "--status"}


def test_site_texts_name_only_options_the_client_docker_sh_parses():
    """A site copies these commands verbatim; an option docker.sh does not know aborts its start."""
    yaml = pytest.importorskip("yaml")
    template = yaml.safe_load(TEMPLATE.read_text())
    parsed = set(re.findall(r"^\s*(--[A-Za-z_]+)\)", template["docker_cln_sh"], re.M))
    texts = GUIDE.read_text() + template["readme_fc"]
    named = set(re.findall(r"(?<![\w-])--[A-Za-z][A-Za-z_-]*", texts)) - NOT_DOCKER_SH
    assert MIGRATION_FLAGS <= named, "the texts describe the migration commands"
    if not MIGRATION_FLAGS <= parsed:
        # Until #640 is merged; afterwards this check covers the migration options as well.
        named -= MIGRATION_FLAGS
    assert named <= parsed, f"options named for sites that docker.sh does not parse: {sorted(named - parsed)}"


def test_site_texts_say_swarm_learning():
    yaml = pytest.importorskip("yaml")
    template = yaml.safe_load(TEMPLATE.read_text())
    texts = [("guide", GUIDE.read_text())] + [(k, template[k]) for k in ("readme_fc", "readme_fs", "readme_am")]
    for name, text in texts:
        assert "federated" not in text.lower(), f"{name}: partner-facing text says swarm learning"
