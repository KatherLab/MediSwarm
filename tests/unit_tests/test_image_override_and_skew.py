"""A kit must outlive a backbone/algorithm update, and skew must be visible.

The backbone (the NVFlare fork), the training code and the job configs all ship in
the IMAGE, not in the startup kit, and docker.sh pulls by tag on every run. So the
only thing that tied a kit to a code change was the pinned tag in docker.sh. With
an override, a site moves to a new image without a new kit.

That makes version skew possible, and nothing detected it before: `expected_version`
existed in the roster but was only ever a display fallback, never compared. All
clients in a federated run must execute the same code, so a stale site has to be
loud, not silent.
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"
HEARTBEAT = REPO_ROOT / "kit_live_sync" / "build_heartbeat.sh"

yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(REPO_ROOT / "server_tools"))


@pytest.fixture(scope="module")
def docker_sh():
    template = yaml.safe_load(TEMPLATE.read_text())
    return re.sub(r"\{~~[a-z_]+~~\}", "jefftud/odelia:1.5.0-provisioned", template["docker_cln_sh"])


# --------------------------------------------------------------- image override

def test_docker_sh_accepts_an_image_override(docker_sh):
    assert "--image)" in docker_sh, "docker.sh cannot be pointed at a different image"


def test_image_conf_is_sourced_so_the_override_persists(docker_sh):
    """A site must be able to set it once, not pass a flag on every run."""
    assert "image.conf" in docker_sh


def test_image_precedence_is_cli_then_env_then_default(docker_sh):
    assert 'DOCKER_IMAGE="${CLI_IMAGE:-${MEDISWARM_IMAGE:-$DOCKER_IMAGE}}"' in docker_sh


def test_env_is_captured_before_image_conf_is_sourced(docker_sh):
    """--image > env > image.conf: a pre-exported MEDISWARM_IMAGE must beat image.conf.

    image.conf is `source`d, so if the env is not captured first, a MEDISWARM_IMAGE=
    line in image.conf silently clobbers the operator's env override (#449 follow-up).
    """
    src = docker_sh
    cap = src.index('ENV_IMAGE="${MEDISWARM_IMAGE:-}"')
    srcconf = src.index('. "$DIR/image.conf"')
    restore = src.index('MEDISWARM_IMAGE="${ENV_IMAGE:-')
    assert cap < srcconf, "env must be captured BEFORE image.conf is sourced"
    assert srcconf < restore, "the captured env must be re-applied AFTER sourcing"


def test_provisioned_tag_is_still_the_default(docker_sh):
    """With no override, a kit must behave exactly as before."""
    assert "DOCKER_IMAGE=jefftud/odelia:1.5.0-provisioned" in docker_sh


def test_heartbeat_reports_the_image_actually_running():
    """kit_version is only what the kit CLAIMS; once overridable it can be a stale lie."""
    script = HEARTBEAT.read_text()
    assert "image_id" in script and "docker inspect" in script
    assert "HB_IMAGE_ID" in script and '"image_id"' in script


# ------------------------------------------------------------- skew detection

@pytest.fixture(scope="module")
def skew():
    pytest.importorskip("fastapi")
    import app
    return app.compute_version_skew


def _site(name, image_id, kit_version="1.5.0", expected=""):
    return {
        "site": name,
        "image_id": image_id,
        "image_ref": f"jefftud/odelia:{kit_version}",
        "kit_version": kit_version,
        "expected_version": expected,
    }


def test_no_skew_when_every_site_runs_the_same_image(skew):
    rows = [_site("UKA_1", "sha256:aaa"), _site("USZ_1", "sha256:aaa"), _site("CAM_1", "sha256:aaa")]
    assert skew(rows) == {}


def test_the_odd_site_out_is_flagged(skew):
    """The failure this exists to catch: one site quietly on stale code."""
    rows = [
        _site("UKA_1", "sha256:aaa"),
        _site("USZ_1", "sha256:aaa"),
        _site("CAM_1", "sha256:STALE", kit_version="1.4.0"),
    ]
    result = skew(rows)
    assert set(result) == {"CAM_1"}
    assert "1.4.0" in result["CAM_1"]


def test_kit_version_mismatch_against_the_roster_is_flagged(skew):
    rows = [
        _site("UKA_1", "sha256:aaa", kit_version="1.5.0", expected="1.5.0"),
        _site("USZ_1", "sha256:aaa", kit_version="1.5.0", expected="1.6.0"),
    ]
    assert "USZ_1" in skew(rows)


def test_a_single_site_cannot_be_skewed(skew):
    """One site is the majority by definition -- do not cry wolf."""
    assert skew([_site("UKA_1", "sha256:aaa")]) == {}


def test_sites_without_version_data_are_ignored(skew):
    assert skew([{"site": "X", "image_id": "", "kit_version": ""}]) == {}
