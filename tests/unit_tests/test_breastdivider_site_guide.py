"""docs/BREASTDIVIDER_SITE_GUIDE.md must only tell sites things that are true of the code.

Partner instructions have drifted from the code before (see test_partner_handbook_is_real.py).
Here: every docker.sh and reconcile flag exists, folder, variable and version names match
data_layout.py, the prose says "swarm learning", and every message the troubleshooting table
quotes is one the node really produces.
"""
import gzip
import re
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs" / "BREASTDIVIDER_SITE_GUIDE.md"
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"
RECONCILE = REPO_ROOT / "scripts" / "site" / "reconcile_odelia_dataset.py"
CUSTOM = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
if str(CUSTOM) not in sys.path:
    sys.path.insert(0, str(CUSTOM))

import data_layout as dl  # noqa: E402


@pytest.fixture(scope="module")
def guide():
    return GUIDE.read_text()


def test_docker_sh_flags_exist(guide):
    """Sites keep their 1.8.1 kits; v1.8.1's docker.sh has the same flags (checked by hand
    with git show v1.8.1:docker_config/master_template.yml when the guide was written)."""
    yaml = pytest.importorskip("yaml")
    docker_sh = yaml.safe_load(TEMPLATE.read_text())["docker_cln_sh"]
    commands = re.findall(r"docker\.sh(?:[^\n`\\]|\\\n)*", guide)        # follows "\" line continuations
    flags = {flag for command in commands for flag in re.findall(r"--[A-Za-z_]{3,}", command)}
    assert {"--config", "--preflight_check", "--data_dir", "--job"} <= flags
    flags |= {flag for flag in ("--image", "--docker_options") if flag in guide}     # named in the prose
    missing = sorted(flag for flag in flags if f"{flag})" not in docker_sh)
    assert not missing, f"the guide uses docker.sh flags that do not exist: {missing}"
    assert "--env CONFIG=${CLI_CONFIG:-${CONFIG:-unilateral}}" in docker_sh       # --config reaches the container


def test_reconcile_flags_exist(guide):
    commands = re.findall(r"reconcile_odelia_dataset\.py[^\n`]*", guide)
    assert commands
    source = RECONCILE.read_text()
    for flag in {f for command in commands for f in re.findall(r"--[A-Za-z_]{3,}", command)} | {"--list"}:
        assert f'"{flag}"' in source, f"reconcile_odelia_dataset.py has no {flag}"


def test_names_match_the_code(guide):
    data, metadata, fallback = dl.CONFIG_FOLDERS["unilateral_BD"]
    for name in (data, metadata, fallback, dl.DEFAULT_IMAGE_FILE, dl.PROCESSING_LOG, dl.STRICT_ENV,
                 dl.DATA_FOLDER_ENV, dl.METADATA_FOLDER_ENV, dl.IMAGE_FILE_ENV, "unilateral_BD"):
        assert name in guide
    for version in dl.BD_ACCEPTED_VERSIONS:
        assert version in guide
    assert "--mode=normal" in guide and dl.BD_EXPECTED_MODE == "normal"


def test_says_swarm_learning(guide):
    assert "swarm learning" in guide
    assert "federated" not in guide.lower()


def test_the_bd_package_is_not_shipped_in_the_image():
    """MEVIS licenses BreastDivider for ODELIA research use only; the image is public."""
    dockerfile = (REPO_ROOT / "docker_config" / "Dockerfile_ODELIA").read_text()
    assert "breastdivider" not in dockerfile.lower() and "odelia-preprocessing" not in dockerfile.lower()


# ---------------------------------------------------------------------------
# the troubleshooting table quotes real messages
# ---------------------------------------------------------------------------

def _crop(path: Path, spacing):
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, 4, 4, 2, 1, 1, 1, 1)
    struct.pack_into("<8f", header, 76, 1.0, *spacing, 1.0, 0.0, 0.0, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(bytes(header) + b"\0" * 68)


BD_LOG = ("x [INFO] Command: python BD_main.py /data/S --single-site --mode=normal --subs 1\n"
          "x [INFO] Package version: 0.2.2\nx [INFO] Git commit: a084145\n"
          "x [INFO] Worker command: python BD_unilateralize.py --mode normal --xy_resolution 256\n"
          "x [PROGRESS] Segmentation report: 0 untrusted-fallback case(s)\n")


def _issues(root: Path, name: str, config: str, spacings, log=None, image_file="Sub_1.nii.gz"):
    images = root / name / dl.CONFIG_FOLDERS[config][0]
    uids = []
    for i, spacing in enumerate(spacings):
        uid = f"E{i}_left"
        _crop(images / uid / image_file, spacing)
        uids.append(uid)
    if log is not None:
        (images / "processing.log").write_text(log)
    return list(dl.check_crop_format(images, config, uids).issues)


def _real_messages(tmp_path):
    varied = [(0.6 + 0.01 * i, 0.7 + 0.01 * i, 4.0 + 0.1 * i) for i in range(6)]
    messages = []
    messages += _issues(tmp_path, "nolog", "unilateral_BD", varied)
    messages += _issues(tmp_path, "unilateralize", "unilateral_BD", varied,
                        log="x [INFO] Command: python unilateralize.py /data/S --single-site\n")
    messages += _issues(tmp_path, "version", "unilateral_BD", varied, log=BD_LOG.replace("0.2.2", "0.3.0"))
    messages += _issues(tmp_path, "tight", "unilateral_BD", varied,
                        log=BD_LOG.replace("--mode=normal", "--mode=tight").replace("--mode normal", "--mode tight"))
    messages += _issues(tmp_path, "xy", "unilateral_BD", varied, log=BD_LOG.replace("256", "192"))
    messages += _issues(tmp_path, "grid", "unilateral_BD", [dl.FIXED_GRID_SPACING_MM] * 4, log=BD_LOG)
    messages += _issues(tmp_path, "constant", "unilateral_BD", [(1.0, 1.0, 1.0)] * 4, log=BD_LOG)
    messages += _issues(tmp_path, "nosub1", "unilateral_BD", varied, log=BD_LOG, image_file="Sub_2.nii.gz")
    messages += _issues(tmp_path, "bdinstd", "unilateral", varied, log=BD_LOG)
    site = tmp_path / "SITE_1"
    (site / "metadata_unilateral").mkdir(parents=True)
    messages.append(dl.metadata_fallback_warning(site, dl.resolve_site_layout(site, "unilateral_BD", env={})))
    pytest.importorskip("torchio")
    pytest.importorskip("pandas")
    from data.datasets.dataset_3d_odelia import ODELIA_Dataset3D
    (site / "metadata_unilateral" / "annotation.csv").write_text("UID,PatientID,Age,Lesion\n")
    (site / "metadata_unilateral" / "split.csv").write_text("UID,Fold,Split\n")
    with pytest.raises(FileNotFoundError) as exc:
        ODELIA_Dataset3D.build_manifests(path_root=tmp_path, institutions="SITE_1", config="unilateral_BD")
    messages.append(str(exc.value).replace(str(tmp_path), "/data"))
    return messages


def _pattern(fragment: str) -> str:
    """`<...>` and `...` in the guide stand for any text; everything else must match literally."""
    parts = re.split(r"<[^>]+>|\.\.\.", fragment)
    return ".*?".join(re.escape(part) for part in parts)


def test_troubleshooting_table_quotes_real_messages(guide, tmp_path):
    rows = [line for line in guide.splitlines() if line.startswith("| `")]
    assert len(rows) >= 8
    messages = _real_messages(tmp_path)
    for row in rows:
        first_cell = row.split("|")[1]
        for fragment in re.findall(r"`([^`]+)`", first_cell):
            if fragment.startswith("CONFIG="):
                continue                                    # names the run's setting, not a message
            assert any(re.search(_pattern(fragment), m) for m in messages), (
                f"the guide quotes {fragment!r}, but the node never says that")
