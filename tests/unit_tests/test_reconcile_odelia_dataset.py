"""scripts/site/reconcile_odelia_dataset.py follows the node's folder rules for every CONFIG.

Sites run it before a run and send us the counts. For the BreastDivider run it must read the
folders the node will read (data_unilateral_BD/, metadata_unilateral_BD/ or the fallback),
report the package version and fallback count, and never print a UID without --list.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "site" / "reconcile_odelia_dataset.py"
CUSTOM = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
GENERATOR = (REPO_ROOT / "application" / "jobs" / "ODELIA_ternary_classification" / "app" / "scripts"
             / "create_synthetic_dataset" / "create_synthetic_dataset.py")
if str(CUSTOM) not in sys.path:
    sys.path.insert(0, str(CUSTOM))

import data_layout  # noqa: E402

SITE = "client_A"


def _load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("reconcile_odelia_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args, script=SCRIPT, env=None, check=True):
    result = subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True,
                            env=env)
    if check:
        assert result.returncode == 0, result.stderr
    return result


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    pytest.importorskip("SimpleITK")
    pytest.importorskip("tqdm")
    root = tmp_path_factory.mktemp("reconcile") / "data"
    subprocess.run([sys.executable, str(GENERATOR), str(root), "--bd"], check=True, capture_output=True)
    return root


@pytest.fixture
def site_copy(synthetic_root, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(synthetic_root / SITE, root / SITE)
    return root


def test_standalone_mapping_matches_the_node():
    """A copy of the script used on its own must pick the same folders as the loader."""
    assert _load_script().STANDALONE_CONFIG_FOLDERS == data_layout.CONFIG_FOLDERS


def test_default_config_reads_the_standard_folders(synthetic_root):
    out = _run("--data_dir", synthetic_root / SITE).stdout
    assert "config           : unilateral (images data_unilateral/, metadata metadata_unilateral/" in out
    assert "exam folders     : 30 under data_unilateral/" in out
    assert "crop format      : ok; no processing.log; spacing of 30 sampled crops: all 1.00 x 1.00 x 1.00 mm" in out
    assert "usable by the node (in all three lists): 30" in out
    assert "ID_0" not in out


def test_bd_config_reads_the_bd_folders_and_reports_the_package(site_copy):
    out = _run("--data_dir", site_copy, "--config", "unilateral_BD").stdout   # one level above the site
    assert f"site folder      : {(site_copy / SITE).resolve()}" in out
    assert "config           : unilateral_BD (images data_unilateral_BD/, metadata metadata_unilateral_BD/" in out
    assert "exam folders     : 30 under data_unilateral_BD/" in out
    assert "processing.log: BreastDivider 0.2.2 (commit a084145, mode normal, xy 256, 0 untrusted-fallback" in out
    assert "30 distinct spacings among 30 crops, 0 on the fixed 0.7 x 0.7 x 3 mm grid" in out
    assert "WARNING" not in out
    assert "fold 0 train: 18 used of 18 listed" in out
    assert "ID_0" not in out


def test_several_sites_need_an_explicit_site_folder(synthetic_root):
    result = _run("--data_dir", synthetic_root, "--config", "unilateral", check=False)
    assert result.returncode != 0 and "several site folders" in result.stderr


def test_bd_metadata_fallback_is_reported(site_copy):
    shutil.rmtree(site_copy / SITE / "metadata_unilateral_BD")
    out = _run("--data_dir", site_copy, "--config", "unilateral_BD").stdout
    assert "metadata metadata_unilateral/" in out
    assert "metadata_unilateral_BD/ not found: the node falls back to metadata_unilateral/" in out


def test_config_from_the_environment_and_case_insensitive(site_copy):
    import os
    env = dict(os.environ, CONFIG="UNILATERAL_bd")
    out = _run("--data_dir", site_copy / SITE, env=env).stdout
    assert "config           : unilateral_BD" in out


def test_unknown_config_lists_the_valid_values(site_copy):
    result = _run("--data_dir", site_copy, "--config", "tight", check=False)
    assert result.returncode != 0
    assert "original, unilateral, unilateral_BD" in result.stderr


def test_standard_crops_in_the_bd_folder_are_flagged(site_copy):
    bd = site_copy / SITE / "data_unilateral_BD"
    shutil.rmtree(bd)
    shutil.copytree(site_copy / SITE / "data_unilateral", bd)
    out = _run("--data_dir", site_copy, "--config", "unilateral_BD").stdout
    assert "WARNING: no processing.log" in out
    assert "share one spacing" in out
    assert "ID_0" not in out


def test_image_file_override_and_missing_files(site_copy):
    case = site_copy / SITE / "data_unilateral_BD" / "ID_000_left"
    (case / "Sub_1.nii.gz").rename(case / "Sub_2.nii.gz")
    out = _run("--data_dir", site_copy, "--config", "unilateral_BD").stdout
    assert "exam folders without Sub_1.nii.gz (the node cannot load them): 1" in out
    out = _run("--data_dir", site_copy, "--config", "unilateral_BD", "--image_file", "Sub_2.nii.gz").stdout
    assert "image file Sub_2.nii.gz" in out
    assert "exam folders without Sub_2.nii.gz (the node cannot load them): 29" in out


def test_folder_overrides(site_copy):
    site = site_copy / SITE
    (site / "data_unilateral").rename(site / "crops_v2")
    (site / "metadata_unilateral").rename(site / "labels_v2")
    out = _run("--data_dir", site_copy, "--data_folder", "crops_v2", "--metadata_folder", "labels_v2").stdout
    assert "(images crops_v2/, metadata labels_v2/" in out and "30 under crops_v2/" in out


def test_list_shows_uids_only_on_request(site_copy):
    (site_copy / SITE / "data_unilateral_BD" / "ID_000_left" / "Sub_1.nii.gz").unlink()
    out = _run("--data_dir", site_copy, "--config", "unilateral_BD", "--list").stdout
    assert "ID_000_left" in out


def test_single_file_copy_still_counts(site_copy, tmp_path):
    """Sites may have received the script on its own; the counts must keep working."""
    lonely = tmp_path / "lonely" / "reconcile_odelia_dataset.py"
    lonely.parent.mkdir()
    shutil.copy(SCRIPT, lonely)
    out = _run("--data_dir", site_copy, "--config", "unilateral_BD", script=lonely).stdout
    assert "config           : unilateral_BD (images data_unilateral_BD/, metadata metadata_unilateral_BD/" in out
    assert "crop format      : not checked" in out
    assert "usable by the node (in all three lists): 30" in out
