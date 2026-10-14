"""CONFIG=unilateral_BD and the ODELIA_* folder overrides, end to end through the loader on CPU.

The data come from the generator CI uses (create_synthetic_dataset.py), run with --bd so that
each site also has data_unilateral_BD/ (spacing varying per case, processing.log,
diagnostic PNGs, removals CSV) and metadata_unilateral_BD/ next to the standard folders --
the layout a site has after running BreastDivider.
"""
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
GENERATOR = (REPO_ROOT / "application" / "jobs" / "ODELIA_ternary_classification" / "app" / "scripts"
             / "create_synthetic_dataset" / "create_synthetic_dataset.py")
if str(CUSTOM) not in sys.path:
    sys.path.insert(0, str(CUSTOM))

pytest.importorskip("torchio")
pytest.importorskip("pandas")
pytest.importorskip("SimpleITK")
pytest.importorskip("tqdm")

import data_layout  # noqa: E402
import env_config  # noqa: E402
from data.datasets.dataset_3d_odelia import ODELIA_Dataset3D  # noqa: E402

SITE = "client_A"
ENV_NAMES = ("CONFIG", "ODELIA_DATA_FOLDER", "ODELIA_METADATA_FOLDER", "ODELIA_IMAGE_FILE",
             "ODELIA_STRICT_DATA_FORMAT", "INSTITUTION", "FOLD")


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic") / "data"
    subprocess.run([sys.executable, str(GENERATOR), str(root), "--bd"], check=True, capture_output=True)
    return root


@pytest.fixture
def site_copy(synthetic_root, tmp_path):
    """A private copy of the synthetic tree for tests that change it."""
    root = tmp_path / "data"
    shutil.copytree(synthetic_root, root)
    return root


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _manifest(root, config=None):
    return ODELIA_Dataset3D.build_manifests(path_root=root, institutions=SITE, config=config)[SITE]


def test_generator_writes_a_breastdivider_layout(synthetic_root):
    site = synthetic_root / SITE
    for folder in ("data_unilateral", "metadata_unilateral", "data_unilateral_BD", "metadata_unilateral_BD"):
        assert (site / folder).is_dir()
    bd = site / "data_unilateral_BD"
    assert (bd / "processing.log").is_file() and (bd / "segmentation_cleanup_removals.csv").is_file()
    assert list(bd.glob("*_diagnostic.png"))
    assert sorted(p.name for p in bd.iterdir() if p.is_dir()) == sorted(
        p.name for p in (site / "data_unilateral").iterdir() if p.is_dir())


def test_unilateral_bd_reads_the_bd_folders(synthetic_root, caplog):
    with caplog.at_level(logging.INFO):
        manifest = _manifest(synthetic_root, "unilateral_BD")
    assert (manifest.config, manifest.data_dir, manifest.meta_dir, manifest.image_file) == (
        "unilateral_BD", "data_unilateral_BD", "metadata_unilateral_BD", "Sub_1.nii.gz")
    assert not manifest.layout.metadata_fallback
    # the crawler keeps exam folders only: processing.log, PNGs and the CSV are skipped
    assert len(manifest.image_uids) == 30 and all(u.endswith(("_left", "_right")) for u in manifest.image_uids)
    assert manifest.format_verdict.ok, manifest.format_verdict.issues
    assert manifest.format_verdict.processing_log.tool == "breastdivider"
    assert manifest.image_path("ID_003_left") == synthetic_root / SITE / "data_unilateral_BD" / "ID_003_left" / "Sub_1.nii.gz"
    text = caplog.text
    assert f"ODELIA data for {SITE}: CONFIG=unilateral_BD" in text
    assert str(synthetic_root / SITE / "data_unilateral_BD") in text and "Sub_1.nii.gz" in text
    assert f"ODELIA crop format for {SITE}: ok" in text
    assert "ID_0" not in text                       # no UIDs in the setup log


def test_config_env_selects_bd_and_items_load_on_cpu(synthetic_root, monkeypatch):
    monkeypatch.setenv("CONFIG", "unilateral_BD")
    ds = ODELIA_Dataset3D(path_root=synthetic_root, institutions=SITE, split="train", config=None, fold=0,
                          transform="USE_UNPROCESSED_IMAGES")
    assert ds.config == "unilateral_BD" and ds.data_dir == "data_unilateral_BD"
    assert len(ds) == 18
    item = ds[0]
    path = ds.get_image_path(item["uid"], SITE)
    assert path.parent.parent.name == "data_unilateral_BD"
    image = item["source"]
    assert tuple(image.shape) == (1, 256, 256, 32)
    # the BD crop itself was read: its header spacing, not the 1 x 1 x 1 mm of the standard crop
    assert image.spacing == pytest.approx(data_layout.read_nifti_spacing(path), abs=1e-4)
    assert image.spacing != pytest.approx((1.0, 1.0, 1.0))


def _croporpad_matches_this_repo():
    """The repo's CropOrPad subclass relies on torchio < 1.0 internals (compute_crop_or_pad).
    The image pins torchio==0.20.1 (Dockerfile_ODELIA); the GitHub unit-test job installs the
    latest torchio, where that attribute is gone."""
    import torchio as tio
    return hasattr(tio.CropOrPad((4, 4, 4)), "compute_crop_or_pad")


def test_bd_items_go_through_the_training_transform(synthetic_root):
    if not _croporpad_matches_this_repo():
        pytest.skip("torchio >= 1.0 lacks CropOrPad.compute_crop_or_pad; the image pins torchio==0.20.1")
    ds = ODELIA_Dataset3D(path_root=synthetic_root, institutions=SITE, split="train", config="unilateral_BD",
                          random_flip=True, random_rotate=True, noise=True)
    item = ds[0]
    assert tuple(item["source"].shape) == (1, 32, 224, 224)
    assert "data_unilateral_BD" in str(ds.get_image_path(item["uid"], SITE))


def test_standard_config_on_the_same_tree_is_unchanged_and_quiet(synthetic_root, caplog):
    """The synthetic CI data (constant 1 x 1 x 1 mm) must keep passing the guard."""
    with caplog.at_level(logging.INFO):
        manifest = _manifest(synthetic_root, "unilateral")
    assert (manifest.data_dir, manifest.meta_dir) == ("data_unilateral", "metadata_unilateral")
    assert manifest.format_verdict.ok
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_bd_metadata_falls_back_with_a_warning(site_copy, caplog):
    shutil.rmtree(site_copy / SITE / "metadata_unilateral_BD")
    with caplog.at_level(logging.INFO):
        manifest = _manifest(site_copy, "unilateral_BD")
    assert manifest.data_dir == "data_unilateral_BD" and manifest.meta_dir == "metadata_unilateral"
    assert manifest.layout.metadata_fallback_from == "metadata_unilateral_BD"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("metadata_unilateral_BD not found" in w for w in warnings)
    assert "metadata fallback" in caplog.text


def test_missing_bd_images_fail_fast_naming_the_folder(site_copy):
    shutil.rmtree(site_copy / SITE / "data_unilateral_BD")
    with pytest.raises(FileNotFoundError) as exc:
        _manifest(site_copy, "unilateral_BD")
    assert str(site_copy / SITE / "data_unilateral_BD") in str(exc.value)
    assert "BREASTDIVIDER_SITE_GUIDE" in str(exc.value)


def test_missing_metadata_still_names_the_csv(tmp_path):
    """CI's preflight without data greps for this pandas message."""
    with pytest.raises(FileNotFoundError, match="metadata_unilateral/annotation.csv"):
        _manifest(tmp_path, "unilateral")


def test_unknown_config_fails_with_the_valid_values(synthetic_root, monkeypatch):
    with pytest.raises(ValueError, match="original, unilateral, unilateral_BD"):
        ODELIA_Dataset3D(path_root=synthetic_root, institutions=SITE, split="train", config="unilateral_tight")
    monkeypatch.setenv("CONFIG", "bd")
    with pytest.raises(ValueError, match="Unknown CONFIG='bd'"):
        env_config.resolve_odelia_runtime_settings({"site_name": SITE, "model_name": "MST", "fold": 0,
                                                    "data_dir": str(synthetic_root), "scratch_dir": "/tmp"})


def test_standard_crops_in_the_bd_folder_warn_or_abort_when_strict(site_copy, monkeypatch, caplog):
    bd = site_copy / SITE / "data_unilateral_BD"
    shutil.rmtree(bd)
    shutil.copytree(site_copy / SITE / "data_unilateral", bd)   # standard crops under the BD name, no log
    with caplog.at_level(logging.WARNING):
        manifest = _manifest(site_copy, "unilateral_BD")
    assert not manifest.format_verdict.ok
    assert "ODELIA_STRICT_DATA_FORMAT=1 would abort" in caplog.text
    monkeypatch.setenv("ODELIA_STRICT_DATA_FORMAT", "1")
    with pytest.raises(data_layout.CropFormatError, match="data_unilateral_BD"):
        _manifest(site_copy, "unilateral_BD")


def test_bd_crops_under_unilateral_abort_when_strict(site_copy, monkeypatch):
    site = site_copy / SITE
    shutil.rmtree(site / "data_unilateral")
    shutil.copytree(site / "data_unilateral_BD", site / "data_unilateral")
    monkeypatch.setenv("ODELIA_STRICT_DATA_FORMAT", "1")
    with pytest.raises(data_layout.CropFormatError, match="BD_unilateralize"):
        _manifest(site_copy, "unilateral")


def test_folder_and_image_file_overrides(site_copy, monkeypatch):
    """A later preprocessing variant: own image folder and file, no new image needed."""
    site = site_copy / SITE
    shutil.copytree(site / "data_unilateral", site / "data_unilateral_enh")
    for case in (site / "data_unilateral_enh").iterdir():
        if case.is_dir():
            (case / "Sub_1.nii.gz").rename(case / "Sub_2.nii.gz")
    monkeypatch.setenv("ODELIA_DATA_FOLDER", "data_unilateral_enh")
    monkeypatch.setenv("ODELIA_IMAGE_FILE", "Sub_2.nii.gz")
    ds = ODELIA_Dataset3D(path_root=site_copy, institutions=SITE, split="val", config="unilateral",
                          transform="USE_UNPROCESSED_IMAGES")
    manifest = ds.manifests[SITE]
    assert (manifest.data_dir, manifest.meta_dir, manifest.image_file) == (
        "data_unilateral_enh", "metadata_unilateral", "Sub_2.nii.gz")
    assert set(manifest.layout.overrides) == {"ODELIA_DATA_FOLDER", "ODELIA_IMAGE_FILE"}
    assert manifest.format_verdict.ok
    item = ds[0]
    path = ds.get_image_path(item["uid"], SITE)
    assert (path.parent.parent.name, path.name) == ("data_unilateral_enh", "Sub_2.nii.gz")
    assert tuple(item["source"].shape) == (1, 256, 256, 32)


def test_metadata_override(site_copy, monkeypatch):
    site = site_copy / SITE
    shutil.copytree(site / "metadata_unilateral", site / "labels_v2")
    monkeypatch.setenv("ODELIA_METADATA_FOLDER", "labels_v2")
    assert _manifest(site_copy, "unilateral_BD").meta_dir == "labels_v2"


def test_preprocess_cache_is_keyed_by_the_layout(synthetic_root, tmp_path):
    cache = tmp_path / "cache"
    spacings = {}
    for config in ("unilateral", "unilateral_BD"):
        ds = ODELIA_Dataset3D(path_root=synthetic_root, institutions=SITE, split="val", config=config,
                              enable_preprocess_cache=True, preprocess_cache_dir=cache)
        uid = ds.df.loc[0, "UID"]
        ds._load_source_image(ds.get_image_path(uid, SITE), SITE, uid)          # fills the cache
        spacings[config] = ds._load_source_image(ds.get_image_path(uid, SITE), SITE, uid).spacing   # cache hit
    assert (cache / "unilateral" / SITE).is_dir() and (cache / "unilateral_BD" / SITE).is_dir()
    assert {p.name for p in (cache / "unilateral" / SITE).glob("*.npz")} == {
        p.name for p in (cache / "unilateral_BD" / SITE).glob("*.npz")}           # same UID, separate entries
    assert spacings["unilateral"] != pytest.approx(spacings["unilateral_BD"])     # and never each other's volume


def test_log_uid_discrepancies_follows_config(synthetic_root, monkeypatch, caplog):
    monkeypatch.setenv("CONFIG", "unilateral_BD")
    with caplog.at_level(logging.INFO):
        ODELIA_Dataset3D.log_UID_discrepancies(logging.getLogger("t"), path_root=synthetic_root, institutions=[SITE])
    assert f"ODELIA data for {SITE}: CONFIG=unilateral_BD" in caplog.text


def test_prepare_odelia_dataset_records_the_layout(synthetic_root, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for name, value in {"SITE_NAME": SITE, "DATA_DIR": str(synthetic_root), "SCRATCH_DIR": str(scratch),
                        "MODEL_NAME": "MST", "CONFIG": "unilateral_BD"}.items():
        monkeypatch.setenv(name, value)
    env_vars = env_config.load_environment_variables()
    manifests = env_config.build_odelia_manifests(env_vars)
    ds_train, ds_val, ds_test, path_run_dir, run_name = env_config.prepare_odelia_dataset(
        logging.getLogger("t"), manifests=manifests, env_vars=env_vars)
    assert "_unilateral_BD_fold0_" in run_name
    assert (len(ds_train), len(ds_val), len(ds_test)) == (18, 6, 6)
    record_text = (path_run_dir / env_config.DATA_LAYOUT_RECORD).read_text()
    record = json.loads(record_text)[SITE]
    assert record["layout"]["config"] == "unilateral_BD"
    assert record["data_dir"].endswith("data_unilateral_BD") and record["metadata_dir"].endswith("metadata_unilateral_BD")
    assert record["crop_format"]["ok"] is True
    assert "ID_0" not in record_text


def test_layout_record_is_best_effort(synthetic_root, tmp_path, caplog):
    """Provenance must never stop a run."""
    manifests = {SITE: _manifest(synthetic_root, "unilateral_BD")}
    with caplog.at_level(logging.WARNING):
        env_config.write_data_layout_record(tmp_path / "missing" / "run", manifests)
    assert "Could not write data_layout.json" in caplog.text
