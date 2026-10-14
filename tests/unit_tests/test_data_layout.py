"""data_layout.py: which folders CONFIG and the ODELIA_* overrides select, and the crop-format guard.

Standard crops (step3_unilateral.py) sit on a fixed 0.7 x 0.7 x 3 mm grid; BreastDivider fits
its box per breast, so its spacing varies. Both are 256 x 256 x 32, so a site that mixes them
trains without any error and loses about 0.06 AUROC
(docs/PREPROCESSING_BENCHMARK_BREASTDIVIDER.md). These tests pin the evidence the guard uses
and that the default 'unilateral' path, including the synthetic CI data (constant 1 x 1 x 1 mm
spacing), stays quiet.
"""
import gzip
import logging
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
if str(CUSTOM) not in sys.path:
    sys.path.insert(0, str(CUSTOM))

import data_layout as dl  # noqa: E402  (stdlib only: runs without torch)

BD_LOG = """\
2026-09-25 14:13:22 [INFO] Command: /venv/bin/python BD_main.py /data/SITE_1 --single-site --mode=normal --subs 1 --force
2026-09-25 14:13:22 [INFO] Package version: 0.2.2
2026-09-25 14:13:22 [INFO] Package version source: _version.py
2026-09-25 14:13:22 [INFO] Git commit: a084145
2026-09-25 14:13:22 [INFO] Worker command: /venv/bin/python /bdpp/BD_unilateralize.py --mode normal --data_root /data/SITE_1/data --xy_resolution 256 --force --subs 1
2026-09-25 14:13:23 [INFO] root:1782 mode            : normal
2026-09-25 14:13:23 [INFO] root:1788 xy_resolution   : 256 (output grid only; box-finding above is unaffected)
2026-09-25 14:15:48 [PROGRESS] root:939 Segmentation report: 2 untrusted-fallback case(s), 0 component-removal(s) across 0 case(s) -> x.csv
"""
# MEVIS also ships unilateralize.py, which writes the STANDARD crops with the same package version.
UNILATERALIZE_LOG = """\
2026-09-25 14:15:50 [INFO] Command: /venv/bin/python unilateralize.py /data/SITE_1 --single-site --image-interpolation linear
2026-09-25 14:15:50 [INFO] Package version: 0.2.2
2026-09-25 14:15:50 [INFO] Git commit: a084145
"""


# ---------------------------------------------------------------------------
# helpers: NIfTI headers written with the standard library (no SimpleITK needed)
# ---------------------------------------------------------------------------

def _nifti1(spacing, endian="<", ndim=3):
    hdr = bytearray(348)
    struct.pack_into(endian + "i", hdr, 0, 348)
    struct.pack_into(endian + "8h", hdr, 40, ndim, 4, 4, 2, 1, 1, 1, 1)
    struct.pack_into(endian + "8f", hdr, 76, 1.0, *spacing, 1.0, 0.0, 0.0, 0.0)
    hdr[344:348] = b"n+1\0"
    return bytes(hdr) + b"\0" * 4 + b"\0" * (4 * 4 * 2 * 2)


def _nifti2(spacing, endian="<"):
    hdr = bytearray(540)
    struct.pack_into(endian + "i", hdr, 0, 540)
    hdr[4:12] = b"n+2\0\r\n\x1a\n"
    struct.pack_into(endian + "8q", hdr, 16, 3, 4, 4, 2, 1, 1, 1, 1)
    struct.pack_into(endian + "8d", hdr, 104, 1.0, *spacing, 1.0, 0.0, 0.0, 0.0)
    return bytes(hdr) + b"\0" * 4 + b"\0" * 64


def _write_crop(path: Path, spacing, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raw if raw is not None else _nifti1(spacing)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wb") as fh:
            fh.write(data)
    else:
        path.write_bytes(data)


def _crops(images_dir: Path, spacings, image_file="Sub_1.nii.gz"):
    """One exam folder per spacing, named like real UIDs (<exam>_left/_right)."""
    uids = []
    for i, spacing in enumerate(spacings):
        uid = f"EXAM{i // 2:03d}_{'left' if i % 2 == 0 else 'right'}"
        _write_crop(images_dir / uid / image_file, spacing)
        uids.append(uid)
    return uids


def _bd_spacings(n):
    return [(0.51 + 0.013 * i, 0.60 + 0.017 * i, 3.75 + 0.19 * i) for i in range(n)]


# ---------------------------------------------------------------------------
# CONFIG and folder selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (None, "unilateral"), ("", "unilateral"), ("  ", "unilateral"),
    ("unilateral", "unilateral"), ("original", "original"),
    ("unilateral_BD", "unilateral_BD"), ("unilateral_bd", "unilateral_BD"), (" UNILATERAL_BD ", "unilateral_BD"),
])
def test_canonical_config(value, expected):
    assert dl.canonical_config(value, env={}) == expected


def test_canonical_config_reads_the_environment_when_none():
    assert dl.canonical_config(None, env={"CONFIG": "unilateral_BD"}) == "unilateral_BD"


def test_unknown_config_names_the_valid_values():
    with pytest.raises(ValueError) as exc:
        dl.canonical_config("unilateral_BD_tight", env={})
    message = str(exc.value)
    assert "unilateral_BD_tight" in message
    for valid in ("original", "unilateral", "unilateral_BD"):
        assert valid in message
    assert "ODELIA_DATA_FOLDER" in message


@pytest.mark.parametrize("config, data, meta", [
    ("original", "data", "metadata"),
    ("unilateral", "data_unilateral", "metadata_unilateral"),
    ("unilateral_BD", "data_unilateral_BD", "metadata_unilateral_BD"),
])
def test_requested_layout_defaults(config, data, meta):
    layout = dl.requested_layout(config, env={})
    assert (layout.data_folder, layout.metadata_folder, layout.image_file) == (data, meta, "Sub_1.nii.gz")
    assert layout.overrides == ()
    assert not layout.metadata_fallback


def test_overrides_replace_the_defaults_and_are_recorded():
    env = {"ODELIA_DATA_FOLDER": "data_unilateral_enh/", "ODELIA_METADATA_FOLDER": "metadata_enh",
           "ODELIA_IMAGE_FILE": "Sub_2.nii.gz"}
    layout = dl.requested_layout("unilateral", env=env)
    assert (layout.data_folder, layout.metadata_folder, layout.image_file) == (
        "data_unilateral_enh", "metadata_enh", "Sub_2.nii.gz")
    assert set(layout.overrides) == {"ODELIA_DATA_FOLDER", "ODELIA_METADATA_FOLDER", "ODELIA_IMAGE_FILE"}


def test_blank_overrides_are_ignored():
    layout = dl.requested_layout("unilateral_BD", env={"ODELIA_DATA_FOLDER": "", "ODELIA_IMAGE_FILE": "  "})
    assert layout.data_folder == "data_unilateral_BD" and layout.image_file == "Sub_1.nii.gz"
    assert layout.overrides == ()


@pytest.mark.parametrize("value", ["/data/UKA_1/data_unilateral_BD", "../UKA_2/data_unilateral", "a/../../b", "."])
def test_overrides_must_stay_inside_the_site_folder(value):
    with pytest.raises(ValueError, match="ODELIA_DATA_FOLDER"):
        dl.requested_layout("unilateral", env={"ODELIA_DATA_FOLDER": value})


def test_bd_metadata_falls_back_to_the_standard_metadata(tmp_path):
    (tmp_path / "metadata_unilateral").mkdir()
    layout = dl.resolve_site_layout(tmp_path, "unilateral_BD", env={})
    assert layout.metadata_folder == "metadata_unilateral"
    assert layout.metadata_fallback and layout.metadata_fallback_from == "metadata_unilateral_BD"
    warning = dl.metadata_fallback_warning(tmp_path, layout)
    assert str(tmp_path / "metadata_unilateral_BD") in warning and str(tmp_path / "metadata_unilateral") in warning


def test_bd_metadata_is_used_when_present(tmp_path):
    (tmp_path / "metadata_unilateral").mkdir()
    (tmp_path / "metadata_unilateral_BD").mkdir()
    layout = dl.resolve_site_layout(tmp_path, "unilateral_BD", env={})
    assert layout.metadata_folder == "metadata_unilateral_BD" and not layout.metadata_fallback


def test_no_fallback_when_neither_metadata_folder_exists(tmp_path):
    """The CSV error must then name the folder the config asked for."""
    layout = dl.resolve_site_layout(tmp_path, "unilateral_BD", env={})
    assert layout.metadata_folder == "metadata_unilateral_BD" and not layout.metadata_fallback


def test_explicit_metadata_override_never_falls_back(tmp_path):
    (tmp_path / "metadata_unilateral").mkdir()
    layout = dl.resolve_site_layout(tmp_path, "unilateral_BD", env={"ODELIA_METADATA_FOLDER": "metadata_mine"})
    assert layout.metadata_folder == "metadata_mine" and not layout.metadata_fallback


def test_standard_config_has_no_fallback_and_does_not_touch_the_disk(tmp_path):
    layout = dl.resolve_site_layout(tmp_path / "does_not_exist", "unilateral", env={})
    assert layout.metadata_folder == "metadata_unilateral" and not layout.metadata_fallback


def test_cache_namespace_separates_formats_but_keeps_existing_default_caches():
    assert dl.requested_layout("unilateral", env={}).cache_namespace == "unilateral"
    assert dl.requested_layout("unilateral_BD", env={}).cache_namespace == "unilateral_BD"
    other_folder = dl.requested_layout("unilateral", env={"ODELIA_DATA_FOLDER": "data_unilateral_BD"})
    other_file = dl.requested_layout("unilateral", env={"ODELIA_IMAGE_FILE": "Sub_2.nii.gz"})
    names = {other_folder.cache_namespace, other_file.cache_namespace, "unilateral", "unilateral_BD"}
    assert len(names) == 4
    assert "/" not in other_folder.cache_namespace


def test_describe_layout_names_folders_file_fallback_and_overrides(tmp_path):
    layout = dl.DataLayout("unilateral_BD", "data_unilateral_BD", "metadata_unilateral", "Sub_1.nii.gz",
                           overrides=("ODELIA_IMAGE_FILE",), metadata_fallback_from="metadata_unilateral_BD")
    text = dl.describe_layout(tmp_path, layout, 12)
    for part in ("CONFIG=unilateral_BD", str(tmp_path / "data_unilateral_BD"), "(12 exam folders)",
                 str(tmp_path / "metadata_unilateral"), "Sub_1.nii.gz", "metadata_unilateral_BD not found",
                 "ODELIA_IMAGE_FILE"):
        assert part in text


# ---------------------------------------------------------------------------
# NIfTI header reader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["a.nii.gz", "a.nii"])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_reads_nifti1_spacing(tmp_path, name, endian):
    path = tmp_path / name
    _write_crop(path, None, raw=_nifti1((0.64, 0.81, 4.9), endian=endian))
    assert dl.read_nifti_spacing(path) == pytest.approx((0.64, 0.81, 4.9), abs=1e-6)


def test_reads_nifti2_spacing(tmp_path):
    path = tmp_path / "a.nii.gz"
    _write_crop(path, None, raw=_nifti2((0.7, 0.7, 3.0)))
    assert dl.read_nifti_spacing(path) == pytest.approx((0.7, 0.7, 3.0))


def test_other_formats_are_not_checked(tmp_path):
    path = tmp_path / "a.mha"
    path.write_bytes(b"ObjectType = Image\n")
    assert dl.read_nifti_spacing(path) is None


@pytest.mark.parametrize("raw", [b"", b"\0" * 100, b"\x01" * 600])
def test_broken_headers_raise(tmp_path, raw):
    path = tmp_path / "a.nii"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        dl.read_nifti_spacing(path)


def test_header_reader_agrees_with_simpleitk(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    image = sitk.Image(16, 12, 8, sitk.sitkInt16)
    image.SetSpacing((0.637, 0.812, 4.93))
    path = tmp_path / "Sub_1.nii.gz"
    sitk.WriteImage(image, str(path))
    assert dl.read_nifti_spacing(path) == pytest.approx(sitk.ReadImage(str(path)).GetSpacing(), abs=1e-5)


# ---------------------------------------------------------------------------
# processing.log
# ---------------------------------------------------------------------------

def test_reads_a_breastdivider_log(tmp_path):
    (tmp_path / "processing.log").write_text(BD_LOG)
    log = dl.read_processing_log(tmp_path)
    assert log.tool == "breastdivider"
    assert (log.package_version, log.git_commit, log.mode, log.xy_resolution, log.untrusted_fallbacks) == (
        "0.2.2", "a084145", "normal", "256", 2)
    assert "BreastDivider 0.2.2" in log.describe() and "2 untrusted-fallback" in log.describe()


def test_unilateralize_log_is_not_breastdivider(tmp_path):
    """unilateralize.py is a substring of BD_unilateralize.py; the tool must still be told apart."""
    (tmp_path / "processing.log").write_text(UNILATERALIZE_LOG)
    log = dl.read_processing_log(tmp_path)
    assert log.tool == "unilateralize" and log.package_version == "0.2.2"


def test_no_log_is_none(tmp_path):
    assert dl.read_processing_log(tmp_path) is None


def test_large_log_is_read_from_head_and_tail(tmp_path):
    """A site-wide log has a line per case; only its first and last lines are needed."""
    head, report = BD_LOG.splitlines(keepends=True)[:-1], BD_LOG.splitlines(keepends=True)[-1]
    filler = "2026-09-25 14:14:00 [DEBUG] root:800 crop_at_original_resolution_then_resample: shape ok\n" * 20000
    (tmp_path / "processing.log").write_text("".join(head) + filler + report)
    assert (tmp_path / "processing.log").stat().st_size > 2 * 256 * 1024
    log = dl.read_processing_log(tmp_path)
    assert log.tool == "breastdivider" and log.package_version == "0.2.2" and log.untrusted_fallbacks == 2


def test_appended_runs_report_the_last_values(tmp_path):
    second = BD_LOG.replace("0.2.2", "0.2.1").replace("Segmentation report: 2", "Segmentation report: 0")
    (tmp_path / "processing.log").write_text(BD_LOG + second)
    log = dl.read_processing_log(tmp_path)
    assert log.package_version == "0.2.1" and log.untrusted_fallbacks == 0


# ---------------------------------------------------------------------------
# check_crop_format
# ---------------------------------------------------------------------------

def test_breastdivider_crops_pass_under_unilateral_bd(tmp_path):
    images = tmp_path / "data_unilateral_BD"
    uids = _crops(images, _bd_spacings(30))
    (images / "processing.log").write_text(BD_LOG)
    verdict = dl.check_crop_format(images, "unilateral_BD", uids)
    assert verdict.ok, verdict.issues
    assert verdict.spacing.read == 20 and verdict.spacing.distinct == 20
    assert "ok;" in verdict.summary() and "varies" in verdict.summary()


def test_standard_crops_in_the_bd_folder_are_flagged(tmp_path):
    images = tmp_path / "data_unilateral_BD"
    uids = _crops(images, [dl.FIXED_GRID_SPACING_MM] * 10)
    verdict = dl.check_crop_format(images, "unilateral_BD", uids)
    assert any("no processing.log" in issue for issue in verdict.issues)
    assert any("fixed 0.7 x 0.7 x 3 mm" in issue for issue in verdict.issues)


def test_float32_spacing_counts_as_the_fixed_grid(tmp_path):
    """0.7 is stored as 0.699999988 in a NIfTI header."""
    images = tmp_path / "data_unilateral_BD"
    f32 = struct.unpack("<f", struct.pack("<f", 0.7))[0]
    uids = _crops(images, [(f32, f32, 3.0)] * 4)
    (images / "processing.log").write_text(BD_LOG)
    verdict = dl.check_crop_format(images, "unilateral_BD", uids)
    assert verdict.spacing.on_fixed_grid == 4
    assert any("fixed 0.7 x 0.7 x 3 mm" in issue for issue in verdict.issues)


def test_constant_non_standard_spacing_in_the_bd_folder_is_flagged(tmp_path):
    images = tmp_path / "data_unilateral_BD"
    uids = _crops(images, [(1.0, 1.0, 1.0)] * 6)
    (images / "processing.log").write_text(BD_LOG)
    verdict = dl.check_crop_format(images, "unilateral_BD", uids)
    assert [issue for issue in verdict.issues if "share one spacing" in issue]


def test_two_crops_with_one_spacing_are_not_enough_to_judge(tmp_path):
    """Left and right of one exam may share a spacing; constancy needs a few crops."""
    images = tmp_path / "data_unilateral_BD"
    uids = _crops(images, [(0.7, 0.7, 5.1)] * 2)
    (images / "processing.log").write_text(BD_LOG)
    assert dl.check_crop_format(images, "unilateral_BD", uids).ok


@pytest.mark.parametrize("log, expected", [
    (UNILATERALIZE_LOG, "does not name BD_unilateralize"),
    (BD_LOG.replace("0.2.2", "0.3.0"), "package version 0.3.0"),
    (BD_LOG.replace("--mode=normal", "--mode=tight").replace("--mode normal", "--mode tight")
     .replace("mode            : normal", "mode            : tight"), "--mode tight"),
    (BD_LOG.replace("256", "192"), "--xy_resolution 192"),
    ("just some text\n", "does not name BD_unilateralize"),
])
def test_bd_log_problems_are_flagged(tmp_path, log, expected):
    images = tmp_path / "data_unilateral_BD"
    uids = _crops(images, _bd_spacings(6))
    (images / "processing.log").write_text(log)
    verdict = dl.check_crop_format(images, "unilateral_BD", uids)
    assert any(expected in issue for issue in verdict.issues), verdict.issues


def test_accepted_versions_pass(tmp_path):
    for version in dl.BD_ACCEPTED_VERSIONS:
        images = tmp_path / version / "data_unilateral_BD"
        uids = _crops(images, _bd_spacings(6))
        (images / "processing.log").write_text(BD_LOG.replace("0.2.2", version))
        assert dl.check_crop_format(images, "unilateral_BD", uids).ok


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), dl.FIXED_GRID_SPACING_MM])
def test_constant_spacing_passes_under_unilateral(tmp_path, spacing):
    """The synthetic CI data (SimpleITK default 1 x 1 x 1 mm) and real standard crops."""
    images = tmp_path / "data_unilateral"
    uids = _crops(images, [spacing] * 30)
    verdict = dl.check_crop_format(images, "unilateral", uids)
    assert verdict.ok, verdict.issues


def test_mevis_unilateralize_log_passes_under_unilateral(tmp_path):
    images = tmp_path / "data_unilateral"
    uids = _crops(images, [dl.FIXED_GRID_SPACING_MM] * 4)
    (images / "processing.log").write_text(UNILATERALIZE_LOG)
    assert dl.check_crop_format(images, "unilateral", uids).ok


def test_breastdivider_crops_under_unilateral_are_flagged(tmp_path):
    images = tmp_path / "data_unilateral"
    uids = _crops(images, _bd_spacings(8))
    (images / "processing.log").write_text(BD_LOG)
    issues = dl.check_crop_format(images, "unilateral", uids).issues
    assert any("names BD_unilateralize" in issue and "CONFIG=unilateral_BD" in issue for issue in issues)
    assert any("varies" in issue for issue in issues)


def test_varying_spacing_alone_is_flagged_under_unilateral(tmp_path):
    images = tmp_path / "data_unilateral"
    uids = _crops(images, _bd_spacings(5))
    issues = dl.check_crop_format(images, "unilateral", uids).issues
    assert len(issues) == 1 and "varies" in issues[0]


def test_missing_image_files_are_flagged(tmp_path):
    images = tmp_path / "data_unilateral"
    uids = _crops(images, [(1.0, 1.0, 1.0)] * 4)
    (images / "EXAM009_left").mkdir()
    verdict = dl.check_crop_format(images, "unilateral", uids + ["EXAM009_left"])
    assert any("1 of 5 sampled exam folders" in issue and "Sub_1.nii.gz" in issue for issue in verdict.issues)


def test_image_file_override_is_sampled(tmp_path):
    images = tmp_path / "data_unilateral_BD"
    uids = _crops(images, _bd_spacings(6), image_file="Sub_2.nii.gz")
    (images / "processing.log").write_text(BD_LOG)
    assert dl.check_crop_format(images, "unilateral_BD", uids, image_file="Sub_2.nii.gz").ok
    assert not dl.check_crop_format(images, "unilateral_BD", uids).ok          # Sub_1 missing


def test_unreadable_and_non_nifti_files_are_counted_not_fatal(tmp_path):
    images = tmp_path / "data_unilateral"
    uids = _crops(images, [(1.0, 1.0, 1.0)] * 3)
    (images / uids[0] / "Sub_1.nii.gz").write_bytes(b"not gzip")
    verdict = dl.check_crop_format(images, "unilateral", uids)
    assert verdict.ok and verdict.spacing.unreadable == 1 and verdict.spacing.read == 2


def test_original_config_is_not_checked(tmp_path):
    verdict = dl.check_crop_format(tmp_path, "original", [])
    assert not verdict.checked and verdict.ok and "not checked" in verdict.summary()


def test_sample_is_deterministic_and_spread_over_the_folder(tmp_path):
    uids = [f"EXAM{i:03d}_left" for i in range(100)]
    sample = dl._evenly_spaced(uids, 20)
    assert len(sample) == 20 == len(set(sample))
    assert sample[0] == "EXAM000_left" and sample[-1] == "EXAM099_left"
    assert sample == dl._evenly_spaced(list(reversed(uids)), 20)


def test_messages_never_contain_uids(tmp_path):
    """Guard output goes to the console and the coordinator; UIDs must stay at the site."""
    images = tmp_path / "data_unilateral"
    uids = _crops(images, _bd_spacings(8))
    (images / "processing.log").write_text(BD_LOG)
    (images / "EXAM042_left").mkdir()
    verdict = dl.check_crop_format(images, "unilateral", uids + ["EXAM042_left"])
    text = verdict.summary() + " ".join(verdict.issues)
    assert verdict.issues and "EXAM" not in text


# ---------------------------------------------------------------------------
# enforce_crop_format: warn by default, abort with ODELIA_STRICT_DATA_FORMAT=1
# ---------------------------------------------------------------------------

def _bad_verdict(tmp_path):
    images = tmp_path / "data_unilateral_BD"
    uids = _crops(images, [dl.FIXED_GRID_SPACING_MM] * 4)
    return dl.check_crop_format(images, "unilateral_BD", uids)


def test_ok_verdict_is_logged_once_at_info(tmp_path, caplog):
    images = tmp_path / "data_unilateral"
    uids = _crops(images, [(1.0, 1.0, 1.0)] * 3)
    log = logging.getLogger("test_data_layout")
    with caplog.at_level(logging.INFO, logger="test_data_layout"):
        dl.enforce_crop_format(dl.check_crop_format(images, "unilateral", uids), log, "SITE_1", env={})
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "ODELIA crop format for SITE_1: ok" in caplog.records[0].getMessage()


def test_issues_warn_and_continue_by_default(tmp_path, caplog):
    log = logging.getLogger("test_data_layout")
    with caplog.at_level(logging.INFO, logger="test_data_layout"):
        dl.enforce_crop_format(_bad_verdict(tmp_path), log, "SITE_1", env={})
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3                       # two issues + the summary
    assert "ODELIA_STRICT_DATA_FORMAT=1 would abort" in warnings[-1]


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_strict_mode_aborts(tmp_path, value):
    with pytest.raises(dl.CropFormatError) as exc:
        dl.enforce_crop_format(_bad_verdict(tmp_path), logging.getLogger("t"), "SITE_1",
                               env={"ODELIA_STRICT_DATA_FORMAT": value})
    message = str(exc.value)
    assert "ODELIA_STRICT_DATA_FORMAT=1" in message and "data_unilateral_BD" in message
    assert isinstance(exc.value, RuntimeError)


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_strict_mode_off(tmp_path, value):
    dl.enforce_crop_format(_bad_verdict(tmp_path), logging.getLogger("t"), env={"ODELIA_STRICT_DATA_FORMAT": value})


def test_strict_mode_does_not_abort_a_clean_folder(tmp_path):
    images = tmp_path / "data_unilateral_BD"
    uids = _crops(images, _bd_spacings(6))
    (images / "processing.log").write_text(BD_LOG)
    dl.enforce_crop_format(dl.check_crop_format(images, "unilateral_BD", uids), logging.getLogger("t"),
                           env={"ODELIA_STRICT_DATA_FORMAT": "1"})


def test_verdict_record_is_json_ready(tmp_path):
    import json
    record = _bad_verdict(tmp_path).as_dict()
    assert json.loads(json.dumps(record))["ok"] is False and record["issues"]
