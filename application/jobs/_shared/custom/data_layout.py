"""Which folders and which image file the ODELIA loader reads, and whether the crops found
there look like the format the run expects.

Standard library only, on purpose: the loader (data/datasets/dataset_3d_odelia.py),
env_config.py and the site script scripts/site/reconcile_odelia_dataset.py all use it, and
the site script also runs on hosts that have neither torch nor pandas.

Per site folder (``/data/<SITE>/`` inside the container)::

    CONFIG          images               labels                    image file
    original        data/                metadata/                 Sub_1.nii.gz
    unilateral      data_unilateral/     metadata_unilateral/      Sub_1.nii.gz   (default)
    unilateral_BD   data_unilateral_BD/  metadata_unilateral_BD/   Sub_1.nii.gz
                                         (else metadata_unilateral/, with a warning)

ODELIA_DATA_FOLDER, ODELIA_METADATA_FOLDER and ODELIA_IMAGE_FILE override the three
columns, so a later preprocessing variant can be read without building a new image.

Two crop formats with identical tensor shapes (256 x 256 x 32 voxels) exist: the standard
crops (step3_unilateral.py, a fixed 0.7 x 0.7 x 3 mm grid) and BreastDivider crops (MEVIS
odelia-preprocessing 0.2.x, a box fitted to each breast, so the spacing varies per case).
Mixed within one model they cost about 0.06 AUROC without raising any error
(docs/PREPROCESSING_BENCHMARK_BREASTDIVIDER.md, section 4.3). check_crop_format() therefore
looks for evidence of the wrong format: the processing.log BreastDivider writes next to its
crops, and the spacing in a sample of NIfTI headers. It warns; with
ODELIA_STRICT_DATA_FORMAT=1 the loader aborts instead.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence, Tuple

CONFIG_ENV = "CONFIG"
DATA_FOLDER_ENV = "ODELIA_DATA_FOLDER"
METADATA_FOLDER_ENV = "ODELIA_METADATA_FOLDER"
IMAGE_FILE_ENV = "ODELIA_IMAGE_FILE"
STRICT_ENV = "ODELIA_STRICT_DATA_FORMAT"

DEFAULT_CONFIG = "unilateral"
DEFAULT_IMAGE_FILE = "Sub_1.nii.gz"

# CONFIG -> (images folder, metadata folder, metadata folder to fall back to).
CONFIG_FOLDERS = {
    "original": ("data", "metadata", None),
    "unilateral": ("data_unilateral", "metadata_unilateral", None),
    # BreastDivider writes crops but no labels. It names its crops <CaseID>_left/_right from
    # the same bilateral input as the standard crops, so the standard metadata fits unless a
    # site adds exams (e.g. single-breast exams the midline split could not serve).
    "unilateral_BD": ("data_unilateral_BD", "metadata_unilateral_BD", "metadata_unilateral"),
}
VALID_CONFIGS = tuple(CONFIG_FOLDERS)

# The grid of the standard crops: step3_unilateral.py resamples to target_spacing
# (0.7, 0.7, 3) before cropping. BreastDivider starts from the same TARGET_SPACING but fits
# and resizes the box per breast; none of the 1,290 benchmark crops kept it.
FIXED_GRID_SPACING_MM = (0.7, 0.7, 3.0)
# NIfTI stores pixdim as float32 (0.7 reads back as 0.699999988), so compare rounded values.
SPACING_DECIMALS = 3
GUARD_SAMPLE_SIZE = 20

PROCESSING_LOG = "processing.log"
# 0.2.1 and 0.2.2 produce identical crops (benchmark section 5); a later version has not
# been validated against the models of this release.
BD_ACCEPTED_VERSIONS = ("0.2.1", "0.2.2")
BD_EXPECTED_MODE = "normal"
BD_EXPECTED_XY_RESOLUTION = "256"
# A site-wide processing.log has a line per case and output file (20,000 lines for 456
# exams). Everything the guard needs is in the first lines (command, version, commit,
# worker command) and the last ones (segmentation report), so only head and tail are read.
_LOG_SCAN_BYTES = 256 * 1024

_TRUE_VALUES = {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Folder selection
# ---------------------------------------------------------------------------

def canonical_config(config: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """Validate CONFIG and return its canonical spelling.

    ``None`` reads $CONFIG; unset or blank means 'unilateral'. Case is ignored
    (``--config unilateral_bd`` is a typo, not another format). Anything else raises a
    ValueError naming the valid values: the loader used to die with a bare KeyError.
    """
    if config is None:
        config = (os.environ if env is None else env).get(CONFIG_ENV, "")
    raw = str(config).strip()
    if not raw:
        return DEFAULT_CONFIG
    for name in VALID_CONFIGS:
        if raw.lower() == name.lower():
            return name
    raise ValueError(
        f"Unknown CONFIG={raw!r}. Valid values: {', '.join(VALID_CONFIGS)}. To read other "
        f"folders keep one of these and set {DATA_FOLDER_ENV}, {METADATA_FOLDER_ENV} or {IMAGE_FILE_ENV}."
    )


def _relative_override(env: Mapping[str, str], name: str) -> str | None:
    """An override names a path inside the site folder /data/<SITE>/. Another site folder is
    INSTITUTION's job, and an absolute host path is not mounted into the container, so both
    are refused here rather than failing later with a less obvious error."""
    raw = env.get(name, "") or ""
    if not raw.strip():
        return None
    path = PurePosixPath(raw.strip())
    if path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError(f"{name}={raw!r} must be a path relative to the site folder /data/<SITE>/, without '..'")
    return str(path)


@dataclass(frozen=True)
class DataLayout:
    """The folders (relative to /data/<SITE>/) and the image file one site's data is read from."""

    config: str
    data_folder: str
    metadata_folder: str
    image_file: str = DEFAULT_IMAGE_FILE
    overrides: Tuple[str, ...] = ()            # environment variables that replaced a default
    metadata_fallback_from: str | None = None  # the metadata folder that was missing, if any

    @property
    def metadata_fallback(self) -> bool:
        return self.metadata_fallback_from is not None

    @property
    def cache_namespace(self) -> str:
        """Directory name for the preprocess cache of this layout.

        The cache keys decoded volumes by UID, and UIDs are identical across crop formats,
        so the image folder and file must be part of the key. Each CONFIG's default layout
        keeps the plain CONFIG name, which is where existing caches already live.
        """
        if self.data_folder == CONFIG_FOLDERS[self.config][0] and self.image_file == DEFAULT_IMAGE_FILE:
            return self.config
        key = f"{self.config}|{self.data_folder}|{self.image_file}"
        readable = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{self.config}__{self.data_folder}__{self.image_file}")
        return f"{readable}__{hashlib.sha256(key.encode('utf-8')).hexdigest()[:8]}"

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "data_folder": self.data_folder,
            "metadata_folder": self.metadata_folder,
            "image_file": self.image_file,
            "overrides": list(self.overrides),
            "metadata_fallback_from": self.metadata_fallback_from,
        }


def requested_layout(config: str | None = None, env: Mapping[str, str] | None = None) -> DataLayout:
    """The layout CONFIG and the ODELIA_* overrides ask for, before looking at the disk."""
    env = os.environ if env is None else env
    config = canonical_config(config, env)
    data_default, metadata_default, _ = CONFIG_FOLDERS[config]
    chosen = {name: _relative_override(env, name) for name in (DATA_FOLDER_ENV, METADATA_FOLDER_ENV, IMAGE_FILE_ENV)}
    return DataLayout(
        config=config,
        data_folder=chosen[DATA_FOLDER_ENV] or data_default,
        metadata_folder=chosen[METADATA_FOLDER_ENV] or metadata_default,
        image_file=chosen[IMAGE_FILE_ENV] or DEFAULT_IMAGE_FILE,
        overrides=tuple(name for name, value in chosen.items() if value is not None),
    )


def resolve_site_layout(site_dir, config: str | None = None, env: Mapping[str, str] | None = None) -> DataLayout:
    """The layout actually used for one site folder.

    Only a CONFIG default can fall back (unilateral_BD: metadata_unilateral_BD missing ->
    metadata_unilateral); an explicit ODELIA_METADATA_FOLDER never does, and neither does a
    metadata folder that exists but is incomplete -- a folder the site created is meant to
    be used, and the CSV error then names it.
    """
    layout = requested_layout(config, env)
    fallback = CONFIG_FOLDERS[layout.config][2]
    if fallback is None or METADATA_FOLDER_ENV in layout.overrides:
        return layout
    site_dir = Path(site_dir)
    if (site_dir / layout.metadata_folder).is_dir() or not (site_dir / fallback).is_dir():
        return layout
    return dataclasses.replace(layout, metadata_folder=fallback, metadata_fallback_from=layout.metadata_folder)


def describe_layout(site_dir, layout: DataLayout, num_exam_folders: int | None = None) -> str:
    """One log line: the effective folders and file, and why they differ from the defaults."""
    site_dir = Path(site_dir)
    images = f"images {site_dir / layout.data_folder}"
    if num_exam_folders is not None:
        images += f" ({num_exam_folders} exam folders)"
    text = (f"CONFIG={layout.config}, {images}, metadata {site_dir / layout.metadata_folder}, "
            f"image file {layout.image_file}")
    if layout.metadata_fallback:
        text += f", metadata fallback ({layout.metadata_fallback_from} not found)"
    if layout.overrides:
        text += f", overridden by {', '.join(layout.overrides)}"
    return text


def metadata_fallback_warning(site_dir, layout: DataLayout) -> str:
    site_dir = Path(site_dir)
    return (f"ODELIA: {site_dir / layout.metadata_fallback_from} not found, using {site_dir / layout.metadata_folder} "
            f"for CONFIG={layout.config}. BreastDivider keeps the <CaseID>_left/_right names, so the labels fit "
            f"as long as the crops come from the same exams. Copy the folder to "
            f"{layout.metadata_fallback_from}/ to make the choice explicit.")


# ---------------------------------------------------------------------------
# Crop-format evidence
# ---------------------------------------------------------------------------

def read_nifti_spacing(path) -> Tuple[float, float, float] | None:
    """Voxel spacing in mm from a NIfTI-1 or NIfTI-2 header; reads the header only.

    Returns None for a file that is not NIfTI (ODELIA_IMAGE_FILE may name another format;
    the guard then skips the spacing check). Raises OSError or ValueError for a NIfTI file
    that cannot be read.
    """
    name = str(path)
    if name.endswith(".nii.gz"):
        opener = gzip.open
    elif name.endswith(".nii"):
        opener = open
    else:
        return None
    with opener(path, "rb") as fh:
        head = fh.read(540)
    if len(head) < 348:
        raise ValueError(f"truncated NIfTI header ({len(head)} bytes)")
    for endian in "<>":
        (sizeof_hdr,) = struct.unpack(endian + "i", head[:4])
        if sizeof_hdr == 348:                                   # NIfTI-1
            dim = struct.unpack(endian + "8h", head[40:56])
            pixdim = struct.unpack(endian + "8f", head[76:108])
            break
        if sizeof_hdr == 540 and len(head) >= 540:              # NIfTI-2
            dim = struct.unpack(endian + "8q", head[16:80])
            pixdim = struct.unpack(endian + "8d", head[104:168])
            break
    else:
        raise ValueError("not a NIfTI-1 or NIfTI-2 header")
    if dim[0] < 3:
        raise ValueError(f"NIfTI image has {dim[0]} dimensions, expected at least 3")
    return tuple(abs(float(p)) for p in pixdim[1:4])


@dataclass(frozen=True)
class ProcessingLog:
    """What a processing.log next to the crops says about the tool that wrote them."""

    path: str
    tool: str                          # 'breastdivider', 'unilateralize' (MEVIS' standard-crop script),
                                       # 'unknown', or 'unreadable' (then `error` says why)
    package_version: str | None = None
    git_commit: str | None = None
    mode: str | None = None
    xy_resolution: str | None = None
    untrusted_fallbacks: int | None = None
    error: str | None = None

    def describe(self) -> str:
        if self.tool == "unreadable":
            return f"{PROCESSING_LOG} unreadable ({self.error})"
        if self.tool == "unknown":
            return f"{PROCESSING_LOG} from an unknown tool"
        name = "BreastDivider" if self.tool == "breastdivider" else "unilateralize.py"
        details = [f"commit {self.git_commit}" if self.git_commit else None,
                   f"mode {self.mode}" if self.mode else None,
                   f"xy {self.xy_resolution}" if self.xy_resolution else None,
                   f"{self.untrusted_fallbacks} untrusted-fallback case(s)" if self.untrusted_fallbacks is not None
                   else None]
        details = [d for d in details if d]
        return (f"{PROCESSING_LOG}: {name} {self.package_version or '(version not recorded)'}"
                + (f" ({', '.join(details)})" if details else ""))


def _head_and_tail(path: Path, size: int = _LOG_SCAN_BYTES) -> str:
    """The whole file if it is small, else its first and last ``size`` bytes, cut at line
    boundaries so that no truncated line can yield a truncated value."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        end = fh.tell()
        fh.seek(0)
        if end <= 2 * size:
            return fh.read().decode("utf-8", errors="replace")
        head = fh.read(size)
        fh.seek(end - size)
        tail = fh.read(size)
    head = head[: head.rfind(b"\n") + 1]
    tail = tail[tail.find(b"\n") + 1:]
    return (head + tail).decode("utf-8", errors="replace")


def _last_match(pattern: str, text: str) -> str | None:
    value = None
    for match in re.finditer(pattern, text):
        value = next((group for group in match.groups() if group is not None), None)
    return value


def read_processing_log(images_dir) -> ProcessingLog | None:
    """Parse ``<images_dir>/processing.log``; None if there is none.

    BreastDivider's crop step (BD_main.py, worker BD_unilateralize.py) records the command,
    'Package version', 'Git commit' and finally a 'Segmentation report'. MEVIS also ships
    unilateralize.py, which writes the *standard* crops with a processing.log of the same
    package version -- so the version alone proves nothing; the BD_unilateralize worker
    does. When a log was appended to by several runs, the last value wins.
    """
    path = Path(images_dir) / PROCESSING_LOG
    if not path.is_file():
        return None
    try:
        text = _head_and_tail(path)
    except OSError as exc:
        return ProcessingLog(path=str(path), tool="unreadable", error=f"{type(exc).__name__}: {exc.strerror or exc}")
    if "BD_unilateralize" in text or "BD_main.py" in text:
        tool = "breastdivider"
    elif re.search(r"\bunilateralize\.py\b", text):
        tool = "unilateralize"
    else:
        tool = "unknown"
    fallbacks = _last_match(r"Segmentation report:\s*(\d+)\s+untrusted", text)
    return ProcessingLog(
        path=str(path),
        tool=tool,
        package_version=_last_match(r"Package version:\s*(\S+)", text),
        git_commit=_last_match(r"Git commit:\s*(\S+)", text),
        mode=_last_match(r"--mode[= ]([A-Za-z_]+)|\bmode\s+:\s*([A-Za-z_]+)", text),
        xy_resolution=_last_match(r"--xy_resolution[= ](\d+)|\bxy_resolution\s+:\s*(\d+)", text),
        untrusted_fallbacks=int(fallbacks) if fallbacks is not None else None,
    )


@dataclass(frozen=True)
class SpacingSample:
    """Voxel spacings read from the headers of a deterministic sample of crops."""

    requested: int = 0                                 # exam folders sampled
    spacings: Tuple[Tuple[float, float, float], ...] = ()  # rounded, one per readable header
    missing: int = 0                                   # sampled folders without the image file
    unreadable: int = 0
    unsupported: int = 0                               # not NIfTI: spacing not checked

    @property
    def read(self) -> int:
        return len(self.spacings)

    @property
    def distinct(self) -> int:
        return len(set(self.spacings))

    @property
    def on_fixed_grid(self) -> int:
        return sum(1 for spacing in self.spacings if spacing == FIXED_GRID_SPACING_MM)

    def ranges(self) -> str:
        axes = list(zip(*self.spacings))
        return ", ".join(f"{name} {min(values):.2f}-{max(values):.2f}" for name, values in zip("xyz", axes)) + " mm"

    def describe(self) -> str:
        if not self.spacings:
            text = f"spacing not checked (no readable NIfTI header among {self.requested} sampled crops)"
        elif self.distinct == 1:
            x, y, z = self.spacings[0]
            text = f"spacing of {self.read} sampled crops: all {x:.2f} x {y:.2f} x {z:.2f} mm"
        else:
            text = f"spacing of {self.read} sampled crops varies: {self.ranges()}"
        problems = [f"{self.missing} without the image file" if self.missing else None,
                    f"{self.unreadable} unreadable" if self.unreadable else None,
                    f"{self.unsupported} not NIfTI" if self.unsupported else None]
        problems = [p for p in problems if p]
        return text + (f" ({', '.join(problems)})" if problems else "")


def _evenly_spaced(items: Iterable[str], count: int) -> list:
    """Every k-th item of the sorted list: the same sample on every run and at every site."""
    items = sorted(items)
    if len(items) <= count:
        return items
    if count <= 1:
        return items[:count]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def sample_spacing(images_dir, uids: Sequence[str], image_file: str = DEFAULT_IMAGE_FILE,
                   sample_size: int = GUARD_SAMPLE_SIZE) -> SpacingSample:
    images_dir = Path(images_dir)
    sampled = _evenly_spaced(uids, sample_size)
    spacings, missing, unreadable, unsupported = [], 0, 0, 0
    for uid in sampled:
        path = images_dir / uid / image_file
        if not path.is_file():
            missing += 1
            continue
        try:
            spacing = read_nifti_spacing(path)
        except Exception:  # noqa: BLE001 -- OSError, EOFError, zlib.error, ValueError, ...: all just "unreadable"
            # The loader skips a corrupt input and aborts only past a budget; one bad file
            # must not turn this check into a crash of the whole setup.
            unreadable += 1
            continue
        if spacing is None:
            unsupported += 1
            continue
        spacings.append(tuple(round(value, SPACING_DECIMALS) for value in spacing))
    return SpacingSample(len(sampled), tuple(spacings), missing, unreadable, unsupported)


@dataclass(frozen=True)
class CropFormatVerdict:
    config: str
    images_dir: str
    checked: bool = True
    issues: Tuple[str, ...] = ()
    processing_log: ProcessingLog | None = None
    spacing: SpacingSample | None = None

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        if not self.checked:
            return f"not checked for CONFIG={self.config}"
        verdict = "ok" if self.ok else f"{len(self.issues)} issue(s)"
        if self.spacing is None:                  # the check could not gather its evidence
            return verdict
        evidence = [self.processing_log.describe() if self.processing_log else f"no {PROCESSING_LOG}",
                    self.spacing.describe()]
        return f"{verdict}; " + "; ".join(evidence)

    def as_dict(self) -> dict:
        return {"config": self.config, "images_dir": self.images_dir, "checked": self.checked, "ok": self.ok,
                "issues": list(self.issues), "summary": self.summary()}


def _breastdivider_issues(images_dir: Path, log: ProcessingLog | None, spacing: SpacingSample) -> list:
    issues = []
    if log is None:
        issues.append(f"no {PROCESSING_LOG} in {images_dir}: nothing shows these are BreastDivider crops "
                      f"(BD_main.py writes one next to its crops)")
    elif log.tool == "unreadable":
        issues.append(f"could not read {log.path} ({log.error}), so nothing shows these are BreastDivider crops")
    elif log.tool != "breastdivider":
        issues.append(f"{log.path} does not name BD_unilateralize, so these crops were not written by BreastDivider"
                      + (" (it names unilateralize.py, which writes the standard crops)" if log.tool == "unilateralize"
                         else ""))
    else:
        if log.package_version not in BD_ACCEPTED_VERSIONS:
            issues.append(f"BreastDivider package version {log.package_version or 'not recorded'} in {log.path}; "
                          f"this release accepts {' or '.join(BD_ACCEPTED_VERSIONS)}")
        if log.mode is not None and log.mode != BD_EXPECTED_MODE:
            issues.append(f"BreastDivider ran with --mode {log.mode}; the consortium uses --mode {BD_EXPECTED_MODE}")
        if log.xy_resolution is not None and log.xy_resolution != BD_EXPECTED_XY_RESOLUTION:
            issues.append(f"BreastDivider ran with --xy_resolution {log.xy_resolution}; the consortium uses the "
                          f"default {BD_EXPECTED_XY_RESOLUTION}")
    if spacing.read and spacing.on_fixed_grid == spacing.read:
        issues.append(f"all {spacing.read} sampled crops in {images_dir} have the fixed 0.7 x 0.7 x 3 mm spacing of "
                      f"the standard crops; BreastDivider fits the box to each breast, so its spacing varies")
    elif spacing.read >= 3 and spacing.distinct == 1:
        x, y, z = spacing.spacings[0]
        issues.append(f"all {spacing.read} sampled crops in {images_dir} share one spacing ({x:.2f} x {y:.2f} x "
                      f"{z:.2f} mm); BreastDivider fits the box to each breast, so its spacing varies")
    return issues


def _standard_issues(images_dir: Path, log: ProcessingLog | None, spacing: SpacingSample) -> list:
    issues = []
    if log is not None and log.tool == "breastdivider":
        issues.append(f"{log.path} names BD_unilateralize: these look like BreastDivider crops. Keep them in "
                      f"data_unilateral_BD and train them with CONFIG=unilateral_BD")
    if spacing.distinct > 1:
        issues.append(f"the spacing of the {spacing.read} sampled crops in {images_dir} varies ({spacing.ranges()}) "
                      f"like BreastDivider crops; the standard crops share one grid (0.7 x 0.7 x 3 mm)")
    return issues


def check_crop_format(images_dir, config: str, uids: Sequence[str], image_file: str = DEFAULT_IMAGE_FILE,
                      sample_size: int = GUARD_SAMPLE_SIZE) -> CropFormatVerdict:
    """Look for evidence that the crops in ``images_dir`` are not in the format CONFIG expects.

    unilateral_BD expects a processing.log naming BD_unilateralize (accepted package version,
    --mode normal, xy resolution 256) and a spacing that varies between crops. unilateral
    expects no BreastDivider log and one spacing for all crops. 'original' (bilateral, the
    loader resamples it) is not checked. Messages name folders and counts, never UIDs: they
    go to the console and to the log the coordinator sees.
    """
    config = canonical_config(config)
    images_dir = Path(images_dir)
    if config == "original":
        return CropFormatVerdict(config, str(images_dir), checked=False)
    try:
        log = read_processing_log(images_dir)
        spacing = sample_spacing(images_dir, uids, image_file, sample_size)
    except Exception as exc:  # noqa: BLE001 -- a check that cannot run is an issue, not a crash
        return CropFormatVerdict(config, str(images_dir), True,
                                 (f"the crop-format check could not run on {images_dir}: {type(exc).__name__}: {exc}",))
    issues = []
    if spacing.missing:
        issues.append(f"{spacing.missing} of {spacing.requested} sampled exam folders in {images_dir} have no "
                      f"{image_file}")
    if config == "unilateral_BD":
        issues += _breastdivider_issues(images_dir, log, spacing)
    else:
        issues += _standard_issues(images_dir, log, spacing)
    return CropFormatVerdict(config, str(images_dir), True, tuple(issues), log, spacing)


def strict_data_format(env: Mapping[str, str] | None = None) -> bool:
    return ((os.environ if env is None else env).get(STRICT_ENV, "") or "").strip().lower() in _TRUE_VALUES


class CropFormatError(RuntimeError):
    """The crops do not look like CONFIG says, and ODELIA_STRICT_DATA_FORMAT=1."""


def enforce_crop_format(verdict: CropFormatVerdict, logger, institution: str = "",
                        env: Mapping[str, str] | None = None) -> None:
    """Log the verdict once; warn per issue, or raise CropFormatError in strict mode."""
    prefix = "ODELIA crop format" + (f" for {institution}" if institution else "")
    if verdict.ok:
        logger.info(f"{prefix}: {verdict.summary()}")
        return
    for issue in verdict.issues:
        logger.warning(f"{prefix}: {issue}")
    message = f"{prefix} (CONFIG={verdict.config}): {verdict.summary()}"
    if strict_data_format(env):
        raise CropFormatError(f"{message}. Aborting because {STRICT_ENV}=1. Issues: " + " | ".join(verdict.issues)
                              + ". See docs/BREASTDIVIDER_SITE_GUIDE.md.")
    logger.warning(f"{message}. Continuing; {STRICT_ENV}=1 would abort here.")
