from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchio as tio

from data.augmentation.augmentations_3d import (
    ImageOrSubjectToTensor,
    ZNormalization,
    CropOrPad,
    DegenerateImageError,
)
from data_layout import (
    CONFIG_FOLDERS,
    DEFAULT_IMAGE_FILE,
    CropFormatVerdict,
    DataLayout,
    canonical_config,
    check_crop_format,
    describe_layout,
    enforce_crop_format,
    metadata_fallback_warning,
    requested_layout,
    resolve_site_layout,
)

logger = logging.getLogger(__name__)


@dataclass
class ODELIAInstitutionManifest:
    institution: str
    path_root: Path
    config: str
    data_dir: str
    meta_dir: str
    annotation_df: pd.DataFrame
    split_df: pd.DataFrame
    image_uids: List[str]
    image_uids_set: set[str] = field(init=False)
    _uids_in_annotation: List[str] = field(init=False, repr=False)
    _uids_in_images: List[str] = field(init=False, repr=False)
    _uids_in_split_cache: Dict[Tuple[int, str | None], List[str]] = field(default_factory=dict, repr=False)
    _dataframe_cache: Dict[Tuple[int, str | None], pd.DataFrame] = field(default_factory=dict, repr=False)
    # Which file of each exam folder is the input (ODELIA_IMAGE_FILE), how data_dir/meta_dir
    # were chosen (CONFIG, overrides, metadata fallback) and what the crop-format guard found.
    # build_manifests fills all three; a hand-built manifest keeps the defaults.
    image_file: str = DEFAULT_IMAGE_FILE
    layout: DataLayout | None = None
    format_verdict: CropFormatVerdict | None = None

    def __post_init__(self):
        self.image_uids_set = set(self.image_uids)
        self._uids_in_annotation = sorted(self.annotation_df['UID'].tolist())
        self._uids_in_images = sorted(self.image_uids)

    @property
    def path_metadata(self) -> Path:
        return self.path_root / self.institution / self.meta_dir

    @property
    def path_data_root(self) -> Path:
        return self.path_root / self.institution / self.data_dir

    def dataframe(self, fold: int = 0, split: str | None = None, fraction=None) -> pd.DataFrame:
        cache_key = (fold, split)
        if cache_key not in self._dataframe_cache:
            df_split = self.split_df[self.split_df['Fold'] == fold]
            if split is not None:
                df_split = df_split[df_split['Split'] == split]
            merged = df_split.merge(self.annotation_df, on='UID', how='inner')
            merged = merged[merged['UID'].isin(self.image_uids_set)].copy()
            merged['Institution'] = self.institution
            self._dataframe_cache[cache_key] = merged.reset_index(drop=True)

        df = self._dataframe_cache[cache_key]
        if fraction is not None:
            return df.sample(frac=fraction, random_state=0).reset_index(drop=True)
        return df.copy()

    def uids_in_annotation(self) -> List[str]:
        return list(self._uids_in_annotation)

    def uids_in_images(self) -> List[str]:
        return list(self._uids_in_images)

    def uids_in_split(self, fold: int = 0, split: str | None = None) -> List[str]:
        cache_key = (fold, split)
        if cache_key not in self._uids_in_split_cache:
            df_split = self.split_df[self.split_df['Fold'] == fold]
            if split is not None:
                df_split = df_split[df_split['Split'] == split]
            self._uids_in_split_cache[cache_key] = sorted(df_split['UID'].tolist())
        return list(self._uids_in_split_cache[cache_key])

    def image_path(self, uid: str, filename: str | None = None) -> Path:
        return self.path_data_root / uid / (self.image_file if filename is None else filename)


class ODELIAStaticVolumeCache:
    """Optional scratch-side cache for decoded ODELIA NIfTI volumes.

    Entries live under ``<cache_root>/<namespace>/<institution>/<uid>.npz``. The namespace
    defaults to the config name; the loader passes DataLayout.cache_namespace, which also
    covers the image folder and file, so standard and BreastDivider crops of the same UID
    never share an entry.
    """

    def __init__(self, cache_root: str | Path, config: str, version: str = "v1", namespace: str | None = None):
        self.cache_root = Path(cache_root)
        self.config = config
        self.namespace = namespace or config
        self.version = version

    def load_or_create(self, institution: str, uid: str, path_img: Path, loader) -> tio.ScalarImage:
        cached = self._load_cached_image(institution, uid, path_img)
        if cached is not None:
            return cached

        image = loader(path_img)
        data = image.data.detach().cpu().numpy()
        affine = np.asarray(image.affine, dtype=np.float32)
        self._save_cached_image(institution, uid, path_img, data, affine)
        return image

    def _cache_paths(self, institution: str, uid: str) -> Tuple[Path, Path]:
        cache_dir = self.cache_root / self.namespace / institution
        data_path = cache_dir / f"{uid}.npz"
        meta_path = cache_dir / f"{uid}.json"
        return data_path, meta_path

    def _signature(self, path_img: Path) -> Dict[str, str | int]:
        stat = path_img.stat()
        return {
            "source_path": str(path_img.resolve()),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "cache_version": self.version,
        }

    def _load_cached_image(self, institution: str, uid: str, path_img: Path) -> tio.ScalarImage | None:
        data_path, meta_path = self._cache_paths(institution, uid)
        if not data_path.exists() or not meta_path.exists():
            return None

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        if metadata != self._signature(path_img):
            return None

        try:
            with np.load(data_path, allow_pickle=False) as arrays:
                data = arrays['data']
                affine = arrays['affine']
        except (OSError, KeyError, ValueError):
            return None

        tensor = torch.as_tensor(data, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tio.ScalarImage(tensor=tensor, affine=np.asarray(affine, dtype=np.float32))

    def _save_cached_image(self, institution: str, uid: str, path_img: Path, data: np.ndarray, affine: np.ndarray) -> None:
        data_path, meta_path = self._cache_paths(institution, uid)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        signature = self._signature(path_img)

        fd, tmp_npz = tempfile.mkstemp(prefix=f"{uid}_", suffix=".npz", dir=str(data_path.parent))
        os.close(fd)
        tmp_npz_path = Path(tmp_npz)
        tmp_meta_path = tmp_npz_path.with_suffix('.json')

        try:
            np.savez_compressed(tmp_npz_path, data=np.asarray(data, dtype=np.float32), affine=np.asarray(affine, dtype=np.float32))
            with open(tmp_meta_path, 'w', encoding='utf-8') as f:
                json.dump(signature, f, sort_keys=True)
            os.replace(tmp_npz_path, data_path)
            os.replace(tmp_meta_path, meta_path)
        finally:
            if tmp_npz_path.exists():
                tmp_npz_path.unlink(missing_ok=True)
            if tmp_meta_path.exists():
                tmp_meta_path.unlink(missing_ok=True)


class ODELIA_Dataset3D(data.Dataset):
    PATH_ROOT = Path('/data')
    ALL_INSTITUTIONS = ['CAM', 'MHA', 'RSH', 'UKA', 'UMCU', 'VHIO', 'RUMC', 'USZ']
    # Default folders per CONFIG (see data_layout.py). The loader resolves the effective
    # folders with resolve_site_layout(), which also applies the ODELIA_* overrides and the
    # unilateral_BD metadata fallback; these maps stay for callers that look them up.
    DATA_DIR = {config: folders[0] for config, folders in CONFIG_FOLDERS.items()}
    META_DIR = {config: folders[1] for config, folders in CONFIG_FOLDERS.items()}
    CLASS_LABELS = {
        'original': {
            'Lesion_Left': ['No', 'Benign', 'Malignant'],
            'Lesion_Right': ['No', 'Benign', 'Malignant'],
        },
        'unilateral': {
            'Lesion': ['No', 'Benign', 'Malignant'],
        },
        # Same unilateral crops and labels, only cut by BreastDivider: same transforms too.
        'unilateral_BD': {
            'Lesion': ['No', 'Benign', 'Malignant'],
        },
    }

    # Degenerate / corrupt input handling (see DegenerateImageError + find_degenerate_inputs).
    # A bad sample is skipped + resampled (never zero-filled), but the run aborts if too many
    # are bad -- so a systematic data problem is surfaced rather than silently worked around.
    SAME_ITEM_RETRIES = 2            # retry the same item (new random crop) before giving up on it
    MAX_BAD_INPUT_FRACTION = 0.05    # abort once more than this fraction of items are unusable
    MIN_BAD_INPUT_BUDGET = 8         # ...but always tolerate at least this many first

    def __init__(
            self,
            path_root=None,
            institutions=None,
            fold=0,
            labels=None,
            config=None,
            split=None,
            fraction=None,
            transform=None,
            random_flip=False,
            random_rotate=False,
            random_inverse=False,
            noise=False,
            to_tensor=True,
            manifests: Dict[str, ODELIAInstitutionManifest] | None = None,
            enable_preprocess_cache: bool = False,
            preprocess_cache_dir: str | Path | None = None,
            preprocess_cache_version: str = "v1",

    ):
        self.path_root = Path(self.PATH_ROOT if path_root is None else path_root)
        self.split = split
        # Validates CONFIG (unknown -> ValueError naming the valid values); None reads $CONFIG.
        config = canonical_config(config)
        self.config = config
        requested = requested_layout(config)
        self.class_labels = self.CLASS_LABELS[config]
        self.meta_dir = requested.metadata_folder
        self.data_dir = requested.data_folder
        self.labels = list(self.class_labels.keys()) if labels is None else labels
        self.class_labels_num = [len(self.class_labels[l]) for l in self.labels]
        self.institutions = self._normalize_institutions(institutions)

        if manifests is None:
            manifests = self.build_manifests(
                path_root=self.path_root,
                institutions=self.institutions,
                config=config,
            )
        self.manifests = manifests

        if transform is None:
            flip_axes = (0, 1) if config == "original" else (0, 1, 2)
            identity = tio.Lambda(lambda x: x)
            self.transform = tio.Compose([
                tio.ToCanonical() if config == "original" else identity,
                tio.Resample((0.7, 0.7, 3)) if config == "original" else identity,
                tio.Flip((1, 0)),
                CropOrPad((448, 448, 32), random_center=random_rotate) if config == "original" else CropOrPad(
                    (224, 224, 32), random_center=random_rotate),
                ZNormalization(
                    per_channel=True,
                    per_slice=False,
                    masking_method=lambda x: (x > x.min()) & (x < x.max()),
                    percentiles=(0.5, 99.5),
                ),
                tio.OneOf([
                    tio.RandomAffine(
                        scales=0,
                        degrees=(0, 0, 0, 0, 0, 90),
                        translation=0,
                        isotropic=True,
                        default_pad_value='minimum',
                    ) if random_rotate else identity,
                    tio.RandomFlip(flip_axes) if random_flip else identity,
                ]),
                tio.Lambda(
                    lambda x: -x if torch.rand((1,))[0] < 0.5 else x,
                    types_to_apply=[tio.INTENSITY],
                ) if random_inverse else identity,
                tio.RandomNoise(std=(0.0, 0.25)) if noise else identity,
                ImageOrSubjectToTensor() if to_tensor else identity
            ])
        elif transform == 'USE_UNPROCESSED_IMAGES':
            self.transform = tio.Compose([tio.Lambda(lambda x: x)])
            enable_preprocess_cache = False
        else:
            self.transform = transform
            enable_preprocess_cache = False

        if enable_preprocess_cache and preprocess_cache_dir:
            layouts = [m.layout for m in self.manifests.values() if getattr(m, 'layout', None) is not None]
            self.preprocess_cache = ODELIAStaticVolumeCache(
                cache_root=preprocess_cache_dir,
                config=config,
                version=preprocess_cache_version,
                namespace=(layouts[0] if layouts else requested).cache_namespace,
            )
        else:
            self.preprocess_cache = None

        dfs = []
        for institution in self.institutions:
            manifest = self.manifests[institution]
            institution_df = manifest.dataframe(fold=fold, split=split, fraction=fraction)
            self._require_non_empty(institution_df, manifest, institution, fold, split)
            dfs.append(institution_df)
        self.df = pd.concat(dfs).reset_index(drop=True)
        self._apply_uid_exclusions()
        self.item_pointers = self.df.index.tolist()
        self._bad_input_count = 0
        self._bad_input_logged: set = set()
        self._bad_input_reasons: dict = {}

    # Splits that must not be empty. An empty `train` eventually raises deep in
    # PyTorch ("num_samples should be a positive integer value") after minutes of
    # hashing; an empty `val`/`test` is worse -- it raises nothing at all, just
    # yields zero batches, so checkpoint selection silently monitors nothing (#411).
    REQUIRED_SPLITS = ("train", "val", "test")
    # Escape hatch for a site that deliberately holds no test data.
    ALLOW_EMPTY_SPLITS_ENV = "ODELIA_ALLOW_EMPTY_SPLITS"

    @classmethod
    def _splits_allowed_empty(cls):
        raw = os.environ.get(cls.ALLOW_EMPTY_SPLITS_ENV, "")
        return {s.strip() for s in raw.split(",") if s.strip()}

    @classmethod
    def _require_non_empty(cls, df, manifest, institution, fold, split):
        """Fail fast, and say what is actually wrong, when a fold/split is empty."""
        if len(df) > 0 or split is None:
            return
        if split not in cls.REQUIRED_SPLITS or split in cls._splits_allowed_empty():
            return

        folds_present = sorted(int(f) for f in manifest.split_df['Fold'].unique())
        split_csv = manifest.path_metadata / 'split.csv'
        if fold not in folds_present:
            reason = (f"split.csv contains folds {folds_present} -- fold {fold} is not among them")
        else:
            reason = (f"fold {fold} exists in split.csv, but no '{split}' rows survive the join with "
                      f"annotation.csv and the images actually present on disk. If the exam folders "
                      f"are symlinks, check the log above for a DANGLING symlink warning -- their "
                      f"targets must be mounted into the container too")
        raise RuntimeError(
            f"ODELIA: {institution} has no samples for FOLD={fold} / split='{split}'. "
            f"{reason}. Check {split_csv}. "
            f"(Set {cls.ALLOW_EMPTY_SPLITS_ENV}={split} only if this site genuinely has no {split} data.)"
        )

    def __len__(self):
        return len(self.item_pointers)

    def _apply_uid_exclusions(self):
        """Drop UIDs listed (one per line) in $ODELIA_EXCLUDE_UIDS_FILE -- the output of a
        preflight ``find_degenerate_inputs`` scan, so known-bad inputs never enter the run."""
        path = os.environ.get("ODELIA_EXCLUDE_UIDS_FILE", "")
        if not path or not os.path.exists(path):
            return
        with open(path) as f:
            excluded = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        if not excluded:
            return
        before = len(self.df)
        self.df = self.df[~self.df["UID"].isin(excluded)].reset_index(drop=True)
        removed = before - len(self.df)
        if removed:
            logger.warning("ODELIA: excluded %d sample(s) listed in %s (degenerate/corrupt inputs)",
                           removed, path)

    def _bad_input_log_path(self) -> Path:
        scratch = os.environ.get("SCRATCH_DIR") or os.environ.get("ODELIA_HASH_CACHE_DIR") or tempfile.gettempdir()
        return Path(scratch) / "odelia_bad_inputs.jsonl"

    def _record_bad_input(self, uid, institution, reason):
        self._bad_input_reasons[reason] = self._bad_input_reasons.get(reason, 0) + 1
        # Loud, non-identifying warning to the console; the UID goes only to a local jsonl.
        key = (uid, reason)
        if key not in self._bad_input_logged:
            self._bad_input_logged.add(key)
            logger.warning("[ODELIA_BAD_INPUT] reason=%s institution=%s split=%s "
                           "(skipping + resampling; UID in odelia_bad_inputs.jsonl)",
                           reason, institution, self.split)
        try:
            with open(self._bad_input_log_path(), "a") as f:
                json.dump({"uid": uid, "institution": institution, "reason": reason,
                           "split": self.split, "config": self.config}, f)
                f.write("\n")
        except Exception:
            pass

    @staticmethod
    def _format_bad_input_error(count, budget, frac, n, reasons, log_path):
        """Build the abort message with a per-reason breakdown + the right remedy.

        Unreadable inputs (PermissionError/OSError/...) are a file-permission problem
        and must NOT be excluded; genuinely degenerate images should be re-exported or
        excluded. Conflating them once led a site to delete valid data (#416)."""
        breakdown = ", ".join(f"{c}x {r}" for r, c in
                              sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
        access_errs = ("PermissionError", "OSError", "FileNotFoundError", "IsADirectoryError")
        unreadable = sum(c for r, c in reasons.items()
                         if r.startswith("load_error:") and any(t in r for t in access_errs))
        if unreadable and unreadable == count:
            hint = ("ALL failures are file-access errors -- this is a PERMISSION/OWNERSHIP problem "
                    "(the container cannot READ the files), NOT corrupt images. Fix the file "
                    "permissions/ownership; do NOT exclude the data.")
        elif unreadable:
            hint = (f"{unreadable} unreadable (fix file permissions) + {count - unreadable} degenerate "
                    "(re-export or exclude those UIDs).")
        else:
            hint = "Degenerate/corrupt image content -- re-export or exclude the listed UIDs."
        return (f"ODELIA: too many unusable inputs ({count} > {budget} = {frac:.0%} of {n}). "
                f"Reasons: {breakdown or 'unknown'}. {hint} Per-UID details: {log_path}")

    @classmethod
    def _normalize_institutions(cls, institutions) -> List[str]:
        if (institutions is None) or (institutions == "ODELIA"):
            return list(cls.ALL_INSTITUTIONS)
        if isinstance(institutions, str):
            return [institutions]
        return list(institutions)

    @classmethod
    def build_manifests(cls, path_root=None, institutions=None, config=None) -> Dict[str, ODELIAInstitutionManifest]:
        """Read each institution's metadata and list its exam folders.

        This is the one place that decides which folders and file are read (CONFIG plus the
        ODELIA_DATA_FOLDER / ODELIA_METADATA_FOLDER / ODELIA_IMAGE_FILE overrides, see
        data_layout.py), so it also logs that choice and runs the crop-format guard -- once per
        institution and setup, and for every caller: training, preflight and predict.py.
        ``config=None`` reads $CONFIG.
        """
        path_root = Path(cls.PATH_ROOT if path_root is None else path_root)
        config = canonical_config(config)
        normalized_institutions = cls._normalize_institutions(institutions)
        manifests: Dict[str, ODELIAInstitutionManifest] = {}

        for institution in normalized_institutions:
            site_dir = path_root / institution
            layout = resolve_site_layout(site_dir, config)
            if layout.metadata_fallback:
                logger.warning(metadata_fallback_warning(site_dir, layout))
            # Metadata first: a site without any data must still get the error that names
            # its metadata CSV (the CI preflight without data relies on it).
            path_metadata = site_dir / layout.metadata_folder
            annotation_df = pd.read_csv(path_metadata / 'annotation.csv', dtype={'UID': str, 'PatientID': str})
            split_df = pd.read_csv(path_metadata / 'split.csv', dtype={'UID': str})
            path_images = site_dir / layout.data_folder
            if not path_images.is_dir():
                if path_images.is_symlink():
                    hint = ("It is a symlink whose target is not visible inside the container; mount the target "
                            "too, e.g. MEDISWARM_DOCKER_OPTIONS=\"-v /abs/target:/abs/target:ro\".")
                elif config == 'unilateral_BD':
                    hint = ("BreastDivider crops belong in data_unilateral_BD next to data_unilateral, see "
                            "docs/BREASTDIVIDER_SITE_GUIDE.md.")
                else:
                    hint = "Check --data_dir and the CONFIG / ODELIA_DATA_FOLDER of the run."
                raise FileNotFoundError(
                    f"ODELIA: image folder {path_images} does not exist ({describe_layout(site_dir, layout)}). {hint}"
                )
            image_uids = cls.run_item_crawler(path_images)
            logger.info(f"ODELIA data for {institution}: {describe_layout(site_dir, layout, len(image_uids))}")
            verdict = check_crop_format(path_images, config, image_uids, layout.image_file)
            enforce_crop_format(verdict, logger, institution)
            manifests[institution] = ODELIAInstitutionManifest(
                institution=institution,
                path_root=path_root,
                config=config,
                data_dir=layout.data_folder,
                meta_dir=layout.metadata_folder,
                annotation_df=annotation_df,
                split_df=split_df,
                image_uids=image_uids,
                image_file=layout.image_file,
                layout=layout,
                format_verdict=verdict,
            )

        return manifests

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def get_image_path(self, uid: str, institution: str) -> Path:
        return self.manifests[institution].image_path(uid)

    def _load_source_image(self, path_img: Path, institution: str, uid: str):
        if self.preprocess_cache is None:
            return self.load_img(path_img)
        return self.preprocess_cache.load_or_create(
            institution=institution,
            uid=uid,
            path_img=path_img,
            loader=self.load_img,
        )

    def __getitem__(self, index):
        # Skip (resample a different item) for a degenerate or unreadable input instead of
        # crashing the run or -- worse -- silently substituting a zero image. The run still
        # aborts if more than MAX_BAD_INPUT_FRACTION of items are bad, so a systematic data
        # problem is surfaced. Real transform bugs (non-DegenerateImageError) still propagate.
        n = len(self.item_pointers)
        budget = max(self.MIN_BAD_INPUT_BUDGET, int(self.MAX_BAD_INPUT_FRACTION * n))
        for offset in range(n):
            idx = self.item_pointers[(index + offset) % n]
            item = self.df.loc[idx]
            uid = item['UID']
            institution = item['Institution']
            try:
                path_img = self.get_image_path(uid, institution)
                img = self._load_source_image(path_img, institution, uid)
            except Exception as e:  # noqa: BLE001 -- any load failure == an unreadable/corrupt input
                self._record_bad_input(uid, institution, f"load_error:{type(e).__name__}")
            else:
                try:
                    target = np.stack(item[self.labels].values)
                    img = self.transform(img)
                    return {'uid': uid, 'source': img, 'target': target}
                except DegenerateImageError as e:
                    self._record_bad_input(uid, institution, e.reason)
            self._bad_input_count += 1
            if self._bad_input_count > budget:
                raise RuntimeError(self._format_bad_input_error(
                    self._bad_input_count, budget, self.MAX_BAD_INPUT_FRACTION, n,
                    self._bad_input_reasons, self._bad_input_log_path()))
        raise RuntimeError(f"ODELIA: no usable input found while resampling around index {index} ({n} items)")

    def find_degenerate_inputs(self, limit: int | None = None):
        """Preflight input check: list inputs that cannot be loaded (corrupt) or z-normalized
        (constant volume / empty intensity mask). Loads each raw input once, WITHOUT
        augmentation -- so it flags bad *inputs* independently of cropping. Returns
        ``[(uid, institution, reason)]``; feed the UIDs to ``$ODELIA_EXCLUDE_UIDS_FILE`` to
        keep them out of a run, and report them to the site to fix."""
        bad = []
        for count, idx in enumerate(self.item_pointers):
            if limit is not None and count >= limit:
                break
            item = self.df.loc[idx]
            uid = item['UID']
            institution = item['Institution']
            try:
                img = self._load_source_image(self.get_image_path(uid, institution), institution, uid)
            except Exception as e:  # noqa: BLE001
                bad.append((uid, institution, f"load_error:{type(e).__name__}"))
                continue
            x = img.data.float()
            mask = (x > x.min()) & (x < x.max())
            n = int(mask.sum())
            if n < 2:
                bad.append((uid, institution, "empty_or_too_small_mask"))
            elif float(x[mask].std()) == 0.0:
                bad.append((uid, institution, "zero_std"))
        return bad

    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer, dtype={'UID': str})
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index(drop=True)
        return df

    @classmethod
    def run_item_crawler(cls, path_root, **kwargs):
        """List the exam directories under ``path_root``.

        ``DirEntry.is_dir()`` follows symlinks, so a link pointing at a real directory
        is included -- but a *dangling* link (target missing) is silently skipped. That
        is exactly what happens when a site stores its exams as symlinks to an absolute
        host path that is not mounted into the container: every entry disappears here,
        and the failure only surfaces much later as an opaque "has no samples for
        FOLD=.../split=..." error (#481). Count them and say so, loudly, right here.
        """
        found, dangling = [], []
        with os.scandir(path_root) as entries:
            for entry in entries:
                if entry.is_dir():
                    found.append(entry.name)
                elif entry.is_symlink() and not os.path.exists(entry.path):
                    dangling.append(entry.name)
        if dangling:
            shown = sorted(dangling)[:5]
            logger.warning(
                "ODELIA: %d of %d entries under %s are DANGLING symlinks and were skipped "
                "(e.g. %s%s). Their targets do not exist inside the container. If the exams are "
                "symlinks to an absolute host path, that path must ALSO be mounted, e.g. "
                "-v /abs/target:/abs/target:ro -- otherwise every split will come out empty.",
                len(dangling), len(dangling) + len(found), path_root,
                ", ".join(shown), ", ..." if len(dangling) > len(shown) else "",
            )
        return sorted(found)

    @classmethod
    def log_UID_discrepancies(
            cls,
            logger,
            path_root=None,
            institutions=None,
            fold=0,
            log_dataset_details=False,
            manifests: Dict[str, ODELIAInstitutionManifest] | None = None,
            config: str | None = None) -> None:

        def _log_duplicates(uids: List[str], where: str, logger, log_dataset_details) -> None:
            if len(uids) != len(set(uids)):
                logger.error(f'{len(uids) - len(set(uids))} duplicates among {where} UIDs detected, they should be unique')
                if log_dataset_details:
                    for uid in set(uids):
                        count = uids.count(uid)
                        if count > 1:
                            logger.error(f'{uid} appears {count} times')

        def _log_difference(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str, logger, log_dataset_details) -> None:
            difference = set(uids_a).difference(set(uids_b))
            if difference:
                logger.warning(f'{len(difference)} UIDs in {where_a} but not in {where_b} detected, make sure this was intended.')
                if log_dataset_details:
                    logger.warning(f'Difference {where_a}\\{where_b}: ' + ', '.join(sorted(difference)))

        def _log_differences(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str, logger, log_dataset_details) -> None:
            _log_difference(uids_a, uids_b, where_a, where_b, logger, log_dataset_details)
            _log_difference(uids_b, uids_a, where_b, where_a, logger, log_dataset_details)

        def _log_intersection(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str, logger, log_dataset_details) -> None:
            intersection = set(uids_a).intersection(set(uids_b))
            if intersection:
                logger.error(f'{len(intersection)} entries in {where_a}∩{where_b} detected, they should be in one set only.')
                if log_dataset_details:
                    logger.error(f'Entries in {where_a}∩{where_b}: ' + ', '.join(sorted(intersection)))

        def _log_left_right_discrepancies(uids: List[str], where: str, logger, log_dataset_details) -> None:
            def _log_neither_left_nor_right(uids: List[str], where: str, logger, log_dataset_details) -> None:
                neither_left_right = [u for u in uids if ( not u.endswith('_left') and not u.endswith('_right') )]
                if neither_left_right:
                    logger.error(f'{len(neither_left_right)} UIDs among {where} data present that do not end in _left or _right, this should not happen.')
                    if log_dataset_details:
                        neither_left_right.sort()
                        logger.error(f'The following UIDs among {where} data do not end in _left or _right: ' + ', '.join(neither_left_right))

            def _log_one_side_only(uids: List[str], where: str, logger, log_dataset_details) -> None:
                def _report(discrepancies: List[str], present: str, absent: str, where: str, logger, log_dataset_details) -> None:
                    if discrepancies:
                        logger.warning(f'{len(discrepancies)} UIDs among {where} data with {present} present and {absent} missing detected, make sure this was intended.')
                        if log_dataset_details:
                            discrepancies = list(discrepancies)
                            discrepancies.sort()
                            logger.warning(f'For the following UIDs among {where} data, {present} is present and {absent} is missing: ' + ', '.join(discrepancies))

                left  = [u[:-5] for u in uids if u.endswith('_left') ]
                right = [u[:-6] for u in uids if u.endswith('_right')]
                left_not_right = set(left).difference(set(right))
                right_not_left = set(right).difference(set(left))
                _report(left_not_right, '_left', '_right', where, logger, log_dataset_details)
                _report(right_not_left, '_right', '_left', where, logger, log_dataset_details)

            _log_neither_left_nor_right(uids, where, logger, log_dataset_details)
            _log_one_side_only(uids, where, logger, log_dataset_details)

        def _log_left_right_overlap(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str, logger, log_dataset_details) -> None:
            exam_ids_a = {u[:-5] for u in uids_a if u.endswith('_left') }
            exam_ids_b = {u[:-6] for u in uids_b if u.endswith('_right')}

            joint_exam_ids = exam_ids_a.intersection(exam_ids_b)
            if joint_exam_ids:
                logger.error(f'UIDs for the same exam found in {where_a} and {where_b}, this should not happen.')
                if log_dataset_details:
                    joint_exam_ids = list(joint_exam_ids)
                    joint_exam_ids.sort()
                    logger.error(f'The following exams are present in {where_a} and {where_b}: ' + ', '.join(joint_exam_ids))


        # Only used when no manifests are passed in; None reads $CONFIG (it used to be
        # hard-coded to 'unilateral', which would have scanned the wrong folders for BD runs).
        normalized_institutions = cls._normalize_institutions(institutions)
        if manifests is None:
            manifests = cls.build_manifests(path_root=path_root, institutions=normalized_institutions, config=config)

        for institution in normalized_institutions:
            manifest = manifests[institution]
            folds_present = sorted(int(f) for f in manifest.split_df['Fold'].unique())
            if fold not in folds_present:
                raise RuntimeError(
                    f"ODELIA: {institution} cannot train FOLD={fold} -- its split.csv contains "
                    f"folds {folds_present}. Check {manifest.path_metadata / 'split.csv'}."
                )
            uids_in_annotation = manifest.uids_in_annotation()
            uids_in_split = {
                split_name: manifest.uids_in_split(fold=fold, split=split_name)
                for split_name in (None, 'train', 'val', 'test')
            }
            uids_in_images = manifest.uids_in_images()

            if log_dataset_details:
                logger.info('Annoation UIDs: ' + ' '.join(uids_in_annotation))
                logger.info('All split UIDs: ' + ' '.join(uids_in_split[None]))
                logger.info('Training UIDs: ' + ' '.join(uids_in_split['train']))
                logger.info('Validation UIDs: ' + ' '.join(uids_in_split['val']))
                logger.info('Test UIDs: ' + ' '.join(uids_in_split['test']))
                logger.info('Image UIDs ' + ' '.join(uids_in_images))

            for uids, where in (
                    (uids_in_annotation, 'annotation'),
                    (uids_in_split[None], 'all split'),
                    (uids_in_split['train'], 'training'),
                    (uids_in_split['val'], 'validation'),
                    (uids_in_split['test'], 'test'),
                    (uids_in_images, 'image'),
            ):
                _log_duplicates(uids, where, logger, log_dataset_details)

            _log_differences(uids_in_annotation,  uids_in_split[None], 'annotation', 'split',  logger, log_dataset_details)
            _log_differences(uids_in_split[None], uids_in_images,      'split',      'images', logger, log_dataset_details)
            _log_differences(uids_in_annotation,  uids_in_images,      'annotation', 'images', logger, log_dataset_details)

            _log_intersection(uids_in_split['train'], uids_in_split['val'],  'training',   'validation', logger, log_dataset_details)
            _log_intersection(uids_in_split['train'], uids_in_split['test'], 'training',   'test',       logger, log_dataset_details)
            _log_intersection(uids_in_split['val'],   uids_in_split['test'], 'validation', 'test',       logger, log_dataset_details)

            _log_left_right_discrepancies(uids_in_split['train'], 'training',   logger, log_dataset_details)
            _log_left_right_discrepancies(uids_in_split['val'],   'validation', logger, log_dataset_details)
            _log_left_right_discrepancies(uids_in_split['test'],  'test',       logger, log_dataset_details)

            _log_left_right_overlap(uids_in_split['train'], uids_in_split['val'],  'training',   'validation', logger, log_dataset_details)
            _log_left_right_overlap(uids_in_split['train'], uids_in_split['test'], 'training',   'test',       logger, log_dataset_details)
            _log_left_right_overlap(uids_in_split['val'],   uids_in_split['test'], 'validation', 'test',       logger, log_dataset_details)
