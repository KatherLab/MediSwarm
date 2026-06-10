from __future__ import annotations

import json
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

from data.augmentation.augmentations_3d import ImageOrSubjectToTensor, ZNormalization, CropOrPad


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

    def image_path(self, uid: str, filename: str = 'Sub_1.nii.gz') -> Path:
        return self.path_data_root / uid / filename


class ODELIAStaticVolumeCache:
    """Optional scratch-side cache for decoded ODELIA NIfTI volumes."""

    def __init__(self, cache_root: str | Path, config: str, version: str = "v1"):
        self.cache_root = Path(cache_root)
        self.config = config
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
        cache_dir = self.cache_root / self.config / institution
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
    DATA_DIR = {
        'original': 'data',
        'unilateral': 'data_unilateral'
    }
    META_DIR = {
        'original': 'metadata',
        'unilateral': 'metadata_unilateral'
    }
    CLASS_LABELS = {
        'original': {
            'Lesion_Left': ['No', 'Benign', 'Malignant'],
            'Lesion_Right': ['No', 'Benign', 'Malignant'],
        },
        'unilateral': {
            'Lesion': ['No', 'Benign', 'Malignant'],
        }
    }

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
        self.config = config
        self.class_labels = self.CLASS_LABELS[config]
        self.meta_dir = self.META_DIR[config]
        self.data_dir = self.DATA_DIR[config]
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
            self.preprocess_cache = ODELIAStaticVolumeCache(
                cache_root=preprocess_cache_dir,
                config=config,
                version=preprocess_cache_version,
            )
        else:
            self.preprocess_cache = None

        dfs = []
        for institution in self.institutions:
            manifest = self.manifests[institution]
            dfs.append(manifest.dataframe(fold=fold, split=split, fraction=fraction))
        self.df = pd.concat(dfs).reset_index(drop=True)
        self.item_pointers = self.df.index.tolist()

    def __len__(self):
        return len(self.item_pointers)

    @classmethod
    def _normalize_institutions(cls, institutions) -> List[str]:
        if (institutions is None) or (institutions == "ODELIA"):
            return list(cls.ALL_INSTITUTIONS)
        if isinstance(institutions, str):
            return [institutions]
        return list(institutions)

    @classmethod
    def build_manifests(cls, path_root=None, institutions=None, config='unilateral') -> Dict[str, ODELIAInstitutionManifest]:
        path_root = Path(cls.PATH_ROOT if path_root is None else path_root)
        normalized_institutions = cls._normalize_institutions(institutions)
        manifests: Dict[str, ODELIAInstitutionManifest] = {}

        for institution in normalized_institutions:
            path_metadata = path_root / institution / cls.META_DIR[config]
            annotation_df = pd.read_csv(path_metadata / 'annotation.csv', dtype={'UID': str, 'PatientID': str})
            split_df = pd.read_csv(path_metadata / 'split.csv', dtype={'UID': str})
            image_uids = cls.run_item_crawler(path_root / institution / cls.DATA_DIR[config])
            manifests[institution] = ODELIAInstitutionManifest(
                institution=institution,
                path_root=path_root,
                config=config,
                data_dir=cls.DATA_DIR[config],
                meta_dir=cls.META_DIR[config],
                annotation_df=annotation_df,
                split_df=split_df,
                image_uids=image_uids,
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

    @staticmethod
    def _attach_source_path(image: tio.ScalarImage, path_img: Path) -> tio.ScalarImage:
        # TorchIO keeps `.path` for file-backed images, but tensor-backed cached
        # images need it restored so local diagnostics can point to the source.
        image.path = str(path_img)
        return image

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        uid = item['UID']
        institution = item['Institution']

        target = np.stack(item[self.labels].values)
        path_img = self.get_image_path(uid, institution)
        img = self._attach_source_path(
            self._load_source_image(path_img, institution, uid),
            path_img,
        )
        img = self.transform(img)

        return {'uid': uid, 'source': img, 'target': target}

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
        with os.scandir(path_root) as entries:
            return sorted(entry.name for entry in entries if entry.is_dir())

    @classmethod
    def log_UID_discrepancies(
            cls,
            logger,
            path_root=None,
            institutions=None,
            fold=0,
            log_dataset_details=False,
            manifests: Dict[str, ODELIAInstitutionManifest] | None = None) -> None:

        def _log_duplicates(uids: List[str], where: str, logger, log_dataset_details) -> None:
            if len(uids) != len(set(uids)):
                logger.error(f'Duplicates among {where} UIDs detected, they should be unique')
                if log_dataset_details:
                    for uid in set(uids):
                        count = uids.count(uid)
                        if count > 1:
                            logger.error(f'{uid} appears {count} times')

        def _log_difference(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str, logger, log_dataset_details) -> None:
            difference = set(uids_a).difference(set(uids_b))
            if difference:
                logger.warning(f'UIDs in {where_a} but not in {where_b} detected, make sure this was intended.')
                if log_dataset_details:
                    logger.warning(f'Difference {where_a}\\{where_b}: ' + ', '.join(sorted(difference)))

        def _log_differences(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str, logger, log_dataset_details) -> None:
            _log_difference(uids_a, uids_b, where_a, where_b, logger, log_dataset_details)
            _log_difference(uids_b, uids_a, where_b, where_a, logger, log_dataset_details)

        def _log_intersection(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str, logger, log_dataset_details) -> None:
            intersection = set(uids_a).intersection(set(uids_b))
            if intersection:
                logger.error(f'Entries in {where_a}∩{where_b} detected, they should be in one set only.')
                if log_dataset_details:
                    logger.error(f'Entries in {where_a}∩{where_b}: ' + ', '.join(sorted(intersection)))

        config = 'unilateral'
        normalized_institutions = cls._normalize_institutions(institutions)
        if manifests is None:
            manifests = cls.build_manifests(path_root=path_root, institutions=normalized_institutions, config=config)

        for institution in normalized_institutions:
            manifest = manifests[institution]
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

            _log_differences(uids_in_annotation, uids_in_split[None], 'annotation', 'split', logger, log_dataset_details)
            _log_differences(uids_in_split[None], uids_in_images, 'split', 'images', logger, log_dataset_details)
            _log_differences(uids_in_annotation, uids_in_images, 'annotation', 'images', logger, log_dataset_details)

            _log_intersection(uids_in_split['train'], uids_in_split['val'], 'training', 'validation', logger, log_dataset_details)
            _log_intersection(uids_in_split['train'], uids_in_split['test'], 'training', 'test', logger, log_dataset_details)
            _log_intersection(uids_in_split['val'], uids_in_split['test'], 'validation', 'test', logger, log_dataset_details)
