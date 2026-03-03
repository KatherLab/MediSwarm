from pathlib import Path
import pandas as pd
import torch.utils.data as data
import torchio as tio
import torch
import numpy as np
from typing import List, Dict, Tuple, Any

from data.augmentation.augmentations_3d import ImageOrSubjectToTensor, ZNormalization, CropOrPad


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
            labels=None,  # None = all labels or  list of labels
            config=None,  # original, unilateral
            split=None,
            fraction=None,
            transform=None,
            random_flip=False,
            random_rotate=False,
            random_inverse=False,
            noise=False,
            to_tensor=True,

    ):
        self.path_root = Path(self.PATH_ROOT if path_root is None else path_root)
        self.split = split
        self.config = config
        self.class_labels = self.CLASS_LABELS[config]
        self.meta_dir = self.META_DIR[config]
        self.data_dir = self.DATA_DIR[config]
        self.labels = list(self.class_labels.keys()) if labels is None else labels
        self.class_labels_num = [len(self.class_labels[l]) for l in self.labels]  # For CORN Loss -1

        if (institutions is None) or (institutions == "ODELIA"):
            institutions = self.ALL_INSTITUTIONS
        elif isinstance(institutions, str):
            institutions = [institutions]
        self.institutions = institutions

        flip_axes = (0, 1) if config == "original" else (0, 1, 2)  # Do not flip horizontal axis 2, otherwise labels incorrect
        if transform is None:
            self.transform = tio.Compose([
                tio.ToCanonical() if config == "original" else tio.Lambda(lambda x: x),
                tio.Resample((0.7, 0.7, 3)) if config == "original" else tio.Lambda(lambda x: x),

                tio.Flip((1, 0)),  # Just for viewing, otherwise upside down
                CropOrPad((448, 448, 32), random_center=random_rotate) if config == "original" else CropOrPad(
                    (224, 224, 32), random_center=random_rotate),

                ZNormalization(per_channel=True, per_slice=False,
                               masking_method=lambda x: (x > x.min()) & (x < x.max()), percentiles=(0.5, 99.5)),

                tio.OneOf([
                    # tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x), # WARNING: 1,2 if Subject, 2, 3 if tensor
                    tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0, 90), translation=0, isotropic=True,
                                     default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                    tio.RandomFlip(flip_axes) if random_flip else tio.Lambda(lambda x: x),  # WARNING: Padding mask
                ]),
                tio.Lambda(lambda x: -x if torch.rand((1,), )[0] < 0.5 else x,
                           types_to_apply=[tio.INTENSITY]) if random_inverse else tio.Lambda(lambda x: x),
                tio.RandomNoise(std=(0.0, 0.25)) if noise else tio.Lambda(lambda x: x),

                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x)
            ])
        elif transform == 'USE_UNPROCESSED_IMAGES':
            self.transform = tio.Compose([tio.Lambda(lambda x: x)])
        else:
            self.transform = transform

        # Get split
        dfs = []
        for institution in self.institutions:
            path_metadata = self.path_root / institution / self.meta_dir

            df_split = self.load_split(path_metadata / 'split.csv', fold=fold, split=split, fraction=fraction)
            df_split['Institution'] = institution

            df_annot = pd.read_csv(path_metadata / 'annotation.csv', dtype={'UID': str, 'PatientID': str})

            df = df_split.merge(df_annot, on='UID', how='inner')  # uses only UIDs present in both dataframes

            uids_in_imags = self.run_item_crawler(self.path_root/institution/self.DATA_DIR[config])
            df = df[df['UID'].isin(uids_in_imags)]  # limit to UIDs for which an image is present

            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)

        self.item_pointers = df.index.tolist()
        self.df = df

    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        uid = item['UID']
        institution = item['Institution']

        target = np.stack(item[self.labels].values)

        path_folder = self.path_root / institution / self.data_dir / uid
        # img = self.load_img([path_folder/f'{name}.nii.gz' for name in [ 'Pre', 'Sub_1', 'T2']])
        img = self.load_img(path_folder / 'Sub_1.nii.gz')
        img = self.transform(img)

        return {'uid': uid, 'source': img, 'target': target}

    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        # WARNING: PatientID must be read as string otherwise leading zeros are cut off
        df = pd.read_csv(filepath_or_buffer, dtype={'UID': str})
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df

    @classmethod
    def run_item_crawler(cls, path_root, **kwargs):
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir()]

    @classmethod
    def print_UID_discrepancies(cls,
                                path_root=None,
                                institutions=None,
                                fold=0) -> None:

        def _get_uids_in_annotation(path_metadata: Path):
            df_annotation = pd.read_csv(path_metadata / 'annotation.csv', dtype={'UID': str, 'PatientID': str})
            uids_in_annotation = list(df_annotation['UID'])
            uids_in_annotation.sort()
            return uids_in_annotation

        def _get_uids_in_split(path_metadata: Path, fold) -> Dict[str|None, List[str]]:
            df_split = {split: cls.load_split(path_metadata / 'split.csv', fold=fold, split=split, fraction=None) for split in (None, 'train', 'val', 'test')}
            uids_split = {s: list(df_split[s]['UID']) for s in df_split.keys()}
            for l in uids_split.values():
                l.sort()
            return uids_split

        def _get_uids_of_images_present(path_root: Path, config) -> List[str]:
            uids_in_images = cls.run_item_crawler(path_root / institution / cls.DATA_DIR[config])
            uids_in_images.sort()
            return uids_in_images

        def _get_uids(path_metadata: Path, path_root: Path, config) -> Tuple[List[str], Dict[str|None, List[str]], List[str]]:
            return _get_uids_in_annotation(path_metadata), _get_uids_in_split(path_metadata, fold), _get_uids_of_images_present(path_root, config)

        def _print_duplicates(uids: List[str], where: str) -> None:
            if len(uids) != len(set(uids)):
                print(f'ERROR: Duplicates among {where} UIDs detected, they should be unique')
                for u in set(uids):
                    count = uids.count(u)
                    if count > 1:
                        print(f'{u} appears {count} times')

        def _print_difference(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str) -> None:
            difference = set(uids_a).difference(set(uids_b))
            if difference:
                difference = list(difference)
                difference.sort()
                print(f'WARNING: Difference in {where_a}\\{where_b} detected, make sure this was intended:', ', '.join(difference))

        def _print_differences(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str) -> None:
            _print_difference(uids_a, uids_b, where_a, where_b)
            _print_difference(uids_b, uids_a, where_b, where_a)

        def _print_intersection(uids_a: List[str], uids_b: List[str], where_a: str, where_b: str) -> None:
            intersection = set(uids_a).intersection(set(uids_b))
            if intersection:
                intersection = list(intersection)
                intersection.sort()
                print(f'ERROR: Entries in {where_a}∩{where_b} detected, they should be in one set only:', ', '.join(intersection))

        config = 'unilateral'
        path_root = Path(cls.PATH_ROOT if path_root is None else path_root)
        meta_dir = cls.META_DIR[config]
        for institution in institutions:
            path_metadata = path_root / institution / meta_dir
            uids_in_annotation, uids_in_split, uids_in_images = _get_uids(path_metadata, path_root, config)

            if True:
                print('INFO: Annoation UIDs:', ' '.join(uids_in_annotation))
                print('INFO: All split UIDs:', ' '.join(uids_in_split[None]))
                print('INFO: Training UIDs:', ' '.join(uids_in_split['train']))
                print('INFO: Validation UIDs:', ' '.join(uids_in_split['val']))
                print('INFO: Test UIDs:', ' '.join(uids_in_split['test']))
                print('INFO: Image UIDs', ' '.join(uids_in_images))

            for uids, where in ((uids_in_annotation, 'annotation'),
                                (uids_in_split[None], 'all split'),
                                (uids_in_split['train'], 'training'),
                                (uids_in_split['val'], 'validation'),
                                (uids_in_split['test'], 'test'),
                                (uids_in_images, 'image'),) :
                _print_duplicates(uids, where)

            _print_differences(uids_in_annotation, uids_in_split[None], 'annotation', 'split')
            _print_differences(uids_in_split[None], uids_in_images,'split', 'images')
            _print_differences(uids_in_annotation, uids_in_images, 'annotation', 'images')

            _print_intersection(uids_in_split['train'], uids_in_split['val'], 'training', 'validation')
            _print_intersection(uids_in_split['train'], uids_in_split['test'], 'training', 'test')
            _print_intersection(uids_in_split['val'], uids_in_split['test'], 'validation', 'test')
