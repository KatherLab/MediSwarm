from pathlib import Path
import pandas as pd
from data.datasets import SimpleDataset3D

class ODELIA_DUKE_Dataset3D(SimpleDataset3D):
    """
    DUKE Dataset for 3D medical images, extending SimpleDataset3D.

    Args:
        path_root (str): Root directory of the dataset.
        item_pointers (list, optional): List of file paths. Defaults to [].
        crawler_glob (str, optional): Glob pattern for crawling files. Defaults to '*.nii.gz'.
        transform (callable, optional): Transformations to apply to the data. Defaults to None.
        image_resize (tuple, optional): Desired output image size. Defaults to None.
        flip (bool, optional): Whether to apply random flipping. Defaults to False.
        image_crop (tuple, optional): Desired crop size. Defaults to None.
        norm (str, optional): Normalization method. Defaults to 'znorm_clip'.
        to_tensor (bool, optional): Whether to convert images to tensor. Defaults to True.
        sequence (str, optional): Sequence type to use for loading images. Defaults to 'sub'.
    """

    MALIGNANCY = 'Malign'

    def __init__(
            self,
            path_root,
            item_pointers=[],
            crawler_glob='*.nii.gz',
            transform=None,
            image_resize=None,
            flip=False,
            image_crop=None,
            norm='znorm_clip',
            to_tensor=True,
            fold=0,
            split='train',
            sequence='sub'
    ):

        UNILATERAL_IMAGE_SUBDIR = Path('data_unilateral')
        METADATA_SUBDIR = Path('metadata')
        METADATA_FILE = Path('split.csv')
        PATIENT_ID = 'PatientID'
        UID = 'UID'
        CLASS = 'Class'
        FOLD = 'Fold'
        SPLIT = 'Split'

        super().__init__(Path(path_root) / UNILATERAL_IMAGE_SUBDIR, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm, to_tensor)
        df = pd.read_csv(Path(path_root) / METADATA_SUBDIR / METADATA_FILE, dtype={PATIENT_ID: str, UID: str, CLASS: int, FOLD: int, SPLIT: str})
        df = df.loc[df[FOLD] == fold]
        df = df.loc[df[SPLIT] == split]

        existing_folders = {folder.name for folder in Path(self.path_root).iterdir() if folder.is_dir()}
        df = df.loc[df[UID].isin(existing_folders)]

        self.df = pd.DataFrame({UID:             df[UID],
                                self.MALIGNANCY: df[CLASS]})
        self.df = self.df.set_index(UID, drop=True)
        self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()
        self.sequence = sequence

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary with 'uid', 'source', and 'target' keys.
        """
        uid = self.item_pointers[index]
        path_item = [self.path_root / uid / name for name in [f'{self.sequence}.nii.gz']]
        img = self.load_item(path_item)
        target = self.df.loc[uid][self.MALIGNANCY]
        return {'uid': uid, 'source': self.transform(img), 'target': target}

    @classmethod
    def run_item_crawler(cls, path_root, crawler_ext, **kwargs):
        """
        Crawls the directory to find items matching the glob pattern.

        Args:
            path_root (Path): Root directory to start crawling.
            crawler_ext (str): Extension to match files.

        Returns:
            list: List of relative file paths.
        """
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir()]

    def get_labels(self):
        """
        Gets the labels for the dataset items.

        Returns:
            list: List of labels.
        """
        return self.df.loc[self.item_pointers, self.MALIGNANCY].tolist()

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: Number of items.
        """
        return len(self.item_pointers)
