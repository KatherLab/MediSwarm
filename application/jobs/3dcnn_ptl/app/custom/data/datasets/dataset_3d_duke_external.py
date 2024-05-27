from pathlib import Path
import pandas as pd
from data.datasets import SimpleDataset3D

class DUKE_Dataset3D_external(SimpleDataset3D):
    """
    DUKE External Dataset for 3D medical images, extending SimpleDataset3D.

    Args:
        path_root (str): Root directory of the dataset.
        item_pointers (list, optional): List of file paths. Defaults to None.
        crawler_glob (str, optional): Glob pattern for crawling files. Defaults to '*.nii.gz'.
        transform (callable, optional): Transformations to apply to the data. Defaults to None.
        image_resize (tuple, optional): Desired output image size. Defaults to None.
        flip (bool, optional): Whether to apply random flipping. Defaults to False.
        image_crop (tuple, optional): Desired crop size. Defaults to None.
        norm (str, optional): Normalization method. Defaults to 'znorm_clip'.
        to_tensor (bool, optional): Whether to convert images to tensor. Defaults to True.
    """

    def __init__(
        self,
        path_root,
        item_pointers=None,
        crawler_glob='*.nii.gz',
        transform=None,
        image_resize=None,
        flip=False,
        image_crop=None,
        norm='znorm_clip',
        to_tensor=True
    ):
        if item_pointers is None:
            item_pointers = []
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm, to_tensor)
        df = pd.read_csv(self.path_root.parent / 'segmentation_metadata_unilateral.csv')
        df = df[[df.columns[0], df.columns[5]]]  # Only pick relevant columns: Patient ID, Tumor Side, Bilateral
        self.df = df.set_index('PATIENT', drop=True)
        self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary with 'uid', 'source', and 'target' keys.
        """
        uid = self.item_pointers[index]
        path_item = [self.path_root / uid / name for name in ['Sub.nii.gz']]
        img = self.load_item(path_item)
        target = self.df.loc[uid]['Malign']
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
