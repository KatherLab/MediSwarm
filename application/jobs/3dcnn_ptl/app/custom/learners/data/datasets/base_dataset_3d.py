from pathlib import Path
import pandas as pd
from data.datasets import SimpleDataset3D


class BaseDataset3D(SimpleDataset3D):
    """
    A base class for handling 3D datasets with common functionality for loading, transforming, and normalizing images.

    Attributes:
        df (pd.DataFrame): DataFrame containing metadata for the dataset.
    """

    def __init__(self, path_root, item_pointers=None, crawler_glob='*.nii.gz', transform=None, image_resize=None,
                 flip=False, image_crop=None, norm='znorm_clip', to_tensor=True, metadata_file=None, columns=None):
        if item_pointers is None:
            item_pointers = []
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm,
                         to_tensor)
        if metadata_file:
            df = pd.read_csv(metadata_file)
            self.df = df[columns].set_index('PATIENT', drop=True)
            self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()
        else:
            self.df = None

    def get_labels(self):
        """Returns the labels for the dataset items."""
        return self.df['Malign'].values if self.df is not None else []

    @classmethod
    def run_item_crawler(cls, path_root, crawler_glob, **kwargs):
        """Crawls the dataset directory and returns a list of item pointers."""
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir()]

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.item_pointers)


### DUKE Dataset 3D
class DUKE_Dataset3D(BaseDataset3D):
    """
    DUKE 3D Dataset class.

    Attributes:
        sequence (str): The sequence name to look for in the data files.
    """

    def __init__(self, path_root, item_pointers=None, crawler_glob='*.nii.gz', transform=None, image_resize=None,
                 flip=False, image_crop=None, norm='znorm_clip', to_tensor=True, sequence='sub'):
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm,
                         to_tensor, metadata_file=path_root.parent / 'Clinical_and_Other_Features.xlsx',
                         columns=['PatientID', 'Location', 'Bilateral'])
        self.sequence = sequence

    def __getitem__(self, index):
        uid = self.item_pointers[index]
        path_item = [self.path_root / uid / f'{self.sequence}.nii.gz']
        img = self.load_item(path_item)
        target = self.df.loc[uid]['Malign']
        return {'uid': uid, 'source': self.transform(img), 'target': target}


### DUKE Dataset 3D Collaborative
class DUKE_Dataset3D_collab(BaseDataset3D):
    """DUKE 3D Collaborative Dataset class."""

    def __init__(self, path_root, item_pointers=None, crawler_glob='*.nii.gz', transform=None, image_resize=None,
                 flip=False, image_crop=None, norm='znorm_clip', to_tensor=True):
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm,
                         to_tensor, metadata_file=path_root.parent / 'datasheet.csv', columns=['PATIENT', 'Malign'])

    def __getitem__(self, index):
        uid = self.item_pointers[index]
        item_dir = self.path_root / uid
        nii_gz_files = list(item_dir.glob('**/*.nii.gz'))
        file_name = 'SUB_4.nii.gz'
        path_item = item_dir / file_name if len(nii_gz_files) > 1 and (item_dir / file_name) in nii_gz_files else \
        nii_gz_files[0]
        img = self.load_item(path_item)
        target = self.df.loc[uid]['Malign']
        return {'uid': uid, 'source': self.transform(img), 'target': target}


### DUKE Dataset 3D External
class DUKE_Dataset3D_external(BaseDataset3D):
    """DUKE 3D External Dataset class."""

    def __init__(self, path_root, item_pointers=None, crawler_glob='*.nii.gz', transform=None, image_resize=None,
                 flip=False, image_crop=None, norm='znorm_clip', to_tensor=True):
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm,
                         to_tensor, metadata_file=path_root.parent / 'segmentation_metadata_unilateral.csv',
                         columns=['PATIENT', 'Malign'])

    def __getitem__(self, index):
        uid = self.item_pointers[index]
        path_item = [self.path_root / uid / 'Sub.nii.gz']
        img = self.load_item(path_item)
        target = self.df.loc[uid]['Malign']
        return {'uid': uid, 'source': self.transform(img), 'target': target}
