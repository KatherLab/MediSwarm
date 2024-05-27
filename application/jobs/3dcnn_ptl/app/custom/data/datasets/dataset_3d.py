from pathlib import Path
import torch.utils.data as data
import torchio as tio
from data.augmentation.augmentations_3d import ImageToTensor, RescaleIntensity, ZNormalization

class SimpleDataset3D(data.Dataset):
    """
    A simple dataset class for 3D medical images using TorchIO for preprocessing and augmentation.

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
    """

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
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_glob = crawler_glob

        if transform is None:
            self.transform = tio.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.RandomFlip((0, 1, 2)) if flip else tio.Lambda(lambda x: x),
                tio.CropOrPad(image_crop) if image_crop is not None else tio.Lambda(lambda x: x),
                self.get_norm(norm),
                ImageToTensor() if to_tensor else tio.Lambda(lambda x: x)  # [C, W, H, D] -> [C, D, H, W]
            ])
        else:
            self.transform = transform

        if item_pointers:
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_glob)

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.item_pointers)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary with 'uid' and 'source' keys.
        """
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        return {'uid': str(rel_path_item), 'source': self.transform(img)}

    def load_item(self, path_item):
        """
        Loads an image from the given path.

        Args:
            path_item (Path): Path to the image file.

        Returns:
            tio.ScalarImage: Loaded image.
        """
        return tio.ScalarImage(path_item)

    @classmethod
    def run_item_crawler(cls, path_root, crawler_glob, **kwargs):
        """
        Crawls the directory to find items matching the glob pattern.

        Args:
            path_root (Path): Root directory to start crawling.
            crawler_glob (str): Glob pattern to match files.

        Returns:
            list: List of relative file paths.
        """
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'{crawler_glob}')]

    @staticmethod
    def get_norm(norm):
        """
        Returns the normalization transform based on the provided norm string.

        Args:
            norm (str): Normalization method name.

        Returns:
            tio.Transform: The normalization transform.
        """
        if norm is None:
            return tio.Lambda(lambda x: x)
        elif isinstance(norm, str):
            if norm == 'min-max':
                return RescaleIntensity((-1, 1), per_channel=True, masking_method=lambda x: x > 0)
            elif norm == 'min-max_clip':
                return RescaleIntensity((-1, 1), per_channel=True, percentiles=(0.5, 99.5), masking_method=lambda x: x > 0)
            elif norm == 'znorm':
                return ZNormalization(per_channel=True, masking_method=lambda x: x > 0)
            elif norm == 'znorm_clip':
                return ZNormalization(per_channel=True, percentiles=(0.5, 99.5), masking_method=lambda x: x > 0)
            else:
                raise ValueError("Unknown normalization")
        else:
            return norm
