from pathlib import Path
import torch.utils.data as data
import torchio as tio

from data.augmentation.augmentations_3d import ImageToTensor, RescaleIntensity, ZNormalization


class SimpleDataset3D(data.Dataset):
    """
    A simple 3D dataset class that handles loading, transforming, and normalizing 3D medical images.

    Attributes:
        path_root (Path): The root directory of the dataset.
        crawler_glob (str): Glob pattern for crawling the dataset directory.
        transform (callable): Transformation to be applied to the images.
        item_pointers (list): List of relative paths to the dataset items.
    """

    def __init__(
        self,
        path_root: str,
        item_pointers: list = [],
        crawler_glob: str = '*.nii.gz',
        transform: tio.transforms.Transform = None,
        image_resize: tuple = None,
        flip: bool = False,
        image_crop: tuple = None,
        norm: str = 'znorm_clip',
        to_tensor: bool = True,
    ):
        """
        Initializes the dataset with the given parameters.

        Args:
            path_root (str): The root directory of the dataset.
            item_pointers (list, optional): List of item pointers. Defaults to [].
            crawler_glob (str, optional): Glob pattern for crawling the dataset directory. Defaults to '*.nii.gz'.
            transform (callable, optional): Transformation to be applied to the images. Defaults to None.
            image_resize (tuple, optional): Size to resize images to. Defaults to None.
            flip (bool, optional): Whether to apply random flipping. Defaults to False.
            image_crop (tuple, optional): Size to crop or pad images to. Defaults to None.
            norm (str, optional): Normalization method. Defaults to 'znorm_clip'.
            to_tensor (bool, optional): Whether to convert images to tensors. Defaults to True.
        """
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

        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_glob)

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.item_pointers)

    def __getitem__(self, index: int) -> dict:
        """Gets the item at the given index.

        Args:
            index (int): Index of the item.

        Returns:
            dict: A dictionary with 'uid' and 'source' keys.
        """
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        return {'uid': str(rel_path_item), 'source': self.transform(img)}

    def load_item(self, path_item: Path) -> tio.ScalarImage:
        """Loads the item from the given path.

        Args:
            path_item (Path): Path to the item.

        Returns:
            tio.ScalarImage: The loaded image.
        """
        return tio.ScalarImage(path_item)

    @classmethod
    def run_item_crawler(cls, path_root: Path, crawler_glob: str, **kwargs) -> list:
        """Crawls the dataset directory and returns a list of item pointers.

        Args:
            path_root (Path): Root directory of the dataset.
            crawler_glob (str): Glob pattern for crawling the dataset directory.

        Returns:
            list: List of relative paths to the dataset items.
        """
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'{crawler_glob}')]

    @staticmethod
    def get_norm(norm: str) -> tio.transforms.Transform:
        """Gets the normalization transform based on the given norm type.

        Args:
            norm (str): Normalization method.

        Returns:
            tio.transforms.Transform: The normalization transform.

        Raises:
            ValueError: If the normalization method is unknown.
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
