import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler


class DataModule(pl.LightningDataModule):
    """
    LightningDataModule for handling dataset loading and batching.

    Attributes:
        ds_train (object): Training dataset.
        ds_val (object): Validation dataset.
        ds_test (object): Test dataset.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of workers for data loading.
        seed (int): Random seed for reproducibility.
        pin_memory (bool): If True, pin memory for faster data transfer to GPU.
        weights (list): Weights for the weighted random sampler.
    """

    def __init__(
            self,
            ds_train: object = None,
            ds_val: object = None,
            ds_test: object = None,
            batch_size: int = 1,
            num_workers: int = mp.cpu_count(),
            seed: int = 0,
            pin_memory: bool = False,
            weights: list = None
    ):
        """
        Initializes the DataModule with datasets and parameters.

        Args:
            ds_train (object, optional): Training dataset. Defaults to None.
            ds_val (object, optional): Validation dataset. Defaults to None.
            ds_test (object, optional): Test dataset. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 1.
            num_workers (int, optional): Number of workers. Defaults to mp.cpu_count().
            seed (int, optional): Random seed. Defaults to 0.
            pin_memory (bool, optional): Pin memory. Defaults to False.
            weights (list, optional): Weights for sampling. Defaults to None.
        """
        super().__init__()
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop('__class__')
        self.hyperparameters.pop('self')

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.weights = weights

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        Returns:
            DataLoader: DataLoader for the training dataset.

        Raises:
            AssertionError: If the training dataset is not initialized.
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.ds_train is not None:
            if self.weights is not None:
                sampler = WeightedRandomSampler(self.weights, len(self.weights), generator=generator)
            else:
                sampler = RandomSampler(self.ds_train, replacement=False, generator=generator)
            return DataLoader(
                self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                sampler=sampler, generator=generator, drop_last=True, pin_memory=self.pin_memory
            )

        raise AssertionError("A training set was not initialized.")

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: DataLoader for the validation dataset.

        Raises:
            AssertionError: If the validation dataset is not initialized.
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_val is not None:
            return DataLoader(
                self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                generator=generator, drop_last=False, pin_memory=self.pin_memory
            )

        raise AssertionError("A validation set was not initialized.")

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns:
            DataLoader: DataLoader for the test dataset.

        Raises:
            AssertionError: If the test dataset is not initialized.
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_test is not None:
            return DataLoader(
                self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                generator=generator, drop_last=False, pin_memory=self.pin_memory
            )

        raise AssertionError("A test dataset was not initialized.")
