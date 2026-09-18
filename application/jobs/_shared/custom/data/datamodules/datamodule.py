import os

import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler


def _configure_odelia_worker_runtime(_worker_id: int) -> None:
    thread_count = max(1, int(os.environ.get("ODELIA_THREADS_PER_WORKER", "1")))
    interop_threads = max(1, int(os.environ.get("ODELIA_INTEROP_THREADS", "1")))

    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = str(thread_count)

    torch.set_num_threads(thread_count)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass


class DataModule(pl.LightningDataModule):
    """Flexible LightningDataModule with weighted or random sampling support."""

    def __init__(
            self,
            ds_train: object = None,
            ds_val: object = None,
            ds_test: object = None,
            batch_size: int = 1,
            batch_size_val: int = None,
            batch_size_test: int = None,
            num_train_samples: int = None,
            num_workers: int = mp.cpu_count(),
            seed: int = 0,
            pin_memory: bool = False,
            weights: list = None
    ):
        super().__init__()
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop('__class__')
        self.hyperparameters.pop('self')

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test

        self.batch_size = batch_size
        self.batch_size_val = batch_size if batch_size_val is None else batch_size_val
        self.batch_size_test = batch_size if batch_size_test is None else batch_size_test
        self.num_train_samples = num_train_samples
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.weights = weights
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    def train_dataloader(self):
        if self._train_loader is not None:
            return self._train_loader

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.ds_train is not None:
            if self.weights is not None:
                num_samples = len(self.weights) if self.num_train_samples is None else self.num_train_samples
                sampler = WeightedRandomSampler(self.weights, num_samples=num_samples, generator=generator)
            else:
                num_samples = len(self.ds_train) if self.num_train_samples is None else self.num_train_samples
                sampler = RandomSampler(self.ds_train, num_samples=num_samples, replacement=False, generator=generator)

            self._train_loader = DataLoader(
                self.ds_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=sampler,
                generator=generator,
                drop_last=True,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                worker_init_fn=_configure_odelia_worker_runtime if self.num_workers > 0 else None,
            )
            return self._train_loader

        raise AssertionError("A training set was not initialized.")

    def val_dataloader(self):
        if self._val_loader is not None:
            return self._val_loader

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.ds_val is not None:
            self._val_loader = DataLoader(
                self.ds_val,
                batch_size=self.batch_size_val,
                num_workers=self.num_workers,
                shuffle=False,
                generator=generator,
                drop_last=False,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                worker_init_fn=_configure_odelia_worker_runtime if self.num_workers > 0 else None,
            )
            return self._val_loader

        raise AssertionError("A validation set was not initialized.")

    def test_dataloader(self):
        if self._test_loader is not None:
            return self._test_loader

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.ds_test is not None:
            self._test_loader = DataLoader(
                self.ds_test,
                batch_size=self.batch_size_test,
                num_workers=self.num_workers,
                shuffle=False,
                generator=generator,
                drop_last=False,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                worker_init_fn=_configure_odelia_worker_runtime if self.num_workers > 0 else None,
            )
            return self._test_loader

        raise AssertionError("A test set was not initialized.")
