from typing import List, Union
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics import AUROC, Accuracy


class VeryBasicModel(pl.LightningModule):
    """
    A very basic model class extending LightningModule with basic functionality.

    Attributes:
        _step_train (int): Counter for training steps.
        _step_val (int): Counter for validation steps.
        _step_test (int): Counter for test steps.
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1

    def forward(self, x_in):
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        """Step function for training, validation, and testing. Must be implemented by subclasses."""
        raise NotImplementedError

    def _epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]], state: str):
        """Epoch end function."""
        return

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        self._step_train += 1
        return self._step(batch, batch_idx, "train", self._step_train, optimizer_idx)

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val, optimizer_idx)

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test, optimizer_idx)

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self._epoch_end(outputs, "train")
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self._epoch_end(outputs, "val")
        return super().validation_epoch_end(outputs)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self._epoch_end(outputs, "test")
        return super().test_epoch_end(outputs)

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        """Saves the best model checkpoint path.

        Args:
            path_checkpoint_dir (str): Directory to save the checkpoint.
            best_model_path (str): Path to the best model.
        """
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, version=0, **kwargs):
        """Gets the best model checkpoint path.

        Args:
            path_checkpoint_dir (str): Directory containing the checkpoint.
            version (int, optional): Version of the checkpoint. Defaults to 0.

        Returns:
            Path: Path to the best checkpoint.
        """
        path_version = 'lightning_logs/version_' + str(version)
        with open(Path(path_checkpoint_dir) / path_version / 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir) / path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, version=0, **kwargs):
        """Loads the best model checkpoint.

        Args:
            path_checkpoint_dir (str): Directory containing the checkpoint.
            version (int, optional): Version of the checkpoint. Defaults to 0.

        Returns:
            LightningModule: The loaded model.
        """
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir, version)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)

    def load_pretrained(self, checkpoint_path, map_location=None, **kwargs):
        """Loads pretrained weights from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint.
            map_location (str, optional): Device to map the checkpoint. Defaults to None.

        Returns:
            LightningModule: The model with loaded weights.
        """
        if checkpoint_path.is_dir():
            checkpoint_path = self._get_best_checkpoint_path(checkpoint_path, **kwargs)

        with pl_legacy_patch():
            if map_location is not None:
                checkpoint = pl_load(checkpoint_path, map_location=map_location)
            else:
                checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)
        return self.load_weights(checkpoint["state_dict"], **kwargs)

    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        """Loads weights into the model.

        Args:
            pretrained_weights (dict): Pretrained weights.
            strict (bool, optional): Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module’s `state_dict` function. Defaults to True.

        Returns:
            LightningModule: The model with loaded weights.
        """
        filter_fn = kwargs.get('filter', lambda key: key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter_fn(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self


class BasicModel(VeryBasicModel):
    """
    A basic model class with optimizer and learning rate scheduler configurations.

    Attributes:
        optimizer (Optimizer): The optimizer to use.
        optimizer_kwargs (dict): Keyword arguments for the optimizer.
        lr_scheduler (Scheduler): The learning rate scheduler to use.
        lr_scheduler_kwargs (dict): Keyword arguments for the learning rate scheduler.
    """

    def __init__(
            self,
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-2},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def configure_optimizers(self):
        """Configures the optimizers and learning rate schedulers.

        Returns:
            list: List containing the optimizer and optionally the learning rate scheduler.
        """
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]


class BasicClassifier(BasicModel):
    """
    A basic classifier model with loss function and metrics.

    Attributes:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        spatial_dims (int): Number of spatial dimensions.
        loss (Loss): The loss function.
        loss_kwargs (dict): Keyword arguments for the loss function.
        auc_roc (ModuleDict): Dictionary of AUROC metrics.
        acc (ModuleDict): Dictionary of Accuracy metrics.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            spatial_dims: int,
            loss=torch.nn.CrossEntropyLoss,
            loss_kwargs={},
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-2},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            aucroc_kwargs={"task": "binary"},
            acc_kwargs={"task": "binary"}
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims
        self.loss = loss(**loss_kwargs)
        self.loss_kwargs = loss_kwargs

        self.auc_roc = nn.ModuleDict({state: AUROC(**aucroc_kwargs) for state in ["train_", "val_", "test_"]})
        self.acc = nn.ModuleDict({state: Accuracy(**acc_kwargs) for state in ["train_", "val_", "test_"]})

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        """Step function for training, validation, and testing.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            state (str): State of the model ('train', 'val', 'test').
            step (int): Current step.
            optimizer_idx (int): Index of the optimizer.

        Returns:
            Tensor: Loss value.
        """
        source, target = batch['source'], batch['target']
        target = target[:, None].float()
        batch_size = source.shape[0]

        # Run Model
        pred = self(source)

        # Compute Loss
        logging_dict = {}
        logging_dict['loss'] = self.loss(pred, target)

        # Compute Metrics
        with torch.no_grad():
            self.acc[state + "_"].update(pred, target)
            self.auc_roc[state + "_"].update(pred, target)

            # Log Scalars
            for metric_name, metric_val in logging_dict.items():
                self.log(f"{state}/{metric_name}", metric_val.cpu() if hasattr(metric_val, 'cpu') else metric_val,
                         batch_size=batch_size, on_step=True, on_epoch=True)

        return logging_dict['loss']

    def _epoch_end(self, outputs, state):
        """Epoch end function.

        Args:
            outputs (Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]): Outputs of the epoch.
            state (str): State of the model ('train', 'val', 'test').
        """
        batch_size = len(outputs)
        for name, value in [("ACC", self.acc[state + "_"]), ("AUC_ROC", self.auc_roc[state + "_"])]:
            self.log(f"{state}/{name}", value.compute().cpu(), batch_size=batch_size, on_step=False, on_epoch=True)
            value.reset()
