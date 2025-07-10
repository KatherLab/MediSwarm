from typing import List, Union, Any
from pathlib import Path
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy


class VeryBasicModel(pl.LightningModule):
    """
    A very basic model class extending LightningModule with basic functionality.
    """
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1

    def forward(self, x_in):
        raise NotImplementedError

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        raise NotImplementedError

    def _epoch_end(self, outputs: List[Any], state: str):
        return

    def training_step(self, batch, batch_idx):
        self._step_train += 1
        return self._step(batch, batch_idx, "train", self._step_train, 0)

    def validation_step(self, batch: dict, batch_idx: int) -> Any:
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val, 0)

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test, 0)

    def on_train_epoch_end(self):
        self._epoch_end([], "train")

    def on_validation_epoch_end(self):
        self._epoch_end([], "val")

    def on_test_epoch_end(self):
        self._epoch_end([], "test")

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, **kwargs):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir) / path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)

    def load_pretrained(self, checkpoint_path, map_location=None, **kwargs):
        if checkpoint_path.is_dir():
            checkpoint_path = self._get_best_checkpoint_path(checkpoint_path, **kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return self.load_weights(checkpoint["state_dict"], **kwargs)

    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter_fn = kwargs.get('filter', lambda key: key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter_fn(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self


class BasicModel(VeryBasicModel):
    def __init__(
            self,
            optimizer=torch.optim.AdamW,
            optimizer_kwargs=None,
            lr_scheduler=None,
            lr_scheduler_kwargs=None,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3, 'weight_decay': 1e-2}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}]
        return [optimizer]


class BasicClassifier(BasicModel):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            spatial_dims: int,
            loss=torch.nn.CrossEntropyLoss,
            loss_kwargs=None,
            optimizer=torch.optim.AdamW,
            optimizer_kwargs=None,
            lr_scheduler=None,
            lr_scheduler_kwargs=None,
            aucroc_kwargs=None,
            acc_kwargs=None,
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims
        self.loss_kwargs = loss_kwargs or {}
        self.loss = loss(**self.loss_kwargs)

        aucroc_kwargs = aucroc_kwargs or {"task": "binary"}
        acc_kwargs = acc_kwargs or {"task": "binary"}

        self.auc_roc = nn.ModuleDict({state: AUROC(**aucroc_kwargs) for state in ["train_", "val_", "test_"]})
        self.acc = nn.ModuleDict({state: Accuracy(**acc_kwargs) for state in ["train_", "val_", "test_"]})

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        source, target = batch['source'], batch['target']
        target = target[:, None].float()
        target_int = target.int()
        batch_size = source.shape[0]

        pred = self(source)
        loss_val = self.loss(pred, target)

        with torch.no_grad():
            prob = torch.sigmoid(pred)  # logits -> probability
            self.acc[state + "_"].update(prob, target_int)
            self.auc_roc[state + "_"].update(prob, target_int)

        self.log(f"{state}/loss", loss_val, batch_size=batch_size, on_step=True, on_epoch=True)
        return loss_val

    def _epoch_end(self, outputs, state):
        for name, metric in [("ACC", self.acc[state + "_"]), ("AUC_ROC", self.auc_roc[state + "_"])]:
            self.log(f"{state}/{name}", metric.compute(), on_epoch=True)
            metric.reset()
