from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy


class VeryBasicModel(pl.LightningModule):
    """Base LightningModule with training, validation, and test hooks stubbed out."""

    def __init__(self, save_hyperparameters=True):
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1

    def forward(self, x, cond=None):
        raise NotImplementedError

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        raise NotImplementedError

    def _epoch_end(self, state: str):
        return

    def training_step(self, batch: dict, batch_idx: int):
        self._step_train += 1
        return self._step(batch, batch_idx, "train", self._step_train)

    def validation_step(self, batch: dict, batch_idx: int):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val)

    def test_step(self, batch: dict, batch_idx: int):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

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
        filter = kwargs.get('filter', lambda key: key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self


class BasicModel(VeryBasicModel):
    """Extension of VeryBasicModel that includes optimizer and scheduler configuration."""

    def __init__(
            self,
            optimizer=torch.optim.Adam,
            optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-2},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            save_hyperparameters=True
    ):
        super().__init__(save_hyperparameters=save_hyperparameters)
        if save_hyperparameters:
            self.save_hyperparameters()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}]
        return [optimizer]


class BasicClassifier(BasicModel):
    """Generic classifier with dynamic metric and loss configuration based on task type."""

    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims,
            loss_kwargs={},
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={'lr': 1e-4, 'weight_decay': 1e-2},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            aucroc_kwargs={},
            acc_kwargs={},
            save_hyperparameters=True
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims

        loss = torch.nn.CrossEntropyLoss

        self.loss = loss(**loss_kwargs)
        self.loss_kwargs = loss_kwargs

        aucroc_kwargs.update({"task": "multiclass", 'num_classes': out_ch})
        acc_kwargs.update({"task": "multiclass", 'num_classes': out_ch})

        self.auc_roc = nn.ModuleDict({state: AUROC(**aucroc_kwargs) for state in ["train_", "val_", "test_"]})
        self.acc = nn.ModuleDict({state: Accuracy(**acc_kwargs) for state in ["train_", "val_", "test_"]})

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        source = batch['source']
        target = batch['target']
        batch_size = source.shape[0]
        self.batch_size = batch_size

        pred = self(source)
        loss_val = self.compute_loss(pred, target)
        target_squeezed = torch.squeeze(target, 1)  # TODO Why is this necessary and is it the right thing to do?
        self.acc[state + "_"].update(pred, target_squeezed)
        self.auc_roc[state + "_"].update(pred, target_squeezed)

        self.log(f"{state}/loss", loss_val, batch_size=batch_size, on_step=True, on_epoch=True)
        return loss_val

    def _epoch_end(self, state):
        acc_value = self.acc[state + "_"].compute()
        auc_roc_value = self.auc_roc[state + "_"].compute()
        self.log(f"{state}/ACC", acc_value, batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log(f"{state}/AUC_ROC", auc_roc_value, batch_size=self.batch_size, on_step=False, on_epoch=True)
        # For ModelCheckpoint, also log as "val/AUC_ROC" if state == "val"
        if state == "val":
            self.log("val/AUC_ROC", auc_roc_value, batch_size=self.batch_size, on_step=False, on_epoch=True)
        # print some debug information
        print(f"Epoch {self.current_epoch} - {state} ACC: {acc_value:.4f}, AUC_ROC: {auc_roc_value:.4f}")
        self.acc[state + "_"].reset()
        self.auc_roc[state + "_"].reset()

    def compute_loss(self, pred, target):
        target_squeezed = torch.squeeze(target, 1)  # TODO Why is this necessary and is it the right thing to do?
        return self.loss(pred, target_squeezed)

    def logits2labels(self, logits):
        return torch.argmax(logits, dim=1, keepdim=True)

    def logits2probabilities(self, logits):
        return F.softmax(logits, dim=1)


### Wrapper class for wrapping any model into BasicModel ###
class ModelWrapper(BasicClassifier):
    def __init__(
        self,
        backbone: nn.Module,
        in_ch: int,
        num_classes: int,
        spatial_dims: int = 3,
        **kwargs
    ):
        super().__init__(
            in_ch=in_ch,
            out_ch=num_classes,
            spatial_dims=spatial_dims,
            **kwargs
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
