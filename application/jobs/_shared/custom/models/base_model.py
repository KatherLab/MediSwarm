from pathlib import Path
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    CalibrationError,
    F1Score,
    Recall,
    Specificity,
)

# Extended evaluation metrics (#410) are attached to these states only, so the
# per-training-step cost of a run is unchanged.
EXTENDED_METRIC_STATES = ("val_", "test_")
# Bootstrap CIs re-compute each metric N times over the buffered predictions, so
# they stay on the test split even when enabled.
BOOTSTRAP_METRIC_STATES = ("test_",)


def _eval_tier():
    """'default' = macro scalars; 'full' also adds per-class values + calibration."""
    return os.environ.get("ODELIA_EVAL_METRICS", "default").strip().lower()


def _bootstrap_enabled():
    return os.environ.get("ODELIA_EVAL_BOOTSTRAP", "").strip().lower() in ("1", "true", "yes", "on")


def _bootstrap_samples():
    try:
        return max(2, int(os.environ.get("ODELIA_EVAL_BOOTSTRAP_N", "200")))
    except ValueError:
        return 200


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

        if loss_kwargs is None:
            loss_kwargs = {}

        # Store class weights as a buffer so they move to the correct device
        # with the model (e.g. GPU).  nn.CrossEntropyLoss.weight is NOT an
        # nn.Parameter, so Lightning's .to(device) won't move it automatically.
        class_weight = loss_kwargs.pop('weight', None)
        if class_weight is not None:
            if not isinstance(class_weight, torch.Tensor):
                class_weight = torch.tensor(class_weight, dtype=torch.float32)
            self.register_buffer('_class_weight', class_weight)
        else:
            self._class_weight = None

        self.loss_kwargs = loss_kwargs
        # Build loss; actual weight will be injected at forward time via compute_loss
        self.loss = torch.nn.CrossEntropyLoss(**loss_kwargs)

        aucroc_kwargs.update({"task": "multiclass", 'num_classes': out_ch})
        acc_kwargs.update({"task": "multiclass", 'num_classes': out_ch})

        self.auc_roc = nn.ModuleDict({state: AUROC(**aucroc_kwargs) for state in ["train_", "val_", "test_"]})
        self.acc = nn.ModuleDict({state: Accuracy(**acc_kwargs) for state in ["train_", "val_", "test_"]})

        self.eval_tier = _eval_tier()
        self.eval_bootstrap = _bootstrap_enabled()
        self.eval_bootstrap_samples = _bootstrap_samples()
        self.extra_metrics = nn.ModuleDict({
            state: nn.ModuleDict(self._build_extended_metrics(out_ch, state))
            for state in EXTENDED_METRIC_STATES
        })

    def _build_extended_metrics(self, num_classes, state):
        """Metrics beyond ACC/AUROC. All accept raw logits (torchmetrics softmaxes)."""
        kwargs = {"task": "multiclass", "num_classes": num_classes}
        metrics = {
            "F1": F1Score(average="macro", **kwargs),
            "Recall": Recall(average="macro", **kwargs),  # sensitivity
            "Specificity": Specificity(average="macro", **kwargs),
            "PR_AUC": AveragePrecision(average="macro", **kwargs),
        }
        if self.eval_tier == "full":
            metrics.update({
                "ECE": CalibrationError(**kwargs),
                "AUC_ROC_per_class": AUROC(average=None, **kwargs),
                "F1_per_class": F1Score(average=None, **kwargs),
                "Recall_per_class": Recall(average=None, **kwargs),
                "Specificity_per_class": Specificity(average=None, **kwargs),
                "PR_AUC_per_class": AveragePrecision(average=None, **kwargs),
            })
        if self.eval_bootstrap and state in BOOTSTRAP_METRIC_STATES:
            from torchmetrics.wrappers import BootStrapper
            quantile = torch.tensor([0.025, 0.975])
            for name, base in (("ACC_boot", Accuracy(**kwargs)), ("AUC_ROC_boot", AUROC(**kwargs))):
                metrics[name] = BootStrapper(base, num_bootstraps=self.eval_bootstrap_samples, quantile=quantile)
        return metrics

    def compute_epoch_metrics(self, state):
        """Compute every metric for `state` as {log_key: scalar tensor}.

        Pure -- performs no logging, so it is unit-testable without a Trainer.
        The `{state}/ACC` and `{state}/AUC_ROC` keys are preserved verbatim
        because IntimeModelSelector selects the global model on `val/AUC_ROC`.
        """
        key = state + "_"
        values = {
            f"{state}/ACC": self.acc[key].compute(),
            f"{state}/AUC_ROC": self.auc_roc[key].compute(),
        }
        if key not in self.extra_metrics:
            return values
        for name, metric in self.extra_metrics[key].items():
            computed = metric.compute()
            if name.endswith("_boot"):
                base = name[: -len("_boot")]
                low, high = computed["quantile"]
                values[f"{state}/{base}_ci_low"] = low
                values[f"{state}/{base}_ci_high"] = high
            elif name.endswith("_per_class"):
                base = name[: -len("_per_class")]
                for index, value in enumerate(computed):
                    values[f"{state}/{base}_class{index}"] = value
            else:
                values[f"{state}/{name}"] = computed
        return values

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
        if state + "_" in self.extra_metrics:
            for metric in self.extra_metrics[state + "_"].values():
                metric.update(pred, target_squeezed)

        self.log(f"{state}/loss", loss_val, batch_size=batch_size, on_step=True, on_epoch=True)
        return loss_val

    def _epoch_end(self, state):
        metric_values = self.compute_epoch_metrics(state)
        acc_value = metric_values[f"{state}/ACC"]
        auc_roc_value = metric_values[f"{state}/AUC_ROC"]
        for name, value in metric_values.items():
            self.log(name, value, batch_size=self.batch_size, on_step=False, on_epoch=True)
        # For ModelCheckpoint, also log as "val/AUC_ROC" if state == "val"
        if state == "val":
            self.log("val/AUC_ROC", auc_roc_value, batch_size=self.batch_size, on_step=False, on_epoch=True)
        # print some debug information
        print(f"Epoch {self.current_epoch} - {state} ACC: {acc_value:.4f}, AUC_ROC: {auc_roc_value:.4f}")
        self.acc[state + "_"].reset()
        self.auc_roc[state + "_"].reset()
        if state + "_" in self.extra_metrics:
            for metric in self.extra_metrics[state + "_"].values():
                metric.reset()

    def compute_loss(self, pred, target):
        target_squeezed = torch.squeeze(target, 1)  # TODO Why is this necessary and is it the right thing to do?
        # Inject class weights (registered as buffer → auto-moved to device)
        if self._class_weight is not None:
            self.loss.weight = self._class_weight
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
        self.save_hyperparameters(ignore=['backbone'])
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
