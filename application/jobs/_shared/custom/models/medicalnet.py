import os

import torch
import torch.nn as nn

from models import BasicClassifier
from models.medicalnet_resnet import resnet34


class FrozenBatchNorm3d(nn.Module):
    """BatchNorm3d that always normalises with its stored statistics.

    Unlike nn.BatchNorm3d in eval mode, this does not depend on the module's training flag:
    PyTorch Lightning restores each submodule's flag directly after validation, which
    bypasses any train() override, and with batch size 1 (the swarm recipe) train-mode
    BatchNorm turns into per-sample normalisation that the eval-mode network cannot
    reproduce (Duke benchmark, 24 Sep 2026: 0.85 AUROC in train mode, 0.44 in eval mode).
    The statistics stay fixed at the pretrained values; the affine weight and bias train.
    State-dict keys match nn.BatchNorm3d, so checkpoints load either way.
    """

    def __init__(self, bn: nn.BatchNorm3d):
        super().__init__()
        self.eps = bn.eps
        self.weight = nn.Parameter(bn.weight.detach().clone())
        self.bias = nn.Parameter(bn.bias.detach().clone())
        self.register_buffer("running_mean", bn.running_mean.detach().clone())
        self.register_buffer("running_var", bn.running_var.detach().clone())
        self.register_buffer("num_batches_tracked", bn.num_batches_tracked.detach().clone())

    def forward(self, x):
        scale = self.weight * torch.rsqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        return x * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)


def freeze_batchnorm(module: nn.Module) -> int:
    """Replace every nn.BatchNorm3d below module with FrozenBatchNorm3d; returns the count."""
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            setattr(module, name, FrozenBatchNorm3d(child))
            count += 1
        else:
            count += freeze_batchnorm(child)
    return count


class _MedicalNetResNet34(nn.Module):
    """MedicalNet 3D-ResNet34 adapted from segmentation to classification."""

    def __init__(self, num_classes: int, pretrained_path: str | None = None, freeze_bn: bool = False):
        super().__init__()
        self.backbone = resnet34(
            sample_input_W=224,
            sample_input_H=224,
            sample_input_D=32,
            shortcut_type="A",
            no_cuda=False,
            num_seg_classes=1,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(512, num_classes)

        if pretrained_path:
            self._load_pretrained(pretrained_path)
        if freeze_bn:
            # After loading, so the frozen statistics are the pretrained ones.
            freeze_batchnorm(self.backbone)

    def _load_pretrained(self, pretrained_path: str) -> None:
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
            if not key.removeprefix("module.").startswith("conv_seg.")
        }
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected MedicalNet checkpoint keys: {unexpected[:5]}")
        if len(missing) == len(self.backbone.state_dict()):
            raise RuntimeError(f"No MedicalNet backbone weights found in {pretrained_path}")

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class MedicalNet(BasicClassifier):
    """MedicalNet 3D-ResNet34 classifier for swarm training."""

    def __init__(
        self,
        n_input_channels: int,
        num_classes: int,
        spatial_dims: int,
        pretrained_path: str | None = None,
        **kwargs,
    ):
        if n_input_channels != 1 or spatial_dims != 3:
            raise ValueError("MedicalNet requires one input channel and spatial_dims=3")
        # Fine-tuning defaults (the BasicClassifier default of lr 1e-3 is for training from
        # scratch): overridable per run through MEDICALNET_LR and MEDICALNET_FREEZE_BN.
        # Keep MEDICALNET_FREEZE_BN=1 for swarm runs: trainable BatchNorm at batch size 1
        # does not survive the switch to eval mode (see FrozenBatchNorm3d).
        kwargs.setdefault("optimizer_kwargs", {
            "lr": float(os.environ.get("MEDICALNET_LR", "1e-4")),
            "weight_decay": 1e-2,
        })
        freeze_bn = os.environ.get("MEDICALNET_FREEZE_BN", "1") == "1"
        super().__init__(n_input_channels, num_classes, spatial_dims, **kwargs)
        self.model = _MedicalNetResNet34(num_classes, pretrained_path, freeze_bn=freeze_bn)

    def forward(self, x):
        return self.model(x)