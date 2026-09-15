import torch
import torch.nn as nn

from models import BasicClassifier
from models.medicalnet_resnet import resnet34


class _MedicalNetResNet34(nn.Module):
    """MedicalNet 3D-ResNet34 adapted from segmentation to classification."""

    def __init__(self, num_classes: int, pretrained_path: str | None = None):
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
        super().__init__(n_input_channels, num_classes, spatial_dims, **kwargs)
        self.model = _MedicalNetResNet34(num_classes, pretrained_path)

    def forward(self, x):
        return self.model(x)