from models import BasicClassifier
import monai.networks.nets as nets
import torch.nn as nn
from einops import rearrange


class _ResNet(nn.Module):
    """Wrapper for MONAI ResNet models supporting 3D/2D input."""

    def __init__(self, n_input_channels: int, num_classes: int, spatial_dims: int, resnet_variant: int):
        super().__init__()
        Model = {
            10: nets.resnet10,
            18: nets.resnet18,
            34: nets.resnet34,
            50: nets.resnet50,
            101: nets.resnet101,
            152: nets.resnet152
        }.get(resnet_variant)
        if Model is None:
            raise ValueError(f"Unsupported ResNet model number: {resnet_variant}")

        shortcut_type = {
            10: 'B',
            18: 'A',
            34: 'A',
            50: 'B',
            101: 'B',
            152: 'B',
        }.get(resnet_variant)

        bias_downsample = {
            10: False,
            18: True,
            34: True,
            50: False,
            101: False,
            152: False,
        }.get(resnet_variant)

        num_channels = {
            10: 512,
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
            152: 2048,
        }.get(resnet_variant)

        self.model = Model(n_input_channels=n_input_channels, spatial_dims=spatial_dims, num_classes=num_classes,
                           feed_forward=False, shortcut_type=shortcut_type, bias_downsample=bias_downsample, pretrained=True)
        self.model.fc = nn.Linear(num_channels,
                                  num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet(BasicClassifier):
    """ResNet-based classifier using MONAI backbones."""

    def __init__(self, n_input_channels: int, num_classes: int, spatial_dims: int, resnet_variant: int, **kwargs):
        super().__init__(n_input_channels, num_classes, spatial_dims, **kwargs)
        self.model = _ResNet(n_input_channels, num_classes, spatial_dims, resnet_variant)

    def forward(self, x):
        return self.model(x)


'''
class ResNetRegression(BasicRegression):
    """ResNet-based regression model using MONAI backbones."""
    def __init__(self, n_input_channels: int, num_classes: int , spatial_dims: int, resnet_variant: str, **kwargs):
        super().__init__(n_input_channels, num_classes, spatial_dims, **kwargs)
        self.model = _ResNet(n_input_channels, num_classes, resnet_variant)

    def forward(self, x):
        return self.model(x)
'''
