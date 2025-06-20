from models import BasicClassifier
import monai.networks.nets as nets
import torch.nn as nn
from einops import rearrange

class _ResNet(nn.Module):
    """Wrapper for MONAI ResNet models supporting 3D/2D input."""
    def __init__(self, n_input_channels: int, num_classes: int , spatial_dims: int, resnet_variant: str):
        super().__init__()
        Model = {
            18: nets.resnet18,
            34: nets.resnet34,
            50: nets.resnet50,
            101: nets.resnet101,
            152: nets.resnet152
        }.get(resnet_variant)
        if Model is None:
            raise ValueError(f"Unsupported ResNet model number: {resnet_variant}")

        self.model = Model(n_input_channels=n_input_channels, spatial_dims=spatial_dims, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet(BasicClassifier):
    """ResNet-based classifier using MONAI backbones."""
    def __init__(self, n_input_channels: int, num_classes: int , spatial_dims :int, resnet_variant: str, **kwargs):
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
