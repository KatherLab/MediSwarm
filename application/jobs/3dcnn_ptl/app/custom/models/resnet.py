from models import BasicClassifier, BasicRegression
import monai.networks.nets as nets
import torch.nn as nn
from einops import rearrange

class _ResNet(nn.Module):
    """Wrapper for MONAI ResNet models supporting 3D/2D input."""
    def __init__(self, in_ch, out_ch=1, spatial_dims=3, resnet_variant=None):
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

        self.model = Model(n_input_channels=in_ch, spatial_dims=spatial_dims, num_classes=out_ch)

    def forward(self, x):
        return self.model(x)

'''
class ResNet(BasicClassifier):
    """ResNet-based classifier using MONAI backbones."""
    def __init__(self, in_ch, out_ch=1, spatial_dims=3, model=34, **kwargs):
        super().__init__(in_ch, out_ch, spatial_dims, **kwargs)
        self.model = _ResNet(in_ch, out_ch, spatial_dims, model)

    def forward(self, x):
        return self.model(x)
'''

class ResNetRegression(BasicRegression):
    """ResNet-based regression model using MONAI backbones."""
    def __init__(self, in_ch, out_ch, spatial_dims, resnet_variant, **kwargs):
        super().__init__(in_ch, out_ch, spatial_dims, **kwargs)
        self.model = _ResNet(in_ch, out_ch, spatial_dims, resnet_variant)

    def forward(self, x):
        return self.model(x)
