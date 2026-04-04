import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from .base_model import BasicClassifier


class _Swin3D(nn.Module):
    """
    3D Swin Transformer for volumetric MRI classification.
    """

    def __init__(
        self,
        out_ch: int,
        img_size=(64, 224, 224),
        in_ch=1,
        feature_size=48,
        pretrained=False,
    ):
        super().__init__()

        self.encoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_ch,
            out_channels=feature_size,   # dummy
            feature_size=feature_size,
            use_checkpoint=False,
        )

        self.embed_dim = feature_size * 8  # final Swin stage
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(self.embed_dim, out_ch)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        # SwinUNETR returns encoder features at multiple stages
        _, _, _, _, x = self.encoder(x)

        # x: (B, C, D', H', W')
        x = self.pool(x).flatten(1)
        x = self.linear(x)
        return x


class Swin3D(BasicClassifier):
    """
    3D Swin Transformer classifier (full-volume attention).
    """

    def __init__(
        self,
        n_input_channels: int,
        num_classes: int,
        spatial_dims: int,
        img_size=(64, 224, 224),
        feature_size=48,
        optimizer_kwargs={"lr": 1e-6},
        **kwargs,
    ):
        super().__init__(
            n_input_channels,
            num_classes,
            spatial_dims,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

        self.model = _Swin3D(
            out_ch=num_classes,
            img_size=img_size,
            in_ch=n_input_channels,
            feature_size=feature_size,
        )

    def forward(self, x):
        return self.model(x)
