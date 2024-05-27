from .base_model import BasicClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer

class VisionTransformer(BasicClassifier):
    """
    VisionTransformer model for 3D classification tasks.

    Attributes:
        model (nn.Module): The VisionTransformer3D model.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        spatial_dims: int = 3,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = False,
        loss=torch.nn.BCEWithLogitsLoss,
        loss_kwargs: dict = {},
        optimizer=torch.optim.AdamW,
        optimizer_kwargs: dict = {'lr': 1e-4},
        lr_scheduler=None,
        lr_scheduler_kwargs: dict = {},
        aucroc_kwargs: dict = {"task": "binary"},
        acc_kwargs: dict = {"task": "binary"}
    ):
        """
        Initializes the VisionTransformer model with the given parameters.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            spatial_dims (int, optional): Number of spatial dimensions. Defaults to 3.
            model_name (str, optional): Name of the VisionTransformer model. Defaults to 'vit_base_patch16_224'.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to False.
            loss (callable, optional): Loss function. Defaults to torch.nn.BCEWithLogitsLoss.
            loss_kwargs (dict, optional): Keyword arguments for the loss function. Defaults to {}.
            optimizer (Optimizer, optional): Optimizer. Defaults to torch.optim.AdamW.
            optimizer_kwargs (dict, optional): Keyword arguments for the optimizer. Defaults to {'lr': 1e-4}.
            lr_scheduler (Scheduler, optional): Learning rate scheduler. Defaults to None.
            lr_scheduler_kwargs (dict, optional): Keyword arguments for the learning rate scheduler. Defaults to {}.
            aucroc_kwargs (dict, optional): Keyword arguments for AUROC. Defaults to {"task": "binary"}.
            acc_kwargs (dict, optional): Keyword arguments for Accuracy. Defaults to {"task": "binary"}.
        """
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)
        self.model = VisionTransformer3D(model_name, pretrained=pretrained, in_chans=in_ch, num_classes=out_ch)

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the VisionTransformer model.

        Args:
            x_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        print(x_in.shape)
        pred_hor = self.model(x_in)
        print(pred_hor.shape)
        pred_hor = self.model(x_in)
        return pred_hor

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding module for Vision Transformer.

    Attributes:
        proj (nn.Module): The convolutional projection layer.
    """

    def __init__(self, in_chans: int, embed_dim: int, patch_size: tuple):
        """
        Initializes the PatchEmbed3D module.

        Args:
            in_chans (int): Number of input channels.
            embed_dim (int): Embedding dimension.
            patch_size (tuple): Size of the patches.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbed3D module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        B, C, D, H, W = x.shape
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x

class VisionTransformer3D(TimmVisionTransformer):
    """
    3D Vision Transformer model extending TimmVisionTransformer.

    Attributes:
        patch_embed (nn.Module): The 3D Patch Embedding module.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the VisionTransformer3D model with the given parameters.

        Args:
            *args: Positional arguments for the TimmVisionTransformer.
            **kwargs: Keyword arguments for the TimmVisionTransformer.
        """
        super().__init__(*args, **kwargs)
        in_chans = kwargs.get("in_chans", 3)
        embed_dim = kwargs.get("embed_dim", 768)
        patch_size = kwargs.get("patch_size", (2, 16, 16))
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim, patch_size)
