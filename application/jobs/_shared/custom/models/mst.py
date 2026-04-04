import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

from .base_model import BasicClassifier


class TransformerEncoder(Encoder):
    """Override the default forward to match input formatting."""

    def forward(self, x, mask=None, src_key_padding_mask=None):
        src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        mask = ~mask if mask is not None else None
        return super().forward(x=x, context=None, mask=src_key_padding_mask, context_mask=None, attn_mask=mask)


class _MST(nn.Module):
    """Multi-slice transformer for 3D volume input classification or regression."""

    def __init__(
            self,
            out_ch=1,
            backbone_type="dinov2",
            model_size=None,
            slice_fusion_type="transformer"
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.slice_fusion_type = slice_fusion_type

        if backbone_type == "dinov2":
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            self.backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
            self.backbone.mask_token = None
            emb_ch = self.backbone.num_features
        else:
            raise ValueError("Unknown backbone_type")

        self.emb_ch = emb_ch

        if slice_fusion_type == "transformer":
            self.slice_fusion = TransformerEncoder(
                dim=emb_ch,
                heads=12 if emb_ch % 12 == 0 else 8,
                ff_mult=1,
                attn_dropout=0.0,
                pre_norm=True,
                depth=1,
                attn_flash=True,
                ff_no_bias=True,
                rotary_pos_emb=True,
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion_type in ["average", "none"]:
            self.slice_fusion = None
        else:
            raise ValueError("Unknown slice_fusion_type")

        self.linear = nn.Linear(emb_ch, out_ch)

    def forward(self, x):
        B, *_ = x.shape
        x = rearrange(x, 'b c d h w -> (b c d) h w')
        x = x[:, None].repeat(1, 3, 1, 1)  # Gray to RGB

        x = self.backbone(x)  # (B * D, E)
        x = rearrange(x, '(b d) e -> b d e', b=B)

        if self.slice_fusion_type == 'none':
            return x
        elif self.slice_fusion_type == 'transformer':
            x = torch.cat([x, self.cls_token.repeat(B, 1, 1)], dim=1)
            x = self.slice_fusion(x)
        elif self.slice_fusion_type == 'average':
            x = x.mean(dim=1, keepdim=True)

        x = self.linear(x[:, -1])
        return x


class MST(BasicClassifier):
    """MST-based classifier using ViT or ResNet as backbone."""

    def __init__(
            self,
            n_input_channels: int,
            num_classes: int,
            spatial_dims: int,
            backbone_type="dinov2",
            model_size="s",
            slice_fusion_type="transformer",
            optimizer_kwargs={'lr': 1e-6},
            **kwargs
    ):
        super().__init__(n_input_channels, num_classes, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)
        self.mst = _MST(out_ch=num_classes, backbone_type=backbone_type, model_size=model_size,
                        slice_fusion_type=slice_fusion_type)

    def forward(self, x):
        return self.mst(x)
