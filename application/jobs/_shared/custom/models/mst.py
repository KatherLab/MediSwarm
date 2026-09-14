import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from x_transformers import Encoder

from .base_model import BasicClassifier

logger = logging.getLogger(__name__)

# SAM-Med2D ViT-B encoder (contributed by Zhongying Deng, Cambridge): the published checkpoint
# was trained at 256 x 256 with the encoder adapter enabled, so both are fixed here.
SAM_MED2D_IMAGE_SIZE = 256
SAM_MED2D_CHECKPOINT_NAME = "sam-med2d_b.pth"
SAM_MED2D_CHECKPOINT_CANDIDATES = (
    "/MediSwarm/pretrained_weights/" + SAM_MED2D_CHECKPOINT_NAME,
    os.path.join(os.environ.get("TORCH_HOME", "/torch_home"), "hub", "checkpoints", SAM_MED2D_CHECKPOINT_NAME),
)


def resolve_sam_med2d_checkpoint():
    """Path of the SAM-Med2D weights, or None (random init) if none is present.

    `SAM_MED2D_CHECKPOINT` wins when set; otherwise the image's pretrained-weights directory and
    the torch hub cache are tried, in that order.
    """
    env = os.environ.get("SAM_MED2D_CHECKPOINT")
    if env:
        if os.path.isfile(env):
            return env
        raise FileNotFoundError(f"SAM_MED2D_CHECKPOINT={env} does not exist")
    for candidate in SAM_MED2D_CHECKPOINT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def build_sam_med2d_encoder():
    """The SAM-Med2D ViT-B image encoder with adapter layers; pretrained if the weights are found."""
    from .segment_anything import sam_model_registry

    checkpoint = resolve_sam_med2d_checkpoint()
    if checkpoint is None:
        logger.warning(
            "SAM-Med2D weights (%s) not found — the MST_SAMMed2D backbone starts from RANDOM weights. "
            "Fine for a preflight check, not for training; see models/segment_anything/README.md.",
            SAM_MED2D_CHECKPOINT_NAME,
        )
    else:
        logger.info(f"SAM-Med2D backbone weights: {checkpoint}")

    class _Args:
        image_size = SAM_MED2D_IMAGE_SIZE
        sam_checkpoint = checkpoint
        encoder_adapter = True

    sam = sam_model_registry["vit_b"](_Args())
    return sam.image_encoder  # prompt encoder and mask decoder are dropped with `sam`


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
        elif backbone_type == "sammed2d":
            self.backbone = build_sam_med2d_encoder()
            emb_ch = 256  # the encoder neck's output channels (B, 256, 16, 16) at 256 px
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

        if self.backbone_type == "sammed2d":
            # The dataset delivers 224 x 224 slices; the SAM-Med2D encoder is fixed at 256 x 256.
            # Cambridge's experiment cropped to 256 at load time instead; resizing here keeps
            # the shared data pipeline unchanged for every other model.
            if x.shape[-2:] != (SAM_MED2D_IMAGE_SIZE, SAM_MED2D_IMAGE_SIZE):
                x = F.interpolate(x, size=(SAM_MED2D_IMAGE_SIZE, SAM_MED2D_IMAGE_SIZE),
                                  mode="bilinear", align_corners=False)
            x = self.backbone(x)  # (B * D, 256, 16, 16)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B * D, 256)
        else:
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
    """MST-based classifier: a 2D slice encoder (DINOv2 ViT-S by default, or the SAM-Med2D
    ViT-B with `backbone_type="sammed2d"`) followed by transformer slice fusion."""

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
