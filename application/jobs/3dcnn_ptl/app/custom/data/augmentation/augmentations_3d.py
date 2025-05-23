import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from x_transformers import Encoder
from typing import Union, Optional, Sequence
import torchio as tio
from torchio.typing import TypeRangeFloat, TypeTripletInt
from torchio.transforms.transform import TypeMaskingMethod
from torchio import Subject, Image
import numpy as np

from base_model import BasicClassifier, BasicRegression


def _get_resnet_torch(model):
    """Retrieve the specified ResNet model from torchvision."""
    return {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152
    }.get(model)


class TransformerEncoder(Encoder):
    """Override the default forward to match input formatting."""
    def forward(self, x, mask=None, src_key_padding_mask=None):
        src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        mask = ~mask if mask is not None else None
        return super().forward(x=x, context=None, mask=src_key_padding_mask, context_mask=None, attn_mask=mask)


class _MST(nn.Module):
    """Multi-slice transformer for 3D volume input classification or regression."""
    def __init__(self, out_ch=1, backbone_type="dinov2", model_size="s", slice_fusion_type="transformer"):
        super().__init__()
        self.backbone_type = backbone_type
        self.slice_fusion_type = slice_fusion_type

        if backbone_type == "resnet":
            Model = _get_resnet_torch(model_size)
            self.backbone = Model(weights="DEFAULT")
            emb_ch = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_type == "dinov2":
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
        in_ch=1,
        out_ch=1,
        spatial_dims=3,
        backbone_type="dinov2",
        model_size="s",
        slice_fusion_type="transformer",
        optimizer_kwargs={'lr': 1e-6},
        **kwargs
    ):
        super().__init__(in_ch, out_ch, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)
        self.mst = _MST(out_ch=out_ch, backbone_type=backbone_type, model_size=model_size, slice_fusion_type=slice_fusion_type)

    def forward(self, x):
        return self.mst(x)


class MSTRegression(BasicRegression):
    """MST-based regression model."""
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        spatial_dims=3,
        backbone_type="dinov2",
        model_size="s",
        slice_fusion_type="transformer",
        optimizer_kwargs={'lr': 1e-6},
        **kwargs
    ):
        super().__init__(in_ch, out_ch, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)
        self.mst = _MST(out_ch=out_ch, backbone_type=backbone_type, model_size=model_size, slice_fusion_type=slice_fusion_type)

    def forward(self, x):
        return self.mst(x)


class ImageOrSubjectToTensor(object):
    """Converts a torchio Image or Subject to a tensor format by swapping axes."""
    def __call__(self, input: Union[Image, Subject]):
        if isinstance(input, Subject):
            return {key: val.data.swapaxes(1, -1) if isinstance(val, Image) else val for key, val in input.items()}
        else:
            return input.data.swapaxes(1, -1)


def parse_per_channel(per_channel, channels):
    if isinstance(per_channel, bool):
        return [(ch,) for ch in range(channels)] if per_channel else [tuple(range(channels))]
    return per_channel


class ZNormalization(tio.ZNormalization):
    """Z-Normalization with support for per-channel and per-slice options, and percentile-based clipping."""
    def __init__(
        self,
        percentiles: TypeRangeFloat = (0, 100),
        per_channel=True,
        per_slice=False,
        masking_method: TypeMaskingMethod = None,
        **kwargs
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.percentiles = percentiles
        self.per_channel = per_channel
        self.per_slice = per_slice

    def apply_normalization(self, subject: Subject, image_name: str, mask: torch.Tensor) -> None:
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])
        per_slice = parse_per_channel(self.per_slice, image.shape[-1])

        image.set_data(torch.cat([
            torch.cat([
                self._znorm(image.data[chs, ..., sl], mask[chs, ..., sl], image_name, image.path)
                for sl in per_slice
            ], dim=-1)
            for chs in per_channel
        ]))

    def _znorm(self, image_data, mask, image_name, image_path):
        cutoff = torch.quantile(image_data.masked_select(mask).float(), torch.tensor(self.percentiles) / 100.0)
        torch.clamp(image_data, *cutoff.to(image_data.dtype).tolist(), out=image_data)
        standardized = self.znorm(image_data, mask)
        if standardized is None:
            raise RuntimeError(
                f'Standard deviation is 0 for masked values in image "{image_name}" ({image_path})'
            )
        return standardized


class CropOrPad(tio.CropOrPad):
    """Crop or pad a subject with optional random center logic for padding."""
    def __init__(
        self,
        target_shape: Union[int, TypeTripletInt, None] = None,
        padding_mode: Union[str, float] = 0,
        mask_name: Optional[str] = None,
        labels: Optional[Sequence[int]] = None,
        random_center=False,
        **kwargs
    ):
        super().__init__(
            target_shape=target_shape,
            padding_mode=padding_mode,
            mask_name=mask_name,
            labels=labels,
            **kwargs
        )
        self.random_center = random_center

    def _get_six_bounds_parameters(self, parameters: np.ndarray):
        result = []
        for number in parameters:
            ini = np.random.randint(low=0, high=number + 1) if self.random_center else int(np.ceil(number / 2))
            fin = number - ini
            result.extend([ini, fin])
        return tuple(result)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}

        if padding_params is not None:
            if self.random_center:
                padding_params = [
                    np.random.randint(0, s + 1) if i % 2 == 0 else s - val
                    for i, (s, val) in enumerate(zip(padding_params[::2], padding_params[1::2]))
                    for _ in (0, 1)
                ]
            pad = tio.Pad(padding_params, **padding_kwargs)
            subject = pad(subject)

        if cropping_params is not None:
            crop = tio.Crop(cropping_params)
            subject = crop(subject)

        return subject
