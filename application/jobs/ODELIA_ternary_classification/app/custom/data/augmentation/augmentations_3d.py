from typing import Union, Optional, Sequence

import torchio as tio
from torchio.typing import TypeRangeFloat, TypeTripletInt
from torchio.transforms.transform import TypeMaskingMethod
from torchio import Subject, Image

import torch
import numpy as np
import math
import random

import torch
import torch.nn.functional as F


class Random3DAugmentation:

    def __init__(
        self,
        p_rotation=0.2,
        rotation_range_degrees=(-30.0, 30.0),
        p_scaling=0.2,
        scaling=(0.7, 1.4),
        p_noise=0.1,
        noise_variance=(0.0, 0.1),
        p_brightness=0.15,
        brightness_range=(0.75, 1.25),
        p_contrast=0.15,
        contrast_range=(0.75, 1.25),
        mirror_axes=(0, 1, 2),
    ):
        self.p_rotation = p_rotation
        self.rotation_range_degrees = rotation_range_degrees
        self.p_scaling = p_scaling
        self.scaling = scaling
        self.p_noise = p_noise
        self.noise_variance = noise_variance
        self.p_brightness = p_brightness
        self.brightness_range = brightness_range
        self.p_contrast = p_contrast
        self.contrast_range = contrast_range
        self.mirror_axes = mirror_axes

    @staticmethod
    def _rand_uniform(lo, hi):
        return lo + (hi - lo) * random.random()

    @staticmethod
    def _rotation_matrix(ax, ay, az, device, dtype):
        cx, sx = math.cos(ax), math.sin(ax)
        cy, sy = math.cos(ay), math.sin(ay)
        cz, sz = math.cos(az), math.sin(az)

        rx = torch.tensor(
            [[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=device, dtype=dtype
        )
        ry = torch.tensor(
            [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device, dtype=dtype
        )
        rz = torch.tensor(
            [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=device, dtype=dtype
        )
        return rz @ ry @ rx

    def _spatial(self, x):
        apply_rot = random.random() < self.p_rotation
        apply_scale = random.random() < self.p_scaling
        if not apply_rot and not apply_scale:
            return x

        min_deg, max_deg = self.rotation_range_degrees
        if apply_rot:
            ax = self._rand_uniform(min_deg, max_deg) * torch.pi / 180.0
            ay = self._rand_uniform(min_deg, max_deg) * torch.pi / 180.0
            az = self._rand_uniform(min_deg, max_deg) * torch.pi / 180.0
            rot = self._rotation_matrix(ax, ay, az, x.device, x.dtype)
        else:
            rot = torch.eye(3, device=x.device, dtype=x.dtype)

        scale = (
            self._rand_uniform(self.scaling[0], self.scaling[1]) if apply_scale else 1.0
        )
        theta = torch.zeros((1, 3, 4), device=x.device, dtype=x.dtype)
        theta[0, :3, :3] = rot * scale

        x_b = x.unsqueeze(0)
        grid = F.affine_grid(theta, size=x_b.shape, align_corners=False)
        x_b = F.grid_sample(
            x_b,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return x_b.squeeze(0)

    def __call__(self, x):
        x = self._spatial(x)

        if self.mirror_axes is not None:
            for axis in self.mirror_axes:
                if random.random() < 0.5:
                    x = torch.flip(x, dims=[axis + 1])

        if random.random() < self.p_noise:
            noise_std = self._rand_uniform(
                self.noise_variance[0], self.noise_variance[1]
            )
            x = x + torch.randn_like(x) * noise_std

        if random.random() < self.p_brightness:
            brightness = self._rand_uniform(
                self.brightness_range[0], self.brightness_range[1]
            )
            x = x * brightness

        if random.random() < self.p_contrast:
            contrast = self._rand_uniform(
                self.contrast_range[0], self.contrast_range[1]
            )
            mean = x.mean()
            x = (x - mean) * contrast + mean

        return x


class ImageOrSubjectToTensor(object):
    """Converts a torchio Image or Subject to a tensor format by swapping axes."""

    def __call__(self, input: Union[Image, Subject]):
        if isinstance(input, Subject):
            return {key: val.data.swapaxes(1, -1) if isinstance(val, Image) else val for key, val in input.items()}
        else:
            return input.data.swapaxes(1, -1)


def parse_per_channel(per_channel, channels):
    if isinstance(per_channel, bool):
        if per_channel == True:
            return [(ch,) for ch in range(channels)]
        else:
            return [tuple(ch for ch in range(channels))]
    else:
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

        image.set_data(
            torch.cat([
                torch.cat([
                    self._znorm(image.data[chs,][:, :, :, sl, ], mask[chs,][:, :, :, sl, ], image_name, image.path)
                    for sl in per_slice], dim=-1)
                for chs in per_channel])
        )

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
            if self.random_center:
                ini = np.random.randint(low=0, high=number + 1)
            else:
                ini = int(np.ceil(number / 2))
            fin = number - ini
            result.extend([ini, fin])
        return tuple(result)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}

        if padding_params is not None:
            if self.random_center:
                random_padding_params = []
                for i in range(0, len(padding_params), 2):
                    s = padding_params[i] + padding_params[i + 1]
                    r = np.random.randint(0, s + 1)
                    random_padding_params.extend([r, s - r])
                padding_params = random_padding_params
            pad = tio.Pad(padding_params, **padding_kwargs)
            subject = pad(subject)

        if cropping_params is not None:
            crop = tio.Crop(cropping_params)
            subject = crop(subject)

        return subject
