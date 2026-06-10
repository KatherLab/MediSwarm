import json
import os
import time
from pathlib import Path
from typing import Union, Optional, Sequence

import torchio as tio
try:
    from torchio.typing import TypeRangeFloat, TypeTripletInt
except ModuleNotFoundError:
    from typing import Tuple, Union
    TypeRangeFloat = Union[float, Tuple[float, float]]
    TypeTripletInt = Tuple[int, int, int]
from torchio.transforms.transform import TypeMaskingMethod
from torchio import Subject, Image

import torch
import numpy as np
import math
import random

import torch
import torch.nn.functional as F


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


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

    @staticmethod
    def _zero_std_guard_log_path() -> Path:
        scratch_dir = os.getenv("SCRATCH_DIR") or os.getenv("SCRATCHDIR") or "/scratch"
        return Path(scratch_dir) / "odelia_zero_std_guard.jsonl"

    @staticmethod
    def _stringify_image_path(image_path) -> str:
        if image_path is None:
            return ""
        if isinstance(image_path, (list, tuple)):
            return ";".join(str(path) for path in image_path)
        return str(image_path)

    @staticmethod
    def _maybe_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _json_safe_percentiles(percentiles):
        try:
            return torch.as_tensor(percentiles).detach().cpu().tolist()
        except (TypeError, ValueError):
            return str(percentiles)

    def _write_zero_std_guard_event(
        self,
        *,
        reason: str,
        image_name: str,
        image_path,
        masked_voxels: int,
        pre_clamp_std: float | None,
        post_clamp_std: float | None = None,
    ) -> None:
        log_path = self._zero_std_guard_log_path()
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
            "reason": reason,
            "image_name": str(image_name),
            "image_path": self._stringify_image_path(image_path),
            "masked_voxels": int(masked_voxels),
            "pre_clamp_std": self._maybe_float(pre_clamp_std),
            "post_clamp_std": self._maybe_float(post_clamp_std),
            "percentiles": self._json_safe_percentiles(self.percentiles),
        }

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            print(
                f"[ODELIA_ZERO_STD_GUARD_LOG_FAILED] reason={reason} "
                f"image_name={image_name} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            return

        detail = ""
        if os.getenv("ODELIA_ZERO_STD_GUARD_LOG_DETAILS", "").lower() in _TRUTHY_ENV_VALUES:
            detail = f" image_path={record['image_path']}"
        print(
            f"[ODELIA_ZERO_STD_GUARD] reason={reason} image_name={image_name} "
            f"masked_voxels={masked_voxels} details_file={log_path}{detail}",
            flush=True,
        )

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
        masked = image_data.masked_select(mask).float()
        masked_voxels = masked.numel()
        pre_clamp_std = masked.std(unbiased=False).item() if masked_voxels else None

        def _zeros(reason: str, post_clamp_std: float | None = None):
            self._write_zero_std_guard_event(
                reason=reason,
                image_name=image_name,
                image_path=image_path,
                masked_voxels=masked_voxels,
                pre_clamp_std=pre_clamp_std,
                post_clamp_std=post_clamp_std,
            )
            return torch.zeros_like(image_data)

        if masked_voxels < 2:
            return _zeros("empty_or_too_small_mask")

        percentiles = torch.tensor(self.percentiles, device=masked.device, dtype=masked.dtype) / 100.0
        cutoff = torch.quantile(masked, percentiles)
        torch.clamp(image_data, *cutoff.to(image_data.dtype).tolist(), out=image_data)
        standardized = self.znorm(image_data, mask)
        if standardized is None:
            post_clamp_values = image_data.masked_select(mask).float()
            post_clamp_std = post_clamp_values.std(unbiased=False).item() if post_clamp_values.numel() else None
            return _zeros("zero_std_after_masking_or_clipping", post_clamp_std=post_clamp_std)
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
