from typing import Union, Optional, Sequence

import torchio as tio
from torchio.typing import TypeRangeFloat, TypeTripletInt
from torchio.transforms.transform import TypeMaskingMethod
from torchio import Subject, Image

import torch
import numpy as np


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
