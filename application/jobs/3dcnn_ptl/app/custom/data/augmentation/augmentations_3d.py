import torchio as tio
from typing import Tuple, Union, Optional, Dict, Sequence
from numbers import Number
import nibabel as nib
import numpy as np
import torch
from torchio.typing import TypeRangeFloat, TypeTripletInt
from torchio.transforms.transform import TypeMaskingMethod
from torchio import Subject, Image


class SubjectToTensor:
    """Transforms TorchIO Subjects into a Python dict and changes axes order from TorchIO to Torch."""

    def __call__(self, subject: Subject) -> Dict[str, torch.Tensor]:
        """Transforms the given subject.

        Args:
            subject (Subject): The subject to be transformed.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with transformed subject data.
        """
        return {key: val.data.swapaxes(1, -1) if isinstance(val, Image) else val for key, val in subject.items()}


class ImageToTensor:
    """Transforms TorchIO Image into a Numpy/Torch Tensor and changes axes order from TorchIO [B, C, W, H, D] to Torch [B, C, D, H, W]."""

    def __call__(self, image: Image) -> torch.Tensor:
        """Transforms the given image.

        Args:
            image (Image): The image to be transformed.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        return image.data.swapaxes(1, -1)


def parse_per_channel(per_channel: Union[bool, list], channels: int) -> list:
    """Parses the per_channel argument.

    Args:
        per_channel (Union[bool, list]): Whether to apply per channel.
        channels (int): The number of channels.

    Returns:
        list: A list of channel tuples.
    """
    if isinstance(per_channel, bool):
        if per_channel:
            return [(ch,) for ch in range(channels)]
        else:
            return [tuple(ch for ch in range(channels))]
    else:
        return per_channel


class ZNormalization(tio.ZNormalization):
    """Add option 'per_channel' to apply znorm for each channel independently and percentiles to clip values first."""

    def __init__(
            self,
            percentiles: TypeRangeFloat = (0, 100),
            per_channel: Union[bool, list] = True,
            masking_method: TypeMaskingMethod = None,
            **kwargs
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.percentiles = percentiles
        self.per_channel = per_channel

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
    ) -> None:
        """Applies normalization to the given subject.

        Args:
            subject (Subject): The subject to normalize.
            image_name (str): The name of the image to normalize.
            mask (torch.Tensor): The mask tensor.
        """
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])

        image.set_data(torch.cat([
            self._znorm(image.data[chs,], mask[chs,], image_name, image.path)
            for chs in per_channel])
        )

    def _znorm(self, image_data: torch.Tensor, mask: torch.Tensor, image_name: str, image_path: str) -> torch.Tensor:
        """Applies z-normalization to the given image data.

        Args:
            image_data (torch.Tensor): The image data to normalize.
            mask (torch.Tensor): The mask tensor.
            image_name (str): The name of the image.
            image_path (str): The path of the image.

        Returns:
            torch.Tensor: The normalized image data.

        Raises:
            RuntimeError: If standard deviation is 0 for masked values.
        """
        cutoff = torch.quantile(image_data.masked_select(mask).float(), torch.tensor(self.percentiles) / 100.0)
        torch.clamp(image_data, *cutoff.to(image_data.dtype).tolist(), out=image_data)

        standardized = self.znorm(image_data, mask)
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image_path})'
            )
            raise RuntimeError(message)
        return standardized


class RescaleIntensity(tio.RescaleIntensity):
    """Add option 'per_channel' to apply rescale for each channel independently."""

    def __init__(
            self,
            out_min_max: TypeRangeFloat = (0, 1),
            percentiles: TypeRangeFloat = (0, 100),
            masking_method: TypeMaskingMethod = None,
            in_min_max: Optional[Tuple[float, float]] = None,
            per_channel: Union[bool, list] = True,
            # Bool or List of tuples containing channel indices that should be normalized together
            **kwargs
    ):
        super().__init__(out_min_max, percentiles, masking_method, in_min_max, **kwargs)
        self.per_channel = per_channel

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
    ) -> None:
        """Applies normalization to the given subject.

        Args:
            subject (Subject): The subject to normalize.
            image_name (str): The name of the image to normalize.
            mask (torch.Tensor): The mask tensor.
        """
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])

        image.set_data(torch.cat([
            self.rescale(image.data[chs,], mask[chs,], image_name)
            for chs in per_channel])
        )


class Pad(tio.Pad):
    """Fixed version of TorchIO Pad.

    Pads with zeros for LabelMaps independent of padding mode (e.g., don't pad with mean).
    Pads with global (not per axis) 'maximum', 'mean', 'median', 'minimum' if any of these padding modes were selected.
    """

    def apply_transform(self, subject: Subject) -> Subject:
        """Applies padding to the given subject.

        Args:
            subject (Subject): The subject to pad.

        Returns:
            Subject: The padded subject.
        """
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        for image in self.get_images(subject):
            new_origin = nib.affines.apply_affine(image.affine, -np.array(low))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            kwargs: Dict[str, Union[str, float]]
            if isinstance(self.padding_mode, Number):
                kwargs = {
                    'mode': 'constant',
                    'constant_values': self.padding_mode,
                }
            elif isinstance(image, tio.LabelMap):  # FIX
                kwargs = {
                    'mode': 'constant',
                    'constant_values': 0,
                }
            else:
                if self.padding_mode in ['maximum', 'mean', 'median', 'minimum']:
                    if self.padding_mode == 'maximum':
                        constant_values = image.data.min()
                    elif self.padding_mode == 'mean':
                        constant_values = image.data.to(torch.float).mean().to(image.data.dtype)
                    elif self.padding_mode == 'median':
                        constant_values = image.data.median()
                    elif self.padding_mode == 'minimum':
                        constant_values = image.data.min()
                    kwargs = {
                        'mode': 'constant',
                        'constant_values': constant_values,
                    }
                else:
                    kwargs = {'mode': self.padding_mode}
            pad_params = self.bounds_parameters
            paddings = (0, 0), pad_params[:2], pad_params[2:4], pad_params[4:]
            padded = np.pad(image.data, paddings, **kwargs)  # type: ignore[call-overload]  # noqa: E501
            image.set_data(torch.as_tensor(padded))
            image.affine = new_affine
        return subject


class UKA_CropOrPad(tio.CropOrPad):
    """CropOrPad. 
     random_center: Random center for crop and pad if no mask is set otherwise only random padding."""

    def __init__(
        self,
        target_shape: Union[int, TypeTripletInt, None] = None,
        padding_mode: Union[str, float] = 0,
        mask_name: Optional[str] = None,
        labels: Optional[Sequence[int]] = None,
        random_center=False,
        **kwargs,
    ):
        super().__init__(
            target_shape=target_shape,
            padding_mode=padding_mode,
            mask_name=mask_name,
            labels=labels,
            **kwargs
        )
        self.random_center = random_center

    def _get_six_bounds_parameters(self, parameters: np.ndarray) :
        result = []
        for number in parameters:
            if self.random_center:
                ini = np.random.randint(low=0, high=number+1)
            else:
                ini = int(np.ceil(number/2))
            fin = number-ini
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
                    r = np.random.randint(0, s+1)
                    random_padding_params.extend([r, s - r])
                padding_params = random_padding_params
            pad = tio.Pad(padding_params, **padding_kwargs)
            subject = pad(subject)  # type: ignore[assignment]
        if cropping_params is not None:
            crop = tio.Crop(cropping_params)
            subject = crop(subject)  # type: ignore[assignment]
        return subject

class CropOrPad(tio.CropOrPad):
    """Fixed version of TorchIO CropOrPad.

    Pads with zeros for LabelMaps independent of padding mode (e.g., don't pad with mean).
    Pads with global (not per axis) 'maximum', 'mean', 'median', 'minimum' if any of these padding modes were selected.
    """

    def apply_transform(self, subject: Subject) -> Subject:
        """Applies cropping or padding to the given subject.

        Args:
            subject (Subject): The subject to crop or pad.

        Returns:
            Subject: The cropped or padded subject.
        """
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None:
            pad = Pad(padding_params, **padding_kwargs)
            subject = pad(subject)  # type: ignore[assignment]
        if cropping_params is not None:
            crop = tio.Crop(cropping_params)
            subject = crop(subject)  # type: ignore[assignment]
        return subject
