import random
from pathlib import Path

import blosc2
import torch
from torch.utils.data import Dataset


class DummyMRIDataset(Dataset):
    def __init__(
        self,
        num_classes=3,
        volumes=None,
        targets=None,
        augment=None,
        data_dir=None,
    ):
        """
        Args:
            num_classes: Number of classes (3: healthy, benign, malignant)
            volumes: Optional pre-generated list of tensors to share across datasets
            targets: Optional pre-generated list of class targets to share across datasets
            augment: Optional callable augmentation applied to source tensor only
            data_dir: Optional path containing exactly three .b2nd files
        """
        self.num_classes = num_classes
        self.augment = augment

        if volumes is not None and targets is not None:
            self.volumes = volumes
            self.targets = targets
        elif data_dir is not None:
            self.volumes, self.targets = self._load_three_cases_with_random_labels(
                data_dir=data_dir, num_classes=num_classes
            )
        else:
            raise ValueError("Provide either (volumes and targets) or data_dir")

    @staticmethod
    def _ensure_channel_first(vol):
        if vol.ndim == 3:
            return vol.unsqueeze(0)
        if vol.ndim == 4:
            # Accept channel-first directly (C, D, H, W)
            if vol.shape[0] <= 8:
                return vol
            # Fallback for channel-last inputs (D, H, W, C)
            if vol.shape[-1] <= 8:
                return vol.permute(3, 0, 1, 2)
        raise ValueError(f"Unsupported volume shape: {tuple(vol.shape)}")

    @classmethod
    def _load_b2nd_tensor(cls, file_path: Path):
        arr = blosc2.open(urlpath=str(file_path))[:]
        vol = torch.as_tensor(arr, dtype=torch.float32)
        return cls._ensure_channel_first(vol)

    @classmethod
    def _load_three_cases_with_random_labels(cls, data_dir, num_classes):
        data_path = Path(data_dir)
        files = sorted(data_path.glob("*.b2nd"))
        if len(files) != 3:
            raise ValueError(
                f"Expected exactly 3 .b2nd files in {data_path}, found {len(files)}"
            )
        if num_classes != 3:
            raise ValueError(
                "This setup expects num_classes=3 for three-case overfitting"
            )

        volumes = [cls._load_b2nd_tensor(fp) for fp in files]
        shapes = [tuple(v.shape) for v in volumes]
        ref_shape = shapes[0]
        if any(shape != ref_shape for shape in shapes):
            raise ValueError(
                "All 3 input volumes must have identical shape after channel handling. "
                f"Found shapes: {shapes}"
            )
        if ref_shape[0] != 3:
            raise ValueError(
                "This training script is configured for exactly 3-channel inputs. "
                f"Found {ref_shape[0]} channels."
            )
        print(f"Loaded .b2nd volumes with shape: {ref_shape}")

        targets = list(range(num_classes))
        random.shuffle(targets)
        return volumes, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        source = self.volumes[idx].clone()
        if self.augment is not None:
            source = self.augment(source)
        return {
            "source": source,
            "target": torch.tensor([self.targets[idx]]),
        }
