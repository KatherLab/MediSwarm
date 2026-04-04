import argparse
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import blosc2
import nibabel as nib
import numpy as np
import torch

NIFTI_PATTERN = re.compile(r"(.+)_000([0-2])\.nii\.gz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess 3-channel nnU-Net-style NIfTI files into .b2nd files. "
            "Expected files per case: <id>_0000.nii.gz, <id>_0001.nii.gz, <id>_0002.nii.gz."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing *_0000.nii.gz, *_0001.nii.gz and *_0002.nii.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where .b2nd outputs will be written (one file per case id).",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(96, 224, 256),
        metavar=("D", "H", "W"),
        help="Final spatial size after preprocessing (default: 96 224 256).",
    )
    parser.add_argument(
        "--clevel",
        type=int,
        default=8,
        help="Blosc2 compression level (default: 8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output .b2nd files if present.",
    )
    return parser.parse_args()


def collect_cases(input_dir: Path) -> Dict[str, Dict[int, Path]]:
    cases: Dict[str, Dict[int, Path]] = {}
    for file_path in sorted(input_dir.glob("*.nii.gz")):
        match = NIFTI_PATTERN.fullmatch(file_path.name)
        if match is None:
            continue
        case_id = match.group(1)
        channel = int(match.group(2))
        cases.setdefault(case_id, {})[channel] = file_path

    complete_cases = {
        cid: ch for cid, ch in cases.items() if set(ch.keys()) == {0, 1, 2}
    }
    missing_cases = {
        cid: ch for cid, ch in cases.items() if set(ch.keys()) != {0, 1, 2}
    }
    if missing_cases:
        lines = []
        for cid, ch in sorted(missing_cases.items()):
            missing = sorted({0, 1, 2} - set(ch.keys()))
            lines.append(f"{cid}: missing channels {missing}")
        raise ValueError("Incomplete channel triplets detected:\n" + "\n".join(lines))
    if not complete_cases:
        raise ValueError(
            f"No complete cases found in {input_dir}. Expected files like <id>_0000.nii.gz."
        )
    return complete_cases


def load_case(
    channel_files: Dict[int, Path],
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    channels: List[np.ndarray] = []
    spacing: Tuple[float, float, float] | None = None
    shape: Tuple[int, int, int] | None = None

    for c in (0, 1, 2):
        nii = nib.load(str(channel_files[c]))
        arr = nii.get_fdata(dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(
                f"Expected 3D image for {channel_files[c]}, got shape {arr.shape}"
            )

        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(
                f"Shape mismatch in case {channel_files[c].name}: expected {shape}, got {arr.shape}"
            )

        current_spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
        if spacing is None:
            spacing = current_spacing
        elif any(abs(a - b) > 1e-6 for a, b in zip(spacing, current_spacing)):
            raise ValueError(
                f"Spacing mismatch in case {channel_files[c].name}: expected {spacing}, got {current_spacing}"
            )
        channels.append(arr)

    assert spacing is not None
    return np.stack(channels, axis=0), spacing


def crop_to_nonzero(data: np.ndarray) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    nonzero = np.any(data != 0, axis=0)
    if not np.any(nonzero):
        full_bbox = (
            slice(0, data.shape[1]),
            slice(0, data.shape[2]),
            slice(0, data.shape[3]),
        )
        return data, full_bbox

    coords = np.where(nonzero)
    zmin, ymin, xmin = (int(c.min()) for c in coords)
    zmax, ymax, xmax = (int(c.max()) + 1 for c in coords)
    bbox = (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))
    return data[(slice(None),) + bbox], bbox


def normalize_nonzero(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    for c in range(out.shape[0]):
        channel = out[c]
        mask = channel != 0
        if np.any(mask):
            vals = channel[mask]
            mean = float(vals.mean())
            std = float(vals.std())
            if std > 0:
                channel[mask] = (vals - mean) / std
            else:
                channel[mask] = 0.0
        else:
            channel.fill(0.0)
        out[c] = channel
    return out


def resample_to_shape(data: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    tensor = torch.from_numpy(data).unsqueeze(0)  # [1, C, D, H, W]
    resized = torch.nn.functional.interpolate(
        tensor,
        size=tuple(int(x) for x in target_shape),
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0).numpy().astype(np.float32, copy=False)


def comp_blosc2_params(
    image_size: Tuple[int, int, int, int],
    patch_size: Sequence[int],
    bytes_per_pixel: int = 4,
    l1_cache_size_per_core_in_bytes: int = 32768,
    l3_cache_size_per_core_in_bytes: int = 1441792,
    safety_factor: float = 0.8,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    num_channels = image_size[0]
    patch_size = np.array(patch_size)
    block_size = np.array(
        (
            num_channels,
            *[2 ** max(0, math.floor(math.log2(i / 2))) for i in patch_size],
        )
    )

    estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
    while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
        axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
        idx = 0
        picked_axis = axis_order[idx]
        while (
            block_size[picked_axis + 1] == 1
            or block_size[picked_axis + 1] == image_size[picked_axis + 1]
        ):
            idx += 1
            picked_axis = axis_order[idx]
        block_size[picked_axis + 1] = 2 ** max(
            0, math.floor(math.log2(block_size[picked_axis + 1] - 1))
        )
        block_size[picked_axis + 1] = min(
            block_size[picked_axis + 1], image_size[picked_axis + 1]
        )
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        if all(i == j for i, j in zip(block_size, image_size)):
            break

    chunk_size = deepcopy(block_size)
    estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
    while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
        axis_order = np.argsort(chunk_size[1:] / block_size[1:])
        idx = 0
        picked_axis = axis_order[idx]
        while (
            chunk_size[picked_axis + 1] == image_size[picked_axis + 1]
            or patch_size[picked_axis] == 1
        ):
            idx += 1
            picked_axis = axis_order[idx]
        chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
        chunk_size[picked_axis + 1] = min(
            chunk_size[picked_axis + 1], image_size[picked_axis + 1]
        )
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        if all(i == j for i, j in zip(chunk_size, image_size)):
            break

    return tuple(block_size), tuple(chunk_size)


def save_b2nd(
    data: np.ndarray,
    output_file: Path,
    target_shape: Sequence[int],
    clevel: int,
) -> None:
    blosc2.set_nthreads(1)
    blocks, chunks = comp_blosc2_params(
        image_size=tuple(int(x) for x in data.shape),
        patch_size=target_shape,
        bytes_per_pixel=data.itemsize,
    )
    blosc2.asarray(
        np.ascontiguousarray(data),
        urlpath=str(output_file),
        chunks=chunks,
        blocks=blocks,
        cparams={"codec": blosc2.Codec.ZSTD, "clevel": clevel},
        mmap_mode="w+",
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(input_dir)
    target_shape = tuple(int(x) for x in args.target_shape)

    print(f"Found {len(cases)} complete case(s) in {input_dir}")
    print(f"Target shape: {target_shape}")

    for idx, case_id in enumerate(sorted(cases.keys()), start=1):
        output_file = output_dir / f"{case_id}.b2nd"
        if output_file.exists() and not args.overwrite:
            print(f"[{idx}/{len(cases)}] Skipping {case_id} (exists)")
            continue

        data, spacing = load_case(cases[case_id])
        original_shape = tuple(int(x) for x in data.shape[1:])
        data, bbox = crop_to_nonzero(data)
        data = normalize_nonzero(data)
        data = resample_to_shape(data, target_shape)
        save_b2nd(data, output_file, target_shape, args.clevel)

        print(
            f"[{idx}/{len(cases)}] {case_id}: "
            f"orig={original_shape}, spacing={spacing}, "
            f"crop=({bbox[0].start}:{bbox[0].stop}, {bbox[1].start}:{bbox[1].stop}, {bbox[2].start}:{bbox[2].stop}) "
            f"-> out={tuple(data.shape)}"
        )

    print(f"Done. Wrote .b2nd files to {output_dir}")


if __name__ == "__main__":
    main()
