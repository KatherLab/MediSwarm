#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np

NIFTI_PATTERN = re.compile(r"(.+)_000([0-2])\.nii\.gz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate median shape from nnU-Net-style triplets: "
            "<id>_0000.nii.gz, <id>_0001.nii.gz, <id>_0002.nii.gz."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing NIfTI triplets.",
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
        cid: channels
        for cid, channels in cases.items()
        if set(channels.keys()) == {0, 1, 2}
    }
    if not complete_cases:
        raise ValueError(
            f"No complete cases found in {input_dir}. Expected <id>_0000/_0001/_0002.nii.gz."
        )
    return complete_cases


def get_case_shape(channels: Dict[int, Path]) -> Tuple[int, int, int]:
    shapes = []
    for c in (0, 1, 2):
        shape = nib.load(str(channels[c])).shape
        if len(shape) != 3:
            raise ValueError(f"Expected 3D image for {channels[c]}, got shape {shape}")
        shapes.append(tuple(int(x) for x in shape))

    if not (shapes[0] == shapes[1] == shapes[2]):
        raise ValueError(
            f"Channel shape mismatch for case {channels[0].name}: {shapes}"
        )
    return shapes[0]


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    cases = collect_cases(input_dir)
    case_shapes = np.array(
        [get_case_shape(cases[cid]) for cid in sorted(cases.keys())], dtype=np.int64
    )

    median_shape_float = np.median(case_shapes, axis=0)
    median_shape = tuple(int(x) for x in np.rint(median_shape_float))

    print(f"Cases used: {len(case_shapes)}")
    print(f"Median shape (float): {tuple(float(x) for x in median_shape_float)}")
    print(f"Median shape (rounded): {median_shape}")
    print(
        f"Use for preprocessing: --target-shape {median_shape[0]} {median_shape[1]} {median_shape[2]}"
    )


if __name__ == "__main__":
    main()
