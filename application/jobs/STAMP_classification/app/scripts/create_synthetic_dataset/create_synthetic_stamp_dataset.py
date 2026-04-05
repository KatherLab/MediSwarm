#!/usr/bin/env python3
"""Create synthetic STAMP-compatible datasets for integration testing.

Generates 2 sites x 15 patients (5 per class for 3 classes) with:
- H5 feature files matching STAMP 2.4.0's expected format
  (tile-level: 'feats' shape (N_tiles, dim_input), 'coords' shape (N_tiles, 2))
- Clinical CSV tables with PATIENT and ground truth columns

Each class has a distinct feature distribution (mean shift) so models
can learn during short integration test runs.

Usage:
    python create_synthetic_stamp_dataset.py <output_folder>

Output structure:
    <output_folder>/
        client_A/
            features/
                P_000.h5 ... P_014.h5
            clini_table.csv
        client_B/
            features/
                P_000.h5 ... P_014.h5
            clini_table.csv
"""

import csv
import os
import pathlib
import shutil
import sys

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
NUM_PATIENTS_PER_SITE = 15  # 5 per class x 3 classes
SITES = ("client_A", "client_B")
CLASSES = ("class_0", "class_1", "class_2")
DIM_INPUT = 1024  # Standard STAMP feature dimension (e.g., UNI, CTransPath)
TILE_COUNT_RANGE = (20, 60)  # Random number of tiles per patient
COORD_RANGE = (0, 50000)  # Micron coordinate range (realistic WSI coords)
PATIENT_LABEL = "PATIENT"
GROUND_TRUTH_LABEL = "Diagnosis"

# Class-specific mean shifts in feature space so a model can learn
# Each class gets a distinct offset added to random normal features
CLASS_MEAN_SHIFTS = {
    "class_0": -1.0,
    "class_1": 0.0,
    "class_2": 1.0,
}


# ---------------------------------------------------------------------------
# H5 file creation
# ---------------------------------------------------------------------------


def create_h5_feature_file(
    filepath: pathlib.Path,
    rng: np.random.RandomState,
    class_label: str,
    dim_input: int = DIM_INPUT,
) -> None:
    """Create a single STAMP-compatible H5 feature file.

    Format matches STAMP 2.4.0's TileBagDataset expectations:
    - 'feats': float32 array of shape (N_tiles, dim_input)
    - 'coords': float32 array of shape (N_tiles, 2) — tile coordinates in um
    - attrs['feat_type'] = 'tile'

    Features are random normal with a class-specific mean shift so that
    different classes are linearly separable.
    """
    n_tiles = rng.randint(TILE_COUNT_RANGE[0], TILE_COUNT_RANGE[1] + 1)
    mean_shift = CLASS_MEAN_SHIFTS[class_label]

    # Generate features: random normal + class-specific shift
    feats = rng.randn(n_tiles, dim_input).astype(np.float32) * 0.5 + mean_shift

    # Generate coordinates: random positions in a realistic WSI coordinate space
    coords = rng.uniform(
        COORD_RANGE[0], COORD_RANGE[1], size=(n_tiles, 2)
    ).astype(np.float32)

    with h5py.File(filepath, "w") as h5:
        h5.create_dataset("feats", data=feats, compression="gzip", compression_opts=4)
        h5.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
        h5.attrs["feat_type"] = "tile"
        h5.attrs["encoder"] = "synthetic-test"
        # STAMP v2 coordinate format: tile_size + unit="um" tells get_coords()
        # to interpret coords directly as micron positions
        h5.attrs["tile_size"] = 256
        h5.attrs["unit"] = "um"


# ---------------------------------------------------------------------------
# Clinical table creation
# ---------------------------------------------------------------------------


def create_clini_table(
    filepath: pathlib.Path,
    patient_ids: list,
    class_labels: list,
) -> None:
    """Create a STAMP-compatible clinical CSV table.

    Columns: PATIENT, Diagnosis
    This is the minimal table needed by STAMP's load_patient_level_data().
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[PATIENT_LABEL, GROUND_TRUTH_LABEL]
        )
        writer.writeheader()
        for pid, label in zip(patient_ids, class_labels):
            writer.writerow({PATIENT_LABEL: pid, GROUND_TRUTH_LABEL: label})


# ---------------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------------


def create_folder_structure(output_folder: pathlib.Path) -> None:
    """Create clean output directory structure."""
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)
    for site in SITES:
        os.makedirs(output_folder / site / "features", exist_ok=True)


def generate_site_data(
    output_folder: pathlib.Path,
    site: str,
    rng: np.random.RandomState,
) -> None:
    """Generate all data for a single site."""
    site_dir = output_folder / site
    feature_dir = site_dir / "features"

    patient_ids = []
    class_labels = []

    for j in range(NUM_PATIENTS_PER_SITE):
        patient_id = f"P_{j:03d}"
        class_label = CLASSES[j % len(CLASSES)]

        patient_ids.append(patient_id)
        class_labels.append(class_label)

        # Create H5 feature file
        h5_path = feature_dir / f"{patient_id}.h5"
        create_h5_feature_file(h5_path, rng, class_label)

    # Create clinical table
    clini_path = site_dir / "clini_table.csv"
    create_clini_table(clini_path, patient_ids, class_labels)

    print(f"  {site}: {len(patient_ids)} patients, {len(CLASSES)} classes")
    print(f"    Features: {feature_dir}")
    print(f"    Clinical table: {clini_path}")


def main(output_folder: pathlib.Path) -> None:
    """Generate complete synthetic STAMP dataset."""
    print(f"Creating synthetic STAMP dataset in: {output_folder}")
    print(f"  Sites: {SITES}")
    print(f"  Patients per site: {NUM_PATIENTS_PER_SITE}")
    print(f"  Classes: {CLASSES}")
    print(f"  Feature dim: {DIM_INPUT}")
    print()

    rng = np.random.RandomState(SEED)

    create_folder_structure(output_folder)

    for site in SITES:
        generate_site_data(output_folder, site, rng)

    print()
    print("Synthetic STAMP dataset created successfully.")
    print(f"Patient label: {PATIENT_LABEL}")
    print(f"Ground truth label: {GROUND_TRUTH_LABEL}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_folder>")
        sys.exit(1)

    output_folder = pathlib.Path(sys.argv[1])
    main(output_folder)
