#!/usr/bin/env python3
"""Create synthetic STAMP-compatible datasets for integration testing.

Generates 2 sites x 15 patients (5 per latent class for 3 classes) with:
- H5 feature files matching STAMP 2.4.0's expected format
  (tile-level: 'feats' shape (N_tiles, dim_input), 'coords' shape (N_tiles, 2))
- Clinical CSV tables whose ground-truth column(s) depend on the task (#271):
    classification -> a single categorical `Diagnosis` column
    survival       -> continuous `Time` + binary `Event` columns
    regression     -> a single continuous `Target` column

Every task derives its label(s) from the same latent class that shifts the
features, so a model can still learn during short integration-test runs.

Usage:
    python create_synthetic_stamp_dataset.py <output_folder>              # classification
    python create_synthetic_stamp_dataset.py <output_folder> --task survival
    python create_synthetic_stamp_dataset.py <output_folder> --task regression

Output structure (unchanged across tasks; only clini_table.csv columns differ):
    <output_folder>/
        client_A/
            features/
                P_000.h5 ... P_014.h5
            clini_table.csv
        client_B/
            ...
"""

import argparse
import csv
import os
import pathlib
import shutil

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
GROUND_TRUTH_LABEL = "Diagnosis"        # classification
TIME_LABEL = "Time"                     # survival
EVENT_LABEL = "Event"                   # survival (1 = event observed, 0 = censored)
REGRESSION_LABEL = "Target"             # regression
TASKS = ("classification", "survival", "regression")

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


def _class_index(class_label: str) -> int:
    """Map 'class_0'/'class_1'/'class_2' -> 0/1/2."""
    return CLASSES.index(class_label)


def build_label_columns(
    task: str,
    patient_ids: list,
    class_labels: list,
    rng: np.random.RandomState,
) -> tuple:
    """Return (fieldnames, rows) for the task's clinical table.

    Labels are derived from the latent class so they correlate with the
    class-shifted features and remain learnable:
      - classification: Diagnosis = the categorical class
      - survival: Time increases with class index; Event ~ 70% observed
      - regression: Target increases linearly with class index
    """
    if task == "classification":
        fieldnames = [PATIENT_LABEL, GROUND_TRUTH_LABEL]
        rows = [
            {PATIENT_LABEL: pid, GROUND_TRUTH_LABEL: lbl}
            for pid, lbl in zip(patient_ids, class_labels)
        ]
    elif task == "survival":
        fieldnames = [PATIENT_LABEL, TIME_LABEL, EVENT_LABEL]
        rows = []
        for pid, lbl in zip(patient_ids, class_labels):
            idx = _class_index(lbl)
            time = round(float(10.0 + 15.0 * idx + rng.normal(0.0, 2.0)), 3)
            time = max(time, 0.1)
            event = int(rng.random() < 0.7)  # ~30% censored
            rows.append({PATIENT_LABEL: pid, TIME_LABEL: time, EVENT_LABEL: event})
    elif task == "regression":
        fieldnames = [PATIENT_LABEL, REGRESSION_LABEL]
        rows = []
        for pid, lbl in zip(patient_ids, class_labels):
            idx = _class_index(lbl)
            target = round(float(2.0 * idx + rng.normal(0.0, 0.3)), 4)
            rows.append({PATIENT_LABEL: pid, REGRESSION_LABEL: target})
    else:
        raise ValueError(f"Unknown task: {task} (expected one of {TASKS})")
    return fieldnames, rows


def create_clini_table(
    filepath: pathlib.Path,
    patient_ids: list,
    class_labels: list,
    task: str = "classification",
    rng: np.random.RandomState = None,
) -> None:
    """Create a STAMP-compatible clinical CSV table for the given task.

    This is the minimal table needed by STAMP's load_patient_level_data().
    """
    if rng is None:
        rng = np.random.RandomState(SEED)
    fieldnames, rows = build_label_columns(task, patient_ids, class_labels, rng)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    task: str = "classification",
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

        # Create H5 feature file (class-shifted so any task is learnable)
        h5_path = feature_dir / f"{patient_id}.h5"
        create_h5_feature_file(h5_path, rng, class_label)

    # Create clinical table (columns depend on the task)
    clini_path = site_dir / "clini_table.csv"
    create_clini_table(clini_path, patient_ids, class_labels, task=task, rng=rng)

    print(f"  {site}: {len(patient_ids)} patients, task={task}")
    print(f"    Features: {feature_dir}")
    print(f"    Clinical table: {clini_path}")


def main(output_folder: pathlib.Path, task: str = "classification") -> None:
    """Generate complete synthetic STAMP dataset for the given task."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task} (expected one of {TASKS})")

    print(f"Creating synthetic STAMP dataset in: {output_folder}")
    print(f"  Task: {task}")
    print(f"  Sites: {SITES}")
    print(f"  Patients per site: {NUM_PATIENTS_PER_SITE}")
    print(f"  Classes: {CLASSES}")
    print(f"  Feature dim: {DIM_INPUT}")
    print()

    rng = np.random.RandomState(SEED)

    create_folder_structure(output_folder)

    for site in SITES:
        generate_site_data(output_folder, site, rng, task=task)

    print()
    print("Synthetic STAMP dataset created successfully.")
    print(f"Patient label: {PATIENT_LABEL}")
    if task == "classification":
        print(f"Ground truth label: {GROUND_TRUTH_LABEL}")
    elif task == "survival":
        print(f"Survival labels: {TIME_LABEL} (time), {EVENT_LABEL} (event)")
    elif task == "regression":
        print(f"Regression label: {REGRESSION_LABEL}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic STAMP dataset.")
    parser.add_argument("output_folder", type=pathlib.Path, help="Output directory.")
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="classification",
        help="Task type governing the clinical-table label column(s) (default: classification).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.output_folder, task=args.task)
