# Meisen Meister (Dummy Overfit Setup)

This repository contains a compact training setup that demonstrates all main pipeline parts:

- preprocessing NIfTI triplets (`_0000/_0001/_0002`) into `.b2nd` files
- loading 3-channel MRI volumes for classification
- applying 3D augmentations
- running a small overfit training loop with a pretrained encoder

The current training script is intentionally minimal and expects exactly 3 preprocessed cases to overfit on.

## Repository Structure

- `data/preprocess_nifti_to_blosc.py`: preprocess raw `.nii.gz` into `.b2nd`
- `data/dataset.py`: dataset class reading `.b2nd` and building train/val views
- `data/augmentation.py`: random 3D spatial + intensity augmentations
- `models/residual_encoder_classifier.py`: Lightning classification model
- `training/callbacks.py`, `training/schedulers.py`: logging and LR scheduling utilities
- `train_overfit.py`: end-to-end dummy overfit training entrypoint
- `config.py`: central config (data path, model/training/augmentation params)

## Installation

```bash
conda create -n meisenmeister python=3.12 -y
conda activate meisenmeister
pip install -r requirements.txt
```

## 1. Preprocess NIfTI to `.b2nd`

Input folder format:

- one file triplet per case ID
- `<case_id>_0000.nii.gz` (Pre)
- `<case_id>_0001.nii.gz` (Post 1)
- `<case_id>_0002.nii.gz` (Post 2)

You can calculate a dataset median shape first:

```bash
python data/calculate_median_shape.py \
  --input-dir /path/to/nifti_folder
```

Example:

```bash
python data/preprocess_nifti_to_blosc.py \
  --input-dir /path/to/nifti_folder \
  --output-dir /path/to/b2nd_out
```

Optional:

- `--target-shape D H W` (default: `96 224 256`, but should the median shape)
- `--overwrite`
- `--clevel` (blosc2 compression level, default: `8`)

What preprocessing does (per case):

1. stack 3 channels into `(3, D, H, W)`
2. crop to nonzero bounding box
3. nonzero z-score normalization per channel
4. trilinear resample to target shape
5. save `<case_id>.b2nd`

## 2. Run Dummy Overfit Training

Set your `.b2nd` directory in `config.py`:

- `DATA_DIR = Path(...)/your_b2nd_folder`

Then run:

```bash
python train_overfit.py
```

Notes:

- `DummyMRIDataset` currently expects exactly **three** `.b2nd` files in `DATA_DIR`
- all three volumes must have the same shape
- inputs are expected to be 3-channel (`IN_CHANNELS = 3`)

## Default Test Data

`test_data/` already contains 3 `.b2nd` files with shape `(3, 96, 224, 256)`.  
If you leave `DATA_DIR` unchanged in `config.py`, `train_overfit.py` runs directly on those.

## About This Setup

This is a focused, reproducible overfit scaffold to validate:

- preprocessing compatibility
- data loading + augmentation behavior
- model wiring and optimization loop

It is not a full production training pipeline yet.
