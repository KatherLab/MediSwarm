# MediSwarm Participant Guide — STAMP (Computational Pathology)

This guide is for data scientists and medical research sites participating in a
STAMP (Solid Tumor Associative Modeling in Pathology) swarm learning project. If
you are joining an ODELIA (breast MRI) project instead, see
[README.participant.md](README.participant.md).

## Overview

STAMP trains classification, regression, or survival models on **pre-extracted
feature vectors** from whole-slide pathology images (WSIs). Unlike ODELIA, which
trains on raw 3D NIfTI volumes, STAMP expects HDF5 (`.h5`) feature files that
have already been extracted from WSIs using a foundation model (e.g. UNI,
CTransPath, RetCCL).

Only the **training** step is federated. Feature extraction (preprocessing) and
inference (deployment) remain standalone steps that each site runs locally.

## Prerequisites

- **Hardware:** Min. 32 GB RAM, 8 cores, NVIDIA GPU with 12+ GB VRAM, 1 TB storage
  (STAMP is lighter on GPU memory than ODELIA because it operates on extracted
  features rather than full imaging volumes)
- **OS:** Ubuntu 20.04 LTS, 22.04 LTS, or 24.04 LTS
- **Software:** Docker, OpenVPN
- **Data:** Pre-extracted H5 feature files + clinical metadata CSV (see
  [Prepare Dataset](#prepare-dataset) below)

## Setup

0. Add this line to your `/etc/hosts`: `172.24.4.65 dl3.tud.de dl3`
1. Make sure your compute node satisfies the specification and has the necessary
   software installed.
2. Set up the VPN. A VPN is necessary so that the swarm nodes can communicate
   with each other securely across firewalls.
    1. Install OpenVPN
       ```bash
       sudo apt-get install openvpn
       ```
    2. If you have a graphical user interface (GUI), follow this guide to connect
       to the VPN: [VPN setup guide(GUI).pdf](../VPN%20setup%20guide%28GUI%29.pdf)
    3. If you have a command line interface (CLI), follow this guide to connect
       to the VPN: [VPN setup guide(CLI).md](../VPN%20setup%20guide%28CLI%29.md)
    4. You may want to clone this repository or selectively download VPN-related
       scripts for this purpose.

## Prepare Dataset

STAMP requires two inputs per site: a **feature directory** containing one H5
file per slide/patient, and a **clinical table** (CSV) with ground-truth labels.

### Feature Extraction (Prerequisite)

Before joining a swarm project you must extract features from your whole-slide
images using the agreed-upon foundation model and tile size. This is done with
standalone [STAMP](https://github.com/KatherLab/STAMP) preprocessing and
produces one `.h5` file per slide.

Coordinate with your swarm operator to ensure all sites use the **same feature
extractor, tile size, and feature dimension** — mismatched features will cause
training failures.

### Folder Structure

```
<name of your site>/
├── features/
│   ├── slide_001.h5
│   ├── slide_002.h5
│   ├── slide_003.h5
│   └── ...
└── clini_table.csv
```

* The name of your site should match your NVFlare site identifier (e.g.
  `UKA_1`), unless instructed otherwise by your swarm operator.
* The `features/` directory contains one `.h5` file per slide (or per patient,
  depending on feature extraction settings).

### H5 Feature File Format

Each `.h5` file must contain:

| Dataset   | Shape           | Dtype   | Description |
|-----------|-----------------|---------|-------------|
| `feats`   | (N_tiles, dim)  | float32 | Feature vectors (e.g. 1024-dim for UNI) |
| `coords`  | (N_tiles, 2)    | float32 | Tile coordinates (for tile-level features) |

Optional HDF5 attributes: `feat_type` (`tile`, `slide`, or `patient`),
`tile_size`, `unit`.

The feature dimension (`dim`) must match the `STAMP_DIM_INPUT` environment
variable (default: 1024 for UNI/UNI2; use 768 for CTransPath).

### Clinical Table Format

`clini_table.csv` is a CSV file with at minimum the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| Patient ID column | Unique patient identifier (default column name: `PATIENT`) | `P_001` |
| Ground truth column | Class label for classification, or continuous value for regression | `tumor`, `normal`, `adenoma` |

The patient ID column name is configured via `STAMP_PATIENT_LABEL` (default:
`PATIENT`) and the ground truth column via `STAMP_GROUND_TRUTH_LABEL`.

**Important:** The H5 filenames (without `.h5`) must match the patient
identifiers in the clinical table, or a separate slide table must be provided
to map between them.

#### Optional: Slide Table

If patients have multiple slides, provide a slide table (`STAMP_SLIDE_TABLE`)
that maps patient IDs to slide filenames.

### Synthetic Test Data

To verify your setup before using real data, you can generate a synthetic
dataset:

```bash
python application/jobs/STAMP_classification/app/scripts/create_synthetic_dataset/create_synthetic_stamp_dataset.py \
    --output_dir /tmp/stamp_synthetic_data
```

This creates two test sites (`client_A/`, `client_B/`) each with 15 synthetic
patients across 3 classes.

## Prepare Training Participation

1. Extract the startup kit provided by your swarm operator for the current
   experiment.

### Configure STAMP Environment Variables

STAMP uses environment variables (all prefixed with `STAMP_`) to configure the
training. These must be exported **before** calling `docker.sh` — the script
automatically forwards all `STAMP_*` variables into the Docker container.

```bash
# ── Required ──
export STAMP_CLINI_TABLE="/data/<SITE_NAME>/clini_table.csv"
export STAMP_FEATURE_DIR="/data/<SITE_NAME>/features"
export STAMP_GROUND_TRUTH_LABEL="<column name>"   # e.g. "Diagnosis"
export STAMP_PATIENT_LABEL="PATIENT"               # patient ID column name

# ── Task & Model ──
export STAMP_TASK="classification"                 # classification, regression, or survival
export STAMP_MODEL_NAME="vit"                      # vit, mlp, trans_mil, linear, barspoon
export STAMP_DIM_INPUT="1024"                      # must match your feature extractor
export STAMP_NUM_CLASSES="3"                        # number of output classes

# ── Training (defaults shown — adjust as needed) ──
export STAMP_BAG_SIZE="512"                        # tiles per bag
export STAMP_BATCH_SIZE="64"                       # training batch size
export STAMP_MAX_EPOCHS="32"                       # max epochs per round
export STAMP_PATIENCE="16"                         # early stopping patience
export STAMP_SEED="42"                             # random seed
export STAMP_NUM_WORKERS="0"                       # DataLoader workers (0 = main thread)
```

**Note:** The paths in `STAMP_CLINI_TABLE` and `STAMP_FEATURE_DIR` are
**container paths**. The host `--data_dir` is mounted at `/data/` inside the
container (read-only), so if your host data is at
`/mnt/data/stamp/UKA_1/features/`, pass `--data_dir /mnt/data/stamp` and set
`STAMP_FEATURE_DIR="/data/UKA_1/features"`.

Your swarm operator will tell you the correct values for `STAMP_MODEL_NAME`,
`STAMP_DIM_INPUT`, `STAMP_NUM_CLASSES`, `STAMP_GROUND_TRUTH_LABEL`, and
`STAMP_TASK` — these must be consistent across all sites.

### Available Models

| Model | `STAMP_MODEL_NAME` | Description |
|-------|---------------------|-------------|
| VIT | `vit` | Vision Transformer with ALiBi positional encoding |
| MLP | `mlp` | Multi-layer perceptron |
| TransMIL | `trans_mil` | Transformer-based multiple instance learning |
| Linear | `linear` | Linear classifier |
| Barspoon | `barspoon` | Encoder-decoder transformer |

### Local Testing on Your Data

1. Set up directories:
   ```bash
   export SITE_NAME=<name of your site, e.g., UKA_1>
   export DATADIR=<path to the folder containing your $SITE_NAME directory>
   export SCRATCHDIR=<path to where training can store temporary files>
   mkdir -p $SCRATCHDIR
   ```

2. From the directory where you unpacked the startup kit:
   ```bash
   cd $SITE_NAME/startup
   ```

3. Verify that your Docker/GPU setup is working:
   ```bash
   ./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training 2>&1 | tee dummy_training_console_output.txt
   ```
   * This will pull the Docker image, which might take a while.
   * If you have multiple GPUs and 0 is busy, use a different one.
   * The "training" itself should take less than a minute and does not use STAMP
     — it runs a minimal CNN on CIFAR-10 to verify Docker + GPU.

4. Export your STAMP environment variables (see
   [Configure STAMP Environment Variables](#configure-stamp-environment-variables)
   above), then verify that your local data can be accessed:
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check 2>&1 | tee preflight_check_console_output.txt
   ```
   * This runs 1 epoch of training on your local data as a quick smoke test.
   * Check `preflight_check_console_output.txt` for errors about missing H5
     files, mismatched patient IDs, or incorrect feature dimensions.

5. Check your local dataset for discrepancies:
   * There should be no duplicate patient IDs in the clinical table.
   * Every H5 file in the feature directory should have a corresponding entry in
     the clinical table (and vice versa).
   * Feature dimensions in H5 files should match `STAMP_DIM_INPUT`.
   * You can add `--log_dataset_details` for more detailed output:
     ```bash
     ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check --log_dataset_details
     ```
     (This output may contain confidential patient IDs — do not share it!)

### Run Local Training

To have a baseline for swarm training, train the same model locally:

1. From the directory where you unpacked the startup kit (unless you just ran the
   preflight check):
   ```bash
   cd $SITE_NAME/startup
   ```

2. Make sure your STAMP environment variables are exported, then start local
   training:
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --local_training 2>&1 | tee local_training_console_output.txt
   ```

3. Output files:
   * Console output: `startup/local_training_console_output.txt`
   * Training results are stored in `$SCRATCHDIR/` under a timestamped run
     directory, including:
     * Model checkpoints (`best.ckpt`, `last.ckpt`)
     * Training logs
     * Validation metrics

### Start Swarm Node

#### VPN

1. Connect to VPN as described in [VPN setup guide(GUI).pdf](../VPN%20setup%20guide%28GUI%29.pdf)
   (GUI) or [VPN setup guide(CLI).md](../VPN%20setup%20guide%28CLI%29.md)
   (command line).

#### Start the Client

1. From the directory where you unpacked the startup kit:
   ```bash
   cd $SITE_NAME/startup  # Skip this if you just ran the preflight check
   ```

2. Make sure your STAMP environment variables are exported (see
   [Configure STAMP Environment Variables](#configure-stamp-environment-variables)),
   then start the client:
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
   ```
   If you have multiple GPUs and 0 is busy, use a different one.

3. Console output is captured in `nohup.out`, which may have been created with
   limited permissions in the container, so make it readable if necessary:
   ```bash
   sudo chmod a+r nohup.out
   ```

4. Output files:
   * Console output: `startup/nohup.out`
   * NVFlare log: `log.txt` (in the startup kit root)
   * Training results are stored in `$SCRATCHDIR/` under a timestamped run
     directory
   * NVFlare job artifacts: `<JOB_ID>/app_$SITE_NAME/`
     * Aggregated model: `FL_global_model.pt`
     * Best aggregated model: `best_FL_global_model.pt`

## Troubleshooting

### Container Running Properly?

```bash
docker ps          # Check if stamp_swarm_client_$SITE_NAME is listed
nvidia-smi         # Check if the GPU is busy training
tail -f nohup.out  # Follow training log
```

For any issues, check if the commands above point to problems and contact your
Swarm Operator.

### STAMP Environment Variables Not Forwarded?

If training fails with errors about missing `STAMP_CLINI_TABLE` or
`STAMP_FEATURE_DIR`, verify that:

1. You exported the `STAMP_*` variables in the **same shell session** before
   calling `docker.sh`.
2. The paths use **container paths** (starting with `/data/`) not host paths.
3. Run `env | grep STAMP_` to confirm all variables are set.

### Data Issues?

* **"No H5 files found"** — Check that `STAMP_FEATURE_DIR` points to the
  correct directory inside the container (`/data/<SITE_NAME>/features`).
* **"Patient ID not found"** — Ensure H5 filenames (without `.h5`) match the
  patient ID column in the clinical table.
* **"Feature dimension mismatch"** — Ensure `STAMP_DIM_INPUT` matches the
  dimension of features in your H5 files (check with
  `python -c "import h5py; f=h5py.File('slide.h5','r'); print(f['feats'].shape)"`).

### Connection to Swarm Server Working?

Let the following command run for an hour or so:

```bash
ping dl3.tud.de
```

* If dl3.tud.de cannot be resolved, double-check whether it is contained in
  `/etc/hosts`.
* If it cannot be reached at all, double-check if the VPN connection is working.
* If intermittent packet loss occurs, double-check if your network connection is
  working properly. Creating new VPN credentials and certificate may also help —
  contact your Swarm Operator.

### Further Possible Issues

* Ensure `STAMP_GROUND_TRUTH_LABEL` exactly matches the column name in your
  clinical CSV (including capitalization).
* Ensure `STAMP_PATIENT_LABEL` exactly matches the patient ID column name.
* Feature files and clinical table entries need correct filenames including
  capitalization.
* Symlinks inside the data directory do not work — they are not available inside
  the Docker mount (which is read-only).
* The correct startup kit needs to be used. `SSLCertVerificationError` or
  `authentication failed` may indicate an incorrect startup kit incompatible
  with the current experiment.
* Do not start the VPN connection more than once on the same machine; do not use
  the same credentials on more than one machine at the same time.
* Disk full — see the [ODELIA participant guide](README.participant.md) for
  general disk management tips (Docker image cleanup, checkpoint accumulation,
  etc.).
* Docker storage configuration — if you have a small system partition and large
  data partition, configure Docker's `data-root` accordingly (see
  [ODELIA participant guide](README.participant.md) for details).
