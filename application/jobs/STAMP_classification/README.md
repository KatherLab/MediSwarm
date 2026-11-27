# STAMP Classification — MediSwarm Swarm Learning Job

This job integrates [STAMP](https://github.com/KatherLab/STAMP) (Solid Tumor Associative Modeling in Pathology) with MediSwarm's NVFlare-based swarm learning framework.

## Overview

STAMP analyzes whole-slide images (WSIs) using pre-extracted deep learning features stored in HDF5 files. Unlike ODELIA's 3D CNN pipeline that trains directly on imaging volumes, STAMP works on tile/slide/patient-level feature vectors that have already been extracted from WSIs using foundation models (e.g., UNI, CTransPath, RetCCL).

Only the **training** section of STAMP is integrated into swarm learning. Preprocessing (feature extraction), deployment (inference), and statistics remain standalone workflows.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STAMP Pipeline                        │
│                                                          │
│  Preprocessing ──► Training ──► Deployment ──► Statistics │
│  (standalone)      (swarm)     (standalone)   (standalone)│
└─────────────────────────────────────────────────────────┘

Swarm Training Flow:
  Clinical Table (CSV/XLSX)  ─┐
                               ├──► STAMP DataLoaders ──► Lightning Model
  H5 Feature Files (.h5)    ─┘         │                      │
                                        │     NVFlare patches  │
                                        │     trainer for      │
                                        │     federated        │
                                        │     aggregation      │
                                        ▼                      ▼
                                   SwarmClientController ◄── PTClientAPILauncherExecutor
```

## Files

| File | Purpose |
|------|---------|
| `app/custom/main.py` | NVFlare entry point — swarm loop, local training, preflight check |
| `app/custom/stamp_training.py` | Training bridge — loads STAMP data/model, runs training rounds |
| `app/custom/stamp_model_wrapper.py` | NVFlare persistor bridge — creates model for federated parameter space |
| `app/config/config_fed_client.conf` | NVFlare client configuration (executors, aggregator, persistor) |
| `app/config/config_fed_server.conf` | NVFlare server configuration (swarm controller, rounds) |
| `meta.conf` | Job metadata (name, min_clients, deploy_map) |

## Environment Variables

All STAMP-specific variables use the `STAMP_` prefix to avoid collision with ODELIA variables.

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `SITE_NAME` | NVFlare site identifier | `site-1` |
| `SCRATCH_DIR` | Working directory for outputs | `/scratch` |
| `TRAINING_MODE` | `swarm`, `local_training`, or `preflight_check` | `swarm` |
| `STAMP_CLINI_TABLE` | Path to clinical data CSV/XLSX | `/data/clinical.csv` |
| `STAMP_FEATURE_DIR` | Directory containing H5 feature files | `/data/features/` |

### Optional — Data

| Variable | Default | Description |
|----------|---------|-------------|
| `STAMP_SLIDE_TABLE` | *(empty)* | Path to slide-level metadata table |
| `STAMP_OUTPUT_DIR` | *(auto-generated)* | Output directory for checkpoints and logs |
| `STAMP_TASK` | `classification` | Task type: `classification`, `regression`, or `survival` |
| `STAMP_GROUND_TRUTH_LABEL` | *(empty)* | Column name for ground truth in clinical table |
| `STAMP_PATIENT_LABEL` | `PATIENT` | Column name for patient ID |
| `STAMP_FILENAME_LABEL` | `FILENAME` | Column name for slide filename |
| `STAMP_TIME_LABEL` | *(empty)* | Column name for survival time (survival task only) |
| `STAMP_STATUS_LABEL` | *(empty)* | Column name for event status (survival task only) |

### Optional — Model

| Variable | Default | Description |
|----------|---------|-------------|
| `STAMP_MODEL_NAME` | `vit` | Model architecture: `vit`, `mlp`, `trans_mil`, `linear` (`barspoon` needs STAMP ≥ 2.5.0) |
| `STAMP_FEATURE_TYPE` | *(auto-detect)* | Feature level: `tile`, `slide`, or `patient` |
| `STAMP_DIM_INPUT` | `1024` | Input feature dimension (UNI/UNI2=1024, CTransPath=768) |
| `STAMP_NUM_CLASSES` | `3` | Number of output classes (a binary target such as MSI-High needs `2`) |
| `STAMP_EXPECTED_CATEGORIES` | *(unset)* | Agreed class order, e.g. `no,yes`. When set, every site checks its inferred order before training and refuses to start on a mismatch (#519) |

### Optional — Training

| Variable | Default | Description |
|----------|---------|-------------|
| `STAMP_BAG_SIZE` | `512` | Number of tiles per bag (MIL) |
| `STAMP_BATCH_SIZE` | `64` | Training batch size |
| `STAMP_MAX_EPOCHS` | `32` | Maximum training epochs |
| `STAMP_PATIENCE` | `16` | Early stopping patience |
| `STAMP_MAX_LR` | `1e-4` | Maximum learning rate (OneCycleLR) |
| `STAMP_DIV_FACTOR` | `25.0` | OneCycleLR div_factor |
| `STAMP_EPOCHS_PER_ROUND` | `5` | Local epochs per swarm round at the reference dataset size; scaled by each site's patient count (`STAMP_EPOCHS_REFERENCE_DATASET_SIZE`, default 200; at most `STAMP_EPOCHS_MAX_CAP`, default 20) |
| `STAMP_NUM_WORKERS` | `min(cpu_count, 8)` | DataLoader workers |
| `STAMP_SEED` | `42` | Passed to STAMP's config, but this job does not seed the run with it (STAMP seeds only from its own CLI). The train/validation split is fixed anyway: STAMP splits with `random_state=0` over the clinical table's row order |
| `STAMP_REQUIRE_GPU` | *(unset)* | `1` makes the run fail at the start when the container sees no GPU, instead of training on the CPU in full precision (benchmark jobs set it) |

## Metrics and run safeguards

- **Best model.** `IntimeModelSelector` keeps the global model with the highest `validation_auroc`,
  each site's validation AUROC of the received global model. STAMP 2.4.0 feeds raw logits to that
  metric, which torchmetrics scores as probabilities whenever a sample's logits lie inside [0, 1];
  `stamp_training.use_probabilities_for_validation_auroc` makes it score softmax probabilities, so
  the selected round is the one with the best real AUROC.
- **Scheduler horizon.** The OneCycleLR schedule is sized for the job's own `num_rounds`
  (`resolve_total_rounds`, #520), so the server cannot outrun it.
- **Timeouts** (stock CCWF controllers, no fault tolerance: every site must finish). The server
  aborts after 30 min of silence from one client (`max_status_report_interval`) or 2 h without a
  finished round anywhere (`progress_timeout`, below the clients' `learn_task_timeout` of 8 h, which
  would otherwise aggregate a round without the stuck site). A site waits up to 60 s for the
  aggregator's permission to submit, and the pipes to the training subprocess tolerate 30 min
  without a heartbeat.

## Data Requirements

Each site needs:

1. **Clinical table** (CSV or XLSX) with at minimum:
   - Patient identifier column (default: `PATIENT`)
   - Ground truth label column (for classification/regression)
   - Slide filename column (default: `FILENAME`) if using a slide table

2. **Feature directory** containing HDF5 (`.h5`) files:
   - One file per slide/patient
   - Each file contains a `feats` dataset (N×F float array)
   - Tile-level features also include a `coords` dataset
   - Feature type (tile/slide/patient) is auto-detected from H5 metadata

3. **Slide table** (optional, CSV or XLSX):
   - Maps patients to slide filenames
   - Required when patients have multiple slides

## Supported Models

| Model | `STAMP_MODEL_NAME` | Description |
|-------|---------------------|-------------|
| VIT | `vit` | Vision Transformer with ALiBi positional encoding |
| MLP | `mlp` | Multi-layer perceptron |
| TransMIL | `trans_mil` | Transformer-based MIL |
| Linear | `linear` | Linear classifier |
| Barspoon | `barspoon` | Encoder-decoder transformer — **unavailable**: added in STAMP 2.5.0; image ships 2.4.0 |

## Quick Start

### Local Testing

```bash
export TRAINING_MODE=preflight_check
export SITE_NAME=test-site
export SCRATCH_DIR=/tmp/stamp_test
export STAMP_CLINI_TABLE=/path/to/clinical.csv
export STAMP_FEATURE_DIR=/path/to/features/
export STAMP_GROUND_TRUTH_LABEL=diagnosis
export STAMP_NUM_CLASSES=3
export STAMP_MODEL_NAME=vit

cd application/jobs/STAMP_classification/app/custom
python main.py
```

### Swarm Deployment

Configure the environment variables above in your Docker/deployment configuration, set `TRAINING_MODE=swarm`, and submit the job via NVFlare:

```bash
nvflare job submit -j application/jobs/STAMP_classification
```

For operating a DECADE run (moving the server to a stand-in host, admin commands without the
console, reading a site's job log, the upload ingest, the 2-round test and the failure
signatures seen so far), see [`docs/DECADE_OPERATIONS.md`](../../../docs/DECADE_OPERATIONS.md).

## Dependencies

- [STAMP](https://github.com/KatherLab/STAMP) (`pip install stamp`)
- PyTorch + PyTorch Lightning
- NVFlare 2.5+
- h5py (for reading feature files)

## Differences from Standalone STAMP

| Aspect | Standalone STAMP | MediSwarm STAMP |
|--------|-----------------|-----------------|
| Data splitting | Internal stratified split | Each site has its own data |
| Training loop | Single `trainer.fit()` | NVFlare-controlled rounds |
| Model aggregation | N/A | Weighted federated averaging |
| Privacy | N/A | Only model weights leave a site, through the swarm; no DP noise in this job (the inert PercentilePrivacy filter was removed, #642) |
| Configuration | YAML config file | Environment variables |
| Preprocessing | Integrated pipeline | Standalone (run before training) |
| Deployment | Integrated pipeline | Standalone (run after training) |
