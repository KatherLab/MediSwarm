# Evaluation Scripts

Scripts for evaluating and comparing MediSwarm model performance.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `predict.py` | Run prediction on external test datasets using trained swarm models |
| `benchmark_models.py` | Benchmark all MediSwarm models on a consistent train/val/test split |
| `run_duke_benchmark.sh` | End-to-end Duke dataset benchmark: build, deploy, train, collect, evaluate |
| `plot_aurocs_from_classprob_csvs.py` | Compute and plot AUROCs from class probability CSV files produced during training |
| `parse_logs_and_plot.py` | Parse training logs and plot convergence curves (legacy) |

---

## `predict.py`

Evaluates one or more trained MediSwarm global model checkpoints on an external test dataset. Supports single model, multi-model (all sites), and ensemble prediction.

### Features

- **Single checkpoint**: Evaluate one specific model file
- **Workspace discovery**: Automatically find all `FL_global_model.pt` and `best_FL_global_model.pt` across sites in a swarm workspace
- **Ensemble**: Average class probabilities across multiple models for improved predictions
- **Per-sample CSV**: Output individual predictions with UIDs, ground truth, predicted class, and class probabilities
- **Comprehensive metrics**: Accuracy, AUC-ROC (macro + per-class), F1 (per-class, macro, weighted), precision, recall, confusion matrix

### Prerequisites

- **CUDA GPU** required
- Environment variables:
  - `DATA_DIR` -- Path to the ODELIA dataset (or external test data)
  - `SITE_NAME` -- Site identifier (for dataset loading)
  - `SCRATCH_DIR` -- Scratch/output directory (for default output path)
  - `MODEL_NAME` -- (optional) Model architecture, can also use `--model-name` flag

### Usage

```bash
# Evaluate the best global model from one site:
python predict.py \
    --checkpoint /path/to/best_FL_global_model.pt \
    --model-name MST

# Evaluate all checkpoints from a swarm workspace:
python predict.py \
    --workspace /path/to/workspace/prod_00/ \
    --model-name MST

# Only evaluate best models (skip latest):
python predict.py \
    --workspace /path/to/workspace/prod_00/ \
    --model-name MST \
    --best-only

# Ensemble all site models (averaged probabilities):
python predict.py \
    --workspace /path/to/workspace/prod_00/ \
    --model-name MST \
    --ensemble

# Evaluate on validation split instead of test:
python predict.py \
    --checkpoint /path/to/model.pt \
    --model-name ResNet101 \
    --split val

# Evaluate multiple specific checkpoints:
python predict.py \
    --checkpoint /path/to/site_A/best_FL_global_model.pt \
                 /path/to/site_B/best_FL_global_model.pt \
    --model-name MST \
    --ensemble

# Custom output directory:
python predict.py \
    --checkpoint /path/to/model.pt \
    --model-name MST \
    --output-dir /path/to/results/

# Load a PyTorch Lightning checkpoint (instead of NVFlare state_dict):
python predict.py \
    --checkpoint /path/to/model.ckpt \
    --model-name MST \
    --checkpoint-type lightning
```

### Workspace Structure

When using `--workspace`, the script discovers checkpoints in the standard NVFlare workspace layout:

```
workspace/prod_00/
  <server_fqdn>/
    <job_id>/
      app_UKA/
        FL_global_model.pt
        best_FL_global_model.pt
      app_TUD/
        FL_global_model.pt
        best_FL_global_model.pt
      ...
```

### Output

- **Console**: Detailed per-model metrics and a summary comparison table
- **Per-model CSV** (`predictions_<site>_<kind>.csv`):
  ```
  uid,ground_truth,prediction,prob_class_0,prob_class_1,prob_class_2
  CASE001,0,0,0.85,0.10,0.05
  CASE002,2,2,0.05,0.15,0.80
  ```
- **Ensemble CSV** (`predictions_ensemble.csv`) -- when `--ensemble` is used
- **JSON results** (`prediction_results.json`): All metrics and metadata for each evaluated checkpoint

---

## `benchmark_models.py`

Trains and evaluates all available MediSwarm models (ResNet variants, MST, Swin3D, and optionally challenge models) on a consistent data split, producing a comparison table with accuracy, AUC-ROC, per-class F1 scores, confusion matrices, and training time.

### Prerequisites

- **CUDA GPU** required
- Environment variables must be set (same as normal MediSwarm training):
  - `DATA_DIR` -- Path to the ODELIA dataset
  - `SCRATCH_DIR` -- Scratch/output directory
  - `SITE_NAME` -- Site identifier (e.g., `TUD_1`)
  - `TRAINING_MODE` -- Set to `local_training` or `preflight_check`
  - `MEDISWARM_VERSION` -- Version string (default: `benchmark`)

### Usage

```bash
# Benchmark all built-in models (ResNet10/18/34/50/101/152, MST, Swin3D):
python benchmark_models.py

# Benchmark specific models only:
python benchmark_models.py --models MST ResNet18 Swin3D

# Include challenge models alongside built-in ones:
python benchmark_models.py --include-challenge

# Set custom number of training epochs (default: 5):
python benchmark_models.py --max-epochs 10

# Save results to a specific file:
python benchmark_models.py --output /path/to/results.json

# Dry run -- list models that would be benchmarked without training:
python benchmark_models.py --dry-run
```

### Output

- **Console**: Formatted table with ACC, AUC-ROC, F1-macro, parameter count, and training time per model, followed by per-class F1 detail and confusion matrices.
- **JSON file**: Saved to `<SCRATCH_DIR>/benchmark/benchmark_results.json` (or `--output` path). Results are saved incrementally so partial results survive crashes.

Each model entry in the JSON contains:
```json
{
  "model": "ResNet18",
  "status": "success",
  "accuracy": 0.7500,
  "auroc_macro": 0.8200,
  "per_class_f1": {"0": 0.75, "1": 0.68, "2": 0.80},
  "macro_f1": 0.7433,
  "weighted_f1": 0.7500,
  "confusion_matrix": [[10, 2, 1], [3, 8, 2], [1, 1, 12]],
  "train_time_seconds": 450.0,
  "train_time_per_epoch": 90.0,
  "num_parameters": 11170000,
  "best_checkpoint": "/path/to/best.ckpt"
}
```

---

## `plot_aurocs_from_classprob_csvs.py`

Reads class probability CSV files produced during local and swarm training, computes various AUROCs (macro, one-vs-one pairwise, combined binary), and generates a comprehensive evaluation plot.

### Prerequisites

- Python packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`
- CSV files from MediSwarm training runs (produced by `threedcnn_ptl.py`)

### Expected Directory Structure

```
<data_dir>/
  local/
    CAM/
      site_model_gt_and_classprob_train.csv
      site_model_gt_and_classprob_validation.csv
    ... (further sites)
  swarm/
    CAM/
      site_model_gt_and_classprob_train.csv
      site_model_gt_and_classprob_validation.csv
      aggregated_model_gt_and_classprob_train.csv
      aggregated_model_gt_and_classprob_validation.csv
    ... (further sites)
```

### Usage

```bash
# Analyze current directory (default):
python plot_aurocs_from_classprob_csvs.py

# Analyze a specific directory:
python plot_aurocs_from_classprob_csvs.py /path/to/results/

# Use one-vs-rest instead of one-vs-one ROC-AUC:
python plot_aurocs_from_classprob_csvs.py --roc_auc_type ovr

# Use log scale for label distribution histograms:
python plot_aurocs_from_classprob_csvs.py --logscale_hist
```

### Output

- `evaluation.png` -- Multi-panel plot showing:
  - Label distributions per site (train vs. validation)
  - AUROC curves over epochs for each site and metric type (macro, pairwise, combined)
  - Comparison between local and swarm training

### Verification Steps

The script automatically verifies:
- Label distributions are constant across epochs
- Swarm aggregated and site models have the same label distributions at epoch 0
- Swarm and local training use identical label distributions

---

## `run_duke_benchmark.sh`

End-to-end benchmark pipeline for the Duke Breast MRI dataset on the TUD compute cluster (dl0, dl2, dl3). Orchestrates the full workflow: Docker build, push, deploy, swarm training, result collection, and local model benchmarking.

### Prerequisites

- `deploy_sites.conf` configured with DL0/DL2/DL3 entries (see `deploy_sites.conf.example`)
- `sshpass` and `expect` installed
- Duke dataset available on each site
- GPU available on each site

### Usage

```bash
# Full pipeline (build, deploy, train swarm, benchmark local):
./run_duke_benchmark.sh

# Swarm only (skip local benchmark):
./run_duke_benchmark.sh --skip-local

# Local benchmark only (skip swarm):
./run_duke_benchmark.sh --skip-swarm

# Collect results from a previous swarm run:
./run_duke_benchmark.sh --collect-only

# Custom models and epochs:
./run_duke_benchmark.sh --models MST ResNet18 Swin3D --local-epochs 10

# Dry run (print configuration only):
./run_duke_benchmark.sh --dry-run
```

### Output

Results are saved to `duke_results/<timestamp>/`:
- `benchmark_config.json` -- Run configuration for reproducibility
- `swarm/` -- Collected checkpoints and prediction CSVs from swarm training
- `local/` -- `benchmark_results.json` from `benchmark_models.py`

See `docs/DUKE_BENCHMARK_RESULTS.md` for the results template and analysis.

---

## `parse_logs_and_plot.py` (Legacy)

Parses training console logs to extract AUC-ROC values and plots convergence curves for swarm vs. local training across sites.

> **Note**: This script is kept for historical comparison only. Use `plot_aurocs_from_classprob_csvs.py` instead, as it computes one-vs-one AUROCs which do not hide poor performance on rare classes.

### Expected Directory Structure

```
CAM/
  local_training_console_output.txt
  nohup.out
MHA/
  local_training_console_output.txt
  nohup.out
...
```

### Usage

```bash
python parse_logs_and_plot.py
```

### Output

- `convergence_per_site.png` -- Per-site convergence plots (4x2 grid)
- `convergence_overview.png` -- Overview plots (3x2 grid) comparing swarm and local training
