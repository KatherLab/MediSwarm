# Evaluation Scripts

Scripts for evaluating and comparing MediSwarm model performance.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `benchmark_models.py` | Benchmark all MediSwarm models on a consistent train/val/test split |
| `plot_aurocs_from_classprob_csvs.py` | Compute and plot AUROCs from class probability CSV files produced during training |
| `parse_logs_and_plot.py` | Parse training logs and plot convergence curves (legacy) |

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
