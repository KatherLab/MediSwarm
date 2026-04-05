# Duke Breast MRI Benchmark Results

## Overview

This document records benchmark results for MediSwarm federated learning on the
[Duke Breast MRI](https://doi.org/10.7937/TCIA.e3sv-re93) dataset across the
TUD compute cluster (dl0, dl2, dl3).

The Duke dataset is a public collection of dynamic contrast-enhanced (DCE) MRI
sequences used for ternary classification of breast lesions:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Benign | Benign lesion |
| 1 | Malignant (non-PCR) | Malignant, no pathological complete response |
| 2 | Malignant (PCR) | Malignant, pathological complete response |

## Infrastructure

| Machine | Role | GPU | Dataset Partition |
|---------|------|-----|-------------------|
| dl3.tud.de | Server + Client (TUD_3) | NVIDIA A100 | ~33% of Duke |
| dl0.tud.de | Client (TUD_1) | NVIDIA A100 | ~33% of Duke |
| dl2.tud.de | Client (TUD_2) | NVIDIA A100 | ~33% of Duke |

**Provision file:** `application/provision/project_DUKE_test.yml`

## How to Reproduce

```bash
# 1. Configure deploy_sites.conf with DL0/DL2/DL3 credentials
#    (see deploy_sites.conf.example for template)

# 2. Run the full benchmark pipeline
./scripts/evaluation/run_duke_benchmark.sh \
    --project application/provision/project_DUKE_test.yml \
    --job ODELIA_ternary_classification \
    --local-epochs 10

# 3. Or run individual phases:
./scripts/evaluation/run_duke_benchmark.sh --skip-swarm   # local only
./scripts/evaluation/run_duke_benchmark.sh --skip-local    # swarm only
./scripts/evaluation/run_duke_benchmark.sh --collect-only   # just gather results
```

## Results

> **Status:** Pending first run. Update this section after completing the benchmark.

### Swarm Training (Federated)

| Metric | Value |
|--------|-------|
| Aggregation | FedAvg (InTimeAccumulateWeightedAggregator) |
| Rounds | TBD |
| Clients | 3 (dl0, dl2, dl3) |
| Model | TBD |
| Best AUC-ROC (macro) | TBD |
| Best Accuracy | TBD |
| Training time | TBD |

### Local Training (Single-Site Benchmark)

| Model | ACC | AUC-ROC | F1 (macro) | Params | Time/epoch |
|-------|-----|---------|------------|--------|------------|
| ResNet10 | - | - | - | - | - |
| ResNet18 | - | - | - | - | - |
| ResNet34 | - | - | - | - | - |
| ResNet50 | - | - | - | - | - |
| MST | - | - | - | - | - |
| Swin3D | - | - | - | - | - |

### Swarm vs Local Comparison

| Training Mode | Best Model | AUC-ROC | ACC | Notes |
|---------------|-----------|---------|-----|-------|
| Local (single site) | - | - | - | - |
| Swarm (3 sites) | - | - | - | - |

## Analysis

> To be completed after benchmark run.

### Key Questions

1. **Does federated training improve over local?** Compare swarm AUC-ROC with
   the best single-site local model.
2. **Which model architecture works best?** Review `benchmark_results.json` for
   the local comparison table.
3. **How does data heterogeneity affect convergence?** Check per-round metrics
   across sites.
4. **Is FedProx beneficial?** If tested, compare FedAvg vs FedProx convergence
   and final metrics.

## Files

Results from benchmark runs are saved under `duke_results/<timestamp>/`:

```
duke_results/
  20250405_143000/
    benchmark_config.json       # Run configuration
    swarm/
      predictions/              # predict.py output
      <server>/<job_id>/
        app_TUD_1/
          best_FL_global_model.pt
          FL_global_model.pt
        app_TUD_2/
          ...
    local/
      benchmark_results.json    # benchmark_models.py output
      benchmark_output.log
      ResNet18/                 # Per-model checkpoints
      MST/
      ...
```

Note: `duke_results/` is in `.gitignore` — results are not committed to the repo.
