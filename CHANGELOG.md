# Changelog

All notable changes to MediSwarm are documented in this file.

## [1.3.0] - 2026-04-05

### Added

- **STAMP Classification Pipeline** — Full support for the KatherLab STAMP 2.4.0
  histopathology classification workflow in federated learning.
  - Separate `Dockerfile_STAMP` for STAMP's Python 3.11 + PyTorch 2.7.1 environment
  - `buildDockerImageAndStartupKits.sh` now accepts `-d` / `--dockerfile` flag
    to select between `Dockerfile_ODELIA` and `Dockerfile_STAMP`
  - Synthetic STAMP dataset generator for integration testing
  - STAMP integration tests: preflight check, local training, simulation mode
  - Per-round metrics CSV callback (`STAMPMetricsCallback`) for training
    monitoring with ground-truth/prediction probability output

- **FedProx Aggregation Strategy** — Alternative to FedAvg for improved
  convergence with non-IID data.
  - `FedProxCallback` Lightning callback adds proximal term
    `(mu/2) * ||w_local - w_global||^2` to gradient updates
  - Compatible with both ODELIA (`pytorch_lightning`) and STAMP (`lightning`)
  - Configurable via `FEDPROX_MU` environment variable
  - Documented comparison of FedAvg, FedProx, Scaffold, and FedOpt in
    `docs/AGGREGATION_STRATEGIES.md`

- **CI/CD for STAMP** — Expanded test infrastructure covering both pipelines.
  - Unit tests for `stamp_training.py`, `stamp_model_wrapper.py`, and
    `fedprox_callback.py`
  - STAMP integration tests in `pr-test.yaml` (Docker build + preflight +
    local training + simulation)
  - `unit-tests.yaml` switched from `pytorch-lightning` to unified `lightning`
    package for cross-pipeline compatibility

- **Duke Benchmark Pipeline** — Automated end-to-end benchmarking on the Duke
  Breast MRI dataset.
  - `scripts/evaluation/run_duke_benchmark.sh` orchestrates build, deploy,
    swarm training, result collection, and local model comparison
  - `deploy_and_test.sh` now reads `SITES` and `SERVER_NAME` from
    `deploy_sites.conf` instead of hardcoding them
  - `deploy_sites.conf.example` with dl0/dl2/dl3 templates
  - `docs/DUKE_BENCHMARK_RESULTS.md` results template

- **Architecture Documentation** — Expanded README with Mermaid diagrams.
  - System architecture diagram showing site-to-server topology
  - Training pipeline sequence diagram
  - Supported pipelines comparison table (ODELIA 3D CNN + STAMP)
  - Project structure overview

### Changed

- `deploy_and_test.sh` container matching broadened to include `stamp` and
  `nvflare` alongside `odelia`
- CI `pr-test.yaml` timeout increased from 45 to 60 minutes
- CI cleanup step now kills `stamp` and `nvflare` containers

### Documentation

- `docs/DIFFERENTIAL_PRIVACY.md` — Gap analysis of current `PercentilePrivacy`
  vs formal (epsilon, delta)-DP with Opacus/DP-SGD roadmap
- `docs/DIFFERENTIAL_PRIVACY_DECISION.md` — Architecture decision record
- `docs/AGGREGATION_STRATEGIES.md` — Comparison matrix for federated
  aggregation algorithms
- `docs/MEDISWARM_COMPATIBILITY_GUIDE.md` — Guide for making training code
  MediSwarm-compatible (from v1.2.0)

## [1.2.0] - 2025-02-15

### Added

- STAMP classification job integration (`application/jobs/STAMP_classification/`)
- Docker build optimization with layer reordering
- NVFlare configuration tuning for swarm topology
- Data-size-weighted epoch computation replacing hardcoded per-site values
- Prediction workflow with ensemble evaluation (`scripts/evaluation/predict.py`)
- Best model checkpointing alongside last-epoch checkpoints

## [1.1.0] - 2024-10-01

### Added

- Challenge model integration (teams 1-5)
- Model benchmarking suite (`scripts/evaluation/benchmark_models.py`)
- Automated deploy and test workflow (`deploy_and_test.sh`)
- AUROC plotting from class probability CSVs

## [1.0.0] - 2024-06-01

### Added

- Initial release of MediSwarm
- ODELIA 3D CNN pipeline for breast MRI classification
- NVFlare 2.7.2 swarm learning with peer-to-peer topology
- Docker-based deployment with startup kit provisioning
- Support for multiple model architectures (ResNet, MST, Swin3D)
