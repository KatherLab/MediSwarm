# Changelog

All notable changes to MediSwarm are documented in this file.

## [1.7.0] - 2026-07-29

Swarm evaluation release: a completed swarm run is now scientifically usable.
Training already worked; what was broken was everything needed to *judge* the
result — AUROC never reached us, the kept model was the last round rather than
the best, and the deploy test could report a green PASS while verifying nothing.
Every fix below was confirmed on a real 2-node run, not by unit tests alone.

### Fixed

- **Swarm AUROC is now retrievable (#492)** — AUROC is computed from the
  validation predictions and written to the `val_auroc` column of
  `stamp_metrics_summary.csv` (previously always blank, because the callback
  looked for metric keys STAMP does not log). `live_sync.sh` now locates a swarm
  run directory by listing `runs/<site>/` — the approach local mode already used —
  and syncs `stamp_metrics_summary.csv` / `stamp_gt_predprob_*.csv`, so the metrics
  actually reach the monitoring server instead of staying on the client node.
- **Best-model selection fires (#493)** — `IntimeModelSelector` is keyed on
  `validation_auroc`, the metric the client genuinely reports. It was previously
  `accuracy` (never logged), so selection never triggered and the **last** round's
  model was broadcast as the final model. Runs now produce `best_FL_global_model.pt`
  alongside the last-round global.
- **`finalize_training` runs in swarm mode (#480)** — it is called inside the swarm
  loop; the launcher SIGTERMs the training subprocess at job end, so the post-loop
  call never executed and best/last checkpoint consolidation was silently skipped.
- **Deploy test tests the image it built (no issue — found during 1.7.0 validation)** —
  client kits ship an `image.conf` pinning the release channel, which outranks the
  kit's built-in tag, so every client ran `:current` while the test pre-pulled the new
  image. Clients are now pinned with `--image`.
- **Deploy-test cleanup is scoped (#472)** — `stop_all` matched every
  `stamp_swarm|nvflare` container on every host, so a retry or a concurrent test
  killed a still-training run's clients (SIGKILL ~1 s after round 0, no error in any
  log). Cleanup is now scoped to the container-name suffix of the kit under test.
- **A skipped evaluation is no longer reported as PASS (#476, partial)** — a
  requested-but-unverified evaluation reports `PASSED (training only)`, and `--strict`
  makes it a failure. Running evaluation on the eval site over SSH remains open.

### Changed

- The DECADE 2-site test project uses ports **8102/8103**. The orchestrator also runs
  the long-lived ODELIA server on 8002/8003, so the test server could not bind and
  clients reached the ODELIA server instead, failing with
  `ClientConnectorCertificateError` — a port collision that looks exactly like a
  certificate problem.

## [1.5.0] - 2026-07-02

Swarm-robustness release: the fixes from the June 8-site runs that repeatedly
died at rounds 7–10. A single node's transient drop no longer aborts the whole
run, an aborted run can resume from a mirrored global, and GPU/VPN paths
self-recover. (The 1.4.x series was image-version bumps only and not tracked here.)

### Added

- **Fault-tolerant swarm controllers (#346)** — `FaultTolerantSwarmServerController`
  / `FaultTolerantSwarmClientController` + tolerant gatherer are the default in the
  challenge jobs. A client that errors is pruned and the run continues while
  `>= min_clients` remain, instead of a `FATAL_SYSTEM_ERROR` whole-run abort.
  Set `min_clients` / `min_responses_required` to (participants − allowed failures).
- **Warm-continue / auto-resume (#347)** — `WarmStartablePTFileModelPersistor`
  mirrors the aggregated global to `/scratch/mediswarm_latest_global.pt` on every
  client each round (never stranded on the crashed node) and can resume from it
  (`auto` / `fresh` / `require`). Admin flow: `prepare_odelia_job.sh --warm-start`.
- **VPN auto-recovery (#348)** — `setup_vpntunnel.sh -s` installs the
  `mediswarm-vpn` systemd service (system-stored creds, auto-reconnect) and
  `vpn_health_monitor.sh --install-timer` re-ups `tun0` within ~30 s of a drop.
- **GPU-container watchdog (#343)** — `gpu_container_watchdog.sh` restarts a
  client that has silently lost its GPU; `fix_docker_cgroupfs.sh` switches Docker
  to the cgroupfs driver so a `daemon-reload` (daily apt upgrade) can't strip the GPU.

### Changed

- **24 h swarm timeouts + wait-for-all (#345)** — outer and inner timeouts
  (`peer_read_timeout` / `learn_task_ack_timeout` / `final_result_ack_timeout` /
  `learn_task_timeout`) committed at 86400 s so a transient stall self-heals.
- **Per-round prediction export throttled (#314)** — the ~3.3× round-time cost
  is now final-round-only, the biggest per-round perf win for scaling.
- **DataLoader worker cap 8 → `min(cpu_count, 16)` (#315)** — restores loader
  concurrency on the ~16-CPU deploy nodes; per-node override via `ODELIA_NUM_WORKERS`.
- **NVFlare fork bumped to `MediSwarm-2.7.2`** — includes the CCWF config-phase
  quorum fix (`configure_min_clients`, separate from runtime fault tolerance).

### Fixed

- **CI 3DCNN simulation false-pass (#353)** — the simulation test now asserts a
  successful run instead of silently passing on an aborted one.
- **`nvflare` example pin (CVE-2026-24178)** — bumped `cifar10` requirement to
  `~=2.7.2` (Dashboard authz-bypass advisory; not on the production install path).

### Documentation

- `docs/SWARM_FAILURE_MODES.md`, `docs/TIMEOUTS.md`, and the participant/operator
  guides updated with the failure modes, timeout rationale, and host-side prep
  (cgroupfs fix, VPN service + watchdog, preflight).

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
