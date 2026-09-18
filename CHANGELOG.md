# Changelog

All notable changes to MediSwarm are documented in this file.

## [Unreleased]

### Fixed

- **Live-sync daemon: one SSH connection per kit instead of one per upload (#602)** — every
  rsync and ssh call was a separate login, three to five per 30-second cycle per kit, about
  two logins per second across the consortium at the upload host, each a logind session
  there. On 16 September the host hit logind's session cap and refused all SSH logins. The
  daemon now multiplexes (`ControlMaster=auto`, `ControlPersist=600`): seven calls, one login,
  measured from the dl3 test kit. Reaches the sites with their next kit.

- **Client worker no longer dies at job launch with `OMP: Error #15` (#596)** — the image
  carries three OpenMP runtimes (numpy/MKL, torch, scikit-learn) and LLVM's aborted the
  worker when it found another one initialised first. Intermittent; hit CAM_1 in production on
  8 April and two of twelve launches on the dl3 test bed. `KMP_DUPLICATE_LIB_OK=TRUE` is now set
  in the image, and the deploy-test harness names a site whose worker died instead of reporting
  it as "not configured".

### Added

- **`MST_SAMMed2D` model** — the MST slice-fusion classifier on the SAM-Med2D ViT-B image
  encoder (adapter layers, 256-pixel slices), contributed by Zhongying Deng (University of
  Cambridge) on branch `custom_cam_ZD` and ported onto `main`. Select with
  `MODEL_NAME=MST_SAMMed2D`. The SAM-Med2D weights (`sam-med2d_b.pth`) are not in the repo:
  place them in the build cache to ship them in the image, or set `SAM_MED2D_CHECKPOINT`;
  without them the encoder starts from random weights and says so. Slices are resized from
  the dataset's 224 to 256 inside the model rather than by changing the shared data pipeline.

### Fixed

- **Tolerant swarm mode: pruning is now announced to the surviving clients (#595)** — the
  server prunes a client on an error report, on `CLIENT_DISCONNECTED`, and (new) when it
  never answered the configure task, and sends a `swarm_ft.prune.<workflow>` notice; clients
  drop it from their trainer, aggregator-candidate and result-recipient lists and from a
  gatherer still waiting on it. Before, a site stopped mid-run cost every later round an
  extra hour (`wait_time_after_min_resps_received`), and a site whose worker crashed at
  launch hung the start task until `start_task_timeout` and failed the run. If the pruned
  site was the round's aggregator, the survivors elect a replacement (the first remaining
  candidate), which sets up the round's gatherer; pending submissions are redirected to it
  and already-accepted results are re-submitted. Strict profiles (quorum = site count) are
  unchanged: a disconnected site is logged, not pruned, so a VPN outage remains survivable.
  Failure modes F13 and F14 document all of it.

## [1.8.1] - 2026-09-13

Controller fix: a swarm can no longer start on a site that has not finished configuring.

### Fixed

- **Configure phase requires the starting client (NVFlare fork PR #8; #576)** — with
  `min_clients` below the number of participating sites, the server left the configure
  phase as soon as the quorum answered. If the starting site was not among them, the start
  task reached a client whose persistor did not exist yet and the run died with
  `invalid model learnable: expect Model type but got NoneType`, or stalled ten minutes on
  `TOPIC_UNKNOWN` replies and hung. The starting client is now configured first and on its
  own, a failure is a clear `system_panic` naming it, and the remaining sites are configured
  with the quorum applied to the whole set — without blocking when the starting client
  already satisfies it. Found by the weekly CI run on 13 Sep; consortium runs were shielded
  only because `configure_min_clients` is set to the site count at submission.
- **CI deploy test on the runners (#580)** — the workflow now bakes `min_clients` into the
  test kits (default 2) and the runner-local configs were brought in line with the renamed
  test sites; the release-triggered deploy test passes again for the first time since the
  runners moved in July.

### Changed

- The fork's `TestConfigurePhase` and `TestPrunedStartingClient` re-implemented the
  controller's condition inside the test and asserted on the copy; replaced by tests that
  drive the real `_configure_clients()`.

## [1.8.0] - 2026-09-13

Per-site evaluation and deploy-test release. A completed swarm run now reports what
every site measured, can return per-case predictions from sites that opt in, and
refuses to warm-start from another architecture's weights. The two-node deploy test
runs end to end on real startup kits again, and it is where most of the fixes below
were found. Every training-path change was confirmed on a real 2-node run (dl0 + dl2,
server on dl3), not by unit tests alone.

### Added

- **Per-site metrics reach the coordinator (#534; #525, #441)** — each site's validation
  metrics, including per-class support counts, are published to the server and returned
  in `cross_val_results.json`. Previously the file was `{}` although the server logged
  `Published metrics for N site(s)` (F10). Verified on real kits: both test sites present
  with per-class support.
- **Per-case predictions, opt-in per site (#556, #557; #526, #527, #528)** — a site that
  starts its client with `ODELIA_RETURN_PER_CASE=1` returns per-case class probabilities
  (no identifiers) with its metrics. Off by default; nothing changes for a site that does
  not set it. Needed for regional fine-tuning comparison (D2.5), active learning (D3.2)
  and the testing node (D3.3).
- **Warm-start provenance and a structural guard (#545, #575; #535)** — the mirrored
  global checkpoint carries a `.provenance.json` sidecar naming the model that wrote it.
  A mirror from another architecture is refused by label and, since #575, also by
  comparing parameter names, so an unlabelled mirror is caught too. A refusal now returns
  no model instead of falling through and loading the file anyway (F11).
- **DataLoader worker cap (#575; #574)** — a client never runs more loader workers than its
  training set can feed (at least four samples per worker). Removes the pin-memory
  shared-memory race on tiny sets (F12); no effect at consortium data sizes.
- **Strict swarm runs are exact-client by default (#514)** — large-model result references
  stay alive for the whole run budget, only missing deliveries are retried, repeated
  delivery of a round is idempotent, and strict mode rejects timeout-driven partial
  aggregation.
- **Active learning (D3.2): sample-selection strategies (#555) and the acquisition
  experiment (#563, corrected in #568)** — entropy and margin sampling against a random
  control over the real per-case predictions of the eight-site run. The first result was
  computed on predictions from the wrong architecture and was retracted; the analysis now
  refuses predictions whose malignant AUROC is below 0.60 (E2 guard) and computes its
  verdict instead of stating it.
- **Robust aggregation and a poisoning simulation (#566; D3.4 #529)** — norm-clipped
  aggregation measured against a scaled-update attacker. The size-weighted mean every job
  uses has breakdown point zero; the smallest site can move it arbitrarily.
- **White-hat site-inference probe (#571; MS6 #531)** — the part of the white-hat attack
  TUD can run without partner scheduling: whether the three class probabilities of a case
  reveal which hospital it came from.
- **Privacy accounting MVP (#563; T3.3 phase 2)**.
- **Frozen run history (#560, #573)** — `scripts/analysis/extract_run_history.py` and the
  committed dataset under `workspace/run_history/`. Records are attributed by the
  heartbeat's hostname, not the site directory name, so TU Dresden test machines no
  longer masquerade as hospitals.
- **Supply chain (#446; #395)** — base image pinned by digest, apt left unpinned on purpose,
  CVE scan in CI.
- **Docs** — consortium briefing rewritten for a non-technical audience (#564); deck index
  (#558); evaluation pitfalls E1–E4 (#533, #568); failure modes F11 and F12; manuscript
  evidence matrix with a mechanically enforced anonymisation rule (#569).

### Fixed

- **`finalize_training` in swarm mode now runs through a callback (#502; #480)** — the
  post-loop call could not execute because the launcher terminates the training
  subprocess at job end.
- **STAMP scheduler horizon follows the job's `num_rounds` (#520; #503)** — the client read
  `STAMP_NUM_ROUNDS` from its environment while the server ran the job's `num_rounds`;
  when they disagreed training died with `Tried to step 9 times`.
- **STAMP installs from a release tarball (#523)** — the git clone in the Dockerfile failed
  under rate limiting and looked like a permissions error.
- **The 2-node deploy test can run at all (#544)** — its project sat on the productive
  server's ports and name, so test clients reached the production server and died on
  certificates from another provisioning generation; `min_clients` and the gitignore for
  site configs fixed alongside.
- **The 2-node deploy test runs on real kits (#573)** — clients are `TEST_A_1`/`TEST_B_1`
  instead of hospital names (which polluted the live monitor and a published chart), the
  data folder is decoupled from the FL identity, stale warm-start mirrors are wiped before
  each model, staged jobs can be submitted by absolute path, and server-side artifacts are
  saved before cleanup deletes them.
- **Weekly all-models preflight (#552, #559, #572)** — builds the kits it needs first, covers
  the ResNet variants, verifies that preflight succeeded rather than started, and prints
  readable output.

### Changed

- **`jefftud/odelia:current` is re-tagged to 1.8.0 with this release.** (It was not
  re-tagged for 1.7.0, so sites have been running the 1.6.0-era image since July.)
- Dependencies: actions/checkout 7, actions/setup-python 7, trivy-action 0.36 (#548, #551,
  #549).

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
