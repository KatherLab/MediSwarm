# MediSwarm / NVFlare Timeout Reference

This document describes all timeout settings used in ODELIA/MediSwarm federated
training, from the outermost (CI) layer down to the innermost (per-message)
layer. Timeouts form a hierarchy: outer timeouts must always be larger than inner
ones, or the outer layer will kill a run that was still making progress.

## Timeout Hierarchy (outer to inner)

```
CI workflow timeout (3 days = 4320 min)
  └── Deploy script per-model timeout (7 days = 10080 min)
        └── NVFlare progress_timeout (12 h = 43200 s)
              └── learn_task_timeout (12 h = 43200 s)
                    ├── heartbeat_timeout (30 min = 1800 s)
                    ├── peer_read_timeout (2 h = 7200 s)
                    ├── external_pre_init_timeout (30 min = 1800 s)
                    └── last_result_transfer_timeout (2 h = 7200 s)
```

---

## 1. CI Workflow Timeout

| Setting | Value | File |
|---------|-------|------|
| `timeout-minutes` | **4320** (3 days) | `.github/workflows/odelia-deploy-test.yml` |

The GitHub Actions job-level timeout for the full deploy test suite. This covers
all 6 models sequentially. Must be larger than the sum of all individual model
runs.

**Risk if too low:** GitHub kills the entire workflow mid-training. All results
from completed models are still uploaded as artifacts, but remaining models are
skipped.

---

## 2. Deploy Script Per-Model Timeout

| Setting | Value | File |
|---------|-------|------|
| `TIMEOUT_MINUTES` | **10080** (7 days) | `scripts/deploy/run_deploy_test.sh` |

How long the deploy test script waits for a single model's training to complete
(server to report "finished"). Can be overridden via `--timeout` flag.

**Risk if too low:** The script stops waiting and moves to the next model,
recording the current one as "TIMEOUT". The NVFlare containers are killed.

---

## 3. NVFlare Server-Side Timeouts

These are set in each job's `config_fed_server.conf`:

| Setting | Value | Description |
|---------|-------|-------------|
| `progress_timeout` | **43200 s** (12 h) | Max time without any client reporting progress before the server declares the job failed. |
| `start_task_timeout` | **3600 s** (60 min) | Time for all clients to pull their startup kits, connect, and be ready. |
| `configure_task_timeout` | **1800 s** (30 min) | Time for clients to acknowledge the swarm configuration message. |
| `max_status_report_interval` | **7200 s** (2 h) | Max interval before a client is considered silent. This must cover large-model setup and slow first-round transfer. |

**File:** ODELIA job configs under
`application/jobs/{ODELIA_ternary_classification,challenge_*}/app/config/config_fed_server.conf`.

**Risk if `progress_timeout` is too low:** Long training rounds on large
models/slow GPUs get killed even though they are still making progress. This is
the most common cause of "unexpected abort" on slow hardware.

**Observed failure fixed by the current values:** a 2-round
`challenge_1DivideAndConquer` smoke with MHA + USZ failed when the previous
`max_status_report_interval=300` declared MHA silent during first-round
large-model setup/transfer. The rerun with `max_status_report_interval=7200`
completed and collected both `FL_global_model.pt` files.

---

## 4. NVFlare Client-Side Timeouts (SwarmClientController)

These are set in each job's `config_fed_client.conf`:

| Setting | Value | Description |
|---------|-------|-------------|
| `learn_task_timeout` | **43200 s** (12 h) | Max time for a single training round (all local epochs). The most critical timeout for large models. |
| `learn_task_abort_timeout` | **300 s** (5 min) | Grace period for a training round to finish after an abort is requested. |
| `learn_task_ack_timeout` | **7200 s** (2 h) | Time for the training task acknowledgment, including streaming model weights to peers. |
| `final_result_ack_timeout` | **7200 s** (2 h) | Time for the final aggregated result acknowledgment. |
| `wait_time_after_min_resps_received` | **1800 s** (30 min) | After `min_responses_required` clients have finished a round, wait this long for stragglers before proceeding. |

**File:** ODELIA job configs under
`application/jobs/{ODELIA_ternary_classification,challenge_*}/app/config/config_fed_client.conf`.

**Risk if `learn_task_timeout` is too low:** Slow clients (small GPU, large
model, many epochs) time out during training. The round is marked as failed and
the job may abort. This was the bottleneck in early deploy tests and becomes
more likely as the client count grows.

---

## 5. NVFlare Executor-Level Timeouts (PTClientAPILauncherExecutor)

These control communication between the NVFlare process and the training
subprocess (the actual PyTorch training script):

| Setting | Value | Description |
|---------|-------|-------------|
| `heartbeat_timeout` | **1800 s** (30 min) | Time without a heartbeat from the training subprocess before NVFlare declares it dead. |
| `peer_read_timeout` | **7200 s** (2 h) | Time to wait for a peer to read a sent message (model weight streaming between clients). |
| `external_pre_init_timeout` | **1800 s** (30 min) | Time for the subprocess to call `flare.init()` after launch (covers import time + GPU init). |
| `last_result_transfer_timeout` | **7200 s** (2 h) | Time for the final trained model to transfer from subprocess back to NVFlare. |

**File:** ODELIA job configs under
`application/jobs/{ODELIA_ternary_classification,challenge_*}/app/config/config_fed_client.conf`
(under the `PTClientAPILauncherExecutor` args).

**Note:** The cifar10 job uses `ModelLearnerExecutor` instead of
`PTClientAPILauncherExecutor`, so it does not have `heartbeat_timeout`,
`peer_read_timeout`, `external_pre_init_timeout`, or
`last_result_transfer_timeout`. These are only relevant for subprocess-based
training.

---

## Tuning Guidelines

### For Larger Models (>500 MB)
- Increase `learn_task_timeout` and `progress_timeout` proportionally
- Consider increasing `peer_read_timeout` and `learn_task_ack_timeout` for
  slow VPN connections

### For More Clients (>4 sites)
- Increase `wait_time_after_min_resps_received` to allow slower sites to catch up
- Increase `start_task_timeout` if sites connect over WAN/VPN
- Keep `max_status_report_interval` generous for large-model first-round setup;
  `300 s` is unsafe for ODELIA 1DivideAndConquer over multi-site WAN/VPN.

### For More Rounds (>20)
- Ensure `progress_timeout` > `learn_task_timeout` (otherwise the server may
  declare "no progress" during a long round)
- The deploy script timeout (`TIMEOUT_MINUTES`) should be >
  `num_rounds * avg_round_time_minutes`

### Environment Variable Overrides
- `EPOCHS_MAX_CAP` (default: 10) — caps the number of local epochs per round,
  which directly affects round duration. Set in the container environment via
  `docker.sh` or docker-compose. Lower values = faster rounds = less likely to
  hit `learn_task_timeout`.

---

## Webviewer Error Detection

The MediSwarm webviewer (`server_tools/app.py`, deployed at `/srv/mediswarm/`)
automatically detects timeout-related errors by scanning console logs. When a
timeout is hit, the run status shows **"error"** with a descriptive reason:

| Pattern | Displayed Reason |
|---------|-----------------|
| `TaskCompletionStatus.TIMEOUT` | NVFlare task timed out (check learn_task_timeout / configure_task_timeout) |
| `learn_task.*timed out` | Training round exceeded learn_task_timeout (12h) |
| `progress_timeout.*exceeded` | No progress reported within progress_timeout (12h) |
| `peer_read_timeout.*exceeded` | P2P model transfer timed out (peer_read_timeout 2h) |
| `heartbeat_timeout.*exceeded` | Subprocess heartbeat lost (heartbeat_timeout 30min) |
| `external_pre_init_timeout` | Subprocess failed to call flare.init() within 30min |
| `last_result_transfer_timeout` | Final result transfer timed out (2h) |
| `configure_task_timeout` | Client configuration timed out (configure_task_timeout 30min) |
| `start_task_timeout` | Client start timed out (start_task_timeout 60min) |

These appear as red badges with hover tooltips on the dashboard.

---

## Reference: Deploy Test Timing (Observed)

From the 4-node deploy test with `challenge_1DivideAndConquer` (20 rounds):

- ~19 min per round (includes training + P2P model exchange)
- ~380 min total (6.3 hours) for 20 rounds
- Model size: ~689 MB (3D CNN)
- Sites: RUMC_1 (22 samples), MHA_1 (~50 samples), CAM_1 (~200 samples), UMCU_1 (~150 samples)
