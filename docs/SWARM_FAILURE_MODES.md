# ODELIA Swarm — Failure Modes & Operator Runbook

A catalogue of the failure modes seen in real multi-site ODELIA swarm runs, how to
**detect** each (ideally before a run, via the pre-flight checks), how to **fix** it,
and how to **prevent** it. Plus an operator diagnostic playbook.

Most of these are now caught automatically by the pre-run checks in the startup-kit
`docker.sh` (`--dummy_training` / `--preflight_check` / `--start_client`); see
[`docker_config/master_template.yml`](../docker_config/master_template.yml) (`_preflight_host_checks`).

## Quick reference

| ID | Symptom | Root cause | Fix |
|----|---------|-----------|-----|
| F1 | Run aborts: `client X didn't report status for N seconds` | Slow site silent during its long round > `max_status_report_interval` | 24 h timeouts (committed in the job configs) |
| F2 | `NVML: Unknown Error` in container; GPU "vanishes" mid-run | Daily `daemon-reload` strips GPU on cgroup v2 + **systemd** Docker driver | `scripts/client_node_setup/fix_docker_cgroupfs.sh` (→ cgroupfs) |
| F3 | Client "healthy" but never registers after a restart | Stale `daemon_pid.fl` lock makes `start.sh` refuse to launch | Delete `daemon_pid.fl` in the kit root, relaunch |
| F4 | Instant crash: `OSError: Read-only file system` | Preprocessing cache under the read-only `/data` mount | Point `ODELIA_PREPROCESS_CACHE_DIR` under `/scratch` |
| F5 | Run aborts: `MODEL_UNRECOGNIZED` from a slow site | Swarm advanced a round without the slow site; its late model rejected | Wait-for-all (`min_responses_required = #clients`) |
| F6 | Run aborts after a node goes `deemed disconnected` | VPN tunnel drop for that site | Stabilise the VPN (keepalive / network); wait-for-all bounds it by `learn_task_timeout` |
| F7 | `DataLoader worker killed by signal: Bus error … shared memory` | `/dev/shm` too small for the dataloader workers | `--ipc=host` (already set for swarm clients) / `--shm-size` |

---

## F1 — Slow-node status timeout
- **Symptom (server log):** `Aborting current RUN due to FATAL_SYSTEM_ERROR received: client UKA_1 didn't report status for 7200 seconds`.
- **Root cause:** The largest-data site (UKA ≈ 9× the others) trains a single round for hours and reports no status in between. The `SwarmServerController.max_status_report_interval` declares it dead and aborts the job.
- **Detection:** per-round wall-clock far exceeds the status interval; site finishes `start_learn_task` then goes silent.
- **Fix:** the challenge/MST job configs now use **24 h** for `max_status_report_interval`, `progress_timeout`, and `learn_task_timeout`. These ship with the submitted job (byoc) — **no startup-kit rebuild needed**; the change takes effect on the next job submission.
- **Prevention:** keep timeouts well above the slowest site's per-round time; see [`docs/TIMEOUTS.md`](TIMEOUTS.md).

## F2 — GPU "NVML: Unknown Error" mid-run (the GPU vanishes)
- **Symptom:** in-container `nvidia-smi` returns `Failed to initialize NVML: Unknown Error` while the **host** GPU is fine; training crashes with `This example does not work without GPU`. Recurs roughly daily.
- **Root cause:** the host's daily `apt-daily-upgrade` runs `systemctl daemon-reload`, which on **cgroup v2 + Docker's `systemd` cgroup driver** re-applies device cgroup rules and strips the GPU from *already-running* containers.
- **Detection:** the pre-run check warns when the Docker cgroup driver is `systemd`; `docker exec <client> nvidia-smi -L` fails while host `nvidia-smi` works.
- **Fix (host-level, durable):** run **`scripts/client_node_setup/fix_docker_cgroupfs.sh`** (as root) — switches Docker to the `cgroupfs` driver and validates that a `daemon-reload` no longer drops the GPU. Then recreate the client (see F3). The change persists across reboots.
- **Prevention:** use the `cgroupfs` driver on all client hosts; the pre-run check flags `systemd`.

## F3 — Stale `daemon_pid.fl` blocks client restart
- **Symptom:** after `docker rm -f` of a client, the new container is `Up … (healthy)` (the healthcheck only runs `nvidia-smi`) but the site never registers; `nohup.out` shows `There seems to be one instance, pid=N, running … remove daemon_pid.fl`.
- **Root cause:** an unclean stop leaves a `daemon_pid.fl` lock in the kit root; `start.sh` refuses to launch a "second" instance.
- **Detection:** the pre-run check (for `--start_client`) detects the lock and removes it when no client container is running.
- **Fix:** delete `daemon_pid.fl` in the kit root, then relaunch `./docker.sh --start_client`.
- **Prevention:** always clear the lock when recreating a client; the pre-run check now does this automatically.

## F4 — Read-only preprocessing-cache path
- **Symptom:** instant crash `OSError: [Errno 30] Read-only file system: '/data/.../odelia_preprocess_cache'`.
- **Root cause:** `ODELIA_PREPROCESS_CACHE_DIR` was set under `/data`, which is mounted **read-only** in the container.
- **Detection:** the pre-run check fails if `ODELIA_PREPROCESS_CACHE_DIR` is under `/data`.
- **Fix:** set it to a container path under `/scratch`, e.g. `ODELIA_PREPROCESS_CACHE_DIR=/scratch/odelia_preprocess_cache`.
- **Prevention:** always keep the cache under `/scratch` (writable); never under `/data`.

## F5 — Slow-node desync (`MODEL_UNRECOGNIZED`)
- **Symptom:** with `min_responses_required = 2`, fast sites finish round *r* and the swarm advances to *r+1*; when the slow site finally submits its round-*r* model, the aggregator rejects it: `cwf.error: MODEL_UNRECOGNIZED`, then the run aborts.
- **Root cause:** the swarm did not wait for the slow site, so its contribution became stale.
- **Fix:** **wait-for-all** — set `min_responses_required` to the number of participating clients, so a round cannot advance until every site (including the slowest) submits.
- **Tradeoff:** wait-for-all is robust against desync but **fragile to drops** — if any client drops mid-round, the round stalls until `learn_task_timeout`. Set `min_responses_required` to the actual participating-client count for the run.

## F6 — VPN tunnel drop ("deemed disconnected")
- **Symptom (server log):** `received dead job report for client X` → `Client X is deemed disconnected!`; the site's heartbeats stop, then re-appear later. With wait-for-all, the round stalls until the site returns or `learn_task_timeout` fires.
- **Root cause:** the site's VPN tunnel drops (observed: a ~2.5 h GoodAccess outage for the Aachen node). Host stays up; only the VPN path to the server is down.
- **Detection:** site shows `cannot send to 'server': target_unreachable` continuously; the pre-run check warns if the server `host:port` is not reachable at start.
- **Fix / prevention:** stabilise the VPN (keepalive / dead-peer-detection / dedicated gateway; raise with the VPN provider). The 24 h timeouts give a long window for a brief drop to self-heal before the run aborts.

## F7 — Shared-memory (`/dev/shm`) bus error
- **Symptom:** `ERROR: Unexpected bus error … insufficient shared memory (shm)` / `DataLoader worker … killed by signal: Bus error`.
- **Root cause:** the container's default 64 MB `/dev/shm` is too small for multi-worker 3D dataloaders.
- **Fix:** swarm clients run with `--ipc=host` (shares the host `/dev/shm`); ad-hoc containers can use `--shm-size=16g`.
- **Prevention:** keep `--ipc=host` on swarm clients (already in `master_template.yml`).

---

## Operator diagnostic playbook

**Drive the live server** via the admin startup kit (run `./fl_admin.sh` in the odelia image with `--net=host`, username line first):
- `check_status server` — registered clients + last-connect times (a frozen last-connect = a stale/dead client).
- `check_status client` — per-client job state (`No Reply` = client registered but not responding).
- `list_jobs [<job_id>]` — job history/status.
- `abort_job <job_id>` (needs a `y` confirmation) — stop a hung/failed run.
- `submit_job <path>` — paths can be in-image (`/MediSwarm/application/jobs/<job>`) or mounted.

**Live job log** (server side): `/<server-kit>/<job_id>/log.txt` inside the server container — round events (`start_learn_task` / `finished_learn_task` / `better_aggregation`).

**Post-mortem of a FINISHED job:** the run workspace is moved to the job store at
`/tmp/nvflare/jobs-storage/<job_id>/workspace` (a zip) inside the server container — read `log_error.txt` for the abort reason and `log.txt` for the round timeline.

**Per-site logs:** each client keeps its NVFlare log at the kit root (`log.txt`), console at `startup/nohup.out`, and the global model under `<job_id>/app_<SITE>/{FL_global_model.pt,best_FL_global_model.pt}`.

**Monitoring a long run:** poll `list_jobs <job_id>` on a low frequency and act on a *round advance*, an error, or a terminal state — rather than watching continuously. Rounds are gated by the slowest site.

## Before every run
Run the pre-flight checks from the startup kit, **even if they passed before** — environments drift between runs (auto-updates, driver/IT changes, read-only mounts, re-synced or corrupted data):

```bash
./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training      # GPU + container smoke
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check   # data + 1 epoch
```

Both now print a `Pre-run checks` block covering F2/F3/F4/F6. Reserve enough time before the scheduled run to resolve anything they flag.
