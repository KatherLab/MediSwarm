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
| F8 | Run trains on a **subset**: `clients [...] did not configure within timeout but min_clients=N allows proceeding` | Controller stops waiting once `configure_min_clients` answer; slower sites lose the key exchange | Set `configure_min_clients` **= number of participating sites** |
| F9 | Site never appears in `check_status server`, container reports `(healthy)` for weeks | Startup kit older than the server's provisioning generation → `ClientConnectorCertificateError` | Re-issue the current startup kit to that site |
| F10 | `cross_val_results.json` is `{}` although the server logged `Published metrics for N site(s)` | Two components act on the same `END_RUN`; `ValidationJsonGenerator` writes the file before the later-listed collector publishes into it | Publish on `ABOUT_TO_END_RUN`, which is fired strictly earlier (already fixed in `per_site_metrics.py`) |

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
- **Fallback (host stuck on `systemd`):** install `scripts/client_node_setup/gpu_container_watchdog.sh install` — a 2-min systemd timer that `docker restart`s a GPU-less (`unhealthy`) client **only when no GPU job is running**, so an idle container auto-recovers without disrupting an active training (#343).

## F3 — Stale `daemon_pid.fl` blocks client restart
- **Symptom:** after `docker rm -f` of a client, the new container is `Up … (healthy)` (the healthcheck only runs `nvidia-smi`) but the site never registers; `nohup.out` shows `There seems to be one instance, pid=N, running … remove daemon_pid.fl`.
- **Root cause:** an unclean stop leaves a `daemon_pid.fl` lock in the kit root; `start.sh` refuses to launch a "second" instance.
- **Detection:** the pre-run check (for `--start_client`) detects the lock and removes it when no client container is running.
- **Fix:** delete `daemon_pid.fl` in the kit root, then relaunch `./docker.sh --start_client`.
- **Prevention:** always clear the lock when recreating a client; the pre-run check now does this automatically.

**F3b — a host reboot mid-run triggers this silently, and the run then hangs for a full day.**
Observed 2026-09-04 on UKA during job `24cdf247`: the node rebooted (`uptime` showed 1:16 while the
run had started two hours earlier), Docker auto-restarted the client at boot, `start.sh` hit a
`daemon_pid.fl` dated **five weeks earlier** and refused to launch. The container then sat as
`Up (healthy)` running only `/bin/bash` -- no FL client, no training process, GPU at 0 %.

Server-side the consequence is worse than the site outage: the swarm logs
`Client UKA_1 is deemed disconnected!`, the remaining peers wait on a gather that can never
complete, and with `progress_timeout = 86400` the job stays `RUNNING` for **24 hours** before
anything fails. The server log simply stops advancing.

- **Detection:** `uptime` on the node; GPU utilisation 0 % with the job still `RUNNING`;
  `docker exec <client> ps -eo pid,args` showing no `python3 -m nvflare...`; server `log.txt`
  mtime frozen while job status is still `RUNNING`.
- **Recovery:** remove `daemon_pid.fl` **and** `pid.fl` from the kit root, relaunch the client
  (`docker exec -d <client> /bin/bash -c 'cd /startupkit/startup && nohup ./start.sh >> ../nohup.out 2>&1'`
  restarts it inside the existing container without recreating it), confirm
  `Successfully registered client:<SITE>`, then abort the wedged job.
- **Aborting:** `abort_job <job_id>` needs a `y` confirmation -- an automated session that does not
  send it will appear to hang on the abort itself. Only if the abort genuinely will not take should
  you kill that job's `runner_process` in the server container (`ps -eo pid,args | grep runner_process`,
  match the job id, `kill -TERM`); the job then lands as `FINISHED:EXECUTION_EXCEPTION`. This stops
  one job, not the server.
- **Worth knowing:** the warm-start mirror is only rewritten when a round completes, so a run killed
  mid-round leaves the previous global intact. Verified here: the `87c5bbee` mirror was still
  byte-identical afterwards. See #535 for the mirror's lack of provenance guarding.

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
- **Recovery:** if any abort mode (F1/F5/F6) does fire, the run can be resumed without losing progress — each client mirrors the latest global to `/scratch/mediswarm_latest_global.pt` every round. See [Recover an aborted run](../assets/readme/README.operator.md#recover-an-aborted-run).

## F7 — Shared-memory (`/dev/shm`) bus error
- **Symptom:** `ERROR: Unexpected bus error … insufficient shared memory (shm)` / `DataLoader worker … killed by signal: Bus error`.
- **Root cause:** the container's default 64 MB `/dev/shm` is too small for multi-worker 3D dataloaders.
- **Fix:** swarm clients run with `--ipc=host` (shares the host `/dev/shm`); ad-hoc containers can use `--shm-size=16g`.
- **Prevention:** keep `--ipc=host` on swarm clients (already in `master_template.yml`).

---

## F8 — Silent client dropout during configuration (the run trains on a subset)
- **Symptom (server job log):** `client configuration took 5.8 seconds` followed by
  `clients ['RUMC_1', 'UKA_1'] did not configure within timeout but min_clients=5 allows proceeding; they remain as participants and may rejoin in a later round`.
  The run then proceeds happily with the remaining sites and **reports success** — the missing sites contribute nothing.
- **Symptom (dropped site's `startup/nohup.out`):**
  ```
  CoreCell WARNING ... CH=credential_manager TP=key_exchange ... no connection to child <SITE>.<job_id>
  CoreCell ERROR   ... CH=credential_manager TP=key_exchange ... cannot forward req: no path
  ```
  then ~15 min later `PipeHandler: peer gone: no heartbeat for 900.0s`, `No best checkpoint found.`,
  `No last checkpoint found.` — the training subprocess started, idled, and exited having trained nothing.
- **Root cause:** the controller waits only until **`configure_min_clients`** sites have answered `swarm_config`, then advances. A site whose job cell takes a second or two longer to come up (worker process launched, cell not yet registered) receives the p2p `key_exchange` before its child cell exists — `no path` — and is left behind. It is a **race, not a site fault**: the affected sites are healthy, correctly provisioned, and their data loads.
- **Detection:** in the job log, count `successfully configured client` lines — it must equal the number of participating sites. `check_status server` showing all sites *registered* is **not** sufficient: registered ≠ configured ≠ training.
- **Fix:** submit the job with `configure_min_clients` (and `min_clients`) equal to the participating-site count:
  ```bash
  ./prepare_odelia_job.sh --job ODELIA_ternary_classification --warm-start fresh \
      --num-rounds N --min-clients 7 --configure-min-clients 7
  ```
  Job configs ship with the submitted job (byoc) — **no startup-kit rebuild needed**.
- **Prevention:** for any **benchmark** run, `configure_min_clients` must equal the number of sites, because a consortium benchmark requires every site to contribute. The default (`min_clients = 5`) exists for resilience during casual runs, but it will silently drop the **slowest** sites — which on a real consortium tend to be the **largest** ones, i.e. exactly the data you least want to lose.
- **Observed 2026-07-31:** a 7-site run dropped `RUMC_1` and `UKA_1` at 5.8 s with `min_clients=5`; re-submitting the identical job with `--min-clients 7 --configure-min-clients 7` configured all seven. Neither site needed any change.

## F9 — Stale startup kit: client "healthy" but never registers
- **Symptom:** the site is absent from `check_status server` while `docker ps` shows its client `Up N weeks (healthy)`. The site believes it is connected.
- **Symptom (client `startup/nohup.out`), repeating every ~10 s:**
  ```
  conn_manager ERROR - Connector [CH0000x ACTIVE http://<server>:8002] failed with exception ClientConnectorCertificateError
  CoreCell WARNING - ... no connection to server
  ```
- **Root cause:** the site is running a startup kit from an **older provisioning generation** than the server. VPN and routing are fine — the TLS handshake is rejected because the kit's certificates were issued by a superseded root.
- **Detection:** compare the client container-name suffix / image tag against the current kit
  (`docker ps --filter name=odelia_swarm_client`), and grep the site's `nohup.out` for `CertificateError`.
  Docker's `(healthy)` only means the process is alive — it says nothing about FL registration.
- **Fix:** issue the current startup kit, extract it into a **fresh empty directory** (do not overwrite an old kit — sites accumulate several), then:
  ```bash
  docker rm -f <old_client_container>
  rm -f <new_kit>/daemon_pid.fl <new_kit>/pid.fl      # see F3
  cd <new_kit>/startup && ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
  ```
  Success looks like `Successfully registered client:<SITE> for project ...` / `Connected to server`.
- **Prevention:** after every kit roll-out, verify the expected site count in `check_status server` rather than assuming; a silently-stale site can persist for weeks.
- **Observed 2026-07-31:** `RSH_1` had been looping on `ClientConnectorCertificateError` for **three weeks** on a v1.5.0 kit while reporting `(healthy)`; installing the current kit registered it immediately.

## F10 — Empty `cross_val_results.json` from event ordering

- **Symptom:** every peer-to-peer run leaves `cross_site_val/cross_val_results.json` as `{}` — on a
  two-day 20-round run and on a 52-second smoke run alike — so per-site figures have to be collected
  by logging into each hospital by hand.
- **What makes it deceptive:** the server log says the metrics arrived.
  ```
  PerSiteMetricCollector - INFO - Published metrics for 8 site(s):
      CAM_1, MHA_1, RSH_1, RUMC_1, UKA_1, UMCU_1, USZ_1, VHIO_1
  ```
  Every hop works — `ClientLogger` → `MLflowWriter` → metric `CellPipe` → `MetricRelay` →
  `fed.analytix_log_stats` → `ClientFedEventRunner` → `ServerFedEventRunner` → the collector.
  Nothing is missing and nothing errors.
- **Root cause:** ordering *within a single event*. `ValidationJsonGenerator` dumps the JSON in its
  own `END_RUN` handler; `AnalyticsReceiver.finalize` also runs on `END_RUN`; and
  `fire_event_to_components` invokes components in the order they are listed in
  `config_fed_server.conf`. With `json_generator` listed first — as it is in all seven job configs —
  the file is written and only then does the collector publish into the generator's in-memory dict.
- **Why it reads as a missing metric:** `ServerRunner` logs `END_RUN fired` *after* the dispatch
  returns, so the publish line appears ~2 s *before* it in the log. The two look sequential when
  they are in fact the same event.
- **Fix:** publish on `EventType.ABOUT_TO_END_RUN`, which is fired strictly before `END_RUN`, so the
  result does not depend on how a job config happens to be ordered. `END_RUN` remains a fallback;
  `finalize` is idempotent.
- **Prevention:** when two components react to the same event and one consumes what the other
  produces, do not rely on config order — separate them onto different events. And note that CI was
  green throughout: `pytest.importorskip("nvflare")` skipped the collector's tests entirely because
  the workflow never installed NVFlare (cf. #416/#423). A skipped test file is not a passing one.
- **Observed 2026-09-04:** job `7c6e72c6` reported all eight sites and still wrote `{}`.

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

**Immediately after submitting a job, verify participation** — a run that silently trains on a
subset still reports success (F8):

```bash
# 1. every participating site must have registered
#    (admin console) check_status server   -> "Registered clients: <N>"
# 2. every participating site must have CONFIGURED for this job
grep -c 'successfully configured client ' <server-kit>/<job_id>/log.txt   # must equal <N>
# 3. and each must reach the round
grep 'updated status of client' <server-kit>/<job_id>/log.txt | grep start_learn_task
```

Registered != configured != training. Check all three.
