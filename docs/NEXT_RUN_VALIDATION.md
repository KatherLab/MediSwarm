# Next Swarm Run — Validation Checklist & Runbook

A pickup-able runbook for the next robust swarm run and the deferred validations that need a
multi-site window. Any operator (or chat) can resume from here.

> **Status of the things this validates**
> - Warm-continue (`auto`/`fresh`/`require`) + abort-recovery — merged (#361, #365); 2-node validated, **needs production-scale**.
> - Fault-tolerance prune-and-continue — merged (#356); **induced single-node-drop not yet exercised live** (#346).
> - Best-model selection `key_metric=val/AUC_ROC` — merged (#365, fixes #364); **confirm it fires at scale**.
> - Prediction-export throttle — merged (#358); **re-measure the GPU export cost to re-tune `ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS`**.
> - Degenerate-input guard — PR #368 (supersedes #340); run `find_degenerate_inputs` per site.

---

## 0. Credentials & secrets — READ FIRST

**Passwords and admin credentials are NOT stored in this repo** (committed secrets persist in git
history forever, are flagged by the repo's GitGuardian check, and can't be truly deleted). Provide
them at runtime via environment variables, sourced from the operator / the secure channel that
shared them.

- Site connection details + the password **env mechanism** live in the gitignored
  `deploy_sites_*.local.conf` (template: [`deploy_sites_rsh_mha.local.conf.example`](../deploy_sites_rsh_mha.local.conf.example)).
  These files match `deploy_sites_*.local.conf` in `.gitignore` and never get committed.
- Expected env vars (export from the secure channel, do **not** commit):
  ```bash
  export RSH_PASS='…'      # ssh/sudo for asoro@172.24.4.71
  export MHA_PASS='…'      # ssh/sudo for odelia@172.24.4.91
  export USZ_PASS='…'      # ssh/sudo for user@172.24.4.75
  export UKA_PASS='…'      # ssh/sudo for swarm@172.24.4.79   (full-scale runs only)
  # dl servers (server/admin + GPU eval): swarm@dl0/dl2/dl3 — admin password via the secure channel
  ```
- A future chat **on this machine** can reuse the already-populated gitignored
  `deploy_sites_rsh_mha.local.conf`; for the 3-site run, copy it to
  `deploy_sites_3site.local.conf` and add the USZ block (see §2).

### On-disk secrets file (unattended pickup)

For unattended runs / future chats on this machine, the passwords live in **`~/.mediswarm_secrets`**
(`chmod 600`, **outside the repo, never committed**). Source it before a run:
```bash
source ~/.mediswarm_secrets   # exports RSH_PASS, MHA_PASS, USZ_PASS, UKA_PASS, DL_ADMIN_PASS
```
Template — fill with the real values and keep the file out of the repo (do not paste real
passwords into anything tracked by git):
```bash
export RSH_PASS='…'        # asoro@172.24.4.71
export MHA_PASS='…'        # odelia@172.24.4.91
export USZ_PASS='…'        # user@172.24.4.75
export UKA_PASS='…'        # swarm@172.24.4.79  (full-scale runs only)
export DL_ADMIN_PASS='…'   # swarm@dl0 / dl2 / dl3 (server-admin + GPU eval)
```

---

## 1. Scope of this run

Run on **RSH + MHA + USZ** (3 clients). **Leave out UKA for the quick validation** — its data is
~9× the others and a single site dominates round time; use UKA only for a full production run.
Server/admin on **Cosmos / dl3.tud.de**; GPU evaluation on **dl0** (Quadro RTX 6000).

Use a **lightweight model (`5Pimed`)** for the mechanism checks (fast rounds); use
**`1DivideAndConquer` (1DC)** for the realistic production-scale pass (≈690 MB global, 20 rounds).

---

## 2. Participating sites (non-secret operational info)

| Site | Host (VPN) | SSH user | `SITE_NAME` | `DATADIR` (host) | `SCRATCHDIR` (host) | GPU | Pass env |
|---|---|---|---|---|---|---|---|
| **RSH** | 172.24.4.71 | `asoro` | `RSH_1` | `/home/asoro/JULIA_ITERATION/odelia_breast_mri/odelia/RSH` | `…/odelia/RSH/scratch` | `device=0` | `RSH_PASS` |
| **MHA** | 172.24.4.91 | `odelia` | `MHA_1` | `/home/odelia/MediSwarm/data` | `/home/odelia/MediSwarm/data/MHA_1/tmp` | `device=0` | `MHA_PASS` |
| **USZ** | 172.24.4.75 | `user` | `USZ_1` | `/mnt/3aef1f67-f1f1-46a8-9ba1-1387521ef48d/Swarm_learning/Data/Data_all` | `/mnt/3aef1f67-f1f1-46a8-9ba1-1387521ef48d/Swarm_learning/Setup/USZ/scratch` | `device=0` | `USZ_PASS` |
| _UKA (full-scale only)_ | 172.24.4.79 | `swarm` | `UKA_1` | _(per site)_ | _(per site)_ | `device=0` | `UKA_PASS` |

- Data layout note: the model loads `<DATADIR>/<SITE_NAME>/data_unilateral/<uid>/Sub_1.nii.gz`
  (resolved via `env_config`); ignore any stale top-level `data_unilateral` copies.
- `MODEL_NAME=5Pimed` (mechanism) or `1DivideAndConquer` (scale). UKA was given `MODEL_NAME=challenge_5Pimed`.

**dl servers**
| Role | Host | User | Notes |
|---|---|---|---|
| Server + admin | `dl3.tud.de` (= Cosmos / agh1, VPN 172.24.4.65) | `swarm` | `/srv/mediswarm/live` live monitor here |
| GPU eval | `dl0` (`dd-dl0`, key-based) | `swarm` | Quadro RTX 6000 — eval works here (Cosmos RTX 5070 fails `no kernel image`) |

`deploy_sites_3site.local.conf` (gitignored) skeleton — copy from the `.example` and add USZ:
```bash
CLIENT_SITES=(RSH MHA USZ)
SERVER_NAME=dl3.tud.de
ADMIN_USER=jiefu.zhu@tu-dresden.de
RSH_HOST=172.24.4.71;  RSH_USER=asoro;  RSH_PASS_ENV=RSH_PASS;  RSH_SITE_NAME=RSH_1
RSH_DATADIR=/home/asoro/JULIA_ITERATION/odelia_breast_mri/odelia/RSH
RSH_SCRATCHDIR=/home/asoro/JULIA_ITERATION/odelia_breast_mri/odelia/RSH/scratch
RSH_DEPLOY_DIR=/home/asoro/mediswarm_run; RSH_GPU="device=0"
MHA_HOST=172.24.4.91;  MHA_USER=odelia;  MHA_PASS_ENV=MHA_PASS;  MHA_SITE_NAME=MHA_1
MHA_DATADIR=/home/odelia/MediSwarm/data
MHA_SCRATCHDIR=/home/odelia/MediSwarm/data/MHA_1/tmp
MHA_DEPLOY_DIR=/home/odelia/mediswarm_run; MHA_GPU="device=0"
USZ_HOST=172.24.4.75;  USZ_USER=user;    USZ_PASS_ENV=USZ_PASS;  USZ_SITE_NAME=USZ_1
USZ_DATADIR=/mnt/3aef1f67-f1f1-46a8-9ba1-1387521ef48d/Swarm_learning/Data/Data_all
USZ_SCRATCHDIR=/mnt/3aef1f67-f1f1-46a8-9ba1-1387521ef48d/Swarm_learning/Setup/USZ/scratch
USZ_DEPLOY_DIR=/home/user/mediswarm_run; USZ_GPU="device=0"
```

---

## 3. Pre-flight (per site, before the run)

- [ ] **Kits rebuilt from `main`** (now carry warm-continue `auto` + FT + the `key_metric` fix) and redistributed.
- [ ] **Docker GPU/cgroup** fixed: `sudo ./scripts/client_node_setup/fix_docker_cgroupfs.sh`, recreate the client container.
- [ ] **VPN as a service**: `sudo ./scripts/client_node_setup/setup_vpntunnel.sh -d <Site> -n -s` + `vpn_health_monitor.sh --install-timer`.
- [ ] **Live-sync** works: `ssh -o BatchMode=yes mediswarm-upload@dl3.tud.de 'echo ok'` ⇒ `ok`.
- [ ] **Pre-run checks**: `./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training` then `--preflight_check` ⇒ green.
- [ ] **`/scratch`** writable, persistent (not tmpfs), and has room for the ≈690 MB mirror (1DC).
- [ ] **Degenerate-input scan** (#368): run `find_degenerate_inputs` per site; exclude any hits via `ODELIA_EXCLUDE_UIDS_FILE` and report them to the site. (RSH/MHA/USZ scanned clean on 2026-06-29.)

---

## 4. Validation checks (run order)

For each: what to do + the pass bar. Record results into the run's `summary.json` / the live monitor.

1. **Normal run completes** — submit the job (5Pimed first, then 1DC); all rounds finish; each site has `FL_global_model.pt`. *Baseline.*
2. **Best-model selection fires (#364)** — client logs show `global best metric is <number>` (not `None`) + `GLOBAL_BEST_MODEL_AVAILABLE` + `WarmStart: mirrored best global`. Confirm whether challenge eval pulls **best** vs **latest** (it changed).
3. **Warm-continue continuity (#347)** — after a clean run, resubmit `--warm-start continue`; each client logs `will warm-start from checkpoint /scratch/mediswarm_latest_global.pt (mode=require)`; the continue mirror md5 == the prior run's mirror md5 (byte-identical).
4. **Abort-recovery (#347, the real scenario)** — kill the run mid-round (or `docker stop` a client) **after** a round has mirrored; resubmit `--warm-start continue`; it **resumes from the surviving mirror and completes**. The ≈690 MB mirror copies cleanly each round without bloating `/scratch` or slowing rounds.
5. **Induced single-node-drop (FT, #346)** — with training underway, `docker stop` one client; server logs `FaultTolerant: client … pruning and continuing` and the round completes with the rest (≥ `min_clients`); restore the node afterward.
6. **GPU eval on dl0** — collect the globals; run `scripts/evaluation/predict.py` on **dl0** (Quadro RTX 6000); valid metrics, no `no kernel image` error.
7. **Export-throttle re-measure (#358)** — time one round with `ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS=1` vs `0` on a representative node (GPU inference, post-`b4a4bc8`); set the production default from the measured per-round cost.

---

## 5. Tooling

- **2-node warm-continue + abort-recovery** (RSH/MHA pattern): [`scripts/deploy/run_warm_continue_test.sh`](../scripts/deploy/run_warm_continue_test.sh)
  — phases: `negative_continue`, `fresh`, `continue` (continuity hash), `fresh_probe`, `abort_recovery`.
  Note: it asserts exactly 2 clients; for the 3-site run use the standard deploy flow (below) or extend `CLIENT_SITES`.
  ```bash
  cp deploy_sites_rsh_mha.local.conf.example deploy_sites_rsh_mha.local.conf   # fill in (gitignored)
  export RSH_PASS='…' MHA_PASS='…'
  scripts/deploy/run_warm_continue_test.sh --conf deploy_sites_rsh_mha.local.conf \
    --project application/provision/project_warm_continue_rsh_mha.yml --job challenge_5pimed --model 5Pimed
  ```
- **General multi-site deploy/smoke**: [`scripts/deploy/run_deploy_test.sh`](../scripts/deploy/run_deploy_test.sh) (uses a `deploy_sites*.conf`).
- **Warm-continue job prep (operator)**: `<admin-kit>/startup/prepare_odelia_job.sh --job <JOB> --warm-start fresh|continue` → `submit_job <printed path>` (see [README.operator.md](../assets/readme/README.operator.md#recover-an-aborted-run)).
- **Degenerate-input preflight (#368)**: `ODELIA_Dataset3D.find_degenerate_inputs()` → lists constant/corrupt inputs; write hits to `$ODELIA_EXCLUDE_UIDS_FILE` to exclude.
- **Failure modes / recovery**: [`docs/SWARM_FAILURE_MODES.md`](SWARM_FAILURE_MODES.md).

---

## 6. Recording template (per run)

```
run_id, git_sha, docker_image, model, sites=[RSH,MHA,USZ]
per-round wall-time (and with/without prediction export)
mirror md5 per site (fresh vs continue → continuity)
best-AUROC + whether best_FL_global_model.pt updated
abort-recovery: aborted round → resumed round (pass/fail)
node-drop: dropped site → run continued (pass/fail)
eval (dl0): per-site test AUROC
degenerate-input scan: per-site bad count (expect 0)
```

---

## 7. Related issues / PRs

- #347 warm-continue · #346 fault-tolerance · #364 key_metric (fixed) · #358 export throttle ·
  #368 degenerate-input guard (supersedes #340) · #361/#365 merged warm-continue work.
