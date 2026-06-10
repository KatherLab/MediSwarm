# MediSwarm Deploy Test Runbook

Operational notes for running the 2-node ODELIA challenge model deploy tests
(Cosmos = server/admin, dl0 = RUMC_1, dl2 = MHA_1).

---

## Infrastructure

| Role | Machine | Tailscale hostname | Site name |
|------|---------|--------------------|-----------|
| Server + admin | Cosmos (localhost) | `hd-cosmos` / `100.100.101.100` | — |
| Client 1 | dl0 | `dd-dl0` | `RUMC_1` |
| Client 2 | dl2 | `dd-dl2` | `MHA_1` |

**SSH access from Cosmos:**
```bash
ssh swarm@dd-dl0   # key-based (cosmos ~/.ssh/id_ed25519 authorized on dl0/dl2)
ssh swarm@dd-dl2
```
Password fallback: `Ekfz_ekfz` (sshpass used by deploy scripts)

**DNS is managed by the deploy runner** — `run_deploy_test.sh` updates each
remote `/etc/hosts` entry for `dl3.tud.de` before client startup. By default it
uses Cosmos's Tailscale IP. If a site needs a LAN/VPN route instead, add a
per-site override to the deploy config:

```bash
# Example: force USZ to reach Cosmos over the LAN route used in the June 2026 smoke test.
USZ_SERVER_IP_OVERRIDE=172.24.4.65
```

If two logical clients share one host, their `*_SERVER_IP_OVERRIDE` values must
match because `/etc/hosts` is host-wide.

---

## Data Paths

| Machine | Site | Data root (host) | Notes |
|---------|------|-----------------|-------|
| dl2 | MHA_1 | `/mnt/sda1/Odelia_challange/ODELIA_Challenge_unilateral` | Has `MHA_1/` subfolder |
| dl0 | RUMC_1 | `/mnt/dlhd0/medswarmdata` | Folder is `RUMC`, not `RUMC_1` — symlink required |

**RUMC_1 symlink on dl0** (one-time setup, already done):
```bash
cd /mnt/dlhd0/medswarmdata && ln -sfn RUMC RUMC_1
```
**IMPORTANT: must be a relative symlink.** An absolute symlink (`→ /mnt/dlhd0/medswarmdata/RUMC`) breaks inside Docker because the host path is bind-mounted as `/data`, not `/mnt/dlhd0/medswarmdata`. Verify with:
```bash
ssh swarm@dd-dl0 "ls -la /mnt/dlhd0/medswarmdata/RUMC_1"
# Must show: RUMC_1 -> RUMC   (relative, not absolute)
```

Scratch dirs:
- dl0: `/mnt/dlhd0/deploy_test`
- dl2: `/mnt/sda1/deploy_test`

---

## Docker Commands (must match participant README)

Always pass all three flags explicitly:

```bash
# Preflight check
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 \
    --preflight_check --job challenge_5pimed

# Local training (1 epoch for quick verification)
NUM_EPOCHS=1 ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 \
    --local_training --job challenge_2BCN_AIM

# Swarm client (participant-facing command from README)
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
```

Missing `--scratch_dir` or `--GPU` causes interactive prompts in non-interactive SSH sessions
— the script silently defaults to `/mnt/scratch` and `device=0` which may not exist.

---

## Models Under Test

| Job directory | MODEL_NAME | Notes |
|--------------|-----------|-------|
| `ODELIA_ternary_classification` | `MST` | Baseline; uses DINOv2 backbone (slow) |
| `challenge_1DivideAndConquer` | `1DivideAndConquer` | Default for `--preflight_check`/`--local_training`; 3D ResNet variant |
| `challenge_2BCN_AIM` | `2BCN_AIM` | SwinUNETR |
| `challenge_3agaldran` | `3agaldran` | MViTv2 |
| `challenge_4abmil` | `4LME_ABMIL` | Key fix in v1.4.2: was deriving `4abmil` (wrong) |
| `challenge_5pimed` | `5Pimed` | Key fix in v1.4.2: was deriving `5pimed` (wrong) |

MODEL_NAME is set two ways (both needed):
1. Auto-derive in `docker_config/master_template.yml` (case statement for 4abmil/5pimed)
2. Hardcoded in each `config_fed_client.conf` SubprocessLauncher `script` field (for admin-submitted jobs)

NVFlare SubprocessLauncher `KEY=VALUE` prefix parsing fixed in PR #295 (NVFlare submodule bump).

---

## Build & Deploy Checklist

```bash
# 1. Bump version if needed
# edit odelia_image.version, commit, push

# 2. Build image + startup kits (--num-rounds 2 for quick swarm verification)
bash scripts/build/buildDockerImageAndStartupKits.sh \
    -p application/provision/project_deploy_test_2site.yml \
    --use-docker-cache --num-rounds 2

# Note: live sync injection runs INSIDE the Docker build — do NOT run
# _injectLiveSyncIntoStartupKits.sh manually for these kits (the yml
# placeholder is un-substituted outside the container).

# 3. Push image
docker push jefftud/odelia:<VERSION>

# 4. Deploy kits to dl0 and dl2
VERSION=$(bash scripts/build/getVersionNumber.sh)
WORKSPACE="workspace/odelia_deploy_test_${VERSION}_model_test/prod_00"
for SITE_NAME in RUMC_1 MHA_1; do
    HOST=$(grep "${SITE_NAME%_*}" deploy_sites_2node_test.conf ...)
    sshpass -p 'Ekfz_ekfz' scp $WORKSPACE/${SITE_NAME}_${VERSION}.zip swarm@<HOST>:~/deploy_test/
    sshpass -p 'Ekfz_ekfz' ssh swarm@<HOST> "cd ~/deploy_test && unzip -qo ..."
done

# 5. Pre-pull image on remotes
for HOST in dd-dl0 dd-dl2; do
    ssh swarm@$HOST "docker pull jefftud/odelia:$VERSION" &
done; wait

# 6. DNS and live monitor cleanup
# run_deploy_test.sh handles /etc/hosts updates and stale live-sync cleanup.
```

---

## Running Tests

### Preflight + Local Training (dl2 only)
```bash
bash scripts/deploy/run_preflight_localtraining_test.sh \
    --conf deploy_sites_2node_test.conf
```
Logs saved to `workspace/preflight_localtraining_results/`.

### Swarm Training (dl0 + dl2, all 6 models)
```bash
bash scripts/deploy/run_deploy_test.sh \
    --all --conf deploy_sites_2node_test.conf --skip-build
```
- Cosmos runs server + admin; submits jobs via NVFlare admin CLI using `expect`
- Clients started on dl0/dl2 via SSH with `docker.sh --start_client`
- `num_rounds=2` guarantees ≥1 full sync per node
- The deploy runner enables the ODELIA preprocess cache by default:
  `ODELIA_ENABLE_PREPROCESS_CACHE=1` and
  `ODELIA_PREPROCESS_CACHE_DIR=$SCRATCHDIR/odelia_preprocess_cache`.
  Add `<SITE>_PREPROCESS_CACHE=0` in the deploy config to disable it for a
  site, or `<SITE>_PREPROCESS_CACHE_DIR=/path/to/cache` to override the path.
- Before each client launch, stale swarm `live_sync.sh` daemons for that site
  are stopped so the web monitor receives logs for the current run only.

---

## Monitoring Swarm Training

**Server log** (completion signal):
```bash
tail -f /home/jeff/deploy_test/dl3.tud.de/startup/nohup.out
# Look for: "Server runner finished." (clean) or "FATAL_SYSTEM_ERROR" (abort)
```

**Client logs** — most important for diagnosing hangs/crashes:
```bash
# dl0 / RUMC_1
ssh swarm@dd-dl0 "tail -f /home/swarm/deploy_test/RUMC_1/startup/nohup.out"

# dl2 / MHA_1
ssh swarm@dd-dl2 "tail -f /home/swarm/deploy_test/MHA_1/startup/nohup.out"
```

The `wait_for_completion()` function in `run_deploy_test.sh` now polls client `nohup.out`
every 2 minutes and prints the last line per client so hangs are detected early.

**Container status:**
```bash
ssh swarm@dd-dl0 "docker ps --filter name=odelia"
ssh swarm@dd-dl2 "docker ps --filter name=odelia"
```

---

## Known Issues & Quirks

- **Evaluation step** in `run_deploy_test.sh` will be skipped (no UKA_1 data on Cosmos)
  — result shown as `PARTIAL` (train=pass, eval=skipped), which is expected for this test.
- **`_injectLiveSyncIntoStartupKits.sh`** called manually fails because the project YAML
  still has the `__REPLACED_...__` placeholder outside the build container. Injection is
  handled automatically during the Docker build.
- **RUMC folder mismatch**: data folder on dl0 is `RUMC`, not `RUMC_1`. Symlink
  `/mnt/dlhd0/medswarmdata/RUMC_1 → RUMC` must exist before tests.
- **Tailscale IPs change** across sessions — always resolve via hostnames (`dd-dl0`,
  `dd-dl2`) rather than hardcoded IPs. Check with `tailscale status` if SSH fails.
- **nohup.out is appended** across server restarts — `wait_for_completion()` records
  the line count at job start and only scans new lines to avoid false positives from
  previous runs.

---

## Timing Baselines (v1.4.2, 2026-04-17, 2-node RUMC_1+MHA_1)

Reference times from a clean run (training only; evaluation is PARTIAL/skipped):

| Model | Training | Total (incl. eval+teardown) |
|-------|----------|-----------------------------|
| MST | 810s (14 min) | 1717s (29 min) |
| 1DivideAndConquer | 3240s (54 min) | 3667s (61 min) |
| 2BCN_AIM | 1140s (19 min) | 1263s (21 min) |
| 3agaldran | 1380s (23 min) | 1552s (26 min) |
| 4LME_ABMIL | 1110s (18 min) | 1232s (21 min) |
| 5Pimed | 1140s (19 min) | 1252s (21 min) |

**Full suite runtime:** ~3h (excluding MST retry from RUMC_1 symlink fix).
`1DivideAndConquer` dominates due to large 3D ResNet model (~106 MB per sync).
