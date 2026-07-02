# ODELIA Swarm v1.5.0 Rollout Runbook

**Status as of 2026-07-02.** Living handoff doc for the v1.5.0 release + the next
8-site production run. A fresh session (human or Claude) should be able to pick up
from here without re-deriving context.

## TL;DR — what's done, what's pending

| Item | Status |
|---|---|
| Release `v1.5.0` (tag + GitHub Release) | ✅ done — <https://github.com/KatherLab/MediSwarm/releases/tag/v1.5.0> |
| Docker image `jefftud/odelia:1.5.0-dev.260702.be9ef04` | ✅ built + **pushed to DockerHub** (digest `sha256:d4663e21…`) |
| All-site startup kits | ✅ built (see paths below) |
| Run quorum `8 / 7 / 7` | ✅ set via `prepare_odelia_job.sh` flags at job-prep time (**no rebuild** — byoc) |
| Partner email | ✅ drafted (below) — **not yet sent** |
| Server node (dl3 = Cosmos) | ✅ **running** since 2026-07-02 12:11 — container `odelia_swarm_server_flserver_be9ef04` (`on-failure:5`; must be restarted manually after a host reboot: `cd <server kit>/startup && ./docker.sh --no_pull --start_server`) |
| Live-sync SSH keys for UKA / UMCU / VHIO | ⏳ UMCU: key already authorized, needs only the host-key refresh (§4); UKA / VHIO pending pubkeys |
| Run | ⛔ scheduled tomorrow / this weekend |

## 1. Release artifacts

- **Version source:** `odelia_image.version` = `1.5.0`. The build appends `-dev.<YYMMDD>.<shorthash>` (see `scripts/build/getVersionNumber.sh`), giving `1.5.0-dev.260702.be9ef04`.
- **Image:** `jefftud/odelia:1.5.0-dev.260702.be9ef04` (15.6 GB), on DockerHub.
- **Kits:** `workspace/odelia_1.5.0-dev.260702.be9ef04_allsites_test/prod_00/<SITE>_1.5.0-dev.260702.be9ef04.zip`
  - Clinical sites: `UKA_1`, `CAM_1`, `VHIO_1`, `MHA_1`, `RSH_1`, `USZ_1`, `UMCU_1`, `RUMC_1`
  - Also: `TUD_1`, `TUD_2`, `MEVIS_1/2/3`, server `dl3.tud.de`, admin `jiefu.zhu@tu-dresden.de`
- **What shipped (issues closed for this release):** fault-tolerance #346, warm-continue #347, VPN auto-recover #348, GPU watchdog + cgroupfs #343, 24 h timeouts #345, prediction-export throttle #314, worker cap #315, CI fix #353, NVFlare fork → `MediSwarm-2.7.2` (`configure_min_clients`). Full list in `CHANGELOG.md` `## [1.5.0]`.

## 2. How to run the 8-site experiment

Job configs are **not** in the startup kits — the admin kit's `prepare_odelia_job.sh`
patches them at prep time from `/MediSwarm/application/jobs/<job>`. Set the quorum
with flags (no image/kit rebuild needed):

```bash
# in the admin kit's startup dir (jiefu.zhu@tu-dresden.de/startup):
./prepare_odelia_job.sh --job ODELIA_ternary_classification --warm-start fresh \
    --num-rounds 20 --configure-min-clients 8 --min-clients 7 --min-responses 7
# then submit the prepared job from the admin console.
```

Swap `--job` for whichever model is being run (`challenge_1DivideAndConquer`, etc.).
For a resume after an abort, use `--warm-start continue` (loads
`/scratch/mediswarm_latest_global.pt`).

### Why 8 / 7 / 7 (config rationale)

- **`configure_min_clients = 8`** — ALL clients must finish the config phase before
  `swarm_start`. If left unset it falls back to `min_clients`; with `min_clients < #sites`
  the server can start on an unconfigured `starting_client` → `invalid model learnable`
  → immediate abort. This is the CCWF config-phase quorum bug fixed in the
  `MediSwarm-2.7.2` fork. **Set it to the actual participant count.**
- **`min_clients = 7` / `min_responses_required = 7`** — runtime fault tolerance:
  tolerate exactly one transient drop; each round still aggregates ≥7/8 sites
  (protects model quality). Warm-continue covers the rarer 2+ simultaneous drop.
  The shipped default (`5`) is a placeholder — too tolerant for 8 sites.
- **`num_rounds = 20`** — full run (multi-day at 8 sites over VPN). Warm-continue
  makes an abort recoverable; alternatively chain shorter (e.g. 5-round) jobs.
- **Timeouts** (already committed in the job configs at 24 h / 86400 s):
  `peer_read_timeout`, `learn_task_timeout`, `learn_task_ack_timeout`,
  `final_result_ack_timeout`, `progress_timeout`, `max_status_report_interval`.
  A transient stall self-heals instead of aborting. Good as-is for slow VPNs.
- **If fewer than 8 sites are ready:** scale to `configure = N`, `min = N-1`,
  `responses = N-1`.

### Not yet validated at scale (go/no-go)

Fault-tolerance (#346) and warm-continue (#347) are merged defaults and validated at
2–3 sites, but the induced single-node-drop and kill-mid-run→resume tests at 8-node
scale are still pending (tracked in #316). Treat this run partly as that validation:
set the quorum conservatively, confirm `/scratch` is writable **and persists across
runs** on every client, and be ready to `--warm-start continue` rather than restart.

## 3. Live-monitor (log upload) status

Monitor host = **this server** (the box with `/srv/mediswarm/live`; sites reach it as
`mediswarm-upload@dl3.tud.de` over the VPN). Re-check with **newest file per site**
(directory mtime is misleading):

```bash
for s in UKA_1 CAM_1 VHIO_1 MHA_1 RSH_1 USZ_1 UMCU_1 RUMC_1; do
  d="/srv/mediswarm/live/$s"
  [ -d "$d" ] && echo "$s: $(find "$d" -type f -printf '%TF %TR\n' 2>/dev/null | sort | tail -1)" \
              || echo "$s: (never uploaded)"
done
```

Status 2026-07-02 ~13:15:

| Site | Newest upload | Status |
|---|---|---|
| RUMC_1 | 07-02 13:13 | ✅ working |
| CAM_1 | 07-02 12:16 | ✅ working |
| MHA_1 | 07-02 12:16 | ✅ working |
| RSH_1 | 07-02 12:16 | ✅ working (recovered since the 06-25 snapshot) |
| USZ_1 | 07-02 12:16 | ✅ working |
| UKA_1 | 06-25 09:57 | ⚠️ stopped ~1 week ago |
| UMCU_1 | 04-13 | ❌ dead since April |
| VHIO_1 | — | ❌ never configured |

**Only UKA, UMCU, VHIO need action.**

## 4. Authorize a site's live-sync SSH key (monitor host = this server)

`mediswarm-upload` (uid 1012, `/home/mediswarm-upload`) has ~14 keys authorized.
When a site replies with their public key:

```bash
echo 'PASTE_SITE_PUBKEY_LINE' | sudo tee -a /home/mediswarm-upload/.ssh/authorized_keys
sudo chown mediswarm-upload:mediswarm-upload /home/mediswarm-upload/.ssh/authorized_keys
sudo chmod 600 /home/mediswarm-upload/.ssh/authorized_keys
```

**Always append** — overwriting breaks the working sites. Site verifies from their
node (VPN up): `ssh -o BatchMode=yes mediswarm-upload@dl3.tud.de 'echo ok'` → `ok`.

**"Host key verification failed" ≠ key problem.** The monitor host was re-keyed
since April, so *returning* sites hit this instead of `ok` even with an authorized
key (UMCU did on 2026-07-02 — their key was already in `authorized_keys`; only the
stale `known_hosts` entry needed refreshing). Fix on the **site** node:

```bash
ssh-keygen -R dl3.tud.de 2>/dev/null
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 mediswarm-upload@dl3.tud.de 'echo ok'
```

Monitor-host key fingerprint for out-of-band verification (ED25519):
`SHA256:UrpTNm/Rq/dOPiCV4XVx/Mal5LpQmVS9tbOQJaFBE5E`

## 5. Host-side prep required at every site (partner actions)

1. **Docker cgroup driver** — `sudo scripts/client_node_setup/fix_docker_cgroupfs.sh`
   then recreate the client container. Fixes the "GPU vanished mid-run after a
   daemon-reload" failure (#343). Idempotent. (VHIO was still on the `systemd`
   driver and most needs this.)
2. **VPN as a service** — `setup_vpntunnel.sh -d <Site> -n -s` +
   `vpn_health_monitor.sh --install-timer` (systemd `mediswarm-vpn` + 30 s `tun0`
   watchdog, #348). Not a manual/GUI OpenVPN.
3. **Pre-flight** — `./docker.sh … --dummy_training` then `--preflight_check`; cache
   under `/scratch` (never `/data`); `/scratch` must persist across runs (warm-continue
   mirror lives there); clear stale `daemon_pid.fl` on restart.

## 6. Rebuild procedure (reference — only if source changes)

```bash
# clean tree required; run from repo root:
./scripts/build/buildDockerImageAndStartupKits.sh -p application/provision/project_Odelia_allsites.yml
docker push jefftud/odelia:<printed image tag>
```
`--use-docker-cache` speeds a re-build (heavy pip layers cached). A source change
bumps the git hash → new image tag → new kits. NOTE: job-config/quorum changes do
**not** need a rebuild (use the `prepare_odelia_job.sh` flags in §2).

## 7. Combined partner email (send-ready)

> **Subject: New ODELIA swarm kits (v1.5.0) — please prepare your node before the next run (by end of day today)**
>
> Dear partners,
>
> Attached are your new startup kits (**v1.5.0**) with substantial robustness fixes from the recent runs: faster rounds, tolerance for a single node's transient drop, automatic resume after an abort, and GPU- and VPN-auto-recovery. We plan to run **tomorrow / this weekend**, so please complete the steps below **by end of day today** and reply to confirm.
>
> **Steps 1–3 are for everyone. Step 4 is specific to your site.**
>
> **1. Fix the Docker GPU/cgroup driver (most important).** From your kit's `scripts/client_node_setup/`:
> ```bash
> sudo ./fix_docker_cgroupfs.sh      # switches Docker to the cgroupfs driver
> ```
> then recreate your client container. One-time; persists across reboots. (Safe to re-run.)
>
> **2. Run the VPN as an auto-recovering service** (not a manual/GUI connection):
> ```bash
> sudo ./scripts/client_node_setup/setup_vpntunnel.sh -d <YourSite> -n -s
> sudo ./scripts/client_node_setup/vpn_health_monitor.sh --install-timer
> ```
> Verify: `systemctl status mediswarm-vpn mediswarm-vpn-health.timer`.
>
> **3. Run the pre-flight checks before the run:**
> ```bash
> ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training
> ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check
> ```
> Resolve anything flagged and reply with the result. Keep your cache under `/scratch` (**never `/data`**), and ensure `/scratch` is writable and **persists across runs** (auto-resume stores the latest global at `/scratch/mediswarm_latest_global.pt`). Clear any stale `daemon_pid.fl` on restart.
>
> **4. Live-monitor log upload — your site's status and action:**
>
> | Center | Your site | Upload status | What you need to do |
> |---|---|---|---|
> | RUMC | RUMC_1 | ✅ working | Nothing |
> | CAM | CAM_1 | ✅ working | Nothing |
> | MHA | MHA_1 | ✅ working | Nothing |
> | RSH | RSH_1 | ✅ working | Nothing |
> | USZ | USZ_1 | ✅ working | Nothing |
> | UKA | UKA_1 | ⚠️ stopped ~1 week ago | Run the key check below; if it fails, send us your key |
> | UMCU | UMCU_1 | ❌ not since April | Your key is already authorized — just do step 1 below (host-key refresh) and restart your live-sync |
> | VHIO | VHIO_1 | ❌ never set up | Run the key steps below and send us your key; you especially need steps 1 & 2 |
>
> **Live-sync key steps (UKA / UMCU / VHIO only):**
> ```bash
> # 1) Refresh our server's host key (it changed since April), then verify, with the VPN up:
> ssh-keygen -R dl3.tud.de 2>/dev/null
> ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 mediswarm-upload@dl3.tud.de 'echo ok'
> #    prints "ok"                        -> your key already works, nothing to send.
> #    "Permission denied" / 255 / timeout -> do steps 2 and 3:
> # 2) Create an upload key if you don't already have one:
> [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "$(hostname)@mediswarm"
> # 3) Reply to this email with your PUBLIC key line:
> cat ~/.ssh/id_ed25519.pub
> ```
> (Our server's host-key fingerprint, if you want to verify it: ED25519 `SHA256:UrpTNm/Rq/dOPiCV4XVx/Mal5LpQmVS9tbOQJaFBE5E`.)
> We'll authorize your key on our side; then re-run step 1 to confirm `ok`. The upload settings ship inside the kit — you don't edit anything. If step 1 still fails after we authorize your key, it's the VPN path (step 2), not the key.
>
> Please confirm steps 1–4 **by end of day today**. Thank you!
