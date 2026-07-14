# NVFlare fork — canonical MediSwarm patch set

MediSwarm vendors NVFlare as a git submodule at `docker_config/NVFlare`
(`KatherLab/NVFlare_MediSwarm`, branch `MediSwarm-2.7.2`), built from local source
into the image (`docker_config/Dockerfile_ODELIA`, `ARG NVF_VERSION=2.7.2`). This
file documents **what MediSwarm changes on top of upstream NVFlare** so a rebase
onto a new upstream (e.g. 2.8.0, see #392) replays a known, curated list instead of
re-discovering it. Tracked by #399.

## Canonical functional patches (keep, replay on rebase)

| Area | Upstream file(s) | What / why |
|---|---|---|
| **FT swarm controller** | `nvflare/app_common/ccwf/server_ctl.py`, `common.py`, `swarm_server_ctl.py` | Prune-and-continue fault tolerance + separate `configure_min_clients` (config-phase quorum) from `min_clients` (runtime tolerance). Fixes the CCWF config-phase start-client crash. **Upstream did not touch these 3 files 2.7.2→2.8.0 → applies clean.** |
| **SubprocessLauncher env prefix** | `nvflare/app_common/launchers/subprocess_launcher.py` | Support `KEY=VALUE` env-var prefixes in the launched script command. |
| **Kit / Docker startup** | `nvflare/lighter/templates/master_template.yml` (overridden by `docker_config/master_template.yml` at build, `Dockerfile_ODELIA:267`) | MediSwarm `docker.sh` startup scripts; `--restart=unless-stopped` (#393). **Heavily reworked upstream (+506 lines 2.7.2→2.8.0) — the main rebase cost.** |
| **Package identity** | `setup.py` | MediSwarm package name + version. |
| **Slow-VPN timeouts** | (timeout defaults) | 100 h (360000 s) task/ack timeouts for slow cross-site VPNs. |
| **Code-integrity check** | (startup) | Verify code integrity before starting training. |
| **Dashboard zip path** | `nvflare/dashboard/application/blob.py` | Fix the dashboard zip path + README header. |

Representative commits (as of `MediSwarm-2.7.2`): `29866b360` (FT/configure-quorum),
`f2b979db4` (env prefix), `61d151a4b` (master_template), `410365c59` (package),
`d81d51d0d` (timeouts), `c24852640` (code integrity), `de53becfb` (dashboard).

## Churn to drop when curating the rebase branch

These add no net functionality and only make the diff noisier — squash out / fold in:

- **Diagnostic add→remove pair:** `a5faa1710` *Add diagnostic timing instrumentation to
  FLCallback* → `a98bf7156` *use print()* → `33fb7dd60` *Remove diagnostic instrumentation*.
  Net zero; drop all three.
- **Standalone formatting commits:** `79854f343`, `72dddb27c` (`ci: black-format …`) —
  fold into their functional parents rather than keeping as separate commits.

## Open review item

- `nvflare/app_common/ccwf/common.py` **`START_TASK_TIMEOUT = 10`** (10 s). Upstream 2.8.0
  flagged this default as too short (#4567). Against our slow-VPN sites, a 10 s
  start-task window is a candidate contributor to the p2p round-start behavior — review
  and raise if implicated. (Changing it requires an image rebuild; validate on the 2-node
  canary.)

## Upstreaming candidates (shrink long-term divergence)

- The **FT swarm controller** (`ccwf/server_ctl.py` prune-and-continue + `configure_min_clients`)
  and the **SubprocessLauncher env-prefix** are generally useful — file upstream PRs/issues
  to NVIDIA/NVFlare so they leave our fork.

## How this de-risks the 2.8.0 rebase (#392)

The 2.8.0 evaluation (`docs/NVFLARE_2.8.0_EVALUATION.md`) found the FT controller files
untouched upstream (clean apply) and `master_template.yml` heavily reworked (the real
conflict). Replaying **this** curated list — minus the churn above — is #392's P1.
