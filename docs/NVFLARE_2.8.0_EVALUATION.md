# NVFlare 2.8.0 — Upgrade Evaluation

**Date:** 2026-07-06 · **Author:** platform · **Status:** evaluation only (no code/kit changes)

> **Update 2026-07-07 — the stall-relevance premise below is now disproven.** A
> controlled clean-infra diagnostic (server on dl3 + 2 dl clients over Tailscale,
> 1DC) closed round 0 end-to-end — register → p2p scatter → multi-epoch training →
> **p2p gather ("Contribution from node_A ACCEPTED")** → round 0→1. So the partner
> "0 rounds completed" is **environmental** (VPN drops mid-round on ~30 min
> multi-epoch rounds + site exec/data errors), **not** a platform/streaming bug.
> **Net effect on this evaluation: 2.8.0 is a nice-to-have (security + streaming
> robustness), NOT a fix for the stall.** Drop the "pull-forward if the canary
> fixes the stall" trigger; the remaining rationale (clean rebase, security fixes)
> still holds, and the upgrade stays deferred.

## Verdict

**Upgrade is worthwhile and low-risk on the core, but NOT this week.** Keep the
production fork on **`MediSwarm-2.7.2`** for the current run/rollout. Schedule the
rebase-and-rebuild **after we land one clean 2.7.2 run** — *unless* a security fix
below is judged urgent, in which case pull it forward.

The strongest reason to do it soon (not just housekeeping): **2.8.0 fixes cluster in
exactly the layers implicated in our round-0 "aggregator can't close the round"
p2p-gather-stall** — large-model streaming and peer-model download reliability.

## Current state

- Fork: `KatherLab/NVFlare_MediSwarm`, branch `MediSwarm-2.7.2`, as a git **submodule**
  at `docker_config/NVFlare`, pip-installed from local source into the image
  (`docker_config/Dockerfile_ODELIA:265`, `ARG NVF_VERSION=2.7.2`; same in `Dockerfile_STAMP`).
- Our divergence from upstream `2.7.2`: **~10 functional commits / 17 files**, clustered in
  the fault-tolerant swarm controller (`ccwf/`), the subprocess launcher, and the
  kit-generation template (`master_template.yml`).
- Upstream delta `2.7.2 → 2.8.0`: **426 commits**.

## Why not this week

1. **Fixes none of our current failures.** All 5 blocked sites are site-side
   (USZ/CAM crashes, UKA VPN hang, RUMC stale cert, VHIO not onboarded) — orthogonal
   to the NVFlare version.
2. **Invalidates every startup kit.** New image → regenerated kits → **all 8 sites must
   redeploy** — the exact sites we're already struggling to bring online. It resets the
   rollout mid-incident.
3. **Re-validation risk at the worst time**, right before the Thu Jul 9 go/no-go.

## Relevant 2.8.0 changes

### Stall-relevant (streaming / peer-model download / swarm) — the real upside
| Area | Commit(s) | Why it matters to us |
|---|---|---|
| Large-model streaming reliability | `972a1318d` (#4714), pass-through zero-copy `a365ebb71` (#4210/#4289) | Our model is ~90M params (~689 MB); the aggregator streams/downloads peer models — our stall symptom |
| Incomplete / corrupt download guards | `f74c106f8` (#4725), `961e540d9` finished-download ref retry (#4708) | "Aggregator never completes the gather" = a peer-model download that doesn't finish/ retries wrong |
| Aborted-job download race | `9ce76b091` FLARE-2952 (#4607), `2d34a09fd` in-flight tensor cleanup (#4501) | Race conditions in the fetch path |
| Swarm controller / config | `a1ad9904e` min_clients default + `start_task_timeout` (#4568), `0f8cc4c85` swarm+tensor-streaming (#4141/#4146), `621f4984a` (#4024) | Same subsystem our FT patch lives in |
| `START_TASK_TIMEOUT` default | `eb84d0e49` (#4567) — flags `common.py:81 START_TASK_TIMEOUT = 10` (10 s) as too short | Worth checking against our slow-VPN reality even before upgrading |
| Aggregation robustness | `a0ad9b625` skip unsupported metrics (#4223) | Adjacent to the `key_metric`/model-selection issue tracked separately |

**These are candidates to test on a 2-node canary specifically against the stall** — it is
plausible (not proven) that 2.8.0 closes rounds where 2.7.2 hangs.

### Security fixes since 2.7.2 (not on the emergency path, but real)
- `a0a104ceb` load classes from allow-list only (#4701) — FOBS hardening
- `a42125829` bind auth tokens to runtime origins (#4605); `b4fb37c4f` cellnet bye auth (#4569)
- `311977144` **CVE-2026-27903** minimatch bump (#4401, analytics-dashboard — off prod path)
- `a47b8ed1f` critical dependency bump (#4389); `eb6e08d27` distributed provisioning + runtime security (#4380)

None appear to force an emergency upgrade, but for a production medical-FL system the
allow-list class loading and auth-token binding are worth adopting in the normal cycle.

## Conflict surface / effort (our fork onto 2.8.0)

| File(s) | Upstream churn 2.7.2→2.8.0 | Our change | Risk |
|---|---|---|---|
| `ccwf/server_ctl.py`, `common.py`, `swarm_server_ctl.py` (**the FT patch, commit `29866b360`**) | **untouched upstream** | +36 lines | **LOW — clean apply** ✅ |
| `subprocess_launcher.py` | +9 | +17 | LOW–MED |
| `setup.py` (package rename) | +12 | +32 | LOW |
| `master_template.yml` (kit/Docker startup) | **+506 (heavily reworked)** | +71 | **HIGH — main cost; re-port carefully** |
| `flare_agent.py`, `pipe_handler.py`, timeouts | small | small | LOW |

**Overall effort: Medium**, dominated by re-porting the `master_template.yml` kit
customization (the FT controller — the scary part — carries over cleanly). Verify
`#4568` vs our `29866b360` don't disagree on `min_clients` semantics (different files,
so at most a behavioral check, not a merge conflict).

## Recommended path

1. **Now:** this evaluation. No image/kit/submodule-pin changes. *(Optional cheap win
   independent of the upgrade: review `common.py` `START_TASK_TIMEOUT=10` against our
   slow-VPN sites.)*
2. **After one clean 2.7.2 run** (gate): execute the deferred upgrade —
   - **P1** branch `MediSwarm-2.8.0` off upstream `2.8.0`; replay the ~10 commits; resolve
     conflicts (mainly `master_template.yml`); run fork unit tests
     (`test_server_ctl_min_clients.py`, `test_download_complete_gating.py`).
     **Rotate the plaintext PAT** embedded in the submodule remote while there.
   - **P2** bump `NVF_VERSION` (both Dockerfiles) + `odelia_image.version` + docs
     (`README.md`, `docs/MEDISWARM_COMPATIBILITY_GUIDE.md`, `docs/AGGREGATION_STRATEGIES.md`,
     `CHANGELOG.md`); rebuild image; **2-node RSH/MHA canary** — clean swarm cycle + FT
     single-drop + warm-continue, **and explicitly retest the p2p-gather-stall**.
   - **P3** regenerate all kits; staged all-sites redeploy (v1.5.0 runbook); re-validate at scale.
3. **Pull-forward trigger:** if the P2 canary shows 2.8.0 resolves the gather-stall, or a
   security fix is deemed urgent, promote the upgrade ahead of further 2.7.2 runs.
