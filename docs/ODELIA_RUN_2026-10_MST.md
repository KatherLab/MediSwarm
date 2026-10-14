# ODELIA consortium run, October 2026: MST on seven sites

Operational record of the first consortium benchmark run of the October schedule (#591).
Per-site results are consortium-confidential and are kept in `RUN_REPORT.md` next to the
released models on the monitor (`ODELIA_MODELS/release/2026-10_7site_MST_7a678d9a/`).

## Run

| | |
|---|---|
| Job | `7a678d9a-5fec-473f-9b23-e43c41322f1f` (`odelia_mst_20r_7site`) |
| Status | `FINISHED:COMPLETED`, 20 of 20 rounds at all 7 sites |
| Time | 13 Oct 2026, 08:32 to 23:36 UTC, 15 h 04 min |
| Sites | CAM_1, MHA_1, RSH_1, RUMC_1, UKA_1, UMCU_1, USZ_1. VHIO_1 excluded (its node cannot read part of its data) |
| Model | `MODEL_NAME=MST`, pinned in the launcher and in the persistor (#541) |
| Data | unilateral crops of the current preprocessing at every site |
| Image | `jefftud/odelia:1.8.1` (UKA_1 on `1.8.1-cam.261009`, same code plus the CAM weights) |
| Job code | `application/jobs/_shared/custom` of `main` at f3dfd0f, shipped with the job; includes the absent-class AUROC fix (#626) |
| Quorum | `min_clients = configure_min_clients = 7`, all sites mandatory (F8) |
| Timeouts | 24 h task and transfer timeouts, 900 s heartbeat |
| Per-case return | MHA_1 and UMCU_1 (D2.5); RUMC_1 not switched on |

## How the job was built

```bash
J=<admin kit>/local/mediswarm_jobs/odelia_mst_20r_7site
cp -rL application/jobs/ODELIA_ternary_classification "$J"   # custom is a symlink to _shared/custom
rm -rf "${J:?}/app/tests" "${J:?}/app/scripts"
python3 scripts/admin/patch_warm_start_job.py --job-dir "$J" --mode fresh --num-rounds 20 \
    --strict-clients CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1
# pin MODEL_NAME=MST in config_fed_client.conf: launcher script line AND persistor model_name
# meta.conf: deploy_map app = ["server", <the seven sites>]   (without "server" the submit is rejected)
```

## Speed

Rounds took 42 to 52 minutes. UKA_1 (about half of the consortium's training data) finished
last in 18 of 20 rounds with a median of 41 min of training per round; UMCU_1 is next at 31 min;
every other site trains in 6 to 9 min and then waits. A 20-round MST run on this site set takes
about 15 hours.

## Attempts before the run

| Job | Outcome | Cause | Action |
|---|---|---|---|
| `d67132d0` (8 sites, 1-round check) | failed | VHIO_1 could not open 9 of 48 validation images (`OSError`) | VHIO_1 excluded |
| `37e5cfbc` (7 sites, check) | failed | same VHIO_1 file access error | as above |
| `c8f004a7` (7 sites, MST) | failed at start | USZ_1: `CUDA driver initialization failed` inside the running client container | freshly created client container; probable cause F2 (cgroup driver) |
| `7a678d9a` (7 sites, MST) | completed | | |

Node work during preparation: stale `daemon_pid.fl` after reboots at UKA_1 and on the
coordinator (F3, kit fix #630); an unattended NVIDIA driver upgrade at UKA_1 left new
containers without GPU until the host was rebooted (library newer than the loaded kernel
module; hold the driver package with `apt-mark hold`).

## Observations

- The final-round global model scores at least as well as the best-validation one on the
  external challenge test split; release both, recommend the final one.
- The per-site metrics returned to the coordinator (`cross_val_results.json`, labelled
  `swarm_global`) are NOT the aggregated global model: the collector keeps the last streamed
  `val/*` values, which come from each site's local model after its last local epoch of the
  final round. The aggregated final global was never evaluated at the sites; only the central
  external test evaluates the global models. Fixed in 1.9.0 (evaluation-only final round,
  separate labels for local and global metrics).
- At four sites the final local model's top-1 prediction on the validation split is "no lesion"
  for every case, so per-site accuracy equals the no-lesion share. Read AUROC there, not accuracy.
- RUMC_1's clock runs about 1 h 56 min ahead of the coordinator, which offsets its timestamps
  in the server log.
- USZ_1 still uploads no logs to the monitor (#615).
- Per-round global models are not kept (#612); the MS6 attacks need them from a later run.

## External test

`workspace/eval_7a678d9a/run_eval.sh` (not in git) runs `scripts/evaluation/predict.py`
inside `jefftud/odelia:1.8.1` on CPU (`MEDISWARM_ALLOW_CPU_MODEL=1`) for both checkpoints and
the five challenge test sites, then pools the per-site prediction CSVs: one-vs-rest AUROC per
class, macro AUROC, lesion vs none and malignant vs rest, pooled and per site. Check pooled
accuracy against the per-class numbers before trusting a result
([EVALUATION_PITFALLS.md](EVALUATION_PITFALLS.md)).
