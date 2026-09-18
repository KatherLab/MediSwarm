# Duke IID Startup Kit Full Deploy Smoke Test

This smoke test validates startup kits and job wiring before partner
distribution. It is not a model-selection run: all jobs are built with
`num_rounds=2`, and prediction metrics are recorded only to prove inference can
load each final global checkpoint.

## Topology

| Role | Machine | Site | Data |
|------|---------|------|------|
| Server + admin | Cosmos | `dl3.tud.de` server/admin kits | no training data |
| Client | `dl0` | `node_A` | `/mnt/dlhd0/DUKE_iid` |
| Client | `dl2` | `node_B` | `/mnt/sda1/DUKE_iid` |
| Client | `dl3` | `node_C` | `/mnt/swarm_alpha/DUKE_iid` |
| Eval host | `dl0` | `test`, ODELIA institutions | Duke + `/mnt/dlhd0/medswarmdata` |

## One-command Run

Run from a clean `main` checkout:

```bash
scripts/deploy/run_duke_iid_startupkit_smoke.sh \
  --conf deploy_sites_duke_iid.conf \
  --num-rounds 2
```

The wrapper performs:

1. preflight checks for branch, local Docker, remote SSH/Docker/GPU/data paths;
2. Docker image + startup-kit build with `--num-rounds 2`;
3. `docker push jefftud/odelia:<version>`;
4. distributed startup-kit deployment and all six admin-submitted jobs via
   `run_deploy_test.sh --skip-eval`;
5. `dl0` prediction smoke evaluation on Duke test and ODELIA institutions;
6. Markdown + JSON reports under
   `workspace/deploy_test_results/startupkit_smoke_<timestamp>/`.

The underlying build script uses `git archive HEAD` and refuses tracked local
changes, so commit or stash tracked edits before running the full wrapper.

## Manual Steps

If you need to split the run:

```bash
bash scripts/build/buildDockerImageAndStartupKits.sh \
  -p application/provision/project_duke_iid_3site.yml \
  --num-rounds 2

docker push jefftud/odelia:$(scripts/build/getVersionNumber.sh)

RUN_DIR=workspace/deploy_test_results/startupkit_smoke_$(date -u +%Y%m%dT%H%M%SZ)

scripts/deploy/run_deploy_test.sh \
  --all \
  --conf deploy_sites_duke_iid.conf \
  --skip-eval \
  --results-dir "$RUN_DIR/training"

scripts/deploy/run_startupkit_smoke_eval_dl0.sh \
  --conf deploy_sites_duke_iid.conf \
  --checkpoint-root "$RUN_DIR/training" \
  --output-dir "$RUN_DIR/eval" \
  --image jefftud/odelia:$(scripts/build/getVersionNumber.sh)
```

## Pass Criteria

- Server, admin, and `node_A/node_B/node_C` startup kits are generated and used.
- All six jobs are submitted via admin `submit_job`.
- All three clients register for every job.
- Every job completes two rounds without fatal NVFlare errors.
- Every model has exactly three `FL_global_model.pt` files.
- The three final global checkpoints for each model have identical md5 hashes.
- `predict.py` on `dl0` produces prediction outputs for Duke test plus
  `CAM`, `MHA`, `RSH`, `RUMC`, `UKA`, and `UMCU`.

The final report is `RUN.md`; training details are in `training/summary.json`,
and prediction smoke details are in `eval/summary.json` plus `eval/REPORT.md`.
