# Duke IID 2-Client DL Validation

This runbook validates swarm learning on DL servers with `dl3` as the
server/admin host and `dl0`/`dl2` as Duke IID clients.

## Topology

| Role | Machine | Site | Data |
|------|---------|------|------|
| Server + admin | `dl3` | `dl3.tud.de` | none |
| Client | `dl0` | `node_A` | `/mnt/dlhd0/DUKE_iid` |
| Client | `dl2` | `node_B` | `/mnt/sda1/DUKE_iid` |
| Eval host | `dl0` | `test`, ODELIA institutions | Duke + `/mnt/dlhd0/medswarmdata` |

## Run

Run from `dl3`:

```bash
cp deploy_sites_duke_iid_2client.local.conf.example deploy_sites_duke_iid_2client.local.conf
export DL_SWARM_PASS='<password>'

scripts/deploy/run_duke_iid_2client_validation.sh \
  --conf deploy_sites_duke_iid_2client.local.conf \
  --project application/provision/project_duke_iid_2client_dl.yml \
  --job challenge_1DivideAndConquer \
  --model 1DivideAndConquer \
  --smoke-rounds 2 \
  --full-rounds 20 \
  --continue-rounds 2 \
  --drop-rounds 3 \
  --resume-rounds 1
```

The build step uses `git archive HEAD` and refuses tracked local changes. Commit
or stash tracked edits before running the full validation, or use `--skip-build`
after building the image and startup kits separately.

## Phases

- `all_model_smoke`: trains all six ODELIA models for two rounds with
  `min_clients=2`, `configure_min_clients=2`, and `min_responses_required=2`,
  then runs dl0 prediction smoke evaluation.
- `one_dc_full`: trains `1DivideAndConquer` for the configured full-round count,
  verifies byte-identical final globals on `node_A` and `node_B`, and evaluates
  on dl0.
- `negative_continue`: verifies strict continue fails when
  `/scratch/mediswarm_latest_global.pt` is absent.
- `fresh_then_continue`: runs a fresh phase, restarts clients, then verifies a
  strict continue loads the mirrored global checkpoint.
- `single_client_drop`: starts with both clients configured, kills `node_B`
  after both clients mirror a first global, and requires the run to finish with
  fault-tolerance log evidence.
- `abort_recovery`: aborts all NVFlare containers after the first mirrored
  global, restarts server/clients, and verifies strict continue resumes.

Use `--phases` with a comma-separated subset for targeted runs, for example:

```bash
scripts/deploy/run_duke_iid_2client_validation.sh \
  --conf deploy_sites_duke_iid_2client.local.conf \
  --phases negative_continue,fresh_then_continue
```

Results are written to
`workspace/duke_iid_2client_validation/<run_id>/RUN.md` and `summary.json`.
