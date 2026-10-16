# ODELIA consortium runs, October to November 2026: run plan

Operational plan for the consortium runs after the move to the 1.9.3 kit: what each run is
for, the exact job, which sites take part, what has to be in place first, how long it takes and
what to check afterwards. The record of the first October run is
[ODELIA_RUN_2026-10_MST.md](ODELIA_RUN_2026-10_MST.md). Per-site results are
consortium-confidential and are not kept in this repository; each run's results go into
`RUN_REPORT.md` next to its released models on the monitor.

Every run is a named preset of `scripts/admin/odelia_run_plan.py`, which calls
`scripts/admin/build_consortium_job.py` with fixed arguments. The builder validates the job and
writes `job_manifest.json` (code version, flags, hashes of the job code, expected image).
`tests/unit_tests/test_odelia_run_plan.py` builds every preset for the real sites and checks
the pins, and checks that every command below is the one the tool prints.

## How a run is built

```bash
python3 scripts/admin/odelia_run_plan.py --list                      # presets and their settings
python3 scripts/admin/odelia_run_plan.py --show cam                  # the exact builder command
python3 scripts/admin/odelia_run_plan.py --build cam --exclude-sites VHIO_1 \
    --out-dir <admin kit>/local/mediswarm_jobs
# then, in the admin console:
submit_job /fl_admin/local/mediswarm_jobs/odelia_cam_sammed2d_20r_7site
```

- **Sites.** Every preset lists the eight hospitals (CAM_1, MHA_1, RSH_1, RUMC_1, UKA_1,
  UMCU_1, USZ_1, VHIO_1), except the D2.5 regional runs. VHIO_1 joins only once its node can
  read all of its data. Until then build with `--exclude-sites VHIO_1`; the job name then ends
  in `_7site` instead of `_8site`, so the two versions never share a name or a mirror.
- **Quorum.** Strict: every listed site is mandatory and
  `min_clients = configure_min_clients = min_responses_required` = number of sites (F8 in
  [SWARM_FAILURE_MODES.md](SWARM_FAILURE_MODES.md)). These runs are benchmarks: a site that
  fails stops the run, nothing is pruned.
- **Job code.** Built from the release image (`--source image:jefftud/odelia:1.9.3`), so the
  code shipped with the job is the code the sites run. The build host needs the image
  (`docker pull jefftud/odelia:1.9.3`).
- **Rounds.** Each preset names its training rounds; the builder adds one evaluation-only
  round (the final global is evaluated at every site), so `num_rounds` is one more.
- **dl3 test bed.** The same preset builds for the test sites with `--sites TEST_A_1,TEST_B_1`.
  A preset that names a site (starting client, MS6 capture sites) is refused until
  `--starting-client` or `--ms6-capture-sites` names a test site. Add `--source repo` to test
  code that is not in an image yet.

## Before the first run

| What | Where | Needed for |
|---|---|---|
| Kit 1.9.3 (`prod_07`), image `jefftud/odelia:1.9.3` through `:current` | every site | every run |
| Coordinator on the 1.9.3 server kit, archive mounted at `/archive` | coordinator | every run; `cam` |
| Data access fixed at VHIO_1 (in October its node could not open part of its validation images) | VHIO_1 | VHIO_1 in any run |
| Per-case return on: `--env ODELIA_RETURN_PER_CASE=1` in the `MEDISWARM_DOCKER_OPTIONS` line of `startup/image.conf` | RUMC_1 (not yet on); UMCU_1 and MHA_1 (on since October, kept by `--migrate_from`) | D2.5; useful for D3.2 |
| The August model at `/scratch/odelia_87c5bbee_global.pt` (see below) | UMCU_1; MHA_1 | D2.5, D3.2 |
| BreastDivider crops in `data_unilateral_BD/` and `metadata_unilateral_BD/`, checked with the node ([BREASTDIVIDER_SITE_GUIDE.md](BREASTDIVIDER_SITE_GUIDE.md)) | every site | `bd` |

Kit and image check, during the migration and again on the coordinator right before every
submit, for the sites of that run:

```bash
python3 scripts/admin/fleet_status.py --sites <the run's sites> --expect-kit 1.9.3 \
    --expect-image jefftud/odelia:1.9.3 --expect-image-id sha256:CONFIG193
```

One line per site from its newest heartbeat (kit, image, age, verdict and why); every site must
be `READY` (exit 0). `OLD KIT`: not moved to 1.9.3, or the old kit still uploads; `OLD IMAGE`:
the client was started before `:current` moved to 1.9.3 or is pinned, restart it; `STOPPED`,
`STALE`, `NO HEARTBEAT`: no running client to show. `sha256:c527c8c509aa` is 1.9.1's config
digest, which sites report and which the coordinator's own Docker cannot read locally
([RELEASE_RUNBOOK.md](RELEASE_RUNBOOK.md), section 11). `READY` means the client runs the right
code; registration and configuration are checked after submitting (F8 in
[SWARM_FAILURE_MODES.md](SWARM_FAILURE_MODES.md)).

### The start model of D2.5 and D3.2

The August 2026 pan-European model: the final global of job `87c5bbee` (1DivideAndConquer,
eight sites), released as `ODELIA_MODELS/release/2026-08_8site_1DivideAndConquer_87c5bbee/FL_global_model.pt`
on the monitor.

| | |
|---|---|
| File at the starting client | `<scratch dir>/odelia_87c5bbee_global.pt` on the host, `/scratch/odelia_87c5bbee_global.pt` in the container |
| sha256 of the file | `ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d` |
| State digest (format 1) | `cdbfab5748a025f3a88216747ac57208d764a93f9e416e90d9c1063f1ea6b51e` |

Only the starting client reads the file: its persistor checks the sha256 and seeds round 0
from it. Every site then checks the round-0 global it receives against the state digest
(`ODELIA_WARM_START_SHA256`) and stops before any training if it differs, so a missing or
different file at the starting client stops the run instead of training from a new model.
The digest is `state_digest.checkpoint_state_digest()` of the file, computed inside
`jefftud/odelia:1.9.1`; to recompute it, run
`python3 /MediSwarm/application/jobs/_shared/custom/state_digest.py <file>` in that image.

At UMCU_1 (starting client of D2.5 Dutch and of every D3.2 arm) and MHA_1: download the file,
save it as `odelia_87c5bbee_global.pt` in the folder passed to `docker.sh --scratch_dir`, and
check that `sha256sum` prints the value above. The runs write their own mirrors
(`/scratch/mediswarm_latest_global_<job name>.pt`) and never overwrite this file.

## Schedule (proposed; the run board is authoritative)

| When | Run | Preset | Training rounds | Expected duration |
|---|---|---|---|---|
| by Fri 23 Oct | sites on kit and image 1.9.3 | | | |
| Mon 26 Oct | check job | `check` | 1 | about 2 h |
| 27 to 29 Oct | Run A, CAM model | `cam` | 20 | not measured at consortium scale; three days reserved |
| by 30 Oct | start model at UMCU_1 and MHA_1, per-case on at RUMC_1 | | | |
| 2 to 3 Nov | Run B, VHIO model | `vhio` | 20 | about 17 to 21 h |
| 4 to 5 Nov | D2.5 regional fine-tuning | `d25_dutch`; Greek cohort at MHA_1 by local training | 5 | about 6 to 7 h |
| 9 to 11 Nov | D3.2 active learning, four arms | `d32_entropy_b20`, `d32_entropy_b40`, `d32_random_b20`, `d32_random_b40` | 5 each | at most about 13 h per arm |
| 16 to 17 Nov | T3.3 DP utility | `dp_s24` | 20 | about 15 h |
| by 20 Nov | BreastDivider crops at every site | | | |
| 23 to 25 Nov | Run C, BreastDivider crops | `bd` | 20 | about 2 days |

Durations come from the October MST run (rounds of 42 to 52 min, 15 h for 20 rounds on seven
sites), the 2-site test of the VHIO model on dl3 (10 to 40% slower per round than MST), #526
(about 1 h per Dutch round) and #592 (2.2 h per round for 1DivideAndConquer at consortium
scale). The slowest site sets the pace of every round.

## The runs

### 1. Check job (`check`)

**Purpose.** Shows that every node runs the 1.9.3 kit and image, reads its data and
completes a round, including the evaluation-only round, before a multi-day run is started. It
decides whether VHIO_1 takes part.

```bash
python3 scripts/admin/build_consortium_job.py --name odelia_check_mst_1r_8site \
    --base-job ODELIA_ternary_classification --model MST --config unilateral --rounds 1 \
    --sites CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1 \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>
```

**Sites.** All eight. If it fails at VHIO_1 alone, repeat with `--exclude-sites VHIO_1` and
build every later run the same way.

**After.** Besides the checks for every run (below): each site's per-case files appear for the
sites that opted in; RUMC_1 must be among them before D2.5.

### 2. Run A, CAM model (`cam`)

**Purpose.** Trains the CAM model (`MODEL_NAME=MST_SAMMed2D`, weights in the image) on the
consortium. Keeps the per-round models of CAM_1 and RUMC_1 for the MS6 white-hat attacks
([MS6_WHITE_HAT_PROTOCOL.md](MS6_WHITE_HAT_PROTOCOL.md)) and archives every global at TUD.
The export of training-set predictions is off.

```bash
python3 scripts/admin/build_consortium_job.py --name odelia_cam_sammed2d_20r_8site \
    --base-job ODELIA_ternary_classification --model MST_SAMMed2D --config unilateral \
    --rounds 20 --sites CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1 \
    --ms6-capture-sites CAM_1,RUMC_1 --export-train-predictions 0 --archive-globals \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>
```

**Before.** CAM_1 and RUMC_1 need room in their scratch folder for the captures (two models
per round). The coordinator's archive (`/srv/mediswarm/archive` on the host) must be mounted at
`/archive` in the server container, or the archive falls back to the job's run directory.

**After.** CAM_1 and RUMC_1 log `MS6 capture active at <site>: rounds all`; every other site
logs `MS6 capture: <site> is not in ODELIA_MS6_CAPTURE_SITES; nothing is kept here`. `/srv/mediswarm/archive/<job id>/` holds one
`round_<rr>_global.pt` per round, `manifest.json` and `SHA256SUMS`
(`sha256sum -c SHA256SUMS` passes).

### 3. Run B, VHIO model (`vhio`)

**Purpose.** Trains the VHIO model (`MODEL_NAME=MedicalNet`, ResNet-34 weights in the image since
1.9.1) on the consortium.

```bash
python3 scripts/admin/build_consortium_job.py --name odelia_vhio_medicalnet_20r_8site \
    --base-job ODELIA_ternary_classification --model MedicalNet --config unilateral --rounds 20 \
    --sites CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1 \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>
```

**Before.** Every site on 1.9.3: the model refuses to start without its weights, which ship
in the image since 1.9.1.

**After.** Every site logs
`MedicalNet pretrained weights: /MediSwarm/pretrained_weights/resnet_34_23dataset.pth`. A site
on an older image stops before training with an error that names `resnet_34_23dataset.pth`.

### 4. D2.5 regional fine-tuning (`d25_dutch`, `d25_greek`)

**Purpose.** Grant task T2.4, deliverable D2.5 (#526): the August pan-European model
fine-tuned on the Dutch and on the Greek population, compared with the same model on the same
cases with the same evaluation code (`scripts/evaluation/compare_regional_finetune.py`; primary
endpoint malignant vs rest AUROC, paired bootstrap). The comparison needs the sites' per-case
rows of the final global, hence per-case return at RUMC_1, UMCU_1 and MHA_1.

**Rounds: 5 training rounds** (plus the evaluation-only round). #526 sets the epoch budget:
`compute_weighted_epochs` gives RUMC_1 and UMCU_1 one local epoch per round, a Dutch round
takes about 1 h (UMCU_1 is the slowest) and the 5-round run about 5 to 6 h. Five rounds are
five local epochs from the August model at each Dutch site, a run that fits into one day, and
the same fine-tuning length as the D3.2 arms, which start from the same model. (#526 states
one epoch per round for MHA_1 too; by the same formula MHA_1 trains more than one epoch per
round, because its training set is smaller. This matters only once the Greek run has a form
that can run, see below.)

#### Dutch cohort (`d25_dutch`)

```bash
python3 scripts/admin/build_consortium_job.py --name odelia_d25_dutch_5r_2site \
    --base-job challenge_1DivideAndConquer --config unilateral --rounds 5 --sites RUMC_1,UMCU_1 \
    --starting-client UMCU_1 \
    --warm-start source:/scratch/odelia_87c5bbee_global.pt:ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d \
    --warm-start-digest cdbfab5748a025f3a88216747ac57208d764a93f9e416e90d9c1063f1ea6b51e \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>
```

**Sites.** RUMC_1 and UMCU_1, one swarm job; UMCU_1 is the starting client and holds the start
model.

**After.** Both sites log `ODELIA_WARM_START_SHA256: round-0 global verified (... digest
cdbfab5748a0)`. The coordinator's `cross_site_val/per_case_predictions/` holds the test-split
rows of the final global for both sites
([SWARM_EVALUATION_OUTPUTS.md](SWARM_EVALUATION_OUTPUTS.md)).

#### Greek cohort: local fine-tuning at MHA_1 (image 1.9.3)

A swarm job cannot have a single site: NVFlare's swarm server controller
(`ServerSideController.__init__`) raises `Not enough participating_clients: must > 1`, and the
builder refuses the `d25_greek` preset for that reason:

```bash
python3 scripts/admin/build_consortium_job.py --name odelia_d25_greek_5r_1site \
    --base-job challenge_1DivideAndConquer --config unilateral --rounds 5 --sites MHA_1 \
    --starting-client MHA_1 \
    --warm-start source:/scratch/odelia_87c5bbee_global.pt:ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d \
    --warm-start-digest cdbfab5748a025f3a88216747ac57208d764a93f9e416e90d9c1063f1ea6b51e \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>
```

The Greek cohort is one site, so it
needs no swarm: MHA_1 fine-tunes the August model by local training and compares it with the
unchanged model on its own test split, with the same comparison code as the Dutch run. Since
1.9.3 one command does all of it at the site:

```bash
# in MHA_1's 1.9.3 kit, startup folder; the start model in the scratch directory as for d25_dutch
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 \
    --container_name odelia_d25_local --run_script /MediSwarm/scripts/evaluation/d25_local_finetune.sh
```

`--container_name` lets it run beside the site's swarm client; without it docker refuses with
`Conflict. The container name ... is already in use` (F19). Tested on 16 Oct through a real 1.9.3
kit (TUD_2) on the production coordinator's fleet, client running.

`scripts/evaluation/d25_local_finetune.sh` checks the start model's sha256, predicts the test
split with it (`predict.py`), fine-tunes it by local training (`ODELIA_LOCAL_INIT_CHECKPOINT` and
`ODELIA_LOCAL_INIT_SHA256`: the checkpoint must fit the model, as for a received global),
predicts again with the fine-tuned model and runs `compare_regional_finetune.py` on the two
prediction sets, paired by case id. Everything stays under `<scratch dir>/d25_local/<UTC time>/`;
MHA_1 sends `report.json` (AUROCs, bootstrap intervals, support counts, no case identifiers).
Nothing is uploaded: `--run_script` starts no live_sync.

**Epochs: 10** (`D25_EPOCHS`, passed with `--docker_options "--env D25_EPOCHS=..."`): what MHA_1
trains in the 5 rounds of a swarm job, since `compute_weighted_epochs` gives its training set 2
epochs per round. The other settings are `D25_START_MODEL`, `D25_START_SHA256` and `D25_LABEL`
(script header).

**Tested** on the dl3 test bed on 16 Oct with image 1.9.2-rc1 and a 1.9.0 kit, which have the
same code and docker.sh as 1.9.3 (MHA_1 challenge data, 2 epochs): start model verified, `Local training starts from /scratch/odelia_87c5bbee_global.pt
(sha256 ad26080c7116..., 450 tensors)`, 26 paired test cases, `report.json` written.

**After.** MHA_1's console ends with `[OK] done. Please send .../report.json`; the run directory
of the fine-tuning holds `local_init.json` (checkpoint and sha256).

### 5. D3.2 active learning (`d32_entropy_b20`, `d32_entropy_b40`, `d32_random_b20`, `d32_random_b40`)

**Purpose.** Deliverable D3.2 (#592): is a model retrained on actively selected cases better
than one retrained on randomly selected cases? Two strategies (entropy, random) times two
budgets (20 and 40 cases per site), each fine-tuned for 5 rounds from the August model. In
round 0 every site scores its training split with the received global (the August model),
selects its cases and trains only on them; the selection stays at the site
(`<scratch dir>/odelia_al/<job id>/<site>/`). The random arms use seed 0.

```bash
python3 scripts/admin/build_consortium_job.py --name odelia_d32_entropy_b20_5r_8site \
    --base-job challenge_1DivideAndConquer --config unilateral --rounds 5 \
    --sites CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1 --starting-client UMCU_1 \
    --warm-start source:/scratch/odelia_87c5bbee_global.pt:ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d \
    --warm-start-digest cdbfab5748a025f3a88216747ac57208d764a93f9e416e90d9c1063f1ea6b51e \
    --al-strategy entropy --al-budget 20 --al-pool-split train --al-seed 0 \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>

python3 scripts/admin/build_consortium_job.py --name odelia_d32_entropy_b40_5r_8site \
    --base-job challenge_1DivideAndConquer --config unilateral --rounds 5 \
    --sites CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1 --starting-client UMCU_1 \
    --warm-start source:/scratch/odelia_87c5bbee_global.pt:ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d \
    --warm-start-digest cdbfab5748a025f3a88216747ac57208d764a93f9e416e90d9c1063f1ea6b51e \
    --al-strategy entropy --al-budget 40 --al-pool-split train --al-seed 0 \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>

python3 scripts/admin/build_consortium_job.py --name odelia_d32_random_b20_5r_8site \
    --base-job challenge_1DivideAndConquer --config unilateral --rounds 5 \
    --sites CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1 --starting-client UMCU_1 \
    --warm-start source:/scratch/odelia_87c5bbee_global.pt:ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d \
    --warm-start-digest cdbfab5748a025f3a88216747ac57208d764a93f9e416e90d9c1063f1ea6b51e \
    --al-strategy random --al-budget 20 --al-pool-split train --al-seed 0 \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>

python3 scripts/admin/build_consortium_job.py --name odelia_d32_random_b40_5r_8site \
    --base-job challenge_1DivideAndConquer --config unilateral --rounds 5 \
    --sites CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1 --starting-client UMCU_1 \
    --warm-start source:/scratch/odelia_87c5bbee_global.pt:ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d \
    --warm-start-digest cdbfab5748a025f3a88216747ac57208d764a93f9e416e90d9c1063f1ea6b51e \
    --al-strategy random --al-budget 40 --al-pool-split train --al-seed 0 \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>
```

**Sites.** All eight; UMCU_1 is the starting client of every arm and holds the start model (the
same file as for D2.5).

**Duration.** #592 budgets 2.2 h per round, about 11 h per arm, plus the evaluation-only round.
That is an upper bound: after the selection each site trains on 20 or 40 cases, so rounds 1 to 4
are much shorter; round 0 adds one scoring pass over each site's training split. Not yet
measured at consortium scale.

**After.** Every site logs the round-0 digest check (as in D2.5) and one line
`al_subset: round 0 selection <strategy> (budget <n>, seed 0, pool train): <n> of ... cases`;
`cross_val_results.json` carries the `al/*` metrics of every site.

### 6. T3.3 DP utility (`dp_s24`)

**Purpose.** Prices the utility cost of client-level differential privacy (#593): MST with
noise multiplier 2.4 and clip norm 1.0, epsilon at most 10 at delta 1e-5 after the 20 training
rounds (the builder computes 9.84 and refuses more rounds than the budget allows; the
evaluation-only round spends nothing). Per-case return is forced off.
[DIFFERENTIAL_PRIVACY.md](DIFFERENTIAL_PRIVACY.md) describes the mechanism and the guarantee.

```bash
python3 scripts/admin/build_consortium_job.py --name odelia_dp_s24_mst_20r_8site \
    --base-job ODELIA_ternary_classification --model MST --config unilateral --rounds 20 \
    --sites CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1 --dp-sigma 2.4 --dp-clip-norm 1.0 \
    --dp-delta 1e-5 --dp-target-epsilon 10 --source image:jefftud/odelia:1.9.3 \
    --out-dir <out-dir>
```

**After.** Every site streams `dp/epsilon` each round and logs
`DP summary: 20 release(s), epsilon 9.84...` at the end; no per-case files arrive. The
utility comparison is against the October MST run (same model, same sites, no noise).

### 7. Run C, BreastDivider crops (`bd`)

**Purpose.** Trains 1DivideAndConquer on the crops made with MEVIS' BreastDivider
(`CONFIG=unilateral_BD`) instead of the current crops. Mixing the two formats costs about 0.06
AUROC without any error ([PREPROCESSING_BENCHMARK_BREASTDIVIDER.md](PREPROCESSING_BENCHMARK_BREASTDIVIDER.md)),
so the job sets `ODELIA_STRICT_DATA_FORMAT=1`: a site whose crops do not look like
BreastDivider output stops at setup instead of training on the wrong format
([BREASTDIVIDER_SITE_GUIDE.md](BREASTDIVIDER_SITE_GUIDE.md), "For coordinators").

```bash
python3 scripts/admin/build_consortium_job.py --name odelia_bd_1dac_20r_8site \
    --base-job challenge_1DivideAndConquer --config unilateral_BD --rounds 20 \
    --sites CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1 --env ODELIA_STRICT_DATA_FORMAT=1 \
    --source image:jefftud/odelia:1.9.3 --out-dir <out-dir>
```

**Before.** Every site has made its crops, sent the output of `reconcile_odelia_dataset.py
--config unilateral_BD` and passed the preflight check with `--config unilateral_BD`. Sites
start their clients as usual, without `--config`: the job sets it.

**After.** Every site logs `ODELIA data for <site>: CONFIG=unilateral_BD, images
/data/<site>/data_unilateral_BD ...` and `ODELIA crop format for <site>: ok`. A site that
stops with a crop-format error fixes its crops (troubleshooting table in the site guide) and
the job is submitted again.

## After every run

Follow the run from the coordinator's server log:

```bash
python3 scripts/admin/watch_run.py <coordinator kit>/startup/nohup.out --job-id <job id> \
    --follow --interval 300 --finish-grace 60 --stall 180
```

It reports each round (start, end, sites finished, slowest site) and ends with exit status 0
(finished, or running with recent progress), 1 (aborted or ended before its last round),
3 (all rounds done but not finished within an hour, #542) or 4 (no progress for three hours).
Then:

- the job ends `FINISHED:COMPLETED` with `num_rounds` rounds at every listed site;
- `download_job <job id>` in the admin console; `cross_site_val/cross_val_results.json` has
  the metrics of every site, including the final global's (`final_global`, evaluation-only
  round), and the per-case files of the sites that return them
  ([SWARM_EVALUATION_OUTPUTS.md](SWARM_EVALUATION_OUTPUTS.md));
- the job folder's `job_manifest.json` goes with the released models, so the run can be traced
  to the exact job code and flags;
- an aborted run is analysed before anything is resubmitted
  ([SWARM_FAILURE_MODES.md](SWARM_FAILURE_MODES.md)).
