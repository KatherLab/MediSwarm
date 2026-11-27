# Which crops a job reads: current crops and BreastDivider crops

For coordinators. MediSwarm 1.9.3, November 2026.

Since November every ODELIA site holds two crop sets of the same breasts: the **current crops**
(standard ODELIA unilateral crops, used by every run so far) and the **BreastDivider crops** made with
MEVIS' odelia-preprocessing package for Run C. Both have the same tensor shape (256 x 256 x 32
voxels), so a node that reads the wrong set raises no error; the model just gets worse (about 0.06
AUROC when the formats are mixed, [PREPROCESSING_BENCHMARK_BREASTDIVIDER.md](PREPROCESSING_BENCHMARK_BREASTDIVIDER.md)).
This page explains how a job chooses between them, what the coordinator controls, what it does
not, and what to check before a run on either set.

In short:

- **The job decides, the sites do nothing.** Every job built with `odelia_run_plan.py` or
  `build_consortium_job.py` pins `CONFIG`. The `bd` preset reads the BreastDivider crops; every
  other preset reads the current crops. Sites start their client once, without `--config`, and
  never restart it to switch.
- **A site's own `--config` cannot override a built job.** Verified in the 1.9.3 image (below).
- **Not covered:** job folders not made by the builder, runs started at the site (preflight,
  local training, `--run_script`), and folder overrides set in a site's `image.conf`.

## The two folder sets at a site

| Folder (inside `/data/SITE/`) | Content | Read when |
|---|---|---|
| `data_unilateral/`, `metadata_unilateral/` | current crops and labels | `CONFIG=unilateral` (default) |
| `data_unilateral_BD/`, `metadata_unilateral_BD/` | BreastDivider crops and their labels | `CONFIG=unilateral_BD` (Run C) |
| `data/`, `metadata/` | bilateral input | `CONFIG=original` |

The map is `CONFIG_FOLDERS` in `application/jobs/_shared/custom/data_layout.py`. Under
`unilateral_BD` a missing `metadata_unilateral_BD/` falls back to `metadata_unilateral/` with a
warning; an explicit `ODELIA_METADATA_FOLDER` never falls back. Any other `CONFIG` value stops the
node with the list of valid values.

**What the sites were told.** Make the BreastDivider crops in new folders next to the current
ones; nothing is replaced; never run the bundle's optional `unilateralize.py` in the site folder
(it writes into `data_unilateral/`); never copy between `data_unilateral/` and
`data_unilateral_BD/` (handbook sections 1.4 and 7, [BREASTDIVIDER_SITE_GUIDE.md](BREASTDIVIDER_SITE_GUIDE.md),
run board step 12). No backup copy was asked for, and the handbook ties "keep the current crops"
to the October and November runs. Runs on the current crops now continue into December, so the
consortium mail of 26 Nov asks sites to keep `data_unilateral/` and `metadata_unilateral/`
unchanged until further notice, also after Run C. Some sites can no longer remake their current
crops (for example where source series are missing), so those folders are the only copy there.

The BreastDivider steps themselves cannot touch `data_unilateral/`: `BD_run_seg.py` writes
`BD_Output/` and `BD_Input_seq/`, `BD_main.py` writes `data_unilateral_BD/` (0.2.2 bundle), and the
node mounts the data read-only (`-v ...:/data/:ro` in `docker.sh`).

## How the folder is chosen

1. **The container.** `docker.sh` starts every client container with
   `--env CONFIG=${CLI_CONFIG:-${CONFIG:-unilateral}}` (`docker_config/master_template.yml`), so the
   container always has a `CONFIG`: `unilateral`, unless the site passed `--config` or set
   `CONFIG` in its environment.
2. **The job.** `build_consortium_job.py` writes `CONFIG=<--config>` as a leading `KEY=VALUE`
   token of the launcher line and `data_config` into the persistor, and refuses to build if the
   two differ ("CONFIG pins agree"). `--env CONFIG=...` is refused: `CONFIG` belongs to
   `--config` (`MANAGED_ENV`), so a job can never carry two answers.
3. **The training process.** NVFlare's `SubprocessLauncher`
   (`nvflare/app_common/launchers/subprocess_launcher.py`, `_start_external_process`) copies the
   container's environment and then overwrites it with the launcher line's leading `KEY=VALUE`
   tokens. The job's `CONFIG` therefore wins over the container's. On 26 Nov 2026 a probe inside
   `jefftud/odelia:1.9.3` (the image every site runs) confirmed it: container `unilateral_BD`,
   launcher `CONFIG=unilateral` gives `unilateral`; without the pin it gives `unilateral_BD`.
4. **The data loader.** `env_config.py` reads `CONFIG` (default `unilateral`), `data_layout.py`
   resolves the folders, and every setup logs them:

   ```
   ODELIA data for RSH_1: CONFIG=unilateral, images /data/RSH_1/data_unilateral (N exam folders), metadata /data/RSH_1/metadata_unilateral, image file Sub_1.nii.gz
   ODELIA crop format for RSH_1: ok; no processing.log; spacing of 20 sampled crops: all 0.70 x 0.70 x 3.00 mm
   ```

   Each run directory also gets `data_layout.json` (folders, file, fallback, format verdict; no UIDs).

## Switching from the admin side

```bash
python3 scripts/admin/odelia_run_plan.py --list                     # CONFIG of every preset
python3 scripts/admin/odelia_run_plan.py --build bd --out-dir <admin kit>/local/mediswarm_jobs      # BreastDivider crops
python3 scripts/admin/odelia_run_plan.py --build v2_s3 --out-dir <admin kit>/local/mediswarm_jobs   # current crops
# then in the admin console: submit_job /fl_admin/local/mediswarm_jobs/<job name>
```

| Preset | `CONFIG` | Strict format check |
|---|---|---|
| `bd` (Run C) | `unilateral_BD` | yes (`ODELIA_STRICT_DATA_FORMAT=1`) |
| every other preset (`check`, `cam`, `vhio`, `v2_s1`...`v2_s4`, `d25_dutch`, `d32_*`, `dp_s24`) | `unilateral` | no |

For a one-off job use `build_consortium_job.py --config unilateral_BD` (or `unilateral`). The
choice is **one value per job, for all sites**: the job deploys a single app. It cannot be changed
at submit time; switching means building another job. A run that mixes the two sets would mix
formats in one model, which is exactly what the strict check exists to prevent.

## What the switch does not cover

- **Job folders not made by the builder.** Only builder-built folders (they contain
  `job_manifest.json`) pin `CONFIG`. Older hand-built folders and the raw job templates in
  `application/jobs/` pin nothing, so their training process reads whatever `CONFIG` the site's
  container has. Submit only builder-built folders, and keep older folders out of
  `local/mediswarm_jobs/` (for example in an `old/` subfolder) so they cannot be submitted by mistake.
- **Runs started at the site.** `--preflight_check`, `--local_training`, `--external_validation`
  and `--run_script` (for example `d25_local_finetune.sh`) use the site's own `--config` (or the
  default `unilateral`). The handbook's BreastDivider checks pass `--config unilateral_BD` on
  purpose; a site must not start its swarm client with it.
- **Folder overrides at the site.** `ODELIA_DATA_FOLDER`, `ODELIA_METADATA_FOLDER` and
  `ODELIA_IMAGE_FILE` set through `MEDISWARM_DOCKER_OPTIONS` in a site's `image.conf` are not pinned
  by jobs and would apply to every run of that client. No site uses them; the setup line above
  shows the folders actually read.
- **BreastDivider crops inside `data_unilateral/`.** A run on the current crops only warns
  (the strict check is on in `bd` only). The giveaway is a `processing.log` naming BreastDivider in
  the crop format line, or a spacing that "varies like BreastDivider crops" at a site whose current
  crops were on the fixed grid before.
- **Warm starts.** `warm_continue.py` refuses a mirror written by a run on another `CONFIG`
  (`WARM_START_DATA_CONFIG_MISMATCH`) when the job's persistor states `data_config`, which every
  builder-built job does; a sidecar older than that field counts as `unilateral`. Without
  `data_config` the mismatch is only logged. No preset warm-starts a `bd` job.

## Before Run C: what counts as a passed check

- **The count check** (`scripts/site/reconcile_odelia_dataset.py --config unilateral_BD`): the
  crop format line says `ok` and names BreastDivider 0.2.1 or 0.2.2, and there is no `WARNING`
  line. Judge the output, not the tick on the run board.
- **The preflight** prints crop-format issues as a warning and continues ("Continuing;
  ODELIA_STRICT_DATA_FORMAT=1 would abort here"). Any such line means the site **stops at setup
  in Run C**, and with all sites mandatory the whole run stops with it. Typical causes: a
  `processing.log` that does not name `BD_unilateralize` (crops from another tool or an earlier
  pipeline: make them again with 0.2.2), or a sampled folder without `Sub_1.nii.gz` (a stray folder
  such as `segmentation_cleanup_removals` or unlabelled `Sub_2`-only folders: move it out).
- **The sample is deterministic.** The node checks 20 folder names spread evenly over the sorted
  list, always including the first and the last, so a stray folder that sorts first or last is hit
  on every start.
- **Renaming crop folders is harmless** (for example to a site's pseudonymisation scheme); the
  check reads the tool's name from the text of `processing.log`, which must stay as BreastDivider
  wrote it.
- **Missing crops pass silently.** Labelled breasts without a crop folder are dropped from the
  split with only a logged difference (`dataset_3d_odelia.py` keeps the UIDs that have images), so a
  site with partial coverage trains on fewer breasts without an error. Look at the count check's
  `in annotation.csv but no exam folder` line and accept a non-zero value only knowingly.

## Before the next run on the current crops: are they unchanged?

No swarm job has read the current crops since 13 Nov 2026. At the start of the next job on them,
compare each site's setup lines with that site's last run on the current crops, before the first
round finishes:

```bash
# on the coordinator host; run once before the job (baseline) and again after its setup
for site in CAM_1 MHA_1 RSH_1 RUMC_1 UKA_1 UMCU_1 USZ_1 VHIO_1; do
  log=$(ls -t /srv/mediswarm/live/$site/swarm/*/log.txt 2>/dev/null | head -1)
  echo "== $site ${log:-no log}"; [ -n "$log" ] || continue
  grep "ODELIA data for" "$log" | tail -1 | sed 's/.* - INFO - //'
  grep -E "Data hash: [0-9a-f]{8,}" "$log" | tail -1 | sed 's/.* - INFO - //'
  grep "Samples in test set of class" "$log" | tail -3 | sed 's/.* - INFO - //'
done
```

Each client container appends to one console log, and the upload may file it under an older job
id, so the newest log file and its last lines show the newest setup. The same folder count, class counts and data hash as in the
last run mean the same data. The data hash covers the split UIDs and the image content of every
crop used (cached by path, size and modification time), not the label values; the class counts
cover those. A difference without a known reason (a site that announced new data or a new split)
is a reason to stop the job and ask the site. Sites whose logs do not reach the monitor cannot be
checked this way; ask them for the two setup lines from their console.

## Related

- [BREASTDIVIDER_SITE_GUIDE.md](BREASTDIVIDER_SITE_GUIDE.md): how sites make and check the crops.
- [ODELIA_RUN_PLAN_2026-Q4.md](ODELIA_RUN_PLAN_2026-Q4.md): the presets and Run C.
- [PREPROCESSING_BENCHMARK_BREASTDIVIDER.md](PREPROCESSING_BENCHMARK_BREASTDIVIDER.md): why the formats must not be mixed.
