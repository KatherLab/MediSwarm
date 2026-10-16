# BreastDivider crops for the November swarm learning run

For ODELIA sites. MediSwarm 1.9.3, October 2026.

In November the consortium trains 1DivideAndConquer on crops made with MEVIS' BreastDivider
package (odelia-preprocessing) instead of the current crops. Each site makes these crops once,
next to its current data, and checks them with the node before the run. Nothing existing is
replaced: the current crops stay where they are for the other runs.

**Why this needs care.** BreastDivider crops have the same size as the current crops (256 x 256 x
32 voxels), so a node that reads the wrong format raises no error. The model just gets worse:
about 0.06 AUROC in our benchmark when the two formats are mixed
([PREPROCESSING_BENCHMARK_BREASTDIVIDER.md](PREPROCESSING_BENCHMARK_BREASTDIVIDER.md)). Please make
the crops exactly as described here, and send us the check results before the run.

## What you need

- **The BreastDivider bundle from MEVIS, version 0.2.2 or 0.2.1.** Both produce identical crops.
  MEVIS licenses it for research use within ODELIA only, so it is not part of the public MediSwarm
  image; you run it on your host, following MEVIS' `readme.md` for the Python packages.
- **Docker with GPU access on the host** for step 1. The first run pulls the model image
  `ykirchhoff/breastdivider` (14.4 GB).
- **Time.** On our benchmark machine about 6 s per exam: 2 s of GPU segmentation and 4.5 s of
  cropping with 14 CPU workers (456 exams in 47 minutes), so 1,000 exams take a little under two
  hours. Fewer workers take longer.
- **A GPU that is free.** Step 1 and the preflight check use the GPU your swarm client needs, so
  run them outside swarm run windows.
- **Your bilateral images** in `<site folder>/data/<CaseID>/Pre.nii.gz` and `Post_1.nii.gz`, the same
  input the current crops were made from. If they live elsewhere, see "Arrange your data" in MEVIS'
  readme (symbolic links are fine).

Below, `$DATADIR` is the folder you pass to `docker.sh --data_dir`, and `$SITE_NAME` is your site
folder inside it (for example `UKA_1`), as in the participant guide. The site folder is the root
for BreastDivider.

## Step 1: segmentation (Docker and GPU)

From the folder where you unpacked the bundle:

```bash
python BD_run_seg.py "$DATADIR/$SITE_NAME" --single-site --bd_gpu=device=0 --force
```

Use the number of a free GPU in `device=0`. Then check the end of
`$DATADIR/$SITE_NAME/BD_Output/processing.log`: there must be no `[ERROR]` line and no line
`N case(s) produced NO output`.

## Step 2: cropping (CPU)

```bash
python BD_main.py "$DATADIR/$SITE_NAME" --single-site --mode=normal --subs 1 --num_workers=8 --force
```

- Set `--num_workers` to the number of CPU cores you can spare.
- Keep `--mode=normal` and do not pass `--xy_resolution`. Other settings write a differently named
  folder with a different grid.
- `--subs 1` is enough: the node reads `Sub_1.nii.gz`. `--subs 1,2` (MEVIS' example) is fine too.
- `--diagnostic_plots` is optional. It roughly doubles the cropping time; the node ignores the PNGs.
- Use the same bundle version for both steps.
- Do **not** run the bundle's optional `unilateralize.py` in your site folder: it writes into
  `data_unilateral/`, where your current crops are.
- Never copy or rename BreastDivider output into `data_unilateral/`, and never copy current crops
  into `data_unilateral_BD/`.

## Step 3: labels

```bash
cp -r "$DATADIR/$SITE_NAME/metadata_unilateral" "$DATADIR/$SITE_NAME/metadata_unilateral_BD"
```

BreastDivider names its crops `<CaseID>_left` and `<CaseID>_right` like the current crops, so the
same `annotation.csv` and `split.csv` apply. Keep each UID's fold and split, so that the results
stay comparable with earlier runs. The node uses only crops whose UID is listed in both
files, so extra crops (BreastDivider writes both sides even for single-breast exams) are ignored. If
you want to add exams that only BreastDivider can serve, please agree this with us first.

Without `metadata_unilateral_BD/` the node falls back to `metadata_unilateral/` and logs a warning.
The copy makes the choice explicit.

## Resulting layout

```
$DATADIR/$SITE_NAME/
├── data/                      bilateral input (BreastDivider only reads it)
├── data_unilateral/           current crops: keep, the other runs use them
├── metadata_unilateral/       current labels: keep
├── BD_Output/                 step 1: breast masks and their processing.log
├── BD_Input_seq/              step 1: temporary files, failed batches are kept here
├── data_unilateral_BD/        step 2: the crops the November run reads
│   ├── <CaseID>_left/Sub_1.nii.gz
│   ├── <CaseID>_right/Sub_1.nii.gz
│   ├── processing.log         keep it: the node checks it
│   ├── segmentation_cleanup_removals.csv
│   └── <CaseID>_diagnostic.png    (only with --diagnostic_plots)
└── metadata_unilateral_BD/    step 3: annotation.csv and split.csv
```

## Step 4: check with the node (MediSwarm 1.9)

Use your 1.9.3 startup kit; its `docker.sh` passes `--config` to the container and runs the
1.9.3 image, which knows `unilateral_BD`. On an older kit, add `--image jefftud/odelia:1.9.3` to
the `docker.sh` command.

**Counts.** This reads only the metadata, the folder names, `processing.log` and the image headers:

```bash
docker run --rm --network none -u "$(id -u):$(id -g)" -v "$DATADIR":/data:ro \
    --entrypoint python3 jefftud/odelia:1.9.3 \
    /MediSwarm/scripts/site/reconcile_odelia_dataset.py --data_dir /data/$SITE_NAME --config unilateral_BD
```

Please send us what it prints: the counts, the `crop format` line (package version and number of
untrusted-fallback cases) and any `WARNING` line. It prints no UIDs unless you add `--list`; keep
that output at your site. If your exam folders are symbolic links to an absolute path, mount that
path as well (`-v /abs/target:/abs/target:ro`), as for `docker.sh`.

**Preflight check.** One epoch of local training on the new crops (GPU, a few minutes):

```bash
cd <unpacked startup kit>/$SITE_NAME/startup
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --preflight_check \
    --job challenge_1DivideAndConquer --config unilateral_BD 2>&1 | tee preflight_BD_console_output.txt
```

It works when the output contains

```
ODELIA data for <SITE>: CONFIG=unilateral_BD, images /data/<SITE>/data_unilateral_BD (<N> exam folders), ...
ODELIA crop format for <SITE>: ok; processing.log: BreastDivider 0.2.2 (...)
Training completed successfully.
```

and no warning that mentions `ODELIA crop format`. Use `--config` only for these checks: the
November job sets `CONFIG=unilateral_BD` itself, so start your swarm client as usual, without
`--config`.

## What the node checks, and how to fix a warning

When it starts, the node logs which folders it reads and checks that the crops look like
BreastDivider output: a `processing.log` in `data_unilateral_BD/` that names `BD_unilateralize`
(package 0.2.1 or 0.2.2, `--mode normal`, xy resolution 256), and a voxel spacing that varies
between crops. BreastDivider fits a box to each breast, while the current crops all share a fixed
0.7 x 0.7 x 3 mm grid. A failed check is a warning by default, but a run can be configured to stop
on it, so please fix every warning before the run.

| Message contains | Meaning | Fix |
|---|---|---|
| `image folder /data/<SITE>/data_unilateral_BD does not exist` | step 2 did not run, or ran on another root | run steps 1 and 2 with your site folder as root |
| `metadata_unilateral_BD not found, using .../metadata_unilateral` | step 3 skipped | copy the folder (step 3), or accept the fallback |
| `no processing.log in .../data_unilateral_BD` | the log was deleted, or the crops were copied from elsewhere | rerun step 2 with `--force` |
| `does not name BD_unilateralize` | not BreastDivider crops (for example from `unilateralize.py`) | delete `data_unilateral_BD/`, rerun step 2 |
| `this release accepts 0.2.1 or 0.2.2` | another bundle version | rerun steps 1 and 2 with 0.2.2 |
| `--mode tight` or `--xy_resolution ...` | other cropping settings | rerun step 2 exactly as above |
| `fixed 0.7 x 0.7 x 3 mm spacing` or `share one spacing` | current crops in the BreastDivider folder | delete `data_unilateral_BD/`, rerun step 2 |
| `sampled exam folders ... have no Sub_1.nii.gz` | step 2 ran without `--subs 1` | rerun step 2 with `--subs 1` |
| `names BD_unilateralize` or `varies ... like BreastDivider crops` in a run without `CONFIG=unilateral_BD` | BreastDivider crops in `data_unilateral/` | restore your current crops in `data_unilateral/` |

## For coordinators: settings in the job

- `CONFIG` selects the folders: `unilateral` (default) reads `data_unilateral/` and
  `metadata_unilateral/`; `unilateral_BD` reads `data_unilateral_BD/` and `metadata_unilateral_BD/`
  (falling back to `metadata_unilateral/`); `original` reads `data/` and `metadata/`. Any other
  value stops the node with the list of valid values.
- `ODELIA_STRICT_DATA_FORMAT=1` turns the format warnings into an error at setup, before training.
- `ODELIA_DATA_FOLDER`, `ODELIA_METADATA_FOLDER` and `ODELIA_IMAGE_FILE` (default `Sub_1.nii.gz`)
  name other folders or another file inside the site folder, so a later preprocessing variant
  needs no new image. The format check still follows `CONFIG`. For a site-local check they can
  be passed as `--docker_options "-e ODELIA_DATA_FOLDER=..."`.
- Each run directory gets `data_layout.json`: the folders and file the node read, whether the
  metadata fallback was used, and the format check result. It contains no UIDs.
