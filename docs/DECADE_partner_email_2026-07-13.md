# DECADE partner email — first STAMP run (Monday 2026-07-13)

> Draft email to the four DECADE consortium sites (Bonn, Mainz, Düsseldorf,
> Heidelberg). Fill `<OPERATOR_CONTACT>` and `<YOUR NAME>` before sending. Send
> each site its own row from the table below, the *MediSwarm Participant Guide —
> STAMP* (`assets/readme/README.participant.STAMP.md`), and that site's own
> startup-kit zip. TUD hosts the server + admin only (no TUD training client).

---

**Subject:** DECADE swarm — please complete these checks + send us 2 values before Monday 13 July (first STAMP run)

Dear DECADE partners,

Next **Monday 13 July** we will do our first federated (swarm) training run for
DECADE. Unlike the ODELIA breast-MRI runs, DECADE uses **STAMP**, which trains on
**pre-extracted pathology features (H5 files)** rather than raw images. This week,
please (a) send us the two values in the box below, and (b) work through Steps 1–5
on your compute node so we catch any data or environment issues before Monday.
Everything in Steps 1–5 runs **only on your own machine** — no data leaves your
site during these checks. Full reference: the attached *MediSwarm Participant
Guide — STAMP*.

## ⚠️ Please reply with these two values (needed to finalize your startup kit)

The model output layer and label column differ per site, so we need from **each**
site:

1. **`STAMP_NUM_CLASSES`** — how many classes your prediction target has (e.g. 2
   for tumor/normal, 3 for a three-way grade). For a **regression** or
   **survival** target instead of classification, tell us that.
2. **`STAMP_GROUND_TRUTH_LABEL`** — the **exact column name** in your
   `clini_table.csv` that holds the ground truth (including capitalization), and
   the list of its possible values (e.g. `Diagnosis` → `tumor`, `normal`).

All sites must also agree on the **same feature extractor + feature dimension**.
We propose **UNI (dim 1024)** — tell us if you extracted with something else
(e.g. CTransPath = 768).

## Your site identifier

| Site | Your NVFlare `SITE_NAME` |
|------|--------------------------|
| Universitätsklinikum Bonn | `UKB_1` |
| Universitätsmedizin Mainz | `Mainz_1` |
| Universitätsklinikum Düsseldorf | `UKD_1` |
| Universitätsklinikum Heidelberg | `UKHD_1` |

Use your own row's `SITE_NAME` wherever `<SITE_NAME>` appears below.

## Prerequisites

- Node meets spec (≥32 GB RAM, 8 cores, NVIDIA GPU ≥12 GB VRAM, ~1 TB disk),
  Ubuntu 20.04/22.04/24.04, Docker installed.
- **Tailscale**: accept the invitation we sent and connect, so your node can
  reach the swarm server (hosted at TUD). Then add this line to `/etc/hosts`:
  ```
  100.100.101.100  dl3.tud.de  dl3
  ```
  Verify with `ping dl3.tud.de`.
- **Data**: a `features/` folder with one `.h5` per slide/patient and a
  `clini_table.csv` with a patient-ID column and your ground-truth column, laid
  out as:
  ```
  <SITE_NAME>/
  ├── features/
  │   ├── slide_001.h5
  │   └── ...
  └── clini_table.csv
  ```
- Unpack the **startup kit** we send you for this experiment, then
  `cd <SITE_NAME>/startup`.

## Config to export before the data checks

We will send your site's confirmed `STAMP_NUM_CLASSES` and
`STAMP_GROUND_TRUTH_LABEL` (from your reply above) with your kit.

```bash
export SITE_NAME=<your SITE_NAME from the table, e.g. UKB_1>
export DATADIR=<folder that contains your $SITE_NAME directory>
export SCRATCHDIR=<writable scratch folder>; mkdir -p $SCRATCHDIR
export STAMP_CLINI_TABLE="/data/$SITE_NAME/clini_table.csv"
export STAMP_FEATURE_DIR="/data/$SITE_NAME/features"
export STAMP_TASK="classification"
export STAMP_MODEL_NAME="vit"
export STAMP_DIM_INPUT="1024"                 # match your feature extractor
export STAMP_NUM_CLASSES="<n>"                # the value you sent us
export STAMP_GROUND_TRUTH_LABEL="<column>"    # the exact column name you sent us
export STAMP_PATIENT_LABEL="PATIENT"
```

Paths starting `/data/` are **container** paths — `$DATADIR` is mounted at `/data`
read-only inside Docker.

## Step 1 — Dummy training (Docker + GPU sanity, no data needed)

```bash
./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training 2>&1 | tee dummy_training_console_output.txt
```

Pulls the image (may be slow the first time) and runs a ~1-minute CIFAR-10 CNN.
It does **not** use STAMP or your data — it just proves Docker + GPU work. If GPU
0 is busy, use another (`device=1`).

## Step 2 — Preflight check (your data loads + trains 1 epoch)

```bash
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check 2>&1 | tee preflight_check_console_output.txt
```

Runs one epoch on your real data. Check the log for errors about missing H5
files, mismatched patient IDs, or wrong feature dimensions.

## Step 3 — Plausibility check (dataset discrepancies)

```bash
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check --log_dataset_details
```

Same check, but with detailed per-dataset output so you can confirm: no duplicate
patient IDs; every `.h5` has a matching clinical-table row (and vice versa);
feature dimension matches `STAMP_DIM_INPUT`. **This output can contain patient IDs
— please do NOT send it to us**; just tell us pass/fail and any error messages.

## Step 4 — SSH key sharing (so we can collect your run logs)

During the swarm run, a small "live-sync" helper pushes only **log/metadata**
files to our monitoring host so we can help if something stalls. Please create a
key and send us the **public** half:

```bash
ssh-keygen -t ed25519 -C "$(hostname)@mediswarm"   # press Enter for defaults
cat ~/.ssh/id_ed25519.pub                            # send THIS line to us
```

Send the `.pub` line to <OPERATOR_CONTACT>. Never send the private key. We will
authorize it on the monitoring host and confirm.

## Step 5 — Local training (your baseline)

```bash
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --local_training 2>&1 | tee local_training_console_output.txt
```

Trains a model on only your data, so on Monday we can compare the federated model
against your local baseline. Checkpoints/metrics land in `$SCRATCHDIR` under a
timestamped folder.

## On Monday

Once you get our go-ahead, you'll start the swarm client:

```bash
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
```

(same env vars exported). We'll coordinate the exact time.

---

**Please reply by Friday 11 July** with: (1) your `STAMP_NUM_CLASSES` +
`STAMP_GROUND_TRUTH_LABEL` (and feature extractor), (2) your SSH `.pub` key, and
(3) pass/fail for Steps 1–5. Reach out anytime if a step fails.

Thanks,
<YOUR NAME>, DECADE Swarm Operator
