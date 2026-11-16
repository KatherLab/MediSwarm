# MediSwarm Participant Guide — STAMP (Computational Pathology)

This guide is for data scientists and medical research sites participating in a
STAMP (Solid Tumor Associative Modeling in Pathology) swarm learning project. If
you are joining an ODELIA (breast MRI) project instead, see
[README.participant.md](README.participant.md).

## Overview

STAMP trains classification, regression, or survival models on **pre-extracted
feature vectors** from whole-slide pathology images (WSIs). Unlike ODELIA, which
trains on raw 3D NIfTI volumes, STAMP expects HDF5 (`.h5`) feature files that
have already been extracted from WSIs using a foundation model (e.g. UNI,
CTransPath, RetCCL).

Only the **training** step is done as swarm learning. Feature extraction (preprocessing) and
inference (deployment) remain standalone steps that each site runs locally.

## Prerequisites

- **Hardware:** Min. 32 GB RAM, 8 cores, NVIDIA GPU with 12+ GB VRAM, 1 TB storage
  (STAMP is lighter on GPU memory than ODELIA because it operates on extracted
  features rather than full imaging volumes)
- **OS:** Ubuntu 20.04 LTS, 22.04 LTS, or 24.04 LTS
- **Software:** Docker, and the network your project uses: Tailscale (e.g. DECADE) or
  OpenVPN (e.g. ODELIA)
- **Data:** Pre-extracted H5 feature files + clinical metadata CSV (see
  [Prepare Dataset](#prepare-dataset) below)

## Setup

0. Add a line mapping the swarm server to your `/etc/hosts`. Your swarm operator
   will confirm the exact IP for your project:
    * OpenVPN projects (e.g. ODELIA): `<SERVER_IP> dl3.tud.de dl3`
    * Tailscale projects (e.g. DECADE): `<SERVER_TAILSCALE_IP> dl3.tud.de dl3`
      (the server's `100.x.y.z` Tailscale address).
1. Make sure your compute node satisfies the specification and has the necessary
   software installed.
2. Set up the VPN. A VPN is necessary so that the swarm nodes can communicate
   with each other securely across firewalls. Your operator will tell you which
   VPN your project uses.

    **Option A — Tailscale (DECADE):**
    1. Accept the Tailscale invitation your operator emailed and connect your
       node to the tailnet.
    2. Confirm the server is reachable: `ping dl3.tud.de` (after the `/etc/hosts`
       entry in step 0).

    **Option B — OpenVPN (ODELIA):**
    1. Install OpenVPN
       ```bash
       sudo apt-get install openvpn
       ```
    2. If you have a graphical user interface (GUI), follow this guide to connect
       to the VPN: [VPN setup guide(GUI).pdf](../VPN%20setup%20guide%28GUI%29.pdf)
    3. If you have a command line interface (CLI), follow this guide to connect
       to the VPN: [VPN setup guide(CLI).md](../VPN%20setup%20guide%28CLI%29.md)
    4. You may want to clone this repository or selectively download VPN-related
       scripts for this purpose.

3. **Share an SSH public key with your swarm operator (for live sync).** While
   your client runs, and during `--local_training`, a small helper (live sync)
   uploads monitoring files to the operator's host, so the operator can help
   when a node stalls. See
   [What leaves your site](#what-leaves-your-site). Generate a key and send the
   operator the **public** half:
   ```bash
   ssh-keygen -t ed25519 -C "$(hostname)@mediswarm"   # press Enter for defaults
   cat ~/.ssh/id_ed25519.pub                            # send THIS line to your operator
   ```
   Never send the private key. The operator authorizes your public key on the
   monitoring host and confirms.

### What leaves your site

Swarm learning exchanges model weights between the sites; your clinical table
and feature files never leave your node. Live sync (kit 1.7.1 and later)
uploads to the operator's monitoring host:

* logs: `startup/nohup.out`, `log.txt` and live sync's own log, and a heartbeat
  every 30 s (kit and image version, client state, host name and IP address,
  and the folder paths of the kit and of the current run);
* `stamp_metrics_summary.csv` of each run: per epoch the training and validation
  loss, the validation AUROC and the learning rate;
* the swarm's global models of a run, `FL_global_model.pt` and
  `best_FL_global_model.pt`: weights only, which every site of the run holds.
  `MEDISWARM_SYNC_GLOBAL_MODELS=0` in `startup/sync.conf` keeps them at your site.

These stay at your site: your local checkpoints (`*.ckpt`), TensorBoard files
and `hparams.yaml` (STAMP writes your patient IDs into all three), the console
output of `--local_training`, and the per-patient prediction files
`stamp_gt_predprob_*.csv` (uploaded only if you set
`MEDISWARM_SYNC_PREDICTIONS=1` in `startup/sync.conf`). The swarm training code
also keeps patient IDs out of the logs it writes.

To change one of these settings, add the line to `startup/sync.conf`. The
`startup/sync.conf.example` in kit 1.7.1 describes the ODELIA kits; ignore it.

## Prepare Dataset

STAMP requires two inputs per site: a **feature directory** containing one H5
file per slide/patient, and a **clinical table** (CSV) with ground-truth labels.

### Feature Extraction (Prerequisite)

Before joining a swarm project you must extract features from your whole-slide
images using the agreed-upon foundation model and tile size. This is done with
standalone [STAMP](https://github.com/KatherLab/STAMP) preprocessing and
produces one `.h5` file per slide.

Coordinate with your swarm operator to ensure all sites use the **same feature
extractor, tile size, and feature dimension** — mismatched features will cause
training failures.

**STAMP version for extraction (DECADE): 2.5.0.** Extract with standalone STAMP
2.5.0 (it requires Python 3.13). The swarm training image itself runs STAMP
2.4.0 — because NVFlare/PyTorch in the image are on Python 3.11 — but it is
patched to read features extracted by STAMP up to 2.5.0. The feature H5 layout
and the UNI extractor are identical between 2.4.0 and 2.5.0, so this is safe.
If you extract with a STAMP newer than 2.5.0, training will refuse the features
— tell your swarm operator first.

### Folder Structure

```
<name of your site>/
├── features/
│   ├── slide_001.h5
│   ├── slide_002.h5
│   ├── slide_003.h5
│   └── ...
└── clini_table.csv
```

* The name of your site should match your NVFlare site identifier (e.g.
  `UKA_1`), unless instructed otherwise by your swarm operator.
* The `features/` directory contains one `.h5` file per slide (or per patient,
  depending on feature extraction settings).

### H5 Feature File Format

Each `.h5` file must contain:

| Dataset   | Shape           | Dtype   | Description |
|-----------|-----------------|---------|-------------|
| `feats`   | (N_tiles, dim)  | float32 | Feature vectors (e.g. 1024-dim for UNI) |
| `coords`  | (N_tiles, 2)    | float32 | Tile coordinates (for tile-level features) |

Optional HDF5 attributes: `feat_type` (`tile`, `slide`, or `patient`),
`tile_size`, `unit`.

The feature dimension (`dim`) must match the `STAMP_DIM_INPUT` environment
variable (default: 1024 for UNI/UNI2; use 768 for CTransPath).

### Clinical Table Format

`clini_table.csv` is a CSV file with at minimum the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| Patient ID column | Unique patient identifier (default column name: `PATIENT`) | `P_001` |
| Ground truth column | Class label for classification, or continuous value for regression | `tumor`, `normal`, `adenoma` |

The patient ID column name is configured via `STAMP_PATIENT_LABEL` (default:
`PATIENT`) and the ground truth column via `STAMP_GROUND_TRUTH_LABEL`.

**Important:** The H5 filenames (without `.h5`) must match the patient
identifiers in the clinical table, or a separate slide table must be provided
to map between them.

#### Optional: Slide Table (patients with multiple slides)

If any patient has **more than one slide**, your `clini_table.csv` has one row per
*patient*, while `features/` has one `.h5` per *slide*. A **slide table** maps the
two together, so STAMP can pool all of a patient's slides into one bag.

Provide a second CSV with one row per slide — a patient-ID column and a slide
filename column — and point STAMP at it:

```bash
export STAMP_SLIDE_TABLE="/data/<SITE_NAME>/slide_table.csv"
export STAMP_PATIENT_LABEL="PATIENT"     # patient-ID column (both tables)
export STAMP_FILENAME_LABEL="FILENAME"   # slide-filename column (slide table)
```

`slide_table.csv`:

| PATIENT | FILENAME |
|---------|----------|
| P_001   | slide_001.h5 |
| P_001   | slide_002.h5 |
| P_002   | slide_003.h5 |

The `FILENAME` values must match the files in `STAMP_FEATURE_DIR`. Without a slide
table, STAMP expects each H5 filename (minus `.h5`) to equal a patient identifier,
which only works when every patient has exactly one slide.

### Synthetic Test Data

To verify your setup before using real data, you can generate a synthetic
dataset:

```bash
python application/jobs/STAMP_classification/app/scripts/create_synthetic_dataset/create_synthetic_stamp_dataset.py \
    --output_dir /tmp/stamp_synthetic_data
```

This creates two test sites (`client_A/`, `client_B/`) each with 15 synthetic
patients across 3 classes.

## Prepare Training Participation

1. Extract the startup kit provided by your swarm operator for the current
   experiment.

### Configure STAMP Environment Variables

STAMP uses environment variables (all prefixed with `STAMP_`) to configure the
training. These must be exported **before** calling `docker.sh` — the script
automatically forwards all `STAMP_*` variables into the Docker container.

> **Do not run `docker.sh` with `sudo`.** `sudo` resets the environment, so the
> `STAMP_*` variables you exported are *not* visible to `docker.sh` and training
> fails with `KeyError: 'STAMP_CLINI_TABLE'` — even though `echo $STAMP_CLINI_TABLE`
> prints the right value in your own shell.
>
> If you need root to talk to Docker, add your user to the `docker` group once and
> then run without `sudo`:
> ```bash
> sudo usermod -aG docker $USER && newgrp docker
> ./docker.sh ...
> ```
> As a fallback, preserve the environment explicitly: `sudo -E ./docker.sh ...`.
> To see what `docker.sh` will forward: `env | grep '^STAMP_'`.

```bash
# ── Required ──
export STAMP_CLINI_TABLE="/data/<SITE_NAME>/clini_table.csv"
export STAMP_FEATURE_DIR="/data/<SITE_NAME>/features"
export STAMP_GROUND_TRUTH_LABEL="<column name>"   # e.g. "Diagnosis"
export STAMP_PATIENT_LABEL="PATIENT"               # patient ID column name

# ── Task & Model (the values your operator gives for the run) ──
export STAMP_TASK="classification"                 # classification, regression, or survival
export STAMP_MODEL_NAME="vit"                      # vit, mlp, trans_mil, linear
export STAMP_DIM_INPUT="1024"                      # must match your feature extractor
export STAMP_NUM_CLASSES="2"                       # e.g. 2 for a yes/no target
export STAMP_EXPECTED_CATEGORIES="no,yes"          # the agreed class order (checked before training)

# ── Your node ──
export STAMP_NUM_WORKERS="2"                       # DataLoader workers (0 = main thread)
```

Leave the training settings (`STAMP_BAG_SIZE`, `STAMP_BATCH_SIZE`, `STAMP_MAX_LR`,
`STAMP_MAX_EPOCHS`, ...) at their defaults: in a swarm run the job sets the model,
the classes and the training recipe for every site, and your local baseline is
only comparable with the same defaults.

**Note:** The paths in `STAMP_CLINI_TABLE` and `STAMP_FEATURE_DIR` are
**container paths**. The host `--data_dir` is mounted at `/data/` inside the
container (read-only), so if your host data is at
`/mnt/data/stamp/UKA_1/features/`, pass `--data_dir /mnt/data/stamp` and set
`STAMP_FEATURE_DIR="/data/UKA_1/features"`.

Your swarm operator will tell you the correct values for `STAMP_MODEL_NAME`,
`STAMP_DIM_INPUT`, `STAMP_NUM_CLASSES`, `STAMP_GROUND_TRUTH_LABEL`, and
`STAMP_TASK` — these must be consistent across all sites.

### Available Models

| Model | `STAMP_MODEL_NAME` | Description |
|-------|---------------------|-------------|
| VIT | `vit` | Vision Transformer with ALiBi positional encoding |
| MLP | `mlp` | Multi-layer perceptron |
| TransMIL | `trans_mil` | Transformer-based multiple instance learning |
| Linear | `linear` | Linear classifier |
| Barspoon | `barspoon` | Encoder-decoder transformer — **not available**: requires STAMP ≥ 2.5.0, the swarm image ships 2.4.0 |

### Check Your Node

`./docker.sh --status` (from the kit's `startup` folder) is a read-only report:
kit and image version, the client container and its GPU, the `STAMP_*` settings
the running client was started with, live sync, the NVIDIA and Docker cgroup
driver, the swarm server and the lock files. Send its output when you ask the
operator for help. Run it as the user that runs the client (no `sudo`).

### Local Testing on Your Data

1. Set up directories:
   ```bash
   export SITE_NAME=<name of your site, e.g., UKA_1>
   export DATADIR=<path to the folder containing your $SITE_NAME directory>
   export SCRATCHDIR=<path to where training can store temporary files>
   mkdir -p $SCRATCHDIR
   ```

2. From the directory where you unpacked the startup kit:
   ```bash
   cd $SITE_NAME/startup
   ```

3. Verify that your Docker/GPU setup is working:
   ```bash
   ./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training 2>&1 | tee dummy_training_console_output.txt
   ```
   * This will pull the Docker image, which might take a while.
   * If you have multiple GPUs and 0 is busy, use a different one.
   * The "training" itself should take less than a minute and does not use STAMP
     — it runs a minimal CNN on CIFAR-10 to verify Docker + GPU.

4. Export your STAMP environment variables (see
   [Configure STAMP Environment Variables](#configure-stamp-environment-variables)
   above), then verify that your local data can be accessed:
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check 2>&1 | tee preflight_check_console_output.txt
   ```
   * This runs 1 epoch of training on your local data as a quick smoke test.
   * Check `preflight_check_console_output.txt` for errors about missing H5
     files, mismatched patient IDs, or incorrect feature dimensions. The line
     `Total samples: N | Class no: X | Class yes: Y` shows the class counts STAMP
     found (counts only; you can share it).
   * It is safe to run next to a running client: it gets a container name of its
     own and leaves the client alone.

5. Check your local dataset for discrepancies:
   * There should be no duplicate patient IDs in the clinical table.
   * Every H5 file in the feature directory should have a corresponding entry in
     the clinical table (and vice versa).
   * Feature dimensions in H5 files should match `STAMP_DIM_INPUT`.
   * The preflight prints what STAMP finds (for example patients without a feature
     file) as warnings; `--log_dataset_details` adds nothing for STAMP. This output
     can list confidential patient IDs: do not share it. Send your operator only the
     `Total samples` line and pass/fail, without IDs.

### Run Local Training

To have a baseline for swarm training, train the same model locally:

1. From the directory where you unpacked the startup kit (unless you just ran the
   preflight check):
   ```bash
   cd $SITE_NAME/startup
   ```

2. Make sure your STAMP environment variables are exported, then start local
   training:
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --local_training 2>&1 | tee local_training_console_output.txt
   ```

3. Output files:
   * Console output: `startup/local_training_console_output.txt`
   * Training results are stored in `$SCRATCHDIR/runs/$SITE_NAME/` under a
     timestamped run directory (`STAMP_vit_<date>`), including:
     * Model checkpoints (`best_model.ckpt`, `last_model.ckpt`)
     * `stamp_metrics_summary.csv` (per-epoch losses, validation AUROC)
     * Per-patient predictions `stamp_gt_predprob_*.csv` (stay at your site)

### Start Swarm Node

#### VPN

1. Make sure the node is on your project's network: Tailscale for DECADE (see
   Setup, Option A), or OpenVPN for ODELIA as described in
   [VPN setup guide(GUI).pdf](../VPN%20setup%20guide%28GUI%29.pdf) (GUI) or
   [VPN setup guide(CLI).md](../VPN%20setup%20guide%28CLI%29.md) (command line).

#### Move to a New Kit

A new kit replaces the old one on your node; certificates and your site name
stay the same. Only one client per site can be in the swarm, so the old client
must be gone before the new one starts.

1. Check the zip: `sha256sum <SITE>_<version>.zip` must match the kit registry.
2. Stop and remove the old client container:
   `docker ps -a --format '{{.Names}}' | grep stamp_swarm_client` shows it, then
   `docker rm -f stamp_swarm_client_<SITE>_<old id>`.
3. Unzip the new kit into a **new** folder (not over the old one).
4. Copy over your own edits to `startup/sync.conf`, if any (e.g. an `-i` key
   option). Keep the new kit's `startup/image.conf`: it names the image of the run.
5. Export your `STAMP_*` settings and run `./docker.sh ... --preflight_check`. Run
   `./docker.sh ... --start_client` only when your operator announces the run.

`--start_client` refuses to start while a client container of another kit runs
for your site (it prints the `docker rm -f` command), and it stops the old kit's
live-sync helper itself.

#### Start the Client

1. From the directory where you unpacked the startup kit:
   ```bash
   cd $SITE_NAME/startup  # Skip this if you just ran the preflight check
   ```

2. Make sure your STAMP environment variables are exported (see
   [Configure STAMP Environment Variables](#configure-stamp-environment-variables)),
   then start the client:
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
   ```
   If you have multiple GPUs and 0 is busy, use a different one.

3. Console output is captured in `nohup.out`, which may have been created with
   limited permissions in the container, so make it readable if necessary:
   ```bash
   sudo chmod a+r nohup.out
   ```

4. Output files:
   * Console output: `startup/nohup.out`
   * NVFlare log: `log.txt` (in the startup kit root)
   * Training results are stored in `$SCRATCHDIR/` under a timestamped run
     directory
   * NVFlare job artifacts: `<JOB_ID>/app_$SITE_NAME/`
     * Aggregated model: `FL_global_model.pt`
     * Best aggregated model: `best_FL_global_model.pt`

## Troubleshooting

### Container Running Properly?

```bash
./docker.sh --status   # read-only report of this node
docker ps              # Check if stamp_swarm_client_$SITE_NAME is listed
nvidia-smi             # Check if the GPU is busy training
tail -f nohup.out      # Follow training log
```

For any issues, check if the commands above point to problems and contact your
Swarm Operator.

* **Never delete `daemon_pid.fl` or `pid.fl` while the client runs**: without them
  the client starts a second copy of itself, and the two push each other out of
  the swarm. To restart a client, remove its container (`docker rm -f ...`) and run
  `--start_client` again; the new container clears old lock files itself.
* **`Driver/library version mismatch` from `nvidia-smi`**: an automatic update
  replaced the NVIDIA libraries while the old driver is still loaded. A running
  client keeps its GPU, but no new container gets one: reboot the node, then start
  the client again. Hold the driver packages during a run
  (`sudo apt-mark hold <the nvidia packages>`; while the mismatch is present,
  `--status` prints this command with the package names).
* **Docker cgroup driver `systemd`** (`--status` warns): a `systemctl
  daemon-reload`, e.g. by an automatic update, can take the GPU away from a
  running client ("NVML: Unknown Error"). `scripts/client_node_setup/fix_docker_cgroupfs.sh`
  in the kit switches Docker to `cgroupfs` (it restarts Docker, so not during a run).

### STAMP Environment Variables Not Forwarded?

If training fails with errors about missing `STAMP_CLINI_TABLE` or
`STAMP_FEATURE_DIR`, verify that:

1. You exported the `STAMP_*` variables in the **same shell session** before
   calling `docker.sh`.
2. The paths use **container paths** (starting with `/data/`) not host paths.
3. Run `env | grep STAMP_` to confirm all variables are set.

### Data Issues?

* **"No H5 files found"** — Check that `STAMP_FEATURE_DIR` points to the
  correct directory inside the container (`/data/<SITE_NAME>/features`).
* **"Patient ID not found"** — Ensure H5 filenames (without `.h5`) match the
  patient ID column in the clinical table.
* **"Feature dimension mismatch"** — Ensure `STAMP_DIM_INPUT` matches the
  dimension of features in your H5 files (check with
  `python -c "import h5py; f=h5py.File('slide.h5','r'); print(f['feats'].shape)"`).

### Connection to Swarm Server Working?

Let the following command run for an hour or so:

```bash
ping dl3.tud.de
```

* If dl3.tud.de cannot be resolved, double-check whether it is contained in
  `/etc/hosts`.
* If it cannot be reached at all, double-check if the VPN connection is working.
* If intermittent packet loss occurs, double-check if your network connection is
  working properly. Creating new VPN credentials and certificate may also help —
  contact your Swarm Operator.

### Further Possible Issues

* Ensure `STAMP_GROUND_TRUTH_LABEL` exactly matches the column name in your
  clinical CSV (including capitalization).
* Ensure `STAMP_PATIENT_LABEL` exactly matches the patient ID column name.
* Feature files and clinical table entries need correct filenames including
  capitalization.
* Symlinks inside the data directory do not work — they are not available inside
  the Docker mount (which is read-only).
* The correct startup kit needs to be used. `SSLCertVerificationError` or
  `authentication failed` may indicate an incorrect startup kit incompatible
  with the current experiment.
* Do not start the VPN connection more than once on the same machine; do not use
  the same credentials on more than one machine at the same time.
* Disk full — see the [ODELIA participant guide](README.participant.md) for
  general disk management tips (Docker image cleanup, checkpoint accumulation,
  etc.).
* Docker storage configuration — if you have a small system partition and large
  data partition, configure Docker's `data-root` accordingly (see
  [ODELIA participant guide](README.participant.md) for details).
