# MediSwarm Participant Guide

This guide is for data scientists and medical research sites participating in a Swarm Learning project.

## Prerequisites

- Hardware: Min. 32GB RAM, 8 cores, NVIDIA GPU with 24GB VRAM, 4TB storage
- OS: Ubuntu 20.04 LTS, 22.04 LTS, or 24.04 LTS
- Software: Docker, OpenVPN

## Setup
0. Add this line to your `/etc/hosts`: `172.24.4.65 dl3.tud.de dl3`
1. Make sure your compute node satisfies the specification and has the necessary software installed.
2. Set up the VPN. A VPN is necessary so that the swarm nodes can communicate with each other securely across firewalls. For that purpose,
    1. Install OpenVPN
       ```bash
       sudo apt-get install openvpn
       ```
    2. If you have a graphical user interface(GUI), follow this guide to connect to the
       VPN: [VPN setup guide(GUI).pdf](../VPN%20setup%20guide%28GUI%29.pdf)
    3. If you have a command line interface(CLI), follow this guide to connect to the
       VPN: [VPN setup guide(CLI).md](../VPN%20setup%20guide%28CLI%29.md)
    4. You may want to clone this repository or selectively download VPN-related scripts for this purpose.

## Prepare Dataset

The dataset must be in the following format.

### Folder Structure

    ```bash
    <name of your site>
    ├── data_unilateral
    │   ├── ID_001_left
    │   │   └── Sub_1.nii.gz
    │   ├── ID_001_right
    │   │   └── Sub_1.nii.gz
    │   ├── ID_002_left
    │   │   └── Sub_1.nii.gz
    │   ├── ID_002_right
    │   │   └── Sub_1.nii.gz
    │   └── ...
    └── metadata_unilateral
        ├── annotation.csv
        └── split.csv
    ```

* The name of your site should usually end in `_1`, e.g., `UKA_1`, unless you participate with multiple nodes.
* `ID_001`, `ID_002` need to be unique identifiers in your dataset, not specifically of this format
* You might have additional images in the folder like `Pre.nii.gz`, `Post_1.nii.gz`, `Post_2.nii.gz`, `T2.nii.gz`, and you might have additional folders like `data_raw`, `data`, `metadata` etc. These will be ignored and should not cause problems.
* If you clone the repository, you will find a script that generates a synthetic dataset as an example.

### Table Format

#### Annotation

* `annotation.csv` defines the class labels
* The file contains the columns `UID`, `PatientID`, `Age`, `Lesion`
    * `UID` is the identifier used in the folder name, e.g., `ID_001_left`.
    * `PatientID` is the identifier of the patient, in this case, `ID_001`.
    * `Age` is the age of the patient at the time of the scan in days.
       This columns is ignored for our current technical tests and exists only for compatibility with the ODELIA challenge data format. Please ignore discrepancies if age is listed in other units than days.
    * `Lesion` is 0 for no lesion, 1 for benign lesion, and 2 for malicious lesion.

#### Split

* `split.csv` defines the training/validation/test split.
* These splits are hard-coded rather than randomized during training in order to have consistent and documented splits.
* The file contains the columns `UID`, `Split`, and `Fold`.
    * `UID` is the identifier used in the folder name, e.g., `ID_001_left`.
    * `Split` is either `train`, `val`, or `test`. The test set is currently ignored.
    * `Fold` is the 0-based index of the fold (for a potential cross-validation).

## Prepare Training Participation

1. Extract the startup kit provided by swarm operator for the current experiment.

### Local Testing on Your Data

1. Directories
   ```bash
   export SITE_NAME=<name of your site, e.g., UKA_1>
   export DATADIR=<path to the folder in which the directory $SITE_NAME containing your local data in the structure described above is stored>
   export SCRATCHDIR=<path to where the training can store temporary files>
   mkdir -p $SCRATCHDIR
   ```
2. From the directory where you unpacked the startup kit,
   ```bash
   cd $SITE_NAME/startup
   ```
3. Verify that your Docker/GPU setup is working
   ```bash
   ./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training 2>&1 | tee dummy_training_console_output.txt
   ```
    * This will pull the Docker image, which might take a while.
    * If you have multiple GPUs and 0 is busy, use a different one.
    * The “training” itself should take less than minute and does not yield a meaningful classification performance.
4. Verify that your local data can be accessed and the model can be trained locally
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check  2>&1 | tee preflight_check_console_output.txt
   ```
    * Without `--job`, this runs the default ODELIA challenge model: `challenge_1DivideAndConquer` (`MODEL_NAME=1DivideAndConquer`).
    * Training time depends on the size of the local dataset.
    * To test a specific challenge model, use the `--job` flag:
      ```bash
      ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check --job challenge_5pimed
      ```
    * To test the MST baseline instead, use:
      ```bash
      ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check --job ODELIA_ternary_classification
      ```
    * Available jobs: `challenge_1DivideAndConquer` (default), `ODELIA_ternary_classification` (MST), `challenge_2BCN_AIM`, `challenge_3agaldran`, `challenge_4abmil`, `challenge_5pimed`
5. Check your local dataset for discrepancies between expected data and data actually found for training.
   * Check `preflight_check_console_output.txt` for errors and warnings about the dataset.
       * Generally, errors indicate that something is wrong, whereas warnings may indicate properties that may or may not be intended in your dataset.
       * There should be no duplicate UIDs.
       * If there are discrepancies between UIDs listed in `split.csv` and `annotation.csv` and the image files present, make sure this is intended and not an error.
       * There should be no UIDs present in more than one split (training, validation, test).
       * There should be no duplicate image data.
       * If there are discrepancies between left and right images present, make sure this is intended for your dataset and not an error.
   * You can run
       ```bash
       ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check --log_dataset_details
       ```
     to see more detailed output including UIDs for further debugging. (This output may contain confidential UIDs, do not share it!)

### Run Local Training

To have a baseline for swarm training, train the same model in a comparable way on the local data only.

1. From the directory where you unpacked the startup kit (unless you just ran the pre-flight check)
   ```bash
   cd $SITE_NAME/startup
   ```
2. Start local training
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --local_training  2>&1 | tee local_training_console_output.txt
   ```
    * Without `--job`, this trains `challenge_1DivideAndConquer` (`MODEL_NAME=1DivideAndConquer`).
    * This currently runs 100 epochs (somewhat comparable to 20 rounds with 5 epochs each in the swarm case).
    * To train a specific challenge model locally, use the `--job` flag:
      ```bash
      ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --local_training --job challenge_2BCN_AIM 2>&1 | tee local_training_console_output.txt
      ```
    * To train the MST baseline locally, use:
      ```bash
      ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --local_training --job ODELIA_ternary_classification 2>&1 | tee local_training_console_output.txt
      ```
    * **Speed up training (recommended for large datasets).** Enable the on-disk preprocessing cache so each image volume is preprocessed once and stored as a compressed `.npz`; later epochs read from the cache instead of re-decoding NIfTI every time. This removes the data-loading bottleneck and keeps the GPU busy. Set the two environment variables in front of the command:
      ```bash
      ODELIA_ENABLE_PREPROCESS_CACHE=1 \
      ODELIA_PREPROCESS_CACHE_DIR=$SCRATCHDIR/odelia_preprocess_cache \
      ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --local_training 2>&1 | tee local_training_console_output.txt
      ```
        * The first epoch is still slow while the cache is built; subsequent epochs are much faster and GPU utilization rises significantly. If epochs are taking many hours and `nvidia-smi` shows low GPU usage, this is almost certainly the fix.
        * The cache needs free disk space (roughly one compressed volume per case), so point `ODELIA_PREPROCESS_CACHE_DIR` at fast local storage with room to spare. If unset, it defaults to `$SCRATCHDIR/odelia_preprocess_cache`.
        * The cache is reused across runs and auto-invalidates when a source image file changes. The same two variables also speed up `--preflight_check` and the swarm `--start_client` run, so it is worth enabling them everywhere.
3. Output files:
    * Logged output during training: `startup/local_training_console_output.txt`
    * Training results are stored in `$SCRATCHDIR/runs/$SITE_NAME/<RUN_NAME>/`
      * `<RUN_NAME>` has the format `<MODEL>_<CONFIG>_<TIMESTAMP>`, e.g. `MST_unilateral_2026_04_03_120000`
      * Class probabilities: `site_model_gt_and_classprob_{train,validation}.csv`
      * Best checkpoint: `epoch=….ckpt`
      * Last checkpoint: `last.ckpt`
      * TensorBoard logs: `lightning_logs/`

### Start Swarm Node

#### VPN

1. Connect to VPN as described in [VPN setup guide(GUI).pdf](../VPN%20setup%20guide%28GUI%29.pdf) (GUI) or [VPN setup guide(CLI).md](../VPN%20setup%20guide%28CLI%29.md) (command line).

2. **For multi-hour swarm runs, install the VPN as an auto-recovering systemd service (strongly recommended).** A manual/GUI OpenVPN connection that drops and does not reconnect quickly is the most common cause of a long run aborting. Install the tunnel in systemd mode (`mediswarm-vpn.service`, `Restart=always`) plus a health watchdog that re-ups it within ~30 s of a drop:
   ```bash
   # first time: install OpenVPN, store credentials, and register the systemd service
   sudo ./scripts/client_node_setup/setup_vpntunnel.sh -d <YourSite> -n -s
   # install a 30 s health-check timer that restarts the tunnel if the interface drops or the gateway is unreachable
   sudo ./scripts/client_node_setup/vpn_health_monitor.sh --install-timer
   ```
   Verify with `systemctl status mediswarm-vpn mediswarm-vpn-health.timer`. Combined with the swarm's 24 h round timeouts, this lets a run ride out a brief VPN blip instead of aborting.

#### Start the Client

1. From the directory where you unpacked the startup kit:
   ```bash
   cd $SITE_NAME/startup  # Skip this if you just ran the pre-flight check
   ```

> **Pre-run checklist — do this before _every_ swarm run, even if it passed last week.** Environments drift between runs (automatic OS/driver upgrades, IT changes, read-only mounts, re-synced or corrupted data), so a node that worked before can be broken today. Re-run the [dummy training](#local-testing-on-your-data) and [pre-flight check](#local-testing-on-your-data) above and confirm:
> - the dummy training succeeds → the **GPU is usable inside the container**;
> - your preprocessing cache `ODELIA_PREPROCESS_CACHE_DIR` is under `/scratch` (writable), **never under `/data`** (read-only);
> - `dl3.tud.de:8002` is reachable (VPN up);
> - when restarting a client, the old container is removed **and** any stale `daemon_pid.fl` is deleted.
>
> `docker.sh` now also runs **automatic pre-run checks** and prints a `Pre-run checks` block (PASS / WARN / FAIL) with remediation. In particular, if you are on Docker's `systemd` cgroup driver it will warn you to run `scripts/client_node_setup/fix_docker_cgroupfs.sh` to avoid the GPU dropping mid-run (see Troubleshooting below and [`docs/SWARM_FAILURE_MODES.md`](../../docs/SWARM_FAILURE_MODES.md)). Reserve enough time before the scheduled run to fix anything they flag.

2. Start the client:
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
   ```
   If you have multiple GPUs and 0 is busy, use a different one.

3. Console output is captured in `nohup.out`, which may have been created with limited permissions in the container, so
   make it readable if necessary:
   ```bash
   sudo chmod a+r nohup.out
   ```
4. Output files:
    * Console output: `startup/nohup.out`
    * NVFlare log: `log.txt` (in the startup kit root)
    * Training results are stored in `$SCRATCHDIR/runs/$SITE_NAME/<RUN_NAME>/`
      * `<RUN_NAME>` has the format `<MODEL>_<CONFIG>_<TIMESTAMP>`, e.g. `MST_unilateral_2026_04_03_120000`
      * Class probabilities: `{aggregated,site}_model_gt_and_classprob_{train,validation}.csv`
      * Best checkpoint: `epoch=….ckpt`
      * Last checkpoint: `last.ckpt`
      * TensorBoard logs: `lightning_logs/`
    * NVFlare job artifacts: `<JOB_ID>/app_$SITE_NAME/`
      * Aggregated model: `FL_global_model.pt`
      * Best aggregated model: `best_FL_global_model.pt`
      * Code used for training: `custom/`

## Troubleshooting

### Container Running Properly?

You can verify that the container is running properly:

```bash
docker ps          # Check if odelia_swarm_client_$SITE_NAME is listed
nvidia-smi         # Check if the GPU is busy training (it will be idling while waiting for model transfer)
tail -f nohup.out  # Follow training log
```

For any issues, check if the commands above point to problems and contact your Swarm Operator.

### Connection to Swarm Server Working?

Let the following command run for an hour or so

```bash
ping dl3.tud.de
```

* If dl3.tud.de cannot be resolved, double-check whether it is contained in `/etc/hosts`
* If it cannot be reached at all, double-check if the VPN connection is working.
* If intermittent package loss occurs, double-check if your network connection is working properly. Creating new VPN credentials and certificate for connection may also help, contact your Swarm Operator for this purpose.

### VPN Drops During a Run / Node "deemed disconnected"?

A multi-hour run can abort if your tunnel drops and does not come back quickly (the server logs the site as `deemed disconnected`, or a peer reports a transient comms error). To make the tunnel self-heal:

* Run the VPN as the `mediswarm-vpn` **systemd service** (`Restart=always`), not a manual/GUI connection — see [Start Swarm Node → VPN](#vpn).
* Install the health watchdog so a dropped tunnel is restarted within ~30 s:
  ```bash
  sudo ./scripts/client_node_setup/vpn_health_monitor.sh --install-timer
  ```
  It restarts the tunnel immediately if the `tun0` interface disappears, or after a few failed gateway pings. Check its log with `journalctl -t mediswarm-vpn-health`.
* If drops recur, capture your OpenVPN client logs around the outage and report to your Swarm Operator (the gateway provider may need to investigate).

### Training Very Slow / GPU Under-utilized?

If an epoch takes many hours (or more than a day) and `nvidia-smi` shows the GPU mostly idle (e.g. <30% utilization), training is bottlenecked on data loading, not on the GPU. Enable the on-disk preprocessing cache by prefixing the run command with:

```bash
ODELIA_ENABLE_PREPROCESS_CACHE=1 \
ODELIA_PREPROCESS_CACHE_DIR=$SCRATCHDIR/odelia_preprocess_cache \
```

See [Run Local Training](#run-local-training) for details. This caches each preprocessed volume as a compressed `.npz` on fast local storage, so epochs after the first read quickly and the GPU stays fed. It applies to local training, the pre-flight check, and the swarm client.

### GPU Lost Mid-Run (`NVML: Unknown Error`)

If training crashes with `Failed to initialize NVML: Unknown Error` (or `This example does not work without GPU`) **while `nvidia-smi` on the host works fine**, the container lost access to the GPU. This happens on hosts using Docker's **systemd** cgroup driver (the Ubuntu default) when a `systemctl daemon-reload` runs — e.g. the **daily automatic apt upgrade** — which strips the GPU from already-running containers.

* Quick check: `docker exec $(docker ps -q --filter name=odelia_swarm_client) nvidia-smi -L` fails while host `nvidia-smi` works.
* **Durable fix** (persists across reboots, so the daily upgrade no longer breaks it):
  ```bash
  sudo bash scripts/client_node_setup/fix_docker_cgroupfs.sh
  ```
  Then recreate the client (next item). See [`docs/SWARM_FAILURE_MODES.md`](../../docs/SWARM_FAILURE_MODES.md) (F2).

### Client Won't Start After a Restart (stale `daemon_pid.fl`)

If you recreated the client (`docker rm -f …`) and it shows `Up … (healthy)` but never registers with the server, check `nohup.out` for `There seems to be one instance, pid=N, running … remove daemon_pid.fl`. An unclean stop left a lock file that makes `start.sh` refuse to launch.

```bash
docker rm -f $(docker ps -aq --filter name=odelia_swarm_client_$SITE_NAME)
find . -maxdepth 2 -name daemon_pid.fl -delete      # from the kit root ($SITE_NAME/)
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
```
(The automatic pre-run check now clears a stale `daemon_pid.fl` for you when no client container is running.)

### `Read-only file system` Crash

If training crashes immediately with `OSError: [Errno 30] Read-only file system: '/data/...'`, your preprocessing cache points under `/data`, which is mounted read-only. Set it under `/scratch` (the container path), never under `/data`:
```bash
ODELIA_ENABLE_PREPROCESS_CACHE=1 \
ODELIA_PREPROCESS_CACHE_DIR=/scratch/odelia_preprocess_cache \
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
```

### Node "Deemed Disconnected" Mid-Run (VPN drop)

If the operator reports your node was `deemed disconnected` and the run aborted, your VPN tunnel to the server dropped *during* the run (the host stays up; only the VPN path fails). Check `ping dl3.tud.de` over time and `nc -vz dl3.tud.de 8002`. Persistent or intermittent drops should be raised with your network/VPN provider (a keepalive often helps) — in a wait-for-all swarm run, even a brief drop on one node can abort the whole round.

### Further Possible Issues

* Folders where files are located need to have the correct name.
* Image files need to have the correct file name including capitalization.
* The directories listed as identifiers in the tables `annotation.csv` and `split.csv` should all be present and named correctly (including capitalization), only those directories should be present.
* The tables should not have additional or duplicate columns, entries need to have the correct captitalization.
* Image and table folders and files need to be present in the folders specified via `--data_dir`. Symlinks to other locations do not work, they are not available in the Docker mount.
* The correct startup kit needs to be used. `SSLCertVerificationError` or `authentication failed` may indicate an incorrect startup kit incompatible with the current experiment.
* Do not start the VPN connection more than once on the same machine, do not use the same credentials on more than one machine at the same time.
* Disk full. This can have multiple reasons:
  * Failed trainings may have accumulated large logs. Identify which startup kit folders are big (`du -hsc`). Maybe compression is already a solution, otherwise delete/move elsewhere what is no longer needed.
  * Many trainings accumulate many checkpoints (can be GB of data per training). Compression won’t help, possibly delete/move elsewhere what is no longer needed.
  * Intermediate steps or unnecessary input for data conversion may have accumulated.
  * Docker may have accumulated many images. Delete unnecessary old images (in particular on a development workstation, they tend to accumulate quickly). You can use [remove_old_odelia_docker_images.sh](../../scripts/dev_utils/remove_old_odelia_docker_images.sh) to remove all but the latest one (if that is what you want). Afterwards, call `docker system prune`.
* If you have partitioned your system to have a small system partition and a large data partition, you probably want to configure the container storage to happen on the data partition.
  * This can be configured via `echo '{"data-root": "/data/var_lib_docker", "features": {"containerd-snapshotter": true}}' > /etc/docker/daemon.json` (where the containerd-snapshotter may or may not be necessary).
  * If the `data-root` is on an external, network or otherwise slow drive, you need to make sure it is available when the container daemon is started, otherwise you will not see previous containers after a reboot. Maybe `sed -i "s/After=/After=SERVICE_PROVIDING_YOUR_DATA_DRIVE.service /g" /usr/lib/systemd/system/containerd.service` is also helpful for you to configure this.
* The time zone may differ between accounts on the host and jobs run in Docker containers, so file modification dates may have an offset from time stamps in logs.
