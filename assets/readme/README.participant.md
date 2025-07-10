# MediSwarm Participant Guide

This guide is for data scientists and medical research sites participating in a Swarm Learning project.

## Prerequisites

- Hardware: Min. 32GB RAM, 8 cores, NVIDIA GPU with 24GB VRAM, 4TB storage
- OS: Ubuntu 20.04 LTS
- Software: Docker, OpenVPN, Git

## Setup

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
3. TODO anything else?

## Prepare Dataset

1. see Step 3: Prepare Data in [README.md](../../application/jobs/ODELIA_ternary_classification/app/scripts/README.md)

## Prepare Training Participation

1. Extract startup kit provided by swarm operator

### Local Testing on Your Data

1. Directories
   ```bash
   export SITE_NAME=<name of your site>  # this should end in `_1`, e.g., `UKA_1`, unless you participate with multiple nodes
   export DATADIR=<path to the folder in which the directory $SITE_NAME containing your local data is stored>
   export SCRATCHDIR=<path to where the training can store temporary files>
   ```
2. From the directory where you unpacked the startup kit,
   ```bash
   cd $SITE_NAME/startup
   ```
3. Verify that your Docker/GPU setup is working
   ```bash
   ./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training
   ```
    * This will pull the Docker image, which might take a while.
    * If you have multiple GPUs and 0 is busy, use a different one.
    * The “training” itself should take less than minute and does not yield a meaningful classification performance.
4. Verify that your local data can be accessed and the model can be trained locally
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check
   ```
    * Training time depends on the size of the local dataset.

### (Optional) Run Local Training

1. From the directory where you unpacked the startup kit
   ```bash
   cd $SITE_NAME/startup
   ```
2. Start local training
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --local_training
   ```
    * TODO update when handling of the number of epochs has been implemented
3. Output files
    * TODO describe

### Start Swarm Node

#### VPN

1. Connect to VPN as described in [VPN setup guide(GUI).pdf](../VPN%20setup%20guide%28GUI%29.pdf) (GUI) or [VPN setup guide(CLI).md](../VPN%20setup%20guide%28CLI%29.md) (command line).

#### Start the Client

1. From the directory where you unpacked the startup kit:
   ```bash
   cd $SITE_NAME/startup  # Skip this if you just ran the pre-flight check
   ```

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
    - **Training logs and checkpoints** are saved under:
      ```
      $SCRATCHDIR/runs/$SITE_NAME/<MODEL_TASK_CONFIG_TIMESTAMP>/
      ```
    - **Best checkpoint** usually saved as `best.ckpt` or `last.ckpt`
    - **Prediction results**, if enabled, will appear in subfolders of the same directory
    - **TensorBoard logs**, if activated, are stored in their respective folders inside the run directory
    - TODO what is enabled/activated should be hard-coded, adapt accordingly

5. (Optional) You can verify that the container is running properly:
   ```bash
   docker ps          # Check if odelia_swarm_client_$SITE_NAME is listed
   nvidia-smi         # Check if the GPU is busy training (it will be idling while waiting for model transfer)
   tail -f nohup.out  # Follow training log
   ```
For any issues, check if the commands above point to problems and contact your Swarm Operator.

### Configurable Parameters for docker.sh

TODO consider what should be described and recommended as configurable here, given that the goal of the startup kits is
to ensure everyone runs the same training

When launching the client using `./docker.sh`, the following environment variables are automatically passed into the
container. You can override them to customize training behavior:

| Environment Variable | Default         | Description                                                          |
|----------------------|-----------------|----------------------------------------------------------------------|
| `SITE_NAME`          | *from flag*     | Name of your local site, e.g. `TUD_1`, passed via `--start_client`   |
| `DATA_DIR`           | *from flag*     | Path to the host folder that contains your local data                |
| `SCRATCH_DIR`        | *from flag*     | Path for saving training outputs and temporary files                 |
| `GPU_DEVICE`         | `device=0`      | GPU identifier to use inside the container (or `all`)                |
| `MODEL`              | `MST`           | Model architecture, choices: `MST`, `ResNet`                         |
| `INSTITUTION`        | `ODELIA`        | Institution name, used to group experiment logs                      |
| `CONFIG`             | `unilateral`    | Configuration schema for dataset (e.g. label scheme)                 |
| `NUM_EPOCHS`         | `1` (test mode) | Number of training epochs (used in preflight/local training)         |
| `TRAINING_MODE`      | derived         | Internal use. Automatically set based on flags like `--start_client` |

These are injected into the container as `--env` variables. You can modify their defaults by editing `docker.sh` or
exporting before run:

```bash
export MODEL=ResNet
export CONFIG=original
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=1 --start_client
```
