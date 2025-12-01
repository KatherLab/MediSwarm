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
    * Training time depends on the size of the local dataset.

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
    * This currently runs 100 epochs (somewhat comparable to 20 rounds with 5 epochs each in the swarm case).
3. Output files
    * Same as for the swarm training (see below).

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
    - TODO describe prediction results once implemented
    - **TensorBoard logs** are stored in their respective folders inside the run directory

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

### Further Possible Issues

* Folders where files are located need to have the correct name.
* Image files need to have the correct file name including capitalization.
* The directories listed as identifiers in the tables `annotation.csv` and `split.csv` should all be present and named correctly (including capitalization), only those directories should be present.
* The tables should not have additional or duplicate columns, entries need to have the correct captitalization.
* Image and table folders and files need to be present in the folders specified via `--data_dir`. Symlinks to other locations do not work, they are not available in the Docker mount.
* The correct startup kit needs to be used. `SSLCertVerificationError` or `authentication failed` may indicate an incorrect startup kit incompatible with the current experiment.
* Do not start the VPN connection more than once on the same machine or on more than one machine at the same time.
