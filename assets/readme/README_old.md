# Introduction

MediSwarm is an open-source project dedicated to advancing medical deep learning through swarm intelligence, leveraging
the NVFlare platform. Developed in collaboration with the Odelia consortium, this repository aims to create a
decentralized and collaborative framework for medical research and applications.

## Key Features

- **Swarm Learning:** Utilizes swarm intelligence principles to improve model performance and adaptability.
- **NVFlare Integration:** Built on NVFlare, providing robust and scalable federated learning capabilities.
- **Data Privacy:** Ensures data security and compliance with privacy regulations by keeping data local to each
  institution.
- **Collaborative Research:** Facilitates collaboration among medical researchers and institutions for enhanced
  outcomes.
- **Extensible Framework:** Designed to support various medical applications and easily integrate with existing
  workflows.

## Prerequisites

### Hardware recommendations

* 64 GB of RAM (32 GB is the absolute minimum)
* 16 CPU cores (8 is the absolute minimum)
* an NVIDIA GPU with 48 GB of RAM (24 GB is the minimum)
* 8 TB of Storage (4 TB is the absolute minimum)

We demonstrate that the system can run on lightweight hardware like this. For less than 10k EUR, you can configure
systems from suppliers like Lambda, Dell Precision, and Dell Alienware.

### Operating System

* Ubuntu 20.04 LTS

### Software

* Docker
* openvpn
* git

### Cloning the repository

    ```bash
    git clone https://github.com/KatherLab/MediSwarm.git --recurse-submodules
    ```

* The last argument is necessary because we are using a git submodule for the (ODELIA fork of
  NVFlare)[https://github.com/KatherLab/NVFlare_MediSwarm]
* If you have cloned it without this argument, use `git submodule update --init --recursive`

### VPN

A VPN is necessary so that the swarm nodes can communicate with each other securely across firewalls. For that purpose,

1. Install OpenVPN
   ```bash
   sudo apt-get install openvpn
   ```
2. If you have a graphical user interface(GUI), follow this guide to connect to the
   VPN: [VPN setup guide(GUI).pdf](assets/VPN%20setup%20guide%28GUI%29.pdf)
3. If you have a command line interface(CLI), follow this guide to connect to the
   VPN: [VPN setup guide(CLI).md](assets/VPN%20setup%20guide%28CLI%29.md)

# Usage for Swarm Participants

## Setup

1. Make sure your compute node satisfies the specification and has the necessary software installed.
2. Clone the repository and connect the client node to the VPN as described above. TODO is cloning the repository
   necessary for swarm participants?
3. TODO anything else?

## Prepare Dataset

1. see Step 3: Prepare Data in (this document)[application/jobs/ODELIA_ternary_classification/app/scripts/README.md]

## Prepare Training Participation

1. Extract startup kit provided by swarm operator

## Run Pre-Flight Check

1. Directories
   ```bash
   export SITE_NAME=<name of your site>  # TODO should be defined above, also needed for dataset location
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

## Configurable Parameters for docker.sh

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

## Start Swarm Node

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
   docker ps  # Check if odelia_swarm_client_$SITE_NAME is listed
   nvidia-smi  # Check if the GPU is busy training (it will be idling while waiting for model transfer)
   tail -f nohup.out  # Follow training log
   ```

## Run Local Training

1. From the directory where you unpacked the startup kit
   ```bash
   cd $SITE_NAME/startup
   ```
2. Start local training
   ```bash
   /docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU all --local_training
   ```
    * TODO update when handling of the number of epochs has been implemented
3. Output files
    * TODO describe

# Usage for MediSwarm and Application Code Developers

## Versioning of ODELIA Docker Images

If needed, update the version number in file (odelia_image.version)[odelia_image.version]. It will be used automatically
for the Docker image and startup kits.

## Build the Docker Image and Startup Kits

The Docker image contains all dependencies for administrative purposes (dashboard, command-line provisioning, admin
console, server) as well as for running the 3DCNN pipeline under the pytorch-lightning framework.
The project description specifies the swarm nodes etc. to be used for a swarm training.

```bash
cd MediSwarm
./buildDockerImageAndStartupKits.sh -p application/provision/<PROJECT DESCRIPTION.yml>
```

1. Make sure you have no uncommitted changes.
2. If package versions are still not available, you may have to check what the current version is and update the
   `Dockerfile` accordingly. Version numbers are hard-coded to avoid issues due to silently different versions being
   installed.
3. After successful build (and after verifying that everything works as expected, i.e., local tests, building startup
   kits, running local trainings in the startup kit), you can manually push the image to DockerHub, provided you have
   the necessary rights. Make sure you are not re-using a version number for this purpose.

## Running Local Tests

   ```bash
   ./runTestsInDocker.sh
   ```

You should see

1. several expected errors and warnings printed from unit tests that should succeed overall, and a coverage report
2. output of a successful simulation run with two nodes
3. output of a successful proof-of-concept run run with two nodes
4. output of a set of startup kits being generated
5. output of a dummy training run using one of the startup kits
6. TODO update this to what the tests output now

Optionally, uncomment running NVFlare unit tests in `_runTestsInsideDocker.sh`.

## Distributing Startup Kits

Distribute the startup kits to the clients.

## Running the Application

1. **CIFAR-10 example:**
   See [cifar10/README.md](application/jobs/cifar10/README.md)
2. **Minimal PyTorch CNN example:**
   See [application/jobs/minimal_training_pytorch_cnn/README.md](application/jobs/minimal_training_pytorch_cnn/README.md)
3. **3D CNN for classifying breast tumors:**
   See [ODELIA_ternary_classification/README.md](application/jobs/ODELIA_ternary_classification/README.md)

## Contributing Application Code

1. Take a look at application/jobs/minimal_training_pytorch_cnn for a minimal example how pytorch code can be adapted to
   work with NVFlare
2. Take a look at application/jobs/ODELIA_ternary_classification for a more relastic example of pytorch code that can
   run in the swarm
3. Use the local tests to check if the code is swarm-ready
4. TODO more detailed instructions

# Usage for Swarm Operators

## Setting up a Swarm

Production mode is designed for secure, real-world deployments. It supports both local and remote setups, whether
on-premise or in the cloud. For more details, refer to
the [NVFLARE Production Mode](https://nvflare.readthedocs.io/en/2.4.1/real_world_fl.html).

To set up production mode, follow these steps:

## Edit `/etc/hosts`

Ensure that your `/etc/hosts` file includes the correct host mappings. All hosts need to be able to communicate to the
server node.

For example, add the following line (replace `<IP>` with the server's actual IP address):

```plaintext
<IP>    dl3.tud.de dl3
```

## Create Startup Kits

### Via Script (recommended)

1. Use, e.g., the file `application/provision/project_MEVIS_test.yml`, adapt as needed (network protocol etc.)
2. Call `buildStartupKits.sh /path/to/project_configuration.yml` to build the startup kits
3. Startup kits are generated to `workspace/<name configured in the .yml>/prod_00/`
4. Deploy startup kits to the respective server/clients

### Via the Dashboard (not recommended)

```bash
docker run -d --rm \
     --ipc=host -p 8443:8443 \
    --name=odelia_swarm_admin \
    -v /var/run/docker.sock:/var/run/docker.sock \
    <DOCKER_IMAGE> \
    /bin/bash -c "nvflare dashboard --start --local --cred <ADMIN_USER_EMAIL>:<PASSWORD>"
```

using some credentials chosen for the swarm admin account.

Access the dashboard in a web browser at `https://localhost:8443` log in with these credentials, and configure the
project:

1. enter project short name, name, description
2. enter docker download link: jefftud/odelia:<version string>
3. if needed, enter dates
4. click save
5. Server Configuration > Server (DNS name): <DNS name of server>
6. click make project public

#### Register client per site

Access the dashboard at `https://<DNS name of server>:8443`.

1. register a user
2. enter organziation (corresponding to the site)
3. enter role (e.g., org admin)
4. add a site (note: must not contain spaces, best use alphanumerical name)
5. specify number of GPUs and their memory

#### Approve clients and finish configuration

Access the dashboard at `https://localhost:8443` log in with the admin credentials.

1. Users Dashboard > approve client user
2. Client Sites > approve client sites
3. Project Home > freeze project

## Download startup kits

After setting up the project admin configuration, server and clients can download their startup kits. Store the
passwords somewhere, they are only displayed once (or you can download them again).

## Starting a Swarm Training

1. Connect the *server* host to the VPN as described above.
2. Start the *server* startup kit using the respective `startup/docker.sh` script with the option to start the server
3. Provide the *client* startup kits to the swarm participants (be aware that email providers or other channels may
   prevent encrypted archives)
4. Make sure the participants have started their clients via the respective startup kits, see below
5. Start the *admin* startup kit using the respective `startup/docker.sh` script to start the admin console
6. Deploy a job by `submit_job <job folder>`

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Maintainers

[Jeff](https://github.com/Ultimate-Storm)
[Ole Schwen](mailto:ole.schwen@mevis.fraunhofer.de)
[Steffen Renisch](mailto:steffen.renisch@mevis.fraunhofer.de)

# Contributing

Feel free to dive in! [Open an issue](https://github.com/KatherLab/MediSwarm/issues) or submit pull requests.

# Credits

This project utilizes platforms and resources from the following repositories:

- **[NVFLARE](https://github.com/NVIDIA/NVFlare)**: NVFLARE (NVIDIA Federated Learning Application Runtime Environment)
  is an open-source framework that provides a robust and scalable platform for federated learning applications. We have
  integrated NVFLARE to efficiently handle the federated learning aspects of our project.

Special thanks to the contributors and maintainers of these repositories for their valuable work and support.

---

For more details about NVFLARE and its features, please visit
the [NVFLARE GitHub repository](https://github.com/NVIDIA/NVFlare).
