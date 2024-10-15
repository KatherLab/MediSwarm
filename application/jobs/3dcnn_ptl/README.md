
# Guide to Run 3D CNN PyTorch Lightning Application

## 1. Prepare Example Data

Download example data and prepare a dataset split for two simulated nodes

```bash
# Set the DATADIR variable
export DATADIR=<your_data_directory>

envsetup_scripts/get_dataset_gdown.sh -w $DATADIR
envsetup_scripts/prepare_dataset_split_for_simulation_and_poc.sh -w $DATADIR
```

## 2. Run with Docker Environment

To run the 3D CNN application using a Docker container, follow these steps:

```bash
docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ./docker_config/NVFlare:/workspace/nvflare \
    --gpus=all \
    -v ./:/workspace \
    -v $DATADIR:/data \
    jefftud/nvflare-pt-dev:3dcnn
```

- `DATADIR`: Set this variable to point to your local data directory.
- `--shm-size=16g`: Allocates 16 GB of shared memory.
- `--ipc=host`: Shares the IPC namespace with the host to enable better communication.
- `--ulimit memlock=-1`: Removes memory lock limitations.
- `--ulimit stack=67108864`: Sets the stack size limit to 64 MB.

## 3. Run the FL Simulator

The FL Simulator is a lightweight tool that uses threads to simulate multiple clients. It is useful for quick local testing and debugging. Run the following command to start the simulator:

```bash
nvflare simulator -w /tmp/3dcnn_ptl -n 2 -t 2 application/jobs/3dcnn_ptl -c simulated_node_0,simulated_node_1
```

- `-w /tmp/3dcnn_ptl`: Specifies the working directory.
- `-n 2`: Sets the number of clients.
- `-t 2`: Specifies the number of threads.
- `-c simulated_node_0,simulated_node_1`: Selects the manual data loaders.

For more details, refer to the [NVFLARE Quick Start with Simulator](https://nvflare.readthedocs.io/en/2.4.1/getting_started.html#quick-start-with-simulator).

## 4. Run POC Mode

Proof of Concept (POC) mode enables quick local setups on a single machine. The FL server and clients run in separate processes or Docker containers. To run POC mode:

```bash
nvflare poc prepare -c simulated_node_0 simulated_node_1

nvflare poc prepare-jobs-dir -j application/jobs/

# Start POC
nvflare poc start -ex admin@nvidia.com
# wait until this has finished
nvflare job submit -j application/jobs/3dcnn_ptl
# wait until this has finished
nvflare poc stop
```

For more information on POC mode, see the [NVFLARE POC Commands](https://nvflare.readthedocs.io/en/2.4.1/user_guide/nvflare_cli/poc_command.html).

For more information on POC mode, see the [NVFLARE POC Commands](https://nvflare.readthedocs.io/en/2.4.1/user_guide/nvflare_cli/poc_command.html).

Alternatively, using Docker in Docker, start with

```bash
nvflare poc prepare -c simulated_node_0 simulated_node_1 -d jefftud/nvflare-pt-dev:3dcnn
```
and proceed as above.

## 5. Run Production Mode

Production mode is designed for secure, real-world deployments. It supports both local and remote setups, whether on-premise or in the cloud. For more details, refer to the [NVFLARE Production Mode](https://nvflare.readthedocs.io/en/2.4.1/real_world_fl.html).

To set up production mode, follow these steps:

### Edit `/etc/hosts`

Ensure that your `/etc/hosts` file includes the correct host mappings. For example, add the following line (replace `<IP>` with the server's actual IP address):

```plaintext
<IP>    dl3.tud.de dl3
```

### Start the production setup

```bash
docker run -it --rm \
    --ipc=host \
    -v ./docker_config/NVFlare:/workspace/nvflare \
    -v ./:/workspace \
    -v /var/run/docker.sock:/var/run/docker.sock \
    jefftud/nvflare-pt-dev:nfcore \
    /bin/bash

nvflare dashboard --port 443 --start -i jefftud/nvflare-pt-dev:nfcore
```

Access the dashboard at `https://localhost:443`. After setting up the project admin configuration, clients can download their startup kits and run the following commands to join the server:

```bash
./docker.sh
cd startup
./start.sh
```

The admin can submit jobs and initiate training by logging in with the admin email and submitting the job folder. For more information, refer to the [NVFLARE Dashboard](https://nvflare.readthedocs.io/en/2.4.1/user_guide/dashboard_ui.html).
