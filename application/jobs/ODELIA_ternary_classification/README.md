# Guide to Run 3D CNN PyTorch Lightning Application
* These instructions are only needed for debugging and development purposes.
* For normal operation (swarm training, local training), use the scripts provided via the respective startup kit for your node.

## 1. Run with Docker Environment

To run the 3D CNN application using a Docker container, follow these steps:

```bash
# Set the DATADIR variable
export DATADIR=<your local data directory>
# Use the current image version, `tail -n 1 odelia_image.version` in the main MediSwarm directory
DOCKER_IMAGE=jefftud/odelia:$DOCKER_IMAGE_VERSION
# Run the Docker container
docker run -it --rm \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus=all \
    -v ./:/workspace \
    -v $DATADIR:/data \
    $DOCKER_IMAGE
```

* `--shm-size=16g`: Allocates 16 GB of shared memory.
* `--ulimit memlock=-1`: Removes memory lock limitations.
* `--ulimit stack=67108864`: Sets the stack size limit to 64 MB.
* Unlike in the tests, we mount the code directory into the container so that we're using the application code from the host and do not need to build a new image in each debugging step.

## Run Stand-Alone Training

Before running a swarm dummy training, first make sure the code works in non-swarm mode.

```bash
cd application/jobs/ODELIA_ternary_classification/app/custom/
export TRAINING_MODE="local_training"
export SITE_NAME=<your site name, i.e., the subfolder of $DATADIR where your data is located>
export NUM_EPOCHS=1
./main.py
cd /workspace
```

## Run Swarm Simulation

The FL Simulator is a lightweight tool that uses threads to simulate multiple clients. It is useful for quick local testing and debugging. Run the following command to start the simulator:

```bash
nvflare simulator -w /tmp/ODELIA_ternary_classification -n 2 -t 2 application/jobs/ODELIA_ternary_classification -c simulated_node_0,simulated_node_1
```

* `-w /tmp/ODELIA_ternary_classification`: Specifies the working directory.
* `-n 2`: Sets the number of clients.
* `-t 2`: Specifies the number of threads.
* `-c simulated_node_0,simulated_node_1`: Names the two simulated nodes.

For more details, refer to the [NVFLARE Quick Start with Simulator](https://nvflare.readthedocs.io/en/2.4.1/getting_started.html#quick-start-with-simulator).

## Run Proof-of-Concept Mode

Proof of Concept (POC) mode enables quick local setups on a single machine. The FL server and clients run in separate processes or Docker containers. To run POC mode:

```bash
nvflare poc prepare -c poc_client_0 poc_client_1
nvflare poc prepare-jobs-dir -j application/jobs/

# Start POC
nvflare poc start
```

For more information on POC mode, see the [NVFLARE POC Commands](https://nvflare.readthedocs.io/en/2.4.1/user_guide/nvflare_cli/poc_command.html).

TODO It should also be possible to run the proof of concept mode in separate Docker containers, requiring Docker in Docker. `nvflare poc prepare -c poc_client_0 poc_client_1 -d <name of the Docker image>`. This is currently not working.
