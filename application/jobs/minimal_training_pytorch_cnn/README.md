# Minimal PyTorch CNN Example

This application code is a minimal example of how CNNs implemented in PyTorch can be trained with NVFlare.
Images (2D) are generated on the fly, so that no datasets need to be read from disk.
This serves both as an example for how a local training and a swarm differ in the code and as a fast-running test case for simulation, proof-of-concept training and actual swarm training without requiring external data.

To see how this is run, see the `docker.sh` script from the startup kit for your node or `runTestsInDocker.sh` in the main MediSwarm directory.

# Running the example manually

This should be done for development and debugging purposes only.

## Start Docker Environment

To run the 3D CNN application using a Docker container, first start an interactive container

```bash
# Use the current image version, `tail -n 1 odelia_image.version` in the main MediSwarm directory
DOCKER_IMAGE=jefftud/odelia:$DOCKER_IMAGE_VERSION
# Run the Docker container
docker run -it --rm \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus=all \
    -v ./:/workspace \
    $DOCKER_IMAGE
```

* `--shm-size=16g`: Allocates 16 GB of shared memory.
* `--ulimit memlock=-1`: Removes memory lock limitations.
* `--ulimit stack=67108864`: Sets the stack size limit to 64 MB.
* Unlike in the tests, we mount the code directory into the container so that we're using the application code from the host and do not need to build a new image in each debugging step.

## Run Stand-Alone Dummy Training

Before running a swarm dummy training, first make sure the code works in non-swarm mode. For this purpose, `main.py` is written in such a way that it can be used in stand-alone and swarm mode.

```bash
cd application/jobs/minimal_training_pytorch_cnn/app/custom/
export TRAINING_MODE="local_training"
./main.py
cd /workspace
```

## Run Swarm Simulation

The FL Simulator is a lightweight tool that uses threads to simulate multiple clients. It is useful for quick local testing and debugging. Run the following command to start the simulator:

```bash
nvflare simulator -w /tmp/minimal_training_pytorch_cnn -n 2 -t 2 application/jobs/minimal_training_pytorch_cnn -c simulated_node_0,simulated_node_1
```

* `-w /tmp/minimal_training_pytorch_cnn`: Specifies the working directory.
* `-n 2`: Sets the number of clients.
* `-t 2`: Specifies the number of threads.
* `-c simulated_node_0,simulated_node_1`: Names the two simulated nodes.

For more details, refer to the [NVFLARE Quick Start with Simulator](https://nvflare.readthedocs.io/en/2.4.1/getting_started.html#quick-start-with-simulator).

## Run Proof-of-Concept Mode

Proof of Concept (POC) mode enables quick local setups on a single machine. The FL server and clients run in separate processes or Docker containers started from within the Docker container. To run POC mode:

```bash
nvflare poc prepare -c poc_client_0 poc_client_1
nvflare poc prepare-jobs-dir -j application/jobs/

# Start POC
nvflare poc start
```

For more information on POC mode, see the [NVFLARE POC Commands](https://nvflare.readthedocs.io/en/2.4.1/user_guide/nvflare_cli/poc_command.html).
