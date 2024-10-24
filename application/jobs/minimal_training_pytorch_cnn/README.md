
# Minimal PyTorch CNN Example

This application code is a minimal example of how CNNs implemented in PyTorch can be trained with NVFlare.
Images (2D) are generated on the fly, so that no datasets need to be read from disk.
This serves both as an example for how a local training and a swarm differ in the code and as a fast-running test case for simulation, proof-of-concept training and actual swarm training without requiring external data.

## 1. Run with Docker Environment

To run the 3D CNN application using a Docker container, follow these steps:

```bash
# Set the DATADIR variable
export DATADIR=<your_data_directory>
# Run the Docker container
docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ./docker_config/NVFlare:/workspace/nvflare \
    --gpus=all \
    -v ./:/workspace \
    jefftud/nvflare-pt-dev:3dcnn
```

- `--shm-size=16g`: Allocates 16 GB of shared memory.
- `--ipc=host`: Shares the IPC namespace with the host to enable better communication.
- `--ulimit memlock=-1`: Removes memory lock limitations.
- `--ulimit stack=67108864`: Sets the stack size limit to 64 MB.

## 2. Run the FL Simulator

The FL Simulator is a lightweight tool that uses threads to simulate multiple clients. It is useful for quick local testing and debugging. Run the following command to start the simulator:

```bash
nvflare simulator -w /tmp/minimal_training_pytorch_cnn -n 2 -t 2 application/jobs/minimal_training_pytorch_cnn -c simulated_node_0,simulated_node_1
```

- `-w /tmp/3dcnn_ptl`: Specifies the working directory.
- `-n 2`: Sets the number of clients.
- `-t 2`: Specifies the number of threads.
- `-c simulated_node_0,simulated_node_1`: Names the two simulated nodes.

For more details, refer to the [NVFLARE Quick Start with Simulator](https://nvflare.readthedocs.io/en/2.4.1/getting_started.html#quick-start-with-simulator).

## 3. Run POC Mode

Proof of Concept (POC) mode enables quick local setups on a single machine. The FL server and clients run in separate processes or Docker containers. To run POC mode:

```bash
nvflare poc prepare -c poc_client_0 poc_client_1
nvflare poc prepare-jobs-dir -j application/jobs/
```

Now proceed interactively by

```bash
nvflare poc start
```

For more information on POC mode, see the [NVFLARE POC Commands](https://nvflare.readthedocs.io/en/2.4.1/user_guide/nvflare_cli/poc_command.html).

## 4. Run Production Mode

See the 3dcnn_ptl example.
