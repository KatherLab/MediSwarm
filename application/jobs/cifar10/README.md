
# Minimum example cifar10 Guide

## 1. Run with Docker Environment
To run the project using a Docker container, execute the following command:

```bash
# Start the Docker container
docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ./docker_config/NVFlare:/workspace/nvflare \
    --gpus=all \
    -v ./:/workspace \
    jefftud/nvflare-pt-dev:cifar10
```

- `--shm-size=16g`: Allocates shared memory.
- `--ipc=host`: Shares IPC namespace with the host.
- `--ulimit memlock=-1`: Removes memory locking limits.
- `--ulimit stack=67108864`: Increases the stack size limit.

## 2. Prepare CIFAR-10 Dataset
To prepare the data splits for CIFAR-10, run the following script:

```bash
./application/jobs/cifar10/prepare_data.sh
```

## 3. Run the FL Simulator
The Federated Learning (FL) Simulator is a lightweight tool that uses threads to simulate multiple clients. It’s useful for local testing and debugging. To run the simulator, use the following command:

```bash
nvflare simulator ./application/jobs/cifar10 -w /tmp/nvflare/cifar10 -n 2 -t 2
```

- `-w /tmp/nvflare/cifar10`: Specifies the workspace directory.
- `-n 2`: Sets the number of clients.
- `-t 2`: Defines the number of threads.

For detailed instructions on how to use the simulator, refer to the official [NVFLARE Quick Start with Simulator Guide](https://nvflare.readthedocs.io/en/2.4.1/getting_started.html#quick-start-with-simulator).