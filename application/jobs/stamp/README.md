# STAMP Guide

## 1. Run with Docker Environment
To run the project using a Docker container, execute the following command:

```bash
export DATASET_DIR=/mnt/swarm_beta/stamp_test_data/data/1/
# Start the Docker container
docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ./docker_config/NVFlare:/workspace/nvflare \
    --gpus=all \
    -v ./:/workspace \
    -v $DATASET_DIR:/data \
    jefftud/nvflare-stamp-dev:v1.1.1
```

- `--shm-size=16g`: Allocates shared memory.
- `--ipc=host`: Shares IPC namespace with the host.
- `--ulimit memlock=-1`: Removes memory locking limits.
- `--ulimit stack=67108864`: Increases the stack size limit.

## 2. Run the FL Simulator
The Federated Learning (FL) Simulator is a lightweight tool that uses threads to simulate multiple clients. It’s useful for local testing and debugging. To run the simulator, use the following command:

```bash
nvflare simulator ./application/jobs/stamp -w /tmp/nvflare/stamp -n 2 -t 2
```

- `-w /tmp/nvflare/stamp`: Specifies the workspace directory.
- `-n 2`: Sets the number of clients.
- `-t 2`: Defines the number of threads.

For detailed instructions on how to use the simulator, refer to the official [NVFLARE Quick Start with Simulator Guide](https://nvflare.readthedocs.io/en/2.4.1/getting_started.html#quick-start-with-simulator).