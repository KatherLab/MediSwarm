#!/usr/bin/env bash

docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
     --ulimit stack=67108864 \
    -v ./docker_config/NVFlare:/workspace/nvflare \
    --gpus=all \
    -v ./:/workspace \
    --user $(id -u):$(id -g) \
    --entrypoint=/workspace/_runTestsInsideDocker.sh \
    nvflare-pt-dev:testing
