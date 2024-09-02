#!/usr/bin/env bash

# TODO do we need this at all or do we need any data for testing?
export DATADIR="$(mktemp)"
echo $DATADIR

docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
     --ulimit stack=67108864 \
    -v ./docker_config/NVFlare:/workspace/nvflare \
    --gpus=all \
    -v ./:/workspace \
    -v $DATADIR:/data:ro \
    --user $(id -u):$(id -g) \
    --entrypoint=/workspace/_runTestsInsideDocker.sh \
    nvflare-pt-dev:testing
