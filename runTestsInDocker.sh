#!/usr/bin/env bash

docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
     --ulimit stack=67108864 \
    -v /tmp:/scratch \
    --gpus=all \
    --entrypoint=/MediSwarm/_runTestsInsideDocker.sh \
    nvflare-pt-dev:odelia
