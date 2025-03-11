#!/usr/bin/env bash

DOCKER_IMAGE=jefftud/odelia:`tail -n 1 odelia_image.version`

docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
     --ulimit stack=67108864 \
    -v /tmp:/scratch \
    --gpus=all \
    --entrypoint=/MediSwarm/_runTestsInsideDocker.sh \
    $DOCKER_IMAGE
