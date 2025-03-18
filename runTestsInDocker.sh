#!/usr/bin/env bash

VERSION=`./getVersionNumber.sh`
DOCKER_IMAGE=jefftud/odelia:$VERSION

docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
     --ulimit stack=67108864 \
    -v /tmp:/scratch \
    --gpus=all \
    --entrypoint=/MediSwarm/_runTestsInsideDocker.sh \
    $DOCKER_IMAGE
