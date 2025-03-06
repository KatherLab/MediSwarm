#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: buildStartupKits.sh SWARM_PROJECT.yml"
    exit 1
fi

DOCKER_IMAGE=jefftud/odelia:1.0rc1

docker run --rm -it -u $(id -u):$(id -g) -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v ./:/workspace/ -w /workspace/ $DOCKER_IMAGE /bin/bash -c "nvflare provision -p $1"
