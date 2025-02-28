#!/usr/bin/env bash

DOCKER_IMAGE=nvflare-pt-dev:odelia

# docker build -t $DOCKER_IMAGE . -f docker_config/Dockerfile_nfcore
docker run --rm -it -u $(id -u):$(id -g) -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v ./:/workspace/ -w /workspace/ $DOCKER_IMAGE /bin/bash -c "nvflare provision -p $1"
