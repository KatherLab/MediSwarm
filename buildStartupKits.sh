#!/usr/bin/env bash

docker build -t nvflare-pt-dev:nfcore . -f docker_config/Dockerfile_nfcore
docker run --rm -it -u $(id -u):$(id -g) -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v ./:/workspace/ -w /workspace/ nvflare-pt-dev:nfcore /bin/bash -c "nvflare provision -p $1"
