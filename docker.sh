#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# docker run script for FL client
# local data directory
: ${MY_DATA_DIR:="/mnt/swarm_alpha/odelia_dataset_divided"}
# The syntax above is to set MY_DATA_DIR to /home/flcient/data if this
# environment variable is not set previously.
# Therefore, users can set their own MY_DATA_DIR with
# export MY_DATA_DIR=$SOME_DIRECTORY
# before running docker.sh

# for all gpus use line below 
GPU2USE='--gpus=all'
# for 2 gpus use line below
#GPU2USE='--gpus=2' 
# for specific gpus as gpu#0 and gpu#2 use line below
#GPU2USE='--gpus="device=0,2"'
# to use host network, use line below
NETARG="--net=host"
# FL clients do not need to open ports, so the following line is not needed.
#NETARG="-p 443:443 -p 8003:8003"
DOCKER_IMAGE=jefftud/nvflare-pt-dev:3dcnn
echo "Starting docker with $DOCKER_IMAGE"
mode="${1:--r}"
if [ $mode = "-d" ]
then
  docker run -d --rm --name=mediswarm_root $GPU2USE -u $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v $DIR/..:/workspace/ \
  -v $MY_DATA_DIR:/data/:ro -w /workspace/ --ipc=host $NETARG $DOCKER_IMAGE \
  /bin/bash -c "python -u -m nvflare.private.fed.app.client.client_train -m /workspace -s fed_client.json --set uid=mediswarm_root secure_train=true config_folder=config org=tud"
else
  docker run --rm -it --name=mediswarm_root $GPU2USE -u $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v $DIR/..:/workspace/ \
  -v $MY_DATA_DIR:/data/:ro -w /workspace/ --ipc=host $NETARG $DOCKER_IMAGE /bin/bash
fi
