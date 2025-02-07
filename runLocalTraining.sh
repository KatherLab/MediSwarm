#!/usr/bin/env bash

# TODO think about how this should become part of the startup kit (considering how the code should be made available in that case)

export DATADIR=<your_data_directory>
export SCRATCHDIR=<your_data_directory>
export SITE_NAME=<your_site_name>
export NUM_EPOCHS=<number_of_epochs_in_total>


docker run --rm -it --detach-keys="ctrl-x" --gpus=all -u $(id -u):$(id -g) \
    -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v ./:/code/ \
    -v $DATADIR:/data/:ro -v $DATADIR:/scratch/ -w /code --env TRAINING_MODE='local_training' \
    --ipc=host --net=host nvflare-pt-dev:3dcnn \
    /bin/bash -c "application/jobs/3dcnn_ptl/app/custom/main_standalone.py --site_name $SITE_NAME --num_epochs $NUM_EPOCHS"
