#!/usr/bin/env bash

set -e

run_3dcnn_simulation_mode () {
    # both clients use the same data according to SITE_NAME, there are no separate env variables from which the code could read which client it is
    # change training configuration to run 2 rounds
    cd /MediSwarm
    export TMPDIR=$(mktemp -d)
    if [[ $MODEL_NAME =~ ^[0-9] ]]; then
        # this is a challenges team members name without challenge prefix
        APP_DIR="challenge_"${MODEL_NAME}
    else
        if [[ $MODEL_NAME == challenge* ]]; then
            # already correct app folder name
            APP_DIR=${MODEL_NAME}
        else
            # probably either resnet or mst, use usual ternary folder
            APP_DIR="ODELIA_ternary_classification"
        fi
    fi
    echo "RUN "$APP_DIR 
    cp -RL application/jobs/${APP_DIR} ${TMPDIR}/${APP_DIR}
    sed -i 's/num_rounds = .*/num_rounds = 2/' ${TMPDIR}/${APP_DIR}/app/config/config_fed_server.conf
    export TRAINING_MODE="swarm"
    export SITE_NAME="client_A"
    export DATA_DIR=/data
    export SCRATCH_DIR=/scratch
    export TORCH_HOME=/torch_home
    export MODEL_NAME=${MODEL_NAME}
    export CONFIG=unilateral
    nvflare simulator -w /tmp/${APP_DIR} -n 2 -t 2 ${TMPDIR}/${APP_DIR} -c client_A,client_B
    rm -rf ${TMPDIR}
}

run_3dcnn_simulation_mode
