#!/usr/bin/env bash

set -e

run_3dcnn_simulation_mode () {
    # both clients use the same data according to SITE_NAME, there are no separate env variables from which the code could read which client it is
    # change training configuration to run 2 rounds
    cd /MediSwarm
    export TMPDIR=$(mktemp -d)
    export TRAINING_MODE="swarm"
    export SITE_NAME="client_A"
    export DATA_DIR=/data
    export SCRATCH_DIR=/scratch
    export TORCH_HOME=/torch_home
    export MODEL_NAME="MST"
    export APP_DIR="ODELIA_ternary_classification"
    cp -RL application/jobs/${APP_DIR} ${TMPDIR}/${APP_DIR}
    sed -i 's/num_rounds = .*/num_rounds = 2/' ${TMPDIR}/${APP_DIR}/app/config/config_fed_server.conf
    sed -i 's/min_clients = .*/min_clients = 2/' ${TMPDIR}/${APP_DIR}/app/config/config_fed_server.conf
    sed -i 's/min_responses_required = .*/min_responses_required = 2/' ${TMPDIR}/${APP_DIR}/app/config/config_fed_client.conf
    export CONFIG=unilateral
    nvflare simulator -w /tmp/${APP_DIR} -n 2 -t 2 ${TMPDIR}/${APP_DIR} -c client_A,client_B
    rm -rf ${TMPDIR}
}

run_3dcnn_simulation_mode
