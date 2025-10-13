#!/usr/bin/env bash

set -e

run_3dcnn_simulation_mode () {
    # both clients use the same data according to SITE_NAME, there are no separate env variables from which the code could read which client it is
    # change training configuration to run 2 rounds
    cd /MediSwarm
    export TMPDIR=$(mktemp -d)
    cp -R application/jobs/ODELIA_ternary_classification ${TMPDIR}/ODELIA_ternary_classification
    sed -i 's/num_rounds = .*/num_rounds = 2/' ${TMPDIR}/ODELIA_ternary_classification/app/config/config_fed_server.conf
    export TRAINING_MODE="swarm"
    export SITE_NAME="client_A"
    export DATA_DIR=/data
    export SCRATCH_DIR=/scratch
    export TORCH_HOME=/torch_home
    export MODEL_NAME=MST
    export CONFIG=unilateral
    nvflare simulator -w /tmp/ODELIA_ternary_classification -n 2 -t 2 ${TMPDIR}/ODELIA_ternary_classification -c client_A,client_B
    rm -rf ${TMPDIR}
}

run_3dcnn_simulation_mode
