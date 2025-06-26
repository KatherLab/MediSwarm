#!/usr/bin/env bash

run_3dcnn_simulation_mode () {
    # both clients use the same data according to SITE_NAME, there are no separate env variables from which the code could read which client it is
    # change training configuration to run 2 rounds
    cd /MediSwarm
    export TMPDIR=$(mktemp -d)
    cp -R application/jobs/3dcnn_ptl ${TMPDIR}/3dcnn_ptl
    sed -i 's/num_rounds = .*/num_rounds = 2/' ${TMPDIR}/3dcnn_ptl/app/config/config_fed_server.conf
    export TRAINING_MODE="swarm"
    export SITE_NAME="client_A"
    nvflare simulator -w /tmp/3dcnn_ptl -n 2 -t 2 ${TMPDIR}/3dcnn_ptl -c client_A,client_B
    unset TRAINING_MODE
    unset SITE_NAME
    rm -rf ${TMPDIR}
    unset TMPDIR
}

run_3dcnn_simulation_mode
