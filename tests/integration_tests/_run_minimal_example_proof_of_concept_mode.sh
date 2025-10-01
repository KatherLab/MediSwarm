#!/usr/bin/env bash

set -e

run_minimal_example_proof_of_concept_mode () {
    # run proof-of-concept mode for minimal example
    mkdir -p ~/.nvflare
    cd /MediSwarm
    export TRAINING_MODE="swarm"
    nvflare poc prepare -c poc_client_0 poc_client_1
    nvflare poc prepare-jobs-dir -j application/jobs/
    nvflare poc start -ex admin@nvidia.com
    sleep 15
    echo "Will submit job now after sleeping 15 seconds to allow the background process to complete"
    nvflare job submit -j application/jobs/minimal_training_pytorch_cnn
    sleep 60
    echo "Will shut down now after sleeping 60 seconds to allow the background process to complete"
    sleep 2
    nvflare poc stop
}

run_minimal_example_proof_of_concept_mode
