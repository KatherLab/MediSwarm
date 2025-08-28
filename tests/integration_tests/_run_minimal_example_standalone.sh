#!/usr/bin/env bash

set -e

run_minimal_example_standalone () {
    # run standalone version of minimal example
    cd /MediSwarm/application/jobs/minimal_training_pytorch_cnn/app/custom/
    export TRAINING_MODE="local_training"
    ./main.py
}

run_minimal_example_standalone
