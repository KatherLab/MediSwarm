#!/usr/bin/env bash

set -e

run_minimal_example_simulation_mode () {
    # run simulation mode for minimal example
    cd /MediSwarm
    export TRAINING_MODE="swarm"
    nvflare simulator -w /tmp/minimal_training_pytorch_cnn -n 2 -t 2 application/jobs/minimal_training_pytorch_cnn -c simulated_node_0,simulated_node_1
}

run_minimal_example_simulation_mode
