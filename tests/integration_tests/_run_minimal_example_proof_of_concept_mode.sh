#!/usr/bin/env bash

set -e

run_minimal_example_proof_of_concept_mode () {
    # run proof-of-concept mode for minimal example
    mkdir -p ~/.nvflare
    cd /MediSwarm
    export TRAINING_MODE="swarm"
    nvflare poc prepare -c poc_client_0 poc_client_1
    # NVFlare 2.8.0 removed `nvflare poc prepare-jobs-dir`; `nvflare job submit -j <folder>`
    # now uploads the job directly from its path, so no pre-linking step is needed.
    # NVFlare 2.8.0's `poc start` blocks on a readiness probe (default ~30s) that
    # times out in CI ("server is not reachable") even though the server does come up.
    # --no-wait restores 2.7.2 fire-and-forget; we then wait ourselves before submitting.
    nvflare poc start --no-wait -ex admin@nvidia.com
    sleep 30
    echo "Will submit job now after sleeping 15 seconds to allow the background process to complete"
    nvflare job submit -j application/jobs/minimal_training_pytorch_cnn
    sleep 60
    echo "Will shut down now after sleeping 60 seconds to allow the background process to complete"
    sleep 2
    nvflare poc stop
}

run_minimal_example_proof_of_concept_mode
