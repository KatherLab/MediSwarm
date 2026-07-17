#!/usr/bin/env bash

set -e

run_minimal_example_proof_of_concept_mode () {
    # run proof-of-concept mode for minimal example
    mkdir -p ~/.nvflare
    cd /MediSwarm
    export TRAINING_MODE="swarm"

    # NVFlare 2.8.0 CLI notes:
    #  - `poc prepare-jobs-dir` was removed; `job submit -j <folder>` uploads directly.
    #  - `poc start` blocks on a readiness probe that times out in CI even though the
    #    server comes up -> use --no-wait and wait ourselves.
    nvflare poc prepare -c poc_client_0 poc_client_1
    nvflare poc start --no-wait -ex admin@nvidia.com
    sleep 45   # let the server + clients come up (no readiness probe with --no-wait)

    echo "Submitting job ..."
    nvflare job submit -j application/jobs/minimal_training_pytorch_cnn

    # 2.8.0 runs the job in BACKGROUND service processes that log to files under the POC
    # workspace, NOT to this script's stdout (2.7.2 streamed it). Poll those logs for the
    # training completion marker, then surface them so the caller's grep check can see it.
    POC_WS=/tmp/nvflare/poc/example_project/prod_00
    MARKER="Epoch 9: 100%"
    for _ in $(seq 1 30); do          # up to ~5 min for 10 epochs
        if grep -rqsF "$MARKER" "$POC_WS" 2>/dev/null; then break; fi
        sleep 10
    done

    nvflare poc stop || true

    echo "========== POC job logs (training output) =========="
    # surface the client job logs; the caller checks for "$MARKER" in this output
    grep -rhsF "Epoch" "$POC_WS" 2>/dev/null | tail -60 || true
}

run_minimal_example_proof_of_concept_mode
