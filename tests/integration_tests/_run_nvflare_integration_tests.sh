#!/usr/bin/env bash

set -e

run_nvflare_integration_tests () {
    chmod a+rwX /MediSwarm -R
    cd /MediSwarm/docker_config/NVFlare
    cd tests/integration_test
    for backend in numpy tensorflow pytorch overseer ha auth preflight cifar auto stats xgboost client_api client_api_qa; do
        run_integration_tests.sh -m $backend
    done
    cd ..
}

run_nvflare_integration_tests
