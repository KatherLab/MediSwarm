#!/usr/bin/env bash

prepare () {
    pip install -U protobuf
    pip install pytest tensorflow torch
    export PATH=~/.local/bin:$PATH
    chmod a+rwX /MediSwarm -R
}

run_nvflare_integration_tests () {
    cd /MediSwarm/docker_config/NVFlare
    cd tests/integration_test
    for backend in numpy tensorflow pytorch overseer ha auth preflight cifar auto stats xgboost client_api client_api_qa; do
        echo "Running NVFlare integration tests " $backend
        ./run_integration_tests.sh -m $backend
    done
    cd ..
}

prepare
run_nvflare_integration_tests
