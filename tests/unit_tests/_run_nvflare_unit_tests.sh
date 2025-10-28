#!/usr/bin/env bash

set -e

run_nvflare_unit_tests () {
    chmod a+rwX /MediSwarm -R
    cd /MediSwarm/docker_config/NVFlare
    ./runtest.sh -c -r
    coverage report -m
    cd ..
}

run_nvflare_unit_tests
