#!/usr/bin/env bash

set -e

prepare () {
    chmod a+rwX /MediSwarm -R
}

run_nvflare_unit_tests () {
    cd /MediSwarm/docker_config/NVFlare
    ./runtest.sh -c -r
    coverage report -m
    cd ..
}

prepare
run_nvflare_unit_tests
