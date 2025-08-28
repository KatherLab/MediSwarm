#!/usr/bin/env bash

set -e

run_controller_unit_tests_with_coverage () {
    # run unit tests of ODELIA swarm learning and report coverage
    export MPLCONFIGDIR=/tmp
    cd /MediSwarm/tests/unit_tests/controller
    PYTHONPATH=/MediSwarm/controller/controller python3 -m coverage run --source=/MediSwarm/controller/controller -m unittest discover
    coverage report -m
    rm .coverage
}

run_controller_unit_tests_with_coverage
