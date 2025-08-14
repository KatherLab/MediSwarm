#!/usr/bin/env bash

set -e

if ! grep -q "127.0.0.1 server.local" /etc/hosts; then
    echo "/etc/hosts needs to contain the following line, please add it."
    echo "127.0.0.1 server.local localhost"
    exit 1
fi

if [ -z "$GPU_FOR_TESTING" ]; then
    export GPU_FOR_TESTING="all"
fi

VERSION=$(./getVersionNumber.sh)
DOCKER_IMAGE=jefftud/odelia:$VERSION
PROJECT_DIR="workspace/odelia_${VERSION}_dummy_project_for_testing"
SYNTHETIC_DATA_DIR=$(mktemp -d)
CWD=$(pwd)

check_files_on_github () {
    echo "TODO check files/documentation on github"
}

check_startup_kits () {
    echo "TODO check startup kits"
}

create_synthetic_data () {
    # create synthetic data
    docker run --rm \
        -u $(id -u):$(id -g) \
        -v "$SYNTHETIC_DATA_DIR":/synthetic_data \
        -w /MediSwarm \
        jefftud/odelia:$VERSION \
        /bin/bash -c "python3 application/jobs/ODELIA_ternary_classification/app/scripts/create_synthetic_dataset/create_synthetic_dataset.py /synthetic_data"
}

cleanup_synthetic_data () {
    rm -rf "$SYNTHETIC_DATA_DIR"
}

start_server_and_clients () {
    cd $PROJECT_DIR/prod_00
    cd server.local/startup
    ./docker.sh --no_pull --start_server
    cd ../..
    sleep 10

    cd client_A/startup
    ./docker.sh --no_pull --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir /data/MEVISTwoNodeSwarm/scratch --GPU device=$GPU_FOR_TESTING --start_client
    cd ../..
    cd client_B/startup
    ./docker.sh --no_pull --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir /data/MEVISTwoNodeSwarm/scratch --GPU device=$GPU_FOR_TESTING --start_client
    sleep 5

    cd "$CWD"
}

kill_server_and_clients () {
    docker kill odelia_swarm_server_flserver odelia_swarm_client_client_A odelia_swarm_client_client_B
}

run_docker_gpu_preflight_check () {
    echo "TODO run dummy training locally"
}

run_data_access_preflight_check () {
    echo "TODO run data access preflight check locally"
}

check_output_of_preflight_checks () {
    echo "TODO check output of preflight checks"
}

run_dummy_training_in_swarm () {
    cd $PROJECT_DIR/prod_00
    cd admin@test.odelia/startup
    ../../../../../_testsOutsideDocker_submitDummyTraining.exp
    docker kill fladmin
    sleep 60
}

check_output_of_dummy_training () {
    echo "TODO check output of dummy training"
}

run_tests () {
    check_files_on_github

    check_startup_kits

    create_synthetic_data

    run_docker_gpu_preflight_check
    run_data_access_preflight_check
    check_output_of_preflight_checks

    start_server_and_clients

    run_dummy_training_in_swarm
    check_output_of_dummy_training

    kill_server_and_clients

    cleanup_synthetic_data
}

run_tests
