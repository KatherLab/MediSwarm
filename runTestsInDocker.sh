#!/usr/bin/env bash

VERSION=`./getVersionNumber.sh`
DOCKER_IMAGE=jefftud/odelia:$VERSION
PROJECT_DIR=workspace/odelia_${VERSION}_dummy_project_for_testing
CWD=`pwd`

run_tests () {
    docker run --rm \
           --shm-size=16g \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -v /tmp:/scratch \
           --gpus=all \
           --entrypoint=/MediSwarm/_runTestsInsideDocker.sh \
           $DOCKER_IMAGE
}

prepare_dummy_trainings () {
    rm $PROJECT_DIR -rf
    ./_buildStartupKits.sh tests/provision/dummy_project_for_testing.yml $VERSION
}

run_dummy_training () {
    cd $PROJECT_DIR/prod_00/client_A/startup/
    ./docker.sh  --data_dir /tmp/ --scratch_dir /tmp/scratch --GPU all --no_pull --dummy_training
    cd "$CWD"
}

run_3dcnn_training_with_synthetic_data_preflight_check () {
    SYNTHETIC_DATA_DIR=`mktemp -d`

    application/jobs/3dcnn_ptl/app/scripts/create_synthetic_dataset/create_synthetic_dataset.py "$SYNTHETIC_DATA_DIR"

    cd $PROJECT_DIR/prod_00/client_A/startup/
    ./docker.sh  --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir /tmp/scratch --GPU all --no_pull --preflight_check
    rm -rf "$SYNTHETIC_DATA_DIR"
    cd "$CWD"
}

cleanup_dummy_trainings () {
    rm $PROJECT_DIR -rf
}


run_tests
prepare_dummy_trainings
run_dummy_training
run_3dcnn_training_with_synthetic_data_preflight_check
cleanup_dummy_trainings
