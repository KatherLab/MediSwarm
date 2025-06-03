#!/usr/bin/env bash

VERSION=`./getVersionNumber.sh`
DOCKER_IMAGE=jefftud/odelia:$VERSION

run_tests () {
    docker run -it --rm \
           --shm-size=16g \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -v /tmp:/scratch \
           --gpus=all \
           --entrypoint=/MediSwarm/_runTestsInsideDocker.sh \
           $DOCKER_IMAGE
}

run_dummy_training () {
    ./_buildStartupKits.sh tests/provision/dummy_project_for_testing.yml $VERSION

    PROJECT_DIR=workspace/odelia_${VERSION}_dummy_project_for_testing
    cd $PROJECT_DIR/prod_00/client_A/startup/
    ./docker.sh  --data_dir /tmp/ --scratch_dir /tmp/scratch --GPU all --no_pull --dummy_training
    cd ../../../../../
    rm -rf $PROJECT_DIR
}

run_tests
run_dummy_training
