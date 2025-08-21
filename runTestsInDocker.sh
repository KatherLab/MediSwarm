#!/usr/bin/env bash

set -e

VERSION=$(./getVersionNumber.sh)
DOCKER_IMAGE=jefftud/odelia:$VERSION
PROJECT_DIR="workspace/odelia_${VERSION}_dummy_project_for_testing"
CWD=$(pwd)
if [ -z "$GPU_FOR_TESTING" ]; then
    export GPU_FOR_TESTING="all"
fi


run_tests () {
    echo "[Run] Unit tests inside Docker..."
    docker run --rm \
           --shm-size=16g \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -v /tmp:/scratch \
           --gpus="$GPU_FOR_TESTING" \
           --entrypoint=/MediSwarm/_runTestsInsideDocker.sh \
           "$DOCKER_IMAGE"
}

prepare_dummy_trainings () {
    echo "[Prepare] Startup kits for dummy project..."
    rm -rf "$PROJECT_DIR"
    ./_buildStartupKits.sh tests/provision/dummy_project_for_testing.yml "$VERSION"
}

run_dummy_training () {
    echo "[Run] Dummy training session..."
    cd "$PROJECT_DIR/prod_00/client_A/startup/"
    ./docker.sh --data_dir /tmp/ --scratch_dir /tmp/scratch --GPU "$GPU_FOR_TESTING" --no_pull --dummy_training
    cd "$CWD"
}

run_3dcnn_tests () {
    echo "[Run] Synthetic data + 3D CNN preflight check..."
    SYNTHETIC_DATA_DIR=$(mktemp -d)

    # create synthetic data
    docker run --rm \
        -u $(id -u):$(id -g) \
        -v "$SYNTHETIC_DATA_DIR":/synthetic_data \
        -w /MediSwarm \
        jefftud/odelia:$VERSION \
        /bin/bash -c "python3 application/jobs/ODELIA_ternary_classification/app/scripts/create_synthetic_dataset/create_synthetic_dataset.py /synthetic_data"

    # run tests using synthetic data
    cd "$PROJECT_DIR/prod_00/client_A/startup/"
    # preflight check (standalone) and swarm simulation mode
    ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir /tmp/scratch --GPU "$GPU_FOR_TESTING" --no_pull --preflight_check
    ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir /tmp/scratch --GPU "$GPU_FOR_TESTING" --no_pull --run_script /MediSwarm/_run3DdcnnptlTestsInDocker.sh

    cd "$CWD"

    # clean up synthetic data
    rm -rf "$SYNTHETIC_DATA_DIR" || echo "Warning: cleanup failed"
}


cleanup_dummy_trainings () {
    echo "[Cleanup] Removing dummy workspace..."
    rm -rf "$PROJECT_DIR"
}

check_license_listings () {
    cd "$CWD"/"$PROJECT_DIR/prod_00/admin@test.odelia/startup"
    ADMIN_LICENSES=$( ./docker.sh --no_pull --list_licenses 2>&1  )
    if ! $( echo $ADMIN_LICENSES | grep -q MIT ) || ! $( echo $ADMIN_LICENSES | grep -q "model weights" ); then
        echo "could not list licenses from admin startup kit"
        exit 1
    fi
    cd "$CWD"/"$PROJECT_DIR/prod_00/server.local/startup/"
    SERVER_LICENSES=$( ./docker.sh --no_pull --list_licenses 2>&1  )
    if ! $( echo $SERVER_LICENSES | grep -q MIT ) || ! $( echo $SERVER_LICENSES | grep -q "model weights" ); then
        echo "could not list licenses from server startup kit"
        exit 1
    fi
    cd "$CWD"/"$PROJECT_DIR/prod_00/client_A/startup/"
    CLIENT_LICENSES=$( ./docker.sh --no_pull --list_licenses 2>&1  )
    if ! $( echo $CLIENT_LICENSES | grep -q MIT ) || ! $( echo $CLIENT_LICENSES | grep -q "model weights" ); then
        echo "could not list licenses from client startup kit"
        exit 1
    fi
}

case "$1" in
    run_tests) run_tests ;;
    prepare_dummy_trainings) prepare_dummy_trainings ;;
    run_dummy_training) run_dummy_training ;;
    run_3dcnn_tests) run_3dcnn_tests ;;
    cleanup) cleanup_dummy_trainings ;;
    check_license_listings) check_license_listings;;
    all | "")
        run_tests
        prepare_dummy_trainings
        run_dummy_training
        run_3dcnn_tests
        cleanup_dummy_trainings
        check_license_listings
        ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
esac
