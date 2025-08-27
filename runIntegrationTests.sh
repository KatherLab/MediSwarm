#!/usr/bin/env bash

set -e

VERSION=$(./getVersionNumber.sh)
DOCKER_IMAGE=jefftud/odelia:$VERSION
PROJECT_DIR="workspace/odelia_${VERSION}_dummy_project_for_testing"
SYNTHETIC_DATA_DIR=$(mktemp -d)
SCRATCH_DIR=$(mktemp -d)
CWD=$(pwd)
PROJECT_FILE="tests/provision/dummy_project_for_testing.yml"
if [ -z "$GPU_FOR_TESTING" ]; then
    export GPU_FOR_TESTING="all"
fi


check_files_on_github () {
    echo "[Run] Test whether expected content is available on github"

    CONTENT=$(curl -L https://github.com/KatherLab/MediSwarm/raw/refs/heads/main/LICENSE)
    if echo "$CONTENT" | grep -q "MIT License" ; then
        echo "Downloaded and verified license from github"
    else
        echo "Could not download and verify license"
        exit 1
    fi

    CONTENT=$(curl -L https://github.com/KatherLab/MediSwarm/raw/refs/heads/main/README.md)
    for ROLE in 'Swarm Participant' 'Developer' 'Swarm Operator';
    do
        if echo "$CONTENT" | grep -q "$ROLE" ; then
            echo "Instructions for $ROLE found"
        else
            echo "Instructions for role $ROLE missing"
            exit 1
        fi
    done
}

_run_test_in_docker() {
    echo "[Run]" $1 "inside Docker ..."
    docker run --rm \
           --shm-size=16g \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -v /tmp:/scratch \
           --gpus="$GPU_FOR_TESTING" \
           --entrypoint=/MediSwarm/$1 \
           "$DOCKER_IMAGE"
}

run_local_tests () {
    _run_test_in_docker tests/integration_tests/_run_controller_unit_tests_with_coverage.sh
    _run_test_in_docker tests/integration_tests/_run_minimal_example_standalone.sh
    _run_test_in_docker tests/integration_tests/_run_minimal_example_simulation_mode.sh
    _run_test_in_docker tests/integration_tests/_run_minimal_example_proof_of_concept_mode.sh

    # uncomment the following line to also run NVFlare's unit tests (takes about 2 minutes and will install python packages in the container)
    # run_test_in_docker tests/integration_tests/_run_nvflare_unit_tests.sh
}

create_startup_kits_and_check_contained_files () {
    echo "[Prepare] Startup kits for dummy project ..."

    if ! grep -q "127.0.0.1 server.local" /etc/hosts; then
        echo "/etc/hosts needs to contain the following line, please add it."
        echo "127.0.0.1 server.local localhost"
        exit 1
    fi

    if [ ! -d "$PROJECT_DIR"/prod_00 ]; then
        ./_buildStartupKits.sh $PROJECT_FILE $VERSION
    fi
    if [ -d "$PROJECT_DIR"/prod_01 ]; then
        echo '"$PROJECT_DIR"/prod_01 exists, please remove/rename it'
        exit 1
    fi
    ./_buildStartupKits.sh $PROJECT_FILE $VERSION

    for FILE in 'client.crt' 'client.key' 'docker.sh' 'rootCA.pem';
    do
        if [ -f "$PROJECT_DIR/prod_01/client_A/startup/$FILE" ] ; then
            echo "$FILE found"
        else
            echo "$FILE missing"
            exit 1
        fi
    done

    ZIP_CONTENT=$(unzip -tv "$PROJECT_DIR/prod_01/client_B_${VERSION}.zip")
    for FILE in 'client.crt' 'client.key' 'docker.sh' 'rootCA.pem';
    do
        if echo "$ZIP_CONTENT" | grep -q "$FILE" ; then
            echo "$FILE found in zip"
        else
            echo "$FILE missing in zip"
            exit 1
        fi
    done
}

create_synthetic_data () {
    echo "[Prepare] Synthetic data ..."
    docker run --rm \
        -u $(id -u):$(id -g) \
        -v "$SYNTHETIC_DATA_DIR":/synthetic_data \
        -w /MediSwarm \
        $DOCKER_IMAGE \
        /bin/bash -c "python3 application/jobs/ODELIA_ternary_classification/app/scripts/create_synthetic_dataset/create_synthetic_dataset.py /synthetic_data"
}


run_dummy_training () {
    echo "[Run] Dummy training session..."
    cd "$PROJECT_DIR/prod_00/client_A/startup/"
    ./docker.sh --data_dir /tmp/ --scratch_dir /tmp/scratch --GPU "$GPU_FOR_TESTING" --no_pull --dummy_training
    cd "$CWD"
}

run_3dcnn_tests () {
    echo "[Run] 3D CNN preflight check..."

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

case "$1" in
    check_files_on_github) check_files_on_github ;;
    run_local_tests) run_local_tests ;;
    create_startup_kits) create_startup_kits_and_check_contained_files ;;
    create_synthetic_data) create_synthetic_data ;;
    run_dummy_training) run_dummy_training ;;
    run_3dcnn_tests) run_3dcnn_tests ;;
    cleanup) cleanup_dummy_trainings ;;
    all | "")
        check_files_on_github
        run_local_tests
        create_startup_kits_and_check_contained_files
        create_synthetic_data
        run_dummy_training
        run_3dcnn_tests
        cleanup_dummy_trainings
        ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
esac
