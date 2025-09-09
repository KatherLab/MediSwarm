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


run_unit_tests_controller(){
    echo "[Run] Controller unit tests"
    _run_test_in_docker tests/integration_tests/_run_controller_unit_tests_with_coverage.sh
}

run_dummy_training_standalone(){
    echo "[Run] Minimal example, standalone"
    _run_test_in_docker tests/integration_tests/_run_minimal_example_standalone.sh
}

run_dummy_training_simulation_mode(){
    echo "[Run] Minimal example, simulation mode"
    _run_test_in_docker tests/integration_tests/_run_minimal_example_simulation_mode.sh
}

run_dummy_training_poc_mode(){
    echo "[Run] Minimal example, proof-of-concept mode"
    _run_test_in_docker tests/integration_tests/_run_minimal_example_proof_of_concept_mode.sh
}

run_nvflare_unit_tests(){
    echo "[Run] NVFlare unit tests"
    _run_test_in_docker tests/unit_tests/_run_nvflare_unit_tests.sh
}


create_startup_kits_and_check_contained_files () {
    echo "[Prepare] Startup kits for test project ..."

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


run_docker_gpu_preflight_check () {
    # requires having built a startup kit
    echo "[Run] Docker/GPU preflight check (local dummy training via startup kit) ..."
    cd "$PROJECT_DIR/prod_00/client_A/startup/"
    CONSOLE_OUTPUT=docker_gpu_preflight_check_console_output.txt
    ./docker.sh --scratch_dir "$SCRATCH_DIR"/client_A --GPU device=$GPU_FOR_TESTING --dummy_training --no_pull 2>&1 | tee "$CONSOLE_OUTPUT"

    if grep -q "Epoch 1: 100%" "$CONSOLE_OUTPUT" && grep -q "Training completed successfully" "$CONSOLE_OUTPUT"; then
        echo "Expected output of Docker/GPU preflight check found"
    else
        echo "Missing expected output of Docker/GPU preflight check"
        exit 1
    fi

    cd "$CWD"
}


run_data_access_preflight_check () {
    # requires having built a startup kit and synthetic dataset
    echo "[Run] Data access preflight check..."
    cd "$PROJECT_DIR"/prod_00
    cd client_A/startup
    CONSOLE_OUTPUT=data_access_preflight_check_console_output.txt
    ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_A --GPU device=$GPU_FOR_TESTING --preflight_check --no_pull 2>&1 | tee $CONSOLE_OUTPUT

    if grep -q "Train set: 18, Val set: 6" "$CONSOLE_OUTPUT" && grep -q "Epoch 0: 100%" "$CONSOLE_OUTPUT"; then
        echo "Expected output of Docker/GPU preflight check found"
    else
        echo "Missing expected output of Docker/GPU preflight check"
        exit 1
    fi

    cd "$CWD"
}


start_server_and_clients () {
    echo "[Run] Start server and client Docker containers ..."

    cd "$PROJECT_DIR"/prod_00
    cd server.local/startup
    ./docker.sh --no_pull --start_server
    cd ../..
    sleep 10

    cd client_A/startup
    ./docker.sh --no_pull --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_A --GPU device=$GPU_FOR_TESTING --start_client
    cd ../..
    cd client_B/startup
    ./docker.sh --no_pull --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_B --GPU device=$GPU_FOR_TESTING --start_client
    sleep 5

    cd "$CWD"
}


run_dummy_training_in_swarm () {
    echo "[Run] Dummy training in swarm ..."

    cd "$PROJECT_DIR"/prod_00
    cd admin@test.odelia/startup
    "$CWD"/_testsOutsideDocker_submitDummyTraining.exp
    docker kill fladmin
    sleep 60
    cd "$CWD"

    cd "$PROJECT_DIR"/prod_00/server.local/startup
    CONSOLE_OUTPUT=nohup.out
    for EXPECTED_OUTPUT in 'Total clients: 2' 'updated status of client client_A on round 4' 'updated status of client client_B on round 4' 'all_done=True' 'Server runner finished.';
    do
        if grep -q "$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT"; then
            echo "Expected output $EXPECTED_OUTPUT found"
        else
            echo "Expected output $EXPECTED_OUTPUT missing"
            exit 1
        fi
    done
    cd "$CWD"

    cd "$PROJECT_DIR"/prod_00/client_A/startup
    CONSOLE_OUTPUT=nohup.out
    for EXPECTED_OUTPUT in 'Sending training result to aggregation client' 'Epoch 9: 100%' ;
    do
        if grep -q "$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT"; then
            echo "Expected output $EXPECTED_OUTPUT found"
        else
            echo "Expected output $EXPECTED_OUTPUT missing"
            exit 1
        fi
    done
    cd "$CWD"

    cd "$PROJECT_DIR"/prod_00/client_A/
    FILES_PRESENT=$(find . -type f -name "*.*")
    for EXPECTED_FILE in 'custom/minimal_training.py' 'best_FL_global_model.pt' 'FL_global_model.pt' ;
    do
        if echo "$FILES_PRESENT" | grep -q "$EXPECTED_FILE" ; then
            echo "Expected file $EXPECTED_FILE found"
        else
            echo "Expected file $EXPECTED_FILE missing"
            exit 1
        fi
    done
    cd "$CWD"
}


kill_server_and_clients () {
    echo "[Cleanup] Kill server and client Docker containers ..."
    docker kill odelia_swarm_server_flserver odelia_swarm_client_client_A odelia_swarm_client_client_B
}


cleanup_temporary_data () {
    echo "[Cleanup] Removing synthetic data, scratch directory, dummy workspace ..."
    rm -rf "$SYNTHETIC_DATA_DIR"
    rm -rf "$SCRATCH_DIR"
    rm -rf "$PROJECT_DIR"
}


case "$1" in
    check_files_on_github)
        check_files_on_github
        cleanup_temporary_data
        ;;

    run_local_tests)
        run_unit_tests_controller
        run_dummy_training_standalone
        run_dummy_training_simulation_mode
        run_dummy_training_poc_mode
        # run_nvflare_unit_tests  # uncomment to enable NVFlare unit tests
        cleanup_temporary_data
        ;;

    create_startup_kits)
        create_startup_kits_and_check_contained_files
        cleanup_temporary_data
        ;;

    run_docker_gpu_preflight_check)
        create_startup_kits_and_check_contained_files
        run_docker_gpu_preflight_check
        cleanup_temporary_data
        ;;

    run_data_access_preflight_check)
        create_startup_kits_and_check_contained_files
        create_synthetic_data
        run_data_access_preflight_check
        cleanup_temporary_data
        ;;

    run_dummy_training_in_swarm)
        create_startup_kits_and_check_contained_files
        create_synthetic_data
        start_server_and_clients
        run_dummy_training_in_swarm
        kill_server_and_clients
        cleanup_temporary_data
        ;;

    all | "")
        check_files_on_github
        run_unit_tests_controller
        run_dummy_training_standalone
        run_dummy_training_simulation_mode
        run_dummy_training_poc_mode
        # run_nvflare_unit_tests  # uncomment to enable NVFlare unit tests
        create_startup_kits_and_check_contained_files
        create_synthetic_data
        run_docker_gpu_preflight_check
        run_data_access_preflight_check
        start_server_and_clients
        run_dummy_training_in_swarm
        kill_server_and_clients
        cleanup_temporary_data
        ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
esac

# TODO adapt .github/workflows/pr-test.yaml

# The following does not work yet. It should be included in "all", in ./assets/readme/README.developer.md and in .github/workflows/pr-test.yaml once it works.
#
# run_simulation_mode_in_docker () {
#     # requires having built a startup kit and synthetic dataset
#     echo "[Run] Simulation mode of 3DCNN training in Docker"
#     _run_test_in_docker tests/integration_tests/_run_3dcnn_simulation_mode.sh
# }
#
#     run_simulation_mode_in_docker)
#         create_startup_kits_and_check_contained_files
#         create_synthetic_data
#         run_simulation_mode_in_docker
#         cleanup_temporary_data
#         ;;
