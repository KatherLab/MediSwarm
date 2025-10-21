#!/usr/bin/env bash

set -e

VERSION=$(./getVersionNumber.sh)
CONTAINER_VERSION_SUFFIX=$(git rev-parse --short HEAD)
DOCKER_IMAGE=localhost:5000/odelia:$VERSION
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

    LICENSE_ON_GITHUB=$(curl -L https://github.com/KatherLab/MediSwarm/raw/refs/heads/main/LICENSE)
    if echo "$LICENSE_ON_GITHUB" | grep -q "MIT License" ; then
        echo "Downloaded and verified license from github"
    else
        echo "Could not download and verify license"
        exit 1
    fi

    MAIN_README=$(curl -L https://github.com/KatherLab/MediSwarm/raw/refs/heads/main/README.md)
    for ROLE in 'Swarm Participant' 'Developer' 'Swarm Operator';
    do
        if echo "$MAIN_README" | grep -qie "$ROLE" ; then
            echo "Instructions for $ROLE found"
        else
            echo "Instructions for role $ROLE missing"
            exit 1
        fi
    done

    PARTICIPANT_README=$(curl -L https://github.com/KatherLab/MediSwarm/raw/refs/heads/main/assets/readme/README.participant.md)
    for EXPECTED_KEYWORDS in 'Prerequisites' 'RAM' 'Ubuntu' 'VPN' 'Prepare Dataset' './docker.sh' 'Local Training' 'Start Swarm Node';
    do
        if echo "$PARTICIPANT_README" | grep -qie "$EXPECTED_KEYWORDS" ; then
            echo "Instructions on $EXPECTED_KEYWORDS found"
        else
            echo "Instructions on $EXPECTED_KEYWORDS missing"
            exit 1
        fi
    done

    SWARM_OPERATOR_README=$(curl -L https://github.com/KatherLab/MediSwarm/raw/refs/heads/main/assets/readme/README.operator.md)
    for EXPECTED_KEYWORDS in 'Create Startup Kits' 'Starting a Swarm Training';
    do
        if echo "$SWARM_OPERATOR_README" | grep -qie "$EXPECTED_KEYWORDS" ; then
            echo "Instructions on $EXPECTED_KEYWORDS found"
        else
            echo "Instructions on $EXPECTED_KEYWORDS missing"
            exit 1
        fi
    done

    APC_DEVELOPER_README=$(curl -L https://github.com/KatherLab/MediSwarm/raw/refs/heads/main/assets/readme/README.developer.md)
    for EXPECTED_KEYWORDS in 'Contributing Application Code';
    do
        if echo "$APC_DEVELOPER_README" | grep -qie "$EXPECTED_KEYWORDS" ; then
            echo "Instructions on $EXPECTED_KEYWORDS found"
        else
            echo "Instructions on $EXPECTED_KEYWORDS missing"
            exit 1
        fi
    done

    DUMMY_TRAINING_APC=$(curl -L https://raw.githubusercontent.com/KatherLab/MediSwarm/refs/heads/main/application/jobs/minimal_training_pytorch_cnn/app/custom/main.py)
    for EXPECTED_KEYWORDS in 'python3';
    do
        if echo "$DUMMY_TRAINING_APC" | grep -qie "$EXPECTED_KEYWORDS" ; then
            echo "Dummy Training ApC: $EXPECTED_KEYWORDS found"
        else
            echo "Dummy Training ApC: $EXPECTED_KEYWORDS missing"
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
           -u $(id -u):$(id -g) \
           -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group \
           -v "$SYNTHETIC_DATA_DIR":/data \
           -v "$SCRATCH_DIR":/scratch \
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
    OUTPUT_WITHOUT_GPU=$(docker run --rm \
                             --shm-size=16g \
                             --ipc=host \
                             --ulimit memlock=-1 \
                             --ulimit stack=67108864 \
                             -u $(id -u):$(id -g) \
                             -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group \
                             -v "$SYNTHETIC_DATA_DIR":/data \
                             -v "$SCRATCH_DIR":/scratch \
                             --entrypoint=/MediSwarm/tests/integration_tests/_run_minimal_example_standalone.sh \
                             "$DOCKER_IMAGE" 2>&1 || echo "")
    if echo "$OUTPUT_WITHOUT_GPU" | grep -q "RuntimeError: This example does not work without GPU" ; then
        echo "Verified that minimal example requires GPU"
    else
        echo "Failed to verify that minimal example requires GPU"
        exit 1
    fi

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

    if [ ! -d "$PROJECT_DIR"/prod_00 ]; then
        ./_buildStartupKits.sh $PROJECT_FILE $VERSION $DOCKER_IMAGE
    fi
    if [ -d "$PROJECT_DIR"/prod_01 ]; then
        echo '$PROJECT_DIR/prod_01 exists, please remove/rename it'
        exit 1
    fi
    ./_buildStartupKits.sh $PROJECT_FILE $VERSION $DOCKER_IMAGE

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
           -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group \
           -v "$SYNTHETIC_DATA_DIR":/synthetic_data \
           -w /MediSwarm \
           $DOCKER_IMAGE \
           /bin/bash -c "python3 application/jobs/ODELIA_ternary_classification/app/scripts/create_synthetic_dataset/create_synthetic_dataset.py /synthetic_data"
}


run_list_licenses () {
    cd "$PROJECT_DIR"/prod_00
    cd localhost/startup
    LICENSES_LISTED=$(./docker.sh --list_licenses --no_pull)

    for EXPECTED_KEYWORDS in 'scikit-learn' 'torch' 'nvflare_mediswarm' 'BSD License' 'MIT License';
    do
        if echo "$LICENSES_LISTED" | grep -qie "$EXPECTED_KEYWORDS" ; then
            echo "Instructions on $EXPECTED_KEYWORDS found"
        else
            echo "Instructions on $EXPECTED_KEYWORDS missing"
            exit 1
        fi
    done

    cd "$CWD"
}


run_docker_gpu_preflight_check () {
    # requires having built a startup kit
    echo "[Run] Docker/GPU preflight check (local dummy training via startup kit) ..."
    cd "$PROJECT_DIR/prod_00/client_A/startup/"
    CONSOLE_OUTPUT=docker_gpu_preflight_check_console_output.txt
    # also check that it finishes within one minute
    timeout --signal=kill 1m ./docker.sh --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --dummy_training --no_pull 2>&1 | tee "$CONSOLE_OUTPUT"

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
    ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --preflight_check --no_pull 2>&1 | tee $CONSOLE_OUTPUT

    if grep -q "Train set: 18, Val set: 6" "$CONSOLE_OUTPUT" && grep -q "Epoch 0: 100%" "$CONSOLE_OUTPUT"; then
        echo "Expected output of Docker/GPU preflight check found"
    else
        echo "Missing expected output of Docker/GPU preflight check"
        exit 1
    fi

    cd "$CWD"
}


run_3dcnn_simulation_mode () {
    # requires having built a startup kit and synthetic dataset
    echo "[Run] Simulation mode of 3DCNN training in Docker"
    _run_test_in_docker tests/integration_tests/_run_3dcnn_simulation_mode.sh
}


start_server () {
    echo "[Run] Start server Docker container ..."

    cd "$PROJECT_DIR"/prod_00
    cd localhost/startup
    ./docker.sh --no_pull --start_server
    cd ../..
    sleep 10

    cd "$CWD"
}


start_clients () {
    echo "[Run] Start client Docker containers ..."

    cd "$PROJECT_DIR"/prod_00
    cd client_A/startup
    ./docker.sh --no_pull --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --start_client
    cd ../..
    cd client_B/startup
    ./docker.sh --no_pull --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_B --GPU "$GPU_FOR_TESTING" --start_client
    sleep 8

    cd "$CWD"
}

start_server_and_clients () {
    start_server
    start_clients
}


start_registry_docker_and_push () {
    docker run -d --rm -p 5000:5000 --name local_test_registry_$CONTAINER_VERSION_SUFFIX registry:3
    sleep 3
    docker push localhost:5000/odelia:$VERSION
}


run_container_with_pulling () {
    docker rmi localhost:5000/odelia:$VERSION
    cd "$PROJECT_DIR"/prod_00
    cd localhost/startup
    OUTPUT=$(./docker.sh --list_licenses)

    if echo "$OUTPUT" | grep -qie "Status: Downloaded newer image for localhost:5000/odelia:$VERSION" ; then
        echo "Image pulled successfully"
    else
        echo "Instructions on $EXPECTED_KEYWORDS missing"
        exit 1
    fi

    cd "$CWD"
}


kill_registry_docker () {
    docker kill local_test_registry_$CONTAINER_VERSION_SUFFIX
}


verify_wrong_client_does_not_connect () {
    echo "[Run] Verify that client with outdated startup kit does not connect ..."

    cp -r "$PROJECT_DIR"/prod_01 "$PROJECT_DIR"/prod_wrong_client
    cd "$PROJECT_DIR"/prod_wrong_client
    cd localhost/startup
    ./docker.sh --no_pull --start_server
    cd ../..
    sleep 10

    rm client_A -rf
    tar xvf "$CWD"/tests/integration_tests/outdated_startup_kit.tar.gz
    sed -i 's#DOCKER_IMAGE=localhost:5000/odelia:1.0.1-dev.250919.095c1b7#DOCKER_IMAGE='$DOCKER_IMAGE'#' client_A/startup/docker.sh
    sed -i 's#CONTAINER_NAME=odelia_swarm_client_client_A_095c1b7#CONTAINER_NAME=odelia_swarm_client_client_A_'$CONTAINER_VERSION_SUFFIX'#' client_A/startup/docker.sh

    cd client_A/startup
    ./docker.sh --no_pull --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --start_client
    cd ../..

    sleep 20

    CONSOLE_OUTPUT_SERVER=localhost/startup/nohup.out
    CONSOLE_OUTPUT_CLIENT=client_A/startup/nohup.out

    if grep -q "Total clients: 1" $CONSOLE_OUTPUT_SERVER; then
        echo "Connection with non-authorized client"
        exit 1
    else
        echo "Connection rejected successfully by server"
    fi

    if grep -q "SSLCertVerificationError" $CONSOLE_OUTPUT_CLIENT; then
        echo "Connection rejected successfully by client"
    else
        echo "Could not verify that connection was rejected"
        exit 1
    fi

    docker kill odelia_swarm_server_flserver_$CONTAINER_VERSION_SUFFIX odelia_swarm_client_client_A_$CONTAINER_VERSION_SUFFIX
    sleep 3
    rm -rf "$PROJECT_DIR"/prod_wrong_client

    cd "$CWD"
}


run_dummy_training_in_swarm () {
    echo "[Run] Dummy training in swarm ..."

    cd "$PROJECT_DIR"/prod_00
    cd admin@test.odelia/startup
    "$CWD"/tests/integration_tests/_submitDummyTraining.exp
    docker kill odelia_swarm_admin_$CONTAINER_VERSION_SUFFIX
    sleep 60
    cd "$CWD"

    cd "$PROJECT_DIR"/prod_00/localhost/startup
    CONSOLE_OUTPUT=nohup.out
    for EXPECTED_OUTPUT in 'Total clients: 2' 'updated status of client client_A on round 4' 'updated status of client client_B on round 4' 'all_done=True' 'Server runner finished.' \
                           'Start to the run Job: [0-9a-f]\{8\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{12\}' 'updated status of client client_B on round 4';
    do
        if grep -q --regexp="$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT"; then
            echo "Expected output $EXPECTED_OUTPUT found"
        else
            echo "Expected output $EXPECTED_OUTPUT missing"
            exit 1
        fi
    done
    cd "$CWD"

    cd "$PROJECT_DIR"/prod_00/client_A/startup
    CONSOLE_OUTPUT=nohup.out
    for EXPECTED_OUTPUT in 'Sending training result to aggregation client' 'Epoch 9: 100%' 'val/AUC_ROC';
    do
        if grep -q "$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT"; then
            echo "Expected output $EXPECTED_OUTPUT found"
        else
            echo "Expected output $EXPECTED_OUTPUT missing"
            exit 1
        fi
    done
    cd "$CWD"

    for EXPECTED_OUTPUT in 'validation metric .* from client' 'aggregating [0-9]* update(s) at round [0-9]*';
    do
        if grep -q --regexp="$EXPECTED_OUTPUT" "$PROJECT_DIR"/prod_00/client_?/startup/nohup.out; then
            echo "Expected output $EXPECTED_OUTPUT found"
        else
            echo "Expected output $EXPECTED_OUTPUT missing"
            exit 1
        fi
    done

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

    actualsize=$(wc -c <*/app_client_A/best_FL_global_model.pt)
    if [ $actualsize -le 1048576 ]; then
        echo "Checkpoint file size OK"
    else
        echo "Checkpoint too large: " $actualsize
        exit 1
    fi

    cd "$CWD"
}


kill_server_and_clients () {
    echo "[Cleanup] Kill server and client Docker containers ..."
    docker kill odelia_swarm_server_flserver_$CONTAINER_VERSION_SUFFIX odelia_swarm_client_client_A_$CONTAINER_VERSION_SUFFIX odelia_swarm_client_client_B_$CONTAINER_VERSION_SUFFIX
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
        ;;

    run_unit_tests_controller)
        run_unit_tests_controller
        cleanup_temporary_data
        ;;

    run_dummy_training_standalone)
        run_dummy_training_standalone
        cleanup_temporary_data
        ;;

    run_dummy_training_simulation_mode)
        run_dummy_training_simulation_mode
        cleanup_temporary_data
        ;;

    run_dummy_training_poc_mode)
        run_dummy_training_poc_mode
        cleanup_temporary_data
        ;;

    run_3dcnn_simulation_mode)
        create_synthetic_data
        run_3dcnn_simulation_mode
        cleanup_temporary_data
        ;;

    create_startup_kits)
        create_startup_kits_and_check_contained_files
        cleanup_temporary_data
        ;;

    run_list_licenses)
        create_startup_kits_and_check_contained_files
        run_list_licenses
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

    push_pull_image)
        create_startup_kits_and_check_contained_files
        start_registry_docker_and_push
        run_container_with_pulling
        kill_registry_docker
        # TODO add to CI if we want this (takes several minutes)
        ;;

    check_wrong_startup_kit)
        create_startup_kits_and_check_contained_files
        create_synthetic_data
        verify_wrong_client_does_not_connect
        cleanup_temporary_data
        # TODO add to CI if we want this
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
        run_nvflare_unit_tests
        create_synthetic_data
        run_3dcnn_simulation_mode
        create_startup_kits_and_check_contained_files
        start_registry_docker_and_push
        run_container_with_pulling
        kill_registry_docker
        run_docker_gpu_preflight_check
        run_data_access_preflight_check
        verify_wrong_client_does_not_connect
        start_server_and_clients
        run_dummy_training_in_swarm
        kill_server_and_clients
        cleanup_temporary_data
        ;;

    *) echo "Unknown argument: $1"; exit 1 ;;
esac
