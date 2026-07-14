#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/ci/_error_filter.sh
source "$SCRIPT_DIR/_error_filter.sh"

VERSION=$("$REPO_ROOT/scripts/build/getVersionNumber.sh")
CONTAINER_VERSION_SUFFIX=$(git rev-parse --short HEAD)
DOCKER_IMAGE=localhost:5000/odelia:$VERSION
STAMP_DOCKER_IMAGE=${STAMP_DOCKER_IMAGE:-stamp-test:build-test}
PROJECT_DIR="workspace/odelia_${VERSION}_dummy_project_for_testing"
SYNTHETIC_DATA_DIR=$(mktemp -d)
STAMP_SYNTHETIC_DATA_DIR=$(mktemp -d)
SCRATCH_DIR=$(mktemp -d)
CWD=$(pwd)
PROJECT_FILE="tests/provision/dummy_project_for_testing.yml"
if [ -z "$GPU_FOR_TESTING" ]; then
    export GPU_FOR_TESTING="all"
fi


check_files_in_repo () {
    echo "[Run] Test whether expected content is available in the repo"
    echo "TODO: change this back to checking files curled from github when repository is public again"

    LICENSE_LOCAL=$(cat "$CWD/LICENSE")
    if grep -q "MIT License" <<< "$LICENSE_LOCAL" ; then
        echo "✅ Verified license in repo"
    else
        echo "❌ Could not verify license"
        exit 1
    fi

    MAIN_README=$(cat "$CWD/README.md")
    for ROLE in 'Swarm Participant' 'Developer' 'Swarm Operator';
    do
        if grep -qi "$ROLE" <<< "$MAIN_README" ; then
            echo "✅ Instructions for $ROLE found"
        else
            echo "❌ Instructions for role $ROLE missing"
            exit 1
        fi
    done

    PARTICIPANT_README=$(cat "$CWD/assets/readme/README.participant.md")
    for EXPECTED_KEYWORDS in 'Prerequisites' 'RAM' 'Ubuntu' 'VPN' 'Prepare Dataset' './docker.sh' 'Local Training' 'Start Swarm Node' 'Output files';
    do
        if grep -qi "$EXPECTED_KEYWORDS" <<< "$PARTICIPANT_README" ; then
            echo "✅ Instructions on $EXPECTED_KEYWORDS found"
        else
            echo "❌ Instructions on $EXPECTED_KEYWORDS missing"
            exit 1
        fi
    done

    SWARM_OPERATOR_README=$(cat "$CWD/assets/readme/README.operator.md")
    for EXPECTED_KEYWORDS in 'Create Startup Kits' 'Starting a Swarm Training';
    do
        if grep -qi "$EXPECTED_KEYWORDS" <<< "$SWARM_OPERATOR_README" ; then
            echo "✅ Instructions on $EXPECTED_KEYWORDS found"
        else
            echo "❌ Instructions on $EXPECTED_KEYWORDS missing"
            exit 1
        fi
    done

    APC_DEVELOPER_README=$(cat "$CWD/assets/readme/README.developer.md")
    for EXPECTED_KEYWORDS in 'Contributing Application Code';
    do
        if grep -qi "$EXPECTED_KEYWORDS" <<< "$APC_DEVELOPER_README" ; then
            echo "✅ Instructions on $EXPECTED_KEYWORDS found"
        else
            echo "❌ Instructions on $EXPECTED_KEYWORDS missing"
            exit 1
        fi
    done

    DUMMY_TRAINING_APC=$(cat "$CWD/application/jobs/minimal_training_pytorch_cnn/app/custom/main.py")
    for EXPECTED_KEYWORDS in 'python3';
    do
        if grep -qi "$EXPECTED_KEYWORDS" <<< "$DUMMY_TRAINING_APC"; then
            echo "✅ Dummy Training ApC: $EXPECTED_KEYWORDS found"
        else
            echo "❌ Dummy Training ApC: $EXPECTED_KEYWORDS missing"
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
           -e MODEL_NAME=${MODEL_NAME:-} \
           --gpus="$GPU_FOR_TESTING" \
           --entrypoint=/MediSwarm/$1 \
           "$DOCKER_IMAGE"
}


_run_stamp_test_in_docker() {
    echo "[Run]" $1 "inside STAMP Docker ..."
    docker run --rm \
           --shm-size=16g \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -u $(id -u):$(id -g) \
           -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group \
           -v "$STAMP_SYNTHETIC_DATA_DIR":/data \
           -v "$SCRATCH_DIR":/scratch \
           --gpus="$GPU_FOR_TESTING" \
           --entrypoint=/MediSwarm/$1 \
           "$STAMP_DOCKER_IMAGE"
}


create_synthetic_stamp_data () {
    echo "[Prepare] Synthetic STAMP data ..."
    docker run --rm \
           -u $(id -u):$(id -g) \
           -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group \
           -v "$STAMP_SYNTHETIC_DATA_DIR":/data \
           -w /MediSwarm \
           "$STAMP_DOCKER_IMAGE" \
           /bin/bash -c "python3 application/jobs/STAMP_classification/app/scripts/create_synthetic_dataset/create_synthetic_stamp_dataset.py /data"
}


run_stamp_preflight_check () {
    echo "[Run] STAMP preflight check"
    _run_stamp_test_in_docker tests/integration_tests/stamp/_run_stamp_preflight_check.sh

    echo "TODO check whether test has succeeded"
}


run_stamp_local_training () {
    echo "[Run] STAMP local training"
    _run_stamp_test_in_docker tests/integration_tests/stamp/_run_stamp_local_training.sh

    echo "TODO check whether test has succeeded"
}


run_stamp_simulation_mode () {
    echo "[Run] STAMP simulation mode"
    _run_stamp_test_in_docker tests/integration_tests/stamp/_run_stamp_simulation_mode.sh

    echo "TODO check whether test has succeeded"
}


run_dummy_training_standalone(){
    echo "[Run] Minimal example, standalone"

    OUTPUT=$(docker run --rm \
                    --shm-size=16g \
                    --ipc=host \
                    --ulimit memlock=-1 \
                    --ulimit stack=67108864 \
                    -u $(id -u):$(id -g) \
                    -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group \
                    -v "$SYNTHETIC_DATA_DIR":/data \
                    -v "$SCRATCH_DIR":/scratch \
                    --gpus="$GPU_FOR_TESTING" \
                    --entrypoint=/MediSwarm/tests/integration_tests/_run_minimal_example_standalone.sh \
                    "$DOCKER_IMAGE" 2>&1 || echo "")

    if grep -q "✓ MediSwarm test running dummy training in standalone mode completed." <<< "$OUTPUT" ; then
        echo "✅ Running" $1 "in Docker completed."
    else
        echo "❌ Running" $1 "in Docker failed."
        exit 1
    fi

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

    if grep -q "RuntimeError: This example does not work without GPU" <<< "$OUTPUT_WITHOUT_GPU"; then
        echo "✅ Verified that minimal example requires GPU"
    else
        echo "❌ Failed to verify that minimal example requires GPU"
        exit 1
    fi
}

run_dummy_training_simulation_mode(){
    echo "[Run] Minimal example, simulation mode (capturing output)"

    OUTPUT=$(_run_test_in_docker tests/integration_tests/_run_minimal_example_simulation_mode.sh 2>&1)

    echo "$OUTPUT"

    if grep -qi "Epoch 9: 100%" <<< "$OUTPUT" && ! has_real_error "$OUTPUT"; then
        echo "✅ Minimal example simulation mode succeeded."
    else
        echo "❌ Minimal example simulation mode failed."
        exit 1
    fi
}

run_dummy_training_poc_mode(){
    echo "[Run] Minimal example, proof-of-concept mode (capturing output)"

    OUTPUT=$(_run_test_in_docker tests/integration_tests/_run_minimal_example_proof_of_concept_mode.sh 2>&1)

    echo "$OUTPUT"

    if grep -qi "Epoch 9: 100%" <<< "$OUTPUT" && ! has_real_error "$OUTPUT"; then
        echo "✅ Minimal example proof-of-concept mode succeeded."
    else
        echo "❌ Minimal example proof-of-concept mode failed."
        exit 1
    fi
}

run_nvflare_unit_tests(){
    echo "[Run] NVFlare unit tests"
    docker run --rm \
           --shm-size=16g \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           --gpus="$GPU_FOR_TESTING" \
           --entrypoint=/MediSwarm/tests/unit_tests/_run_nvflare_unit_tests.sh \
           "$DOCKER_IMAGE"
}

run_nvflare_integration_tests(){
    echo "[Run] NVFlare integration tests"
    docker run --rm \
           --shm-size=16g \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           --gpus="$GPU_FOR_TESTING" \
           --entrypoint=/MediSwarm/tests/integration_tests/_run_nvflare_integration_tests.sh \
           "$DOCKER_IMAGE"
}

create_startup_kits_and_check_contained_files () {
    echo "[Prepare] Startup kits for test project ..."

    if [ ! -d "$PROJECT_DIR"/prod_00 ]; then
        ./scripts/build/_buildStartupKits.sh $PROJECT_FILE $VERSION $DOCKER_IMAGE
    fi
    if [ -d "$PROJECT_DIR"/prod_01 ]; then
        echo '$PROJECT_DIR/prod_01 exists, please remove/rename it'
        exit 1
    fi
    ./scripts/build/_buildStartupKits.sh $PROJECT_FILE $VERSION $DOCKER_IMAGE

    for FILE in 'client.crt' 'client.key' 'docker.sh' 'rootCA.pem';
    do
        if [ -f "$PROJECT_DIR/prod_01/client_A/startup/$FILE" ] ; then
            echo "✅ $FILE found"
        else
            echo "❌ $FILE missing"
            exit 1
        fi
    done

    DOCKER_SCRIPT="$PROJECT_DIR/prod_01/client_A/startup/docker.sh"
    for EXPECTED_OUTPUT in "\-\-dummy_training"  \
                           "\-\-preflight_check" \
                           "\-\-local_training"  \
                           "\-\-start_client"    \
                           "\-\-list_licenses"   ;
    do
        if grep -q "$EXPECTED_OUTPUT" "$DOCKER_SCRIPT" ; then
            echo "✅ Expected option $EXPECTED_OUTPUT for docker.sh in client startup kit found"
        else
            cat "$DOCKER_SCRIPT"
            echo "❌ Expected option $EXPECTED_OUTPUT missing in docker.sh in client startup kit"
            exit 1
        fi
    done

    ZIP_CONTENT=$(unzip -tv "$PROJECT_DIR/prod_01/client_B_${VERSION}.zip")
    for FILE in 'client.crt' 'client.key' 'docker.sh' 'rootCA.pem';
    do
        if grep -q "$FILE" <<< "$ZIP_CONTENT"; then
            echo "✅ $FILE found in zip"
        else
            echo "❌ $FILE missing in zip"
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
    # the output has mixed line endings, remove CRs
    cd "$CWD"/"$PROJECT_DIR/prod_00/admin@test.odelia/startup"
    ADMIN_LICENSES=$( ./docker.sh --no_pull --list_licenses  | sed 's/\r//g' )
    cd "$CWD"/"$PROJECT_DIR/prod_00/localhost/startup/"
    SERVER_LICENSES=$( ./docker.sh --no_pull --list_licenses | sed 's/\r//g' )
    cd "$CWD"/"$PROJECT_DIR/prod_00/client_A/startup/"
    CLIENT_LICENSES=$( ./docker.sh --no_pull --list_licenses | sed 's/\r//g' )
    cd "$CWD"

    for license_output in "$ADMIN_LICENSES" "$SERVER_LICENSES" "$CLIENT_LICENSES";
    do
        for expected_keyword in 'scikit-learn' 'torch' 'nvflare_mediswarm' 'BSD License' 'MIT License' 'model weights';
        do
            if grep -q "$expected_keyword" <<< "$license_output"; then
                echo "✅ License check: $expected_keyword found"
            else
                echo "❌ License check: $expected_keyword missing"
                exit 1
            fi
        done
    done
}


run_docker_gpu_preflight_check () {
    # requires having built a startup kit
    echo "[Run] Docker/GPU preflight check (local dummy training via startup kit) ..."
    cd "$PROJECT_DIR/prod_00/client_A/startup/"
    CONSOLE_OUTPUT_FILE=docker_gpu_preflight_check_console_output.txt
    # also check that it finishes within one minute
    timeout --signal=kill 1m ./docker.sh --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --dummy_training --no_pull 2>&1 | tee "$CONSOLE_OUTPUT_FILE"

    if grep -q "Epoch 1: 100%" "$CONSOLE_OUTPUT_FILE" && grep -q "Training completed successfully" "$CONSOLE_OUTPUT_FILE"; then
        echo "✅ Expected output of Docker/GPU preflight check found"
    else
        echo "❌ Missing expected output of Docker/GPU preflight check"
        exit 1
    fi

    cd "$CWD"
}


run_two_containers_in_parallel () {
    # requires having built a startup kit
    echo "[Run] Starting two containers in parallel (local dummy training via startup kit) ..."
    cd "$PROJECT_DIR/prod_00/client_A/startup/"
    CONSOLE_OUTPUT_FILE=docker_gpu_preflight_check_console_output.txt
    timeout --signal=kill 1m ./docker.sh --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --dummy_training --no_pull 2>&1 | tee "$CONSOLE_OUTPUT_FILE" &
    sleep 1

    CONSOLE_OUTPUT_FILE_A=docker_gpu_preflight_check_console_output_a.txt
    timeout --signal=kill 1m ./docker.sh --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --dummy_training --no_pull --container_name MediSwarmODELIATestSecondContainer 2>&1 | tee "$CONSOLE_OUTPUT_FILE_A" &
    sleep 60

    if grep -q "Epoch 1: 100%" "$CONSOLE_OUTPUT_FILE_A" && grep -q "Training completed successfully" "$CONSOLE_OUTPUT_FILE_A"; then
        echo "✅ Expected output of running two containers in parallel found"
    else
        echo "❌ Missing expected output of running two containers in parallel"
        exit 1
    fi

    cd "$CWD"
}


run_data_access_preflight_check () {
    # requires having built a startup kit and synthetic dataset
    echo "[Run] Data access preflight check with unproblematic dataset ..."
    cd "$PROJECT_DIR"/prod_00
    cd client_A/startup
    CONSOLE_OUTPUT_FILE=data_access_preflight_check_console_output.txt
    # Data access assertions do not depend on the default challenge model. Use
    # the lightweight 3D-CNN path so this check stays focused on dataset access
    # and logging rather than challenge-model startup time.
    timeout --signal=kill 5m ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --job ODELIA_ternary_classification --model_name ResNet18 --preflight_check --no_pull 2>&1 | tee $CONSOLE_OUTPUT_FILE

    for EXPECTED_OUTPUT in "Epoch 0: 100%"                                            \
                           "INFO:threedcnn_ptl:Run directory"                         \
                           "INFO:threedcnn_ptl:Run name"                              \
                           "INFO:threedcnn_ptl:Total samples in training set:"        \
                           "INFO:threedcnn_ptl:Total samples in validation set:"      \
                           "INFO:threedcnn_ptl:Total samples in test set:"            \
                           "INFO:threedcnn_ptl:Samples in .* set of class .: . (.*%)" \
                           "INFO:threedcnn_ptl:Total samples in test set:"            ;
    do
        if grep -q --regexp="$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT_FILE"; then
            echo "✅ Expected output $EXPECTED_OUTPUT of data access preflight check with unproblematic dataset found"
        else
            cat "$CONSOLE_OUTPUT_FILE"
            echo "❌ Missing expected output $EXPECTED_OUTPUT of data access preflight check with unproblematic dataset"
            exit 1
        fi
    done

    if grep -q  "ID_0" "$CONSOLE_OUTPUT_FILE" ; then
        echo "❌ Unexpected output (UIDs) of data access preflight check without logging dataset details found"
        exit 1
    else
        echo "✅ No unexpected output of data access preflight check without logging dataset details found"
    fi

    for UNEXPECTED_OUTPUT in "WARNING:threedcnn_ptl:No Samples of class 2 in test set, please make sure this was intended."                                \
                             "ERROR:threedcnn_ptl:Duplicate image data detected. This should not happen."                                                  \
                             "ERROR:threedcnn_ptl:Duplicates among all split UIDs detected, they should be unique"                                         \
                             "WARNING:threedcnn_ptl:UIDs in split but not in images detected, make sure this was intended."                                \
                             "WARNING:threedcnn_ptl:UIDs in images but not in split detected, make sure this was intended."                                \
                             "WARNING:threedcnn_ptl:UIDs in annotation but not in images detected, make sure this was intended."                           \
                             "WARNING:threedcnn_ptl:UIDs in images but not in annotation detected, make sure this was intended."                           \
                             "ERROR:threedcnn_ptl:Entries in training∩validation detected, they should be in one set only."                                \
                             "ERROR:threedcnn_ptl:Entries in training∩test detected, they should be in one set only."                                      \
                             "ERROR:threedcnn_ptl:Entries in validation∩test detected, they should be in one set only."                                    \
                             "ERROR:threedcnn_ptl:UIDs among training data present that do not end in _left or _right, this should not happen."            \
                             "WARNING:threedcnn_ptl:UIDs with among training data _left present and _right missing detected, make sure this was intended." \
                             "WARNING:threedcnn_ptl:UIDs with among training data _right present and _left missing detected, make sure this was intended." \
                             "ERROR:threedcnn_ptl:UIDs for the same exam found in training and validation, this should not happen."                        ;
    do
        if grep -q "$UNEXPECTED_OUTPUT" "$CONSOLE_OUTPUT_FILE"; then
            cat "$CONSOLE_OUTPUT_FILE"
            echo "❌ Unexpected output $UNEXPECTED_OUTPUT of data access preflight check with unproblematic dataset found"
            exit 1
        fi
    done
    echo "✅ No unexpected output of data access preflight check with unproblematic dataset found"

    # wait for container to be cleaned up
    sleep 5
    cd "$CWD"
}


run_data_access_preflight_check_with_problems () {
    # requires having built a startup kit and synthetic dataset
    echo "[Run] Data access preflight check with problematic dataset ..."
    cd "$PROJECT_DIR"/prod_00
    cd client_P/startup
    CONSOLE_OUTPUT_FILE=data_access_preflight_check_console_output.txt
    # timeout may kill epoch before it is finished, this test is only about logging before the epoch is started
    timeout --signal=kill 1m ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_P --GPU "$GPU_FOR_TESTING" --job ODELIA_ternary_classification --model_name ResNet18 --preflight_check --no_pull 2>&1 | tee $CONSOLE_OUTPUT_FILE

    for EXPECTED_OUTPUT in "WARNING:threedcnn_ptl:No Samples of class 2 in test set, please make sure this was intended."                                \
                           "ERROR:threedcnn_ptl:Duplicate image UIDs detected. This should not happen."                                                  \
                           "ERROR:threedcnn_ptl:Duplicate image data detected. This should not happen."                                                  \
                           "ERROR:threedcnn_ptl:Duplicates among all split UIDs detected, they should be unique"                                         \
                           "WARNING:threedcnn_ptl:UIDs in split but not in images detected, make sure this was intended."                                \
                           "WARNING:threedcnn_ptl:UIDs in images but not in split detected, make sure this was intended."                                \
                           "WARNING:threedcnn_ptl:UIDs in annotation but not in images detected, make sure this was intended."                           \
                           "WARNING:threedcnn_ptl:UIDs in images but not in annotation detected, make sure this was intended."                           \
                           "ERROR:threedcnn_ptl:Entries in training∩validation detected, they should be in one set only."                                \
                           "ERROR:threedcnn_ptl:Entries in training∩test detected, they should be in one set only."                                      \
                           "ERROR:threedcnn_ptl:Entries in validation∩test detected, they should be in one set only."                                    \
                           "ERROR:threedcnn_ptl:UIDs among training data present that do not end in _left or _right, this should not happen."            \
                           "WARNING:threedcnn_ptl:UIDs with among training data _left present and _right missing detected, make sure this was intended." \
                           "WARNING:threedcnn_ptl:UIDs with among training data _right present and _left missing detected, make sure this was intended." \
                           "ERROR:threedcnn_ptl:UIDs for the same exam found in training and validation, this should not happen."                        ;
    do
        if grep -q "$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT_FILE"; then
            echo "✅ Expected output $EXPECTED_OUTPUT (including expected warnings and errors) of data access preflight check with problematic dataset found"
        else
            cat "$CONSOLE_OUTPUT_FILE"
            echo "❌ Expected output $EXPECTED_OUTPUT missing in data access preflight check with problematic dataset"
            exit 1
        fi
    done

    if grep -q  "ID_0" "$CONSOLE_OUTPUT_FILE" ; then
        echo "❌ Unexpected output of data access preflight check with problematic dataset without logging dataset details found"
        exit 1
    else
        echo "✅ Output of data access preflight check with problematic dataset contains no unexpected UIDs."
    fi

    # wait for container to be cleaned up
    sleep 5
    cd "$CWD"
}


run_data_access_preflight_check_with_problems_log_details () {
    # requires having built a startup kit and synthetic dataset
    echo "[Run] Data access preflight check with logging dataset details..."
    cd "$PROJECT_DIR"/prod_00

    cd client_P/startup
    CONSOLE_OUTPUT_FILE=data_access_preflight_check_console_output.txt
    # timeout may kill epoch before it is finished, this test is only about logging before the epoch is started
    timeout --signal=kill 1m ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_P --GPU "$GPU_FOR_TESTING" --job ODELIA_ternary_classification --model_name ResNet18 --preflight_check --log_dataset_details --no_pull 2>&1 | tee $CONSOLE_OUTPUT_FILE

    for EXPECTED_OUTPUT in "INFO:threedcnn_ptl:All training data image UIDs, UIDs with hashes:"                                                       \
                           "INFO:threedcnn_ptl:All validation data image UIDs, UIDs with hashes:"                                                     \
                           "INFO:threedcnn_ptl:All test data image UIDs, UIDs with hashes:"                                                           \
                           "INFO:threedcnn_ptl:All training ∪ validation ∪ test data image UIDs, UIDs with hashes:"                                   \
                           "ID_016_left .* appears 3 times"                                                                                           \
                           "Image data with hash .* appears 6 times: ID_000_left, ID_016_left, ID_016_right, ID_998_right, ID_999_left, SomeUID_both" \
                           "WARNING:threedcnn_ptl:Difference split.images: ID_017_left, ID_017_right"                                                 \
                           "WARNING:threedcnn_ptl:Difference images.split: ID_014_left, ID_014_right"                                                 \
                           "WARNING:threedcnn_ptl:Difference annotation.images: ID_017_left, ID_017_right"                                            \
                           "WARNING:threedcnn_ptl:Difference images.annotation: ID_014_left, ID_014_right"                                            \
                           "ERROR:threedcnn_ptl:Entries in training∩validation: ID_016_left, ID_016_right"                                            \
                           "ERROR:threedcnn_ptl:Entries in training∩test: ID_016_left, ID_016_right"                                                  \
                           "ERROR:threedcnn_ptl:Entries in validation∩test: ID_016_left, ID_016_right"                                                \
                           "WARNING:threedcnn_ptl:For the following UIDs among training data, _left is present and _right is missing: ID_999"         \
                           "WARNING:threedcnn_ptl:For the following UIDs among training data, _right is present and _left is missing: ID_998"         \
                           "ERROR:threedcnn_ptl:The following UIDs among training data do not end in _left or _right: SomeUID_both"                   \
                           "ERROR:threedcnn_ptl:The following exams are present in training and validation: ID_016"                                   ;
    do
        if grep -q --regexp="$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT_FILE"; then
            echo "✅ Expected output $EXPECTED_OUTPUT (including expected warnings and errors) of data access preflight check with problematic dataset and logged details found"
        else
            cat "$CONSOLE_OUTPUT_FILE"
            echo "❌ Expected output $EXPECTED_OUTPUT missing in data access preflight check with problematic dataset and logged details"
            exit 1
        fi
    done

    # wait for container to be cleaned up
    sleep 5
    cd "$CWD"
}


run_data_access_preflight_check_without_data () {
    # requires having built a startup kit and _not_ having a synthetic dataset
    echo "[Run] Data access preflight check with logging dataset details..."
    cd "$PROJECT_DIR"/prod_00
    cd client_P/startup
    CONSOLE_OUTPUT_FILE=data_access_preflight_check_console_output.txt
    # also check that it finishes the single round within one minute
    timeout --signal=kill 15s ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_P --GPU "$GPU_FOR_TESTING" --preflight_check --log_dataset_details --no_pull 2>&1 | tee $CONSOLE_OUTPUT_FILE

    if grep -Eq "No such file or directory: '/data/client_P/metadata_unilateral/(annotation|split)\.csv'" "$CONSOLE_OUTPUT_FILE" ; then
        echo "✅ Expected error output of data access preflight check found if no data is present"
    else
        echo "❌ Missing expected output of data access preflight check found if no data is present"
        exit 1
    fi

    cd "$CWD"
}


run_3dcnn_simulation_mode () {
    # requires having built a startup kit and synthetic dataset
    echo "[Run] Simulation mode of 3DCNN training in Docker (capturing output)"

    if ! OUTPUT=$(_run_test_in_docker tests/integration_tests/_run_3dcnn_simulation_mode.sh 2>&1); then
        echo "$OUTPUT"
        echo "❌ 3DCNN simulation mode failed."
        exit 1
    fi

    echo "$OUTPUT"

    if grep -q "=== 3DCNN Simulation Mode PASSED ===" <<< "$OUTPUT"; then
        echo "✅ 3DCNN simulation mode succeeded."
    else
        echo "❌ Missing 3DCNN simulation success marker."
        exit 1
    fi
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
    sleep 10
    docker push localhost:5000/odelia:$VERSION
}


run_container_with_pulling () {
    docker rmi localhost:5000/odelia:$VERSION
    cd "$PROJECT_DIR"/prod_00
    cd localhost/startup
    OUTPUT=$(./docker.sh --list_licenses)

    if grep -qi "Status: Downloaded newer image for localhost:5000/odelia:$VERSION" <<< "$OUTPUT"; then
        echo "✅ Image pulled successfully"
    else
        echo "❌ Instructions on $EXPECTED_KEYWORDS missing"
        exit 1
    fi

    cd "$CWD"
}


kill_registry_docker () {
    docker kill local_test_registry_$CONTAINER_VERSION_SUFFIX
}


verify_wrong_certificates_are_rejected () {
    echo "[Run] Verify that client and admin console with invalid certificate in startup kit do not connect ..."

    # start server
    cp -r "$PROJECT_DIR"/prod_01 "$PROJECT_DIR"/prod_wrong_client
    cd "$PROJECT_DIR"/prod_wrong_client
    cd localhost/startup
    ./docker.sh --no_pull --start_server
    cd ../..
    sleep 10

    # inject invalid certificates from outdated startup kits
    rm client_A -rf
    rm admin@test.odelia/ -rf
    tar xvf "$CWD"/tests/integration_tests/outdated_startup_kit.tar.gz
    sed -i 's#DOCKER_IMAGE=localhost:5000/odelia:1.0.1-dev.250919.095c1b7#DOCKER_IMAGE='$DOCKER_IMAGE'#' client_A/startup/docker.sh
    sed -i 's#CONTAINER_NAME=odelia_swarm_client_client_A_095c1b7#CONTAINER_NAME=odelia_swarm_client_client_A_'$CONTAINER_VERSION_SUFFIX'#' client_A/startup/docker.sh
    sed -i 's#DOCKER_IMAGE=localhost:5000/odelia:1.0.1-dev.251023.e940002#DOCKER_IMAGE='$DOCKER_IMAGE'#' admin@test.odelia/startup/docker.sh
    sed -i 's#CONTAINER_NAME=odelia_swarm_admin_e940002#CONTAINER_NAME=odelia_swarm_admin_'$CONTAINER_VERSION_SUFFIX'#' admin@test.odelia/startup/docker.sh

    # start client and verify that it gets rejected
    cd client_A/startup
    ./docker.sh --no_pull --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --start_client
    cd ../..

    sleep 20

    CONSOLE_OUTPUT_FILE_SERVER=localhost/startup/nohup.out
    CONSOLE_OUTPUT_FILE_CLIENT=client_A/startup/nohup.out

    if grep -q "Total clients: 1" $CONSOLE_OUTPUT_FILE_SERVER; then
        echo "❌ Could not verify that connection to unauthorized client was rejected"
        exit 1
    else
        echo "✅ Connection to unauthorized client rejected successfully by server"
    fi

    if grep -q "SSLCertVerificationError" $CONSOLE_OUTPUT_FILE_CLIENT; then
        echo "✅ Connection to unauthorized server rejected successfully by client"
    else
        echo "❌ Could not verify that connection to unauthorized server was rejected"
        exit 1
    fi

    # start admin console and verify that it gets rejected
    cd admin@test.odelia/startup
    CONSOLE_OUTPUT_FILE_ADMIN=$("$CWD"/tests/integration_tests/_attemptAdminConsoleLogin.exp)
    if grep -q "Communication Error - please try later" <<< "$CONSOLE_OUTPUT_FILE_ADMIN"; then
        echo "✅ Connection by unauthorized admin console rejected successfully"
    else
        echo "❌ Connection with non-authorized admin console"
        exit 1
    fi
    cd ../..

    # cleanup
    docker kill odelia_swarm_server_flserver_$CONTAINER_VERSION_SUFFIX odelia_swarm_client_client_A_$CONTAINER_VERSION_SUFFIX
    sleep 3
    rm -rf "$PROJECT_DIR"/prod_wrong_client

    cd "$CWD"
}


run_dummy_training_in_swarm () {
    echo "[Run] Dummy training in swarm (polling for completion, up to 5 minutes) ..."

    cd "$PROJECT_DIR"/prod_00
    cd admin@test.odelia/startup
    expect -f "$CWD"/tests/integration_tests/_submitDummyTraining.exp
    docker kill odelia_swarm_admin_$CONTAINER_VERSION_SUFFIX
    cd "$CWD"

    # Poll for completion instead of a fixed sleep.  The server log will
    # contain "Server runner finished." once all rounds are done.
    #
    # #388: 30 attempts (300s) was too tight. The self-hosted runner shares a host
    # with image builds and deploy tests, and the observed failures show the swarm
    # still making progress ("Round 2 started", "Contribution ... ACCEPTED") when we
    # gave up — i.e. we were timing out on a *slow* run, not a hung one, and then
    # failing the assertions below. Poll for up to 15 min; a genuinely hung run still
    # fails, just later. Override with CI_SWARM_WAIT_ATTEMPTS if needed.
    local server_log="$PROJECT_DIR/prod_00/localhost/startup/nohup.out"
    local max_attempts=${CI_SWARM_WAIT_ATTEMPTS:-90}
    local attempt=0
    echo "  Waiting for swarm training to finish (checking every 10s, max ${max_attempts}0s) ..."
    while [ $attempt -lt $max_attempts ]; do
        if [ -f "$server_log" ] && grep -q 'Server runner finished\.' "$server_log" 2>/dev/null; then
            echo "  ✅ Server runner finished detected after $((attempt * 10))s"
            break
        fi
        attempt=$((attempt + 1))
        sleep 10
    done
    if [ $attempt -eq $max_attempts ]; then
        echo "  ⚠️  Timed out after ${max_attempts}0s waiting for swarm completion — proceeding to assertions"
    fi

    # check for expected output in server log (clients joined, job ID assigned, 5 rounds, start of round logged, finished training logged)
    cd "$PROJECT_DIR"/prod_00/localhost/startup
    CONSOLE_OUTPUT_FILE=nohup.out
    for EXPECTED_OUTPUT in 'Client: New client client_A.* joined.*' \
                           'Client: New client client_B.* joined.*' \
                           'Client: New client client_.* joined. Sent token: .* Total clients: 1' \
                           'Client: New client client_.* joined. Sent token: .* Total clients: 2' \
                           'Job: [0-9a-f\-]\+ started to run, status changed to RUNNING.' \
                           'updated status of client client_A on round 4: .* action=start_learn_task, all_done=False' \
                           'updated status of client client_B on round 4: .* action=start_learn_task, all_done=False' \
                           'all_done=True' \
                           'Server runner finished.';
    do
        if grep -q --regexp="$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT_FILE"; then
            echo "✅ Expected output $EXPECTED_OUTPUT found"
        else
            cat "$CONSOLE_OUTPUT_FILE"
            echo "❌ Expected output $EXPECTED_OUTPUT missing"
            exit 1
        fi
    done
    cd "$CWD"

    # check for expected output in client log
    cd "$PROJECT_DIR"/prod_00/client_A/startup
    CONSOLE_OUTPUT_FILE=combined_nohup.out
    cat nohup.out ../../client_B/startup/nohup.out > $CONSOLE_OUTPUT_FILE
    for EXPECTED_OUTPUT in 'sending training result to aggregation client' \
                           'Epoch 9: 100%' \
                           'val/AUC_ROC' \
                           'validation metric .* from client' \
                           'aggregating [0-9]* update(s) at round [0-9]*' \
                           'Successfully registered client:client_A for project' \
                           'Got engine after .* seconds' \
                           'Got the new primary SP:' \
                           'accepted learn request from client_.' \
                           'Contribution from client_. ACCEPTED by the aggregator at round .' \
                           'broadcasting learn task of round . to .*; aggregation happens on client_.';
    do
        if grep -q --regexp="$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT_FILE"; then
            echo "✅ Expected output $EXPECTED_OUTPUT found"
        else
            cat "$CONSOLE_OUTPUT_FILE"
            echo "❌ Expected output $EXPECTED_OUTPUT missing"
            exit 1
        fi
    done
    cd "$CWD"

    cd "$PROJECT_DIR"/prod_00/client_A/
    FILES_PRESENT=$(find . -type f -name "*.*")
    for EXPECTED_FILE in 'custom/minimal_training.py' 'best_FL_global_model.pt' 'FL_global_model.pt' ;
    do
        if grep -q "$EXPECTED_FILE" <<< "$FILES_PRESENT"; then
            echo "✅ Expected file $EXPECTED_FILE found"
        else
            echo "❌ Expected file $EXPECTED_FILE missing"
            exit 1
        fi
    done

    actualsize=$(wc -c <*/app_client_A/best_FL_global_model.pt)
    if [ $actualsize -le 1048576 ]; then
        echo "✅ Checkpoint file size OK"
    else
        echo "❌ Checkpoint too large: " $actualsize
        exit 1
    fi

    cd "$CWD"
}


kill_server_and_clients () {
    echo "[Cleanup] Kill server and client Docker containers if running ..."
    docker kill odelia_swarm_server_flserver_$CONTAINER_VERSION_SUFFIX odelia_swarm_client_client_A_$CONTAINER_VERSION_SUFFIX odelia_swarm_client_client_B_$CONTAINER_VERSION_SUFFIX || true
}


run_3dcnn_local_training () {
    # requires having built a startup kit and synthetic dataset
    echo "[Run] 3DCNN local training..."
    cd "$PROJECT_DIR"/prod_00
    cd client_A/startup
    CONSOLE_OUTPUT_FILE=local_training_console_output.txt
    timeout --signal=kill 60m ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --local_training --no_pull 2>&1 | tee $CONSOLE_OUTPUT_FILE

    if grep -q "Epoch 99: 100%" "$CONSOLE_OUTPUT_FILE" && grep -q "Training completed successfully" "$CONSOLE_OUTPUT_FILE"; then
        echo "✅ Expected output of 3DCNN local training found"
    else
        echo "❌ Missing expected output of 3DCNN local training"
        exit 1
    fi

    cd "$CWD"
}


run_3dcnn_training_in_swarm () {
    echo "[Run] 3DCNN training in swarm (polling for completion, up to 60 minutes) ..."

    cd "$PROJECT_DIR"/prod_00
    cd admin@test.odelia/startup
    expect -f "$CWD"/tests/integration_tests/_submit3DCNNTraining.exp
    docker kill odelia_swarm_admin_$CONTAINER_VERSION_SUFFIX
    cd "$CWD"

    # Poll for completion instead of a fixed sleep.  The server log will
    # contain "Server runner finished." once all rounds are done.  We check
    # every 30 seconds for up to 60 minutes (120 iterations).
    local server_log="$PROJECT_DIR/prod_00/localhost/startup/nohup.out"
    local max_attempts=120
    local attempt=0
    echo "  Waiting for 3DCNN swarm training to finish (checking every 30s, max 60min) ..."
    while [ $attempt -lt $max_attempts ]; do
        if [ -f "$server_log" ] && grep -q 'Server runner finished\.' "$server_log" 2>/dev/null; then
            echo "  ✅ Server runner finished detected after $((attempt * 30))s"
            break
        fi
        attempt=$((attempt + 1))
        sleep 30
    done
    if [ $attempt -eq $max_attempts ]; then
        echo "  ⚠️  Timed out after 60min waiting for 3DCNN swarm completion — proceeding to assertions"
    fi

    # check for expected output in server log (clients joined, job ID assigned, 20 rounds)
    cd "$PROJECT_DIR"/prod_00/localhost/startup
    CONSOLE_OUTPUT_FILE=nohup.out
    for EXPECTED_OUTPUT in 'updated status of client client_A on round 19: .* action=start_learn_task, all_done=False' \
                           'updated status of client client_B on round 19: .* action=start_learn_task, all_done=False' \
                           'all_done=True';
    do
        if grep -q --regexp="$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT_FILE"; then
            echo "✅ Expected output $EXPECTED_OUTPUT found"
        else
            cat "$CONSOLE_OUTPUT_FILE"
            echo "❌ Expected output $EXPECTED_OUTPUT missing"
            exit 1
        fi
    done
    cd "$CWD"

    # check for expected output in client log
    cd "$PROJECT_DIR"/prod_00/client_A/startup
    CONSOLE_OUTPUT_FILE=nohup.out
    for EXPECTED_OUTPUT in 'sending training result to aggregation client' \
                           'Epoch 99: 100%';
    do
        if grep -q --regexp="$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT_FILE"; then
            echo "✅ Expected output $EXPECTED_OUTPUT found"
        else
            cat "$CONSOLE_OUTPUT_FILE"
            echo "❌ Expected output $EXPECTED_OUTPUT missing"
            exit 1
        fi
    done
    cd "$CWD"

    cd "$PROJECT_DIR"/prod_00/client_A/
    FILES_PRESENT=$(find . -type f -name "*.*")
    for EXPECTED_FILE in 'custom/threedcnn_ptl.py' 'FL_global_model.pt' ;
    do
        if grep -q "$EXPECTED_FILE" <<< "$FILES_PRESENT"; then
            echo "✅ Expected file $EXPECTED_FILE found"
        else
            echo "❌ Expected file $EXPECTED_FILE missing"
            exit 1
        fi
    done
}


_verify_challenge_preflight_check() {
    MODEL_NAME=$1
    EXPECTED_OUTPUT=$2
    CONSOLE_OUTPUT_FILE=preflight_check_console_output_$MODEL_NAME.txt
    timeout --signal=kill 5m ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir "$SCRATCH_DIR"/client_A --GPU "$GPU_FOR_TESTING" --job $MODEL_NAME --preflight_check --no_pull 2>&1 | tee $CONSOLE_OUTPUT_FILE

    if grep -q "$EXPECTED_OUTPUT" "$CONSOLE_OUTPUT_FILE"; then
        echo "✅ Expected output of $MODEL_NAME preflight check found"
    else
        echo "❌ Missing expected output of $MODEL_NAME preflight check"
        exit 1
    fi
    sleep 5
}


run_all_models_preflight_check () {
    # requires having built a startup kit and synthetic dataset
    echo "[Run] 3DCNN local training..."
    cd "$PROJECT_DIR"/prod_00
    cd client_A/startup
    CONSOLE_OUTPUT_FILE=preflight_check_console_output.txt

    _verify_challenge_preflight_check "challenge_1DivideAndConquer"   "3 | model   | ResidualEncoderClsNetwork"
    _verify_challenge_preflight_check "challenge_2BCN_AIM"            "3 | backbone | SwinUNETRMultiTask"
    _verify_challenge_preflight_check "challenge_3agaldran"           "3 | backbone | Wrapper"
    _verify_challenge_preflight_check "challenge_4abmil"              "3 | backbone | ABMIL_Swin"
    _verify_challenge_preflight_check "challenge_5pimed"              "3 | backbone | Resnet"
    _verify_challenge_preflight_check "ODELIA_ternary_classification" "3 | mst     | _MST"

    cd "$CWD"
}


cleanup_synthetic_data () {
    echo "[Cleanup] Removing synthetic data ..."
    rm -rf "$SYNTHETIC_DATA_DIR"/*
}


# Helper: remove a directory that may contain root-owned files created by
# Docker containers.  Falls back to a disposable Alpine container when the
# regular `rm` fails (which happens on the CI runner because NVFlare runs
# as root inside its container and bind-mounts the workspace).
_rm_rf () {
    local dir="$1"
    [ -z "$dir" ] && return 0
    [ ! -e "$dir" ] && return 0
    rm -rf "$dir" 2>/dev/null || \
        docker run --rm -v "$(cd "$(dirname "$dir")" && pwd)":/ws alpine \
            rm -rf "/ws/$(basename "$dir")"
}

cleanup_temporary_data () {
    echo "[Cleanup] Removing synthetic data directory, scratch directory, dummy workspace ..."
    _rm_rf "$SYNTHETIC_DATA_DIR"
    _rm_rf "$STAMP_SYNTHETIC_DATA_DIR"
    _rm_rf "$SCRATCH_DIR"
    _rm_rf "$PROJECT_DIR"
}

trap cleanup_temporary_data EXIT


case "$1" in
    check_files_in_repo)
        # check_files_in_repo
        echo "❗ checking files in repository currently disabled"
        ;;

    run_nvflare_unit_tests)
        run_nvflare_unit_tests
        ;;

    run_nvflare_integration_tests)
        run_nvflare_integration_tests
        ;;

    run_nvflare_unit_and_integration_tests)
        # TODO add to CI or "all" section if we want this (takes several minutes and fails for insufficient GPU memory)
        run_nvflare_unit_tests
        run_nvflare_integration_tests
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

    run_two_containers_in_parallel)
        create_startup_kits_and_check_contained_files
        run_two_containers_in_parallel
        cleanup_temporary_data
        ;;

    run_data_access_preflight_check)
        create_startup_kits_and_check_contained_files
        create_synthetic_data
        run_data_access_preflight_check
        run_data_access_preflight_check_with_problems
        run_data_access_preflight_check_with_problems_log_details
        cleanup_synthetic_data
        run_data_access_preflight_check_without_data
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
        verify_wrong_certificates_are_rejected
        cleanup_temporary_data
        # TODO add to CI if we want this
        ;;

    run_dummy_training_in_swarm)
        create_startup_kits_and_check_contained_files
        start_server_and_clients
        run_dummy_training_in_swarm
        kill_server_and_clients
        cleanup_temporary_data
        ;;

    run_3dcnn_local_training)
        create_startup_kits_and_check_contained_files
        create_synthetic_data
        run_3dcnn_local_training
        cleanup_temporary_data
        ;;

    run_3dcnn_training_in_swarm)
        create_startup_kits_and_check_contained_files
        create_synthetic_data
        start_server_and_clients
        run_3dcnn_training_in_swarm
        kill_server_and_clients
        cleanup_temporary_data
        ;;

    run_all_models_preflight_check)
        create_synthetic_data
        run_all_models_preflight_check
        cleanup_temporary_data
        ;;

    run_stamp_preflight_check)
        create_synthetic_stamp_data
        run_stamp_preflight_check
        cleanup_temporary_data
        ;;

    run_stamp_local_training)
        create_synthetic_stamp_data
        run_stamp_local_training
        cleanup_temporary_data
        ;;

    run_stamp_simulation_mode)
        create_synthetic_stamp_data
        run_stamp_simulation_mode
        cleanup_temporary_data
        ;;

    run_stamp_all)
        create_synthetic_stamp_data
        run_stamp_preflight_check
        run_stamp_local_training
        run_stamp_simulation_mode
        cleanup_temporary_data
        ;;

    kill_server_and_clients)
        kill_server_and_clients
        ;;

    all | "")
        # check_files_in_repo
        echo "❗ checking files in repository currently disabled"
        run_dummy_training_standalone
        run_dummy_training_simulation_mode
        run_dummy_training_poc_mode
        create_synthetic_data
        run_3dcnn_simulation_mode
        create_startup_kits_and_check_contained_files
        start_registry_docker_and_push
        run_container_with_pulling
        kill_registry_docker
        run_list_licenses
        run_docker_gpu_preflight_check
        run_data_access_preflight_check
        run_data_access_preflight_check_log_details
        cleanup_synthetic_data
        run_data_access_preflight_check_without_data
        create_synthetic_data
        run_3dcnn_local_training
        verify_wrong_certificates_are_rejected
        start_server_and_clients
        run_dummy_training_in_swarm
        run_3dcnn_training_in_swarm
        kill_server_and_clients
        cleanup_temporary_data
        ;;

    *) echo "Unknown argument: $1"; exit 1 ;;
esac
