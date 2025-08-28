#!/usr/bin/env bash

set -e


start_server_and_clients () {
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

kill_server_and_clients () {
    docker kill odelia_swarm_server_flserver odelia_swarm_client_client_A odelia_swarm_client_client_B
}


run_data_access_preflight_check () {
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

run_dummy_training_in_swarm () {
    cd "$PROJECT_DIR"/prod_00
    cd admin@test.odelia/startup
    ../../../../../_testsOutsideDocker_submitDummyTraining.exp
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

run_tests () {
    run_data_access_preflight_check

    start_server_and_clients
    run_dummy_training_in_swarm
    kill_server_and_clients

    cleanup_temporary_data
}

run_tests
