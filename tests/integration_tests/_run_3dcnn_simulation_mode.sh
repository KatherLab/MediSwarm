#!/usr/bin/env bash

set -e

run_3dcnn_simulation_mode () {
    # both clients use the same data according to SITE_NAME, there are no separate env variables from which the code could read which client it is
    cd /MediSwarm
    export TMPDIR=$(mktemp -d)
    export TRAINING_MODE="swarm"
    export SITE_NAME="client_A"
    export DATA_DIR=/data
    export SCRATCH_DIR=/scratch
    export TORCH_HOME=/torch_home
    # Default to a valid lightweight 3D-CNN ternary model when MODEL_NAME is
    # unset/empty. This CI smoke test should exercise ResNet18, not MST.
    export MODEL_NAME="${MODEL_NAME:-ResNet18}"
    export APP_DIR="ODELIA_ternary_classification"
    echo "RUN ${APP_DIR} with MODEL_NAME=${MODEL_NAME}"
    cp -RL application/jobs/${APP_DIR} ${TMPDIR}/${APP_DIR}
    # CI runs only two simulator clients on one GPU. Keep this as a short
    # fault-tolerance smoke: one client may fail under runner memory pressure,
    # but one good client must complete and produce a global model.
    sed -i 's/num_rounds = .*/num_rounds = 1/' ${TMPDIR}/${APP_DIR}/app/config/config_fed_server.conf
    sed -i 's/min_clients = .*/min_clients = 1/' ${TMPDIR}/${APP_DIR}/app/config/config_fed_server.conf
    sed -i '/min_clients =/a\      starting_client = "client_A"\n      result_clients = ["client_A"]\n      aggr_clients = ["client_A"]\n      train_clients = ["client_A", "client_B"]' ${TMPDIR}/${APP_DIR}/app/config/config_fed_server.conf
    sed -i 's/min_responses_required = .*/min_responses_required = 1/' ${TMPDIR}/${APP_DIR}/app/config/config_fed_client.conf
    # Production ODELIA jobs use long timeouts to ride out VPN stalls. This
    # synthetic CI simulation should fail promptly and print the simulator log.
    sed -i \
        -e 's/start_task_timeout = .*/start_task_timeout = 300/' \
        -e 's/progress_timeout = .*/progress_timeout = 600/' \
        -e 's/configure_task_timeout = .*/configure_task_timeout = 300/' \
        ${TMPDIR}/${APP_DIR}/app/config/config_fed_server.conf
    sed -i \
        -e 's/last_result_transfer_timeout = .*/last_result_transfer_timeout = 300/' \
        -e 's/external_pre_init_timeout = .*/external_pre_init_timeout = 300/' \
        -e 's/peer_read_timeout = .*/peer_read_timeout = 300/' \
        -e 's/heartbeat_timeout = .*/heartbeat_timeout = 300/' \
        -e 's/learn_task_timeout = .*/learn_task_timeout = 600/' \
        -e 's/learn_task_abort_timeout = .*/learn_task_abort_timeout = 60/' \
        -e 's/learn_task_ack_timeout = .*/learn_task_ack_timeout = 300/' \
        -e 's/final_result_ack_timeout = .*/final_result_ack_timeout = 300/' \
        -e 's/wait_time_after_min_resps_received = .*/wait_time_after_min_resps_received = 1/' \
        ${TMPDIR}/${APP_DIR}/app/config/config_fed_client.conf
    export CONFIG=unilateral
    # Keep this integration test as an end-to-end smoke test. The default swarm
    # epoch weighting is sized for real training and expands this tiny synthetic
    # dataset to 10 epochs per round, which makes CI hold GPU memory longer than
    # necessary.
    export EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-1}"
    export EPOCHS_REFERENCE_DATASET_SIZE="${EPOCHS_REFERENCE_DATASET_SIZE:-18}"
    export EPOCHS_MAX_CAP="${EPOCHS_MAX_CAP:-1}"
    export ODELIA_NUM_WORKERS="${ODELIA_NUM_WORKERS:-0}"
    export ODELIA_HASH_NUM_WORKERS="${ODELIA_HASH_NUM_WORKERS:-0}"
    local WS=/tmp/${APP_DIR}
    local LOG=${TMPDIR}/sim.log

    # `nvflare simulator` returns exit code 0 even when the federated run aborts
    # with a FATAL_SYSTEM_ERROR, so the run outcome MUST be asserted explicitly
    # (otherwise an aborted simulation silently passes CI -- #353).
    set +e
    nvflare simulator -w ${WS} -n 2 -t 2 ${TMPDIR}/${APP_DIR} -c client_A,client_B 2>&1 | tee ${LOG}
    local RC=${PIPESTATUS[0]}
    set -e

    local fail=""
    if grep -qiE "FATAL_SYSTEM_ERROR|Aborting current RUN|failed to configure clients" ${LOG}; then
        fail="run aborted (FATAL_SYSTEM_ERROR / failed to configure clients)"
    elif [ "${RC}" -ne 0 ]; then
        fail="simulator exited with code ${RC}"
    elif ! find ${WS} -name 'FL_global_model.pt' 2>/dev/null | grep -q .; then
        # a completed run persists a global model; its absence means the run did not finish
        fail="no global model (FL_global_model.pt) was produced"
    fi

    if [ -n "${fail}" ]; then
        echo "=== 3DCNN Simulation Mode FAILED: ${fail} ==="
        rm -rf ${TMPDIR} ${WS}
        exit 1
    fi

    echo "=== 3DCNN Simulation Mode PASSED ==="
    rm -rf ${TMPDIR} ${WS}
}

run_3dcnn_simulation_mode
