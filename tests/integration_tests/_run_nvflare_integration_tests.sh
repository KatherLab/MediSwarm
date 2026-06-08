#!/usr/bin/env bash

prepare () {
    # update protobuf and wandb to defined versions
    pip install -U \
        protobuf==7.34.1 \
        wandb==0.26.1
    # install pytest and tensorflow + dependencies at defined versions
    pip install \
        flatbuffers==25.12.19 \
        gast==0.7.0 \
        google_pasta==0.2.0 \
        h5py==3.14.0 \
        iniconfig==2.3.0 \
        keras==3.12.1 \
        libclang==18.1.1 \
        ml_dtypes==0.5.4 \
        namex==0.1.0 \
        opt_einsum==3.4.0 \
        pluggy==1.6.0 \
        pytest==9.0.3 \
        tensorflow==2.21.0 \
        termcolor==3.3.0

    # install xgboost and nvidia_nccl_cu12, needed for xgboost test
    pip install \
        xgboost==3.2.0 \
        nvidia_nccl_cu12==2.30.4

    export PATH=~/.local/bin:$PATH
    chmod a+rwX /MediSwarm -R
}

run_nvflare_integration_tests () {
    cd /MediSwarm/docker_config/NVFlare
    cd tests/integration_test
    for backend in numpy tensorflow pytorch overseer ha auth preflight cifar auto stats xgboost client_api client_api_qa; do
        test_name="NVFlare integration test for backend $backend"
        echo "⏳ Running $test_name"
        timeout 20m ./run_integration_tests.sh -m $backend
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "✅ $test_name finished"
        elif [ $exit_code -eq 124 ]; then
            echo "❌ $test_name timed out"
        else
            echo "❌ $test_name exited with exit code $exit_code."
        fi
    done
    cd ..
}

prepare
run_nvflare_integration_tests
