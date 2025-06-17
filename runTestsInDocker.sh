#!/usr/bin/env bash
set -e

VERSION=$(./getVersionNumber.sh)
DOCKER_IMAGE=jefftud/odelia:$VERSION
PROJECT_DIR="workspace/odelia_${VERSION}_dummy_project_for_testing"
CWD=$(pwd)

run_tests () {
    echo "[Run] Unit tests inside Docker..."
    docker run --rm \
           --shm-size=16g \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -v /tmp:/scratch \
           --gpus=all \
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
    ./docker.sh --data_dir /tmp/ --scratch_dir /tmp/scratch --GPU all --no_pull --dummy_training
    cd "$CWD"
}

run_3dcnn_training_with_synthetic_data_preflight_check () {
    echo "[Run] Synthetic data + 3D CNN preflight check..."
    SYNTHETIC_DATA_DIR=$(mktemp -d)

    python3 application/jobs/3dcnn_ptl/app/scripts/create_synthetic_dataset/create_synthetic_dataset.py "$SYNTHETIC_DATA_DIR"

    cd "$PROJECT_DIR/prod_00/client_A/startup/"
    ./docker.sh --data_dir "$SYNTHETIC_DATA_DIR" --scratch_dir /tmp/scratch --GPU all --no_pull --preflight_check
    cd "$CWD"

    rm -rf "$SYNTHETIC_DATA_DIR"
}

cleanup_dummy_trainings () {
    echo "[Cleanup] Removing dummy workspace..."
    rm -rf "$PROJECT_DIR"
}

# 支持单步执行
case "$1" in
    run_tests) run_tests ;;
    prepare_dummy_trainings) prepare_dummy_trainings ;;
    run_dummy_training) run_dummy_training ;;
    preflight_check) run_3dcnn_training_with_synthetic_data_preflight_check ;;
    cleanup) cleanup_dummy_trainings ;;
    all | "")  # 默认运行全部
        run_tests
        prepare_dummy_trainings
        run_dummy_training
        run_3dcnn_training_with_synthetic_data_preflight_check
        cleanup_dummy_trainings
        ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
esac
