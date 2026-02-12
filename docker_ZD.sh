#!/usr/bin/env bash
# docker run script for FL client with proper env variable forwarding
# Auto disable TTY in non-interactive CI environments
if [ -t 1 ]; then
    TTY_OPT="-it"
else
    echo "[INFO] No interactive terminal detected, disabling TTY."
    TTY_OPT=""
fi

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir)        MY_DATA_DIR="$2"; shift ;;
        --scratch_dir)     MY_SCRATCH_DIR="$2"; shift ;;
        --GPU)             GPU2USE="$2"; shift ;;
        --no_pull)         NOPULL="1" ;;
        --dummy_training)  DUMMY_TRAINING="1" ;;
        --preflight_check) PREFLIGHT_CHECK="1" ;;
        --local_training)  LOCAL_TRAINING="1" ;;
        --start_client)    START_CLIENT="1" ;;
        --interactive)     INTERACTIVE="1" ;;
        --run_script)      SCRIPT_TO_RUN="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Prompt for parameters if missing
if [[ -z "$DUMMY_TRAINING" && -z "$MY_DATA_DIR" ]]; then
    read -p "Enter the path to your data directory (default: /home/flclient/data): " user_data_dir
    : ${MY_DATA_DIR:="${user_data_dir:-/home/flclient/data}"}
fi

if [ -z "$MY_SCRATCH_DIR" ]; then
    read -p "Enter the path to your scratch directory (default: /mnt/scratch): " user_scratch_dir
    : ${MY_SCRATCH_DIR:="${user_scratch_dir:-/mnt/scratch}"}
fi

if [ -z "$GPU2USE" ]; then
    read -p "Enter the GPU index to use or 'all' (default: device=0): " user_gpu
    : ${GPU2USE:="${user_gpu:-device=0}"}
fi

# Resolve script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p "$MY_SCRATCH_DIR"
chmod -R 777 "$MY_SCRATCH_DIR"

# Networking & Cleanup
NETARG="--net=host"
rm -rf ../pid.fl ../daemon_pid.fl

# Docker image and container name
DOCKER_IMAGE=jefftud/odelia:1.0.1-dev.250901.629dfcd
if [ -z "$NOPULL" ]; then
    echo "Updating docker image"
    docker pull "$DOCKER_IMAGE"
fi

CONTAINER_NAME=odelia_swarm_client_CAM_1
DOCKER_OPTIONS_A="--name=$CONTAINER_NAME --gpus=$GPU2USE -u $(id -u):$(id -g)"
DOCKER_MOUNTS="-v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v $DIR/..:/startupkit/ -v $MY_SCRATCH_DIR:/scratch/ -v /opt/hpe/MediSwarm:/MediSwarm"
if [[ ! -z "$MY_DATA_DIR" ]]; then
    DOCKER_MOUNTS+=" -v $MY_DATA_DIR:/data/:ro"
fi
DOCKER_OPTIONS_B="-w /startupkit/startup/ --ipc=host $NETARG"
DOCKER_OPTIONS="${DOCKER_OPTIONS_A} ${DOCKER_MOUNTS} ${DOCKER_OPTIONS_B}"

# Common ENV vars
ENV_VARS="--env SITE_NAME=CAM_1 \
          --env DATA_DIR=/data \
          --env SCRATCH_DIR=/scratch \
          --env TORCH_HOME=/torch_home \
          --env GPU_DEVICE=$GPU2USE \
          --env MODEL_NAME=MST \
          --env CONFIG=unilateral \
          --env MEDISWARM_VERSION=1.0.1-dev.250901.629dfcd"

# Execution modes
if [[ ! -z "$DUMMY_TRAINING" ]]; then
    docker run -it --rm $TTY_OPT $DOCKER_OPTIONS $ENV_VARS --env TRAINING_MODE=local_training $DOCKER_IMAGE \
    /bin/bash -c "pip install --no-deps appdirs medim; /MediSwarm/application/jobs/minimal_training_pytorch_cnn/app/custom_cam_ZD/main.py"

elif [[ ! -z "$PREFLIGHT_CHECK" ]]; then
    docker run --rm $TTY_OPT $DOCKER_OPTIONS $ENV_VARS --env TRAINING_MODE=preflight_check --env NUM_EPOCHS=1 $DOCKER_IMAGE \
    /bin/bash -c "/MediSwarm/application/jobs/ODELIA_ternary_classification/app/custom/main.py"

elif [[ ! -z "$LOCAL_TRAINING" ]]; then
    docker run -it --rm $TTY_OPT $DOCKER_OPTIONS $ENV_VARS --env TRAINING_MODE=local_training --env NUM_EPOCHS=100 $DOCKER_IMAGE \
    /bin/bash -c "pip install --no-deps appdirs medim; /MediSwarm/application/jobs/ODELIA_ternary_classification/app/custom_cam_ZD/main.py"

elif [[ ! -z "$START_CLIENT" ]]; then
    docker run -d -t --rm $DOCKER_OPTIONS $ENV_VARS --env TRAINING_MODE=swarm $DOCKER_IMAGE \
    /bin/bash -c "nohup ./start.sh >> nohup.out 2>&1 && /bin/bash"

elif [[ ! -z "$INTERACTIVE" ]]; then
    docker run --rm $TTY_OPT --detach-keys="ctrl-x" $DOCKER_OPTIONS $DOCKER_IMAGE /bin/bash

elif [[ ! -z "$SCRIPT_TO_RUN" ]]; then
    docker run --rm $TTY_OPT $DOCKER_OPTIONS $ENV_VARS $DOCKER_IMAGE \
    /bin/bash -c "$SCRIPT_TO_RUN"

else
    echo "❗ One of the following options must be passed:"
    echo "--dummy_training   minimal sanity check for Docker/GPU"
    echo "--preflight_check  verify data access & local training"
    echo "--local_training   train a local model"
    echo "--start_client     launch FL client in swarm mode"
    echo "--interactive      drop into interactive container (for debugging)"
    echo "--run_script       execute script in container (for testing)"
    exit 1
fi
