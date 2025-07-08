#!/usr/bin/env bash

set -e

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist, aborting"
   exit 1
fi

DOCKER_BUILD_ARGS="--no-cache --progress=plain";

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p)                  PROJECT_FILE="$2"; shift ;;
        --use-docker-cache)  DOCKER_BUILD_ARGS="";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$PROJECT_FILE" ]; then
    echo "Usage: buildDockerImageAndStartupKits.sh -p <swarm_project.yml> [--use-docker-cache]"
    exit 1
fi

VERSION=`./getVersionNumber.sh`
DOCKER_IMAGE=jefftud/odelia:$VERSION


# prepare clean version of source code repository clone for building Docker image

CWD=`pwd`
CLEAN_SOURCE_DIR=`mktemp -d`
mkdir $CLEAN_SOURCE_DIR/MediSwarm
cp -r . $CLEAN_SOURCE_DIR/MediSwarm/
cd $CLEAN_SOURCE_DIR/MediSwarm
git clean -x -q -f .
cd docker_config/NVFlare
git clean -x -q -f .
cd ../..
rm .git -rf
chmod a+rX . -R
cd $CWD


# prepare pre-trained model weights for being included in Docker image

MODEL_WEIGHTS_FILE='docker_config/torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth'
MODEL_LICENSE_FILE='docker_config/torch_home_cache/hub/facebookresearch_dinov2_main/LICENSE'
if [[ ! -f $MODEL_WEIGHTS_FILE || ! -f $MODEL_LICENSE_FILE ]]; then
    read -p "Pre-trained model not available. Build the image without them? " -n 1 -r
    if [[ ! $REPLY = ^[Yy]$ ]]; then
        BUILT_WITHOUT_PRETRAINED_WEIGHTS=1
        mkdir $CLEAN_SOURCE_DIR/torch_home_cache
    else
        exit 1
    fi
else
    if echo 2e405cee1bad14912278296d4f42e993 $MODEL_WEIGHTS_FILE | md5sum --check - && echo 153d2db1c329326a2d9f881317ea942e $MODEL_LICENSE_FILE | md5sum --check -; then
        cp -r ./docker_config/torch_home_cache $CLEAN_SOURCE_DIR/torch_home_cache
    else
        exit 1
    fi
fi
chmod a+rX $CLEAN_SOURCE_DIR/torch_home_cache -R


# build and print follow-up steps

docker build $DOCKER_BUILD_ARGS -t $DOCKER_IMAGE $CLEAN_SOURCE_DIR -f docker_config/Dockerfile_ODELIA

echo "Docker image $DOCKER_IMAGE built successfully"
echo "./_buildStartupKits.sh $PROJECT_FILE $VERSION"
./_buildStartupKits.sh $PROJECT_FILE $VERSION
echo "Startup kits built successfully"

rm -rf $CLEAN_SOURCE_DIR

if [ -z BUILT_WITHOUT_PRETRAINED_WEIGHTS ]; then
    echo "If you wish, manually push $DOCKER_IMAGE now"
else
    echo "Now run a dummy training to download the pretrained model weights, export them to docker_config/torch_home_cache/hub, and re-build the image"
fi
