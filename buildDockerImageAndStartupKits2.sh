#!/usr/bin/env bash

set -e

# make sure we are building from a state without local changes
#if ! git diff --quiet || ! git diff --staged --quiet ; then
#   echo "Local changes exist, aborting"
#   exit 1
#fi

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
CONTAINER_VERSION_ID=`git rev-parse --short HEAD`

# prepare clean version of source code repository clone for building Docker image

CWD=`pwd`
CLEAN_SOURCE_DIR=`mktemp -d`
mkdir -p $CLEAN_SOURCE_DIR/MediSwarm
git archive --format=tar HEAD | tar x -C $CLEAN_SOURCE_DIR/MediSwarm/
cd docker_config/NVFlare
git archive --format=tar HEAD | tar x -C $CLEAN_SOURCE_DIR/MediSwarm/docker_config/NVFlare
cd ../..

cd $CLEAN_SOURCE_DIR/MediSwarm
chmod a+rX . -R

# replacements in copy of source code
sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_DOCKER_IMAGE__#'$VERSION'#' docker_config/master_template.yml
sed -i 's#__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__#'$CONTAINER_VERSION_ID'#' docker_config/master_template.yml

echo "Call chache and copy pretrained model weights script" 
bash ./_cacheAndCopyPretrainedModelWeights.sh $CWD $CLEAN_SOURCE_DIR
cd $CWD

echo "current working dir: `$CWD`"

# build and print follow-up steps
CONTAINER_NAME=`grep "      docker_image: " $PROJECT_FILE | sed 's/      docker_image: //' | sed 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'$VERSION'#' | tr -d '\r'`
echo "Container Name: $CONTAINER_NAME"
echo "Docker build args: $DOCKER_BUILD_ARGS"
echo "Clean source dir: $CLEAN_SOURCE_DIR"

echo "--> build command: docker build $DOCKER_BUILD_ARGS -t $CONTAINER_NAME $CLEAN_SOURCE_DIR -f docker_config/Dockerfile_ODELIA"
docker build $DOCKER_BUILD_ARGS -t $CONTAINER_NAME $CLEAN_SOURCE_DIR -f docker_config/Dockerfile_ODELIA

echo "Docker image $CONTAINER_NAME built successfully"
echo "./_buildStartupKits.sh $PROJECT_FILE $VERSION $CONTAINER_NAME"
./_buildStartupKits.sh $PROJECT_FILE $VERSION $CONTAINER_NAME
echo "Startup kits built successfully"

rm -rf $CLEAN_SOURCE_DIR

echo "If you wish, manually push $CONTAINER_NAME now"
