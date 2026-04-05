#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist, aborting"
   exit 1
fi

DOCKER_BUILD_ARGS="--no-cache --progress=plain";
DOCKERFILE="docker_config/Dockerfile_ODELIA"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p)                  PROJECT_FILE="$2"; shift ;;
        -d|--dockerfile)     DOCKERFILE="$2"; shift ;;
        --use-docker-cache)  DOCKER_BUILD_ARGS="";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$PROJECT_FILE" ]; then
    echo "Usage: buildDockerImageAndStartupKits.sh -p <swarm_project.yml> [-d <Dockerfile>] [--use-docker-cache]"
    echo "  -d  Dockerfile to use (default: docker_config/Dockerfile_ODELIA)"
    echo "      For STAMP builds, use: -d docker_config/Dockerfile_STAMP"
    exit 1
fi

if [ ! -f "$DOCKERFILE" ]; then
    echo "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

VERSION=`"$SCRIPT_DIR/getVersionNumber.sh"`
CONTAINER_VERSION_ID=`git rev-parse --short HEAD`

# prepare clean version of source code repository clone for building Docker image

CWD=`pwd`
CLEAN_SOURCE_DIR=`mktemp -d`
mkdir $CLEAN_SOURCE_DIR/MediSwarm
git archive --format=tar HEAD | tar x -C $CLEAN_SOURCE_DIR/MediSwarm/
cd docker_config/NVFlare
git archive --format=tar HEAD | tar x -C $CLEAN_SOURCE_DIR/MediSwarm/docker_config/NVFlare
cd ../..

cd $CLEAN_SOURCE_DIR/MediSwarm
chmod a+rX . -R

# replacements in copy of source code
sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_DOCKER_IMAGE__#'$VERSION'#' docker_config/master_template.yml
sed -i 's#__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__#'$CONTAINER_VERSION_ID'#' docker_config/master_template.yml

# Only cache pretrained model weights for ODELIA builds (STAMP uses pre-extracted
# H5 features and doesn't need DINOv2/challenge weights in the Docker image)
if [[ "$DOCKERFILE" != *"Dockerfile_STAMP"* ]]; then
    ./scripts/build/_cacheAndCopyPretrainedModelWeights.sh $CWD $CLEAN_SOURCE_DIR
fi
cd $CWD

# build and print follow-up steps
CONTAINER_NAME=`grep "      docker_image: " $PROJECT_FILE | sed 's/      docker_image: //' | sed 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'$VERSION'#'`
echo $CONTAINER_NAME

docker build $DOCKER_BUILD_ARGS -t $CONTAINER_NAME $CLEAN_SOURCE_DIR -f $DOCKERFILE

echo "Docker image $CONTAINER_NAME built successfully"
echo "scripts/build/_buildStartupKits.sh $PROJECT_FILE $VERSION $CONTAINER_NAME"
"$SCRIPT_DIR/_buildStartupKits.sh" $PROJECT_FILE $VERSION $CONTAINER_NAME
echo "Startup kits built successfully"

rm -rf $CLEAN_SOURCE_DIR

echo "If you wish, manually push $CONTAINER_NAME now"
