#!/usr/bin/env bash

set -e

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist, aborting"
   exit 1
fi

DOCKER_BUILD_ARGS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p)          PROJECT_FILE="$2"; shift ;;
        --no-cache)  DOCKER_BUILD_ARGS="--no-cache --progress=plain";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$PROJECT_FILE" ]; then
    echo "Usage: buildDockerImageAndStartupKits.sh -p <swarm_project.yml> [--no-cache]"
    exit 1
fi

VERSION=`./getVersionNumber.sh`
DOCKER_IMAGE=jefftud/odelia:$VERSION

# prepare clean version of source code repository clone for building Docker image
CWD=`pwd`
CLEAN_SOURCE_DIR=`mktemp -d`
cp -r . $CLEAN_SOURCE_DIR/
cd $CLEAN_SOURCE_DIR
git clean -f .
chmod a+rX . -R
cd $CWD

docker build $DOCKER_BUILD_ARGS -t $DOCKER_IMAGE $CLEAN_SOURCE_DIR -f docker_config/Dockerfile_ODELIA

rm -rf $CLEAN_SOURCE_DIR

./_buildStartupKits.sh $PROJECT_FILE $VERSION

echo "If you wish, manually push $DOCKER_IMAGE now"
