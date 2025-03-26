#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: buildDockerImageAndStartupKits.sh SWARM_PROJECT.yml"
    exit 1
fi

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist"
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

docker build -t $DOCKER_IMAGE $CLEAN_SOURCE_DIR -f docker_config/Dockerfile_ODELIA --build-arg CLEAN_SOURCE_DIR=$CLEAN_SOURCE_DIR

rm -rf $CLEAN_SOURCE_DIR

./_buildStartupKits.sh $1 $VERSION

echo "If you wish, manually push $DOCKER_IMAGE now"
