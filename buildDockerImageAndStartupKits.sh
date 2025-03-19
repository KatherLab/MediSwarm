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

docker build -t $DOCKER_IMAGE . -f docker_config/Dockerfile_ODELIA

./_buildStartupKits.sh $1 $VERSION

echo "If you wish, manually push $DOCKER_IMAGE now"
