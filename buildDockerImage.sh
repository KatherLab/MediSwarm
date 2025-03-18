#!/usr/bin/env bash

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist"
   exit 1
fi

VERSION=`./getVersionNumber.sh`
DOCKER_IMAGE=jefftud/odelia:$VERSION

docker build -t $DOCKER_IMAGE . -f docker_config/Dockerfile_ODELIA

# replace <DOCKER_IMAGE> by the current name and run the command manually
# docker push <DOCKER_IMAGE>
