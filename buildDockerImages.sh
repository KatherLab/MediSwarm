#!/usr/bin/env bash

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist"
   exit 1
fi

docker build -t nvflare-pt-dev:odelia . -f docker_config/Dockerfile_ODELIA
# TODO enable tagging and pushing
# TODO think about tag name (to be adapted in the project config as well)
