#!/usr/bin/env bash

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist"
   exit 1
fi

docker build -t nvflare-pt-dev:nfcore . -f docker_config/Dockerfile_nfcore
# TODO enable tagging and pushing
# docker tag nvflare-pt-dev:nfcore jefftud/nvflare-pt-dev:nfcore
# docker push jefftud/nvflare-pt-dev:nfcore

docker build -t nvflare-pt-dev:3dcnn . -f docker_config/Dockerfile_3dcnn
# docker tag nvflare-pt-dev:3dcnn jefftud/nvflare-pt-dev:3dcnn
# docker push jefftud/nvflare-pt-dev:3dcnn
