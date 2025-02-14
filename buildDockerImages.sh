#!/usr/bin/env bash

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist"
   exit 1
fi

export DOCKER_BUILT_FROM_MEDISWARM_REVISION=`git rev-parse HEAD`

sed -i 's/DOCKER_BUILT_FROM_MEDISWARM_REVISION  # this should be replaced by buildDockerImages.sh/'$DOCKER_BUILT_FROM_MEDISWARM_REVISION'/' docker_config/master_template.yml
docker build -t nvflare-pt-dev:nfcore . -f docker_config/Dockerfile_nfcore
# TODO enable tagging and pushing
# docker tag nvflare-pt-dev:nfcore jefftud/nvflare-pt-dev:nfcore
# docker push jefftud/nvflare-pt-dev:nfcore
sed -i 's/'$DOCKER_BUILT_FROM_MEDISWARM_REVISION'/DOCKER_BUILT_FROM_MEDISWARM_REVISION  # this should be replaced by buildDockerImages.sh/' docker_config/master_template.yml

docker build -t nvflare-pt-dev:3dcnn . -f docker_config/Dockerfile_3dcnn
# docker tag nvflare-pt-dev:3dcnn jefftud/nvflare-pt-dev:3dcnn
# docker push jefftud/nvflare-pt-dev:3dcnn
