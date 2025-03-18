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

sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'$VERSION'#' $1

docker run --rm -it -u $(id -u):$(id -g) -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v ./:/workspace/ -w /workspace/ $DOCKER_IMAGE /bin/bash -c "nvflare provision -p $1 && ./_generateStartupKitArchives.sh $1 $VERSION"

sed -i 's#'$VERSION'#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#' $1

echo "If you wish, manually push $DOCKER_IMAGE now"
