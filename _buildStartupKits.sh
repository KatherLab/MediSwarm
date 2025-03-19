#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: _buildStartupKits.sh SWARM_PROJECT.yml VERSION_STRING"
    exit 1
fi

PROJECT_YML=$1
VERSION=$2

sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'$VERSION'#' $PROJECT_YML

docker run --rm -it -u $(id -u):$(id -g) -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v ./:/workspace/ -w /workspace/ jefftud/odelia:$VERSION /bin/bash -c "nvflare provision -p $PROJECT_YML && ./_generateStartupKitArchives.sh $PROJECT_YML $VERSION"

sed -i 's#'$VERSION'#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#' $PROJECT_YML
