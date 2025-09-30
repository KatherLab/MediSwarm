#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: _buildStartupKits.sh SWARM_PROJECT.yml VERSION_STRING CONTAINER_NAME [PATH_FOR_VPN_CREDENTIALS]"
    exit 1
fi

PROJECT_YML=$1
VERSION=$2
CONTAINER_NAME=$3
PATH_FOR_VPN_CREDENTIALS=""
if [ "$#" -eq 4 ]; then
    PATH_FOR_VPN_CREDENTIALS=$4
fi

sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'$VERSION'#' $PROJECT_YML

ARGUMENTS="$PROJECT_YML $VERSION $PATH_FOR_VPN_CREDENTIALS"

echo "Building startup kits: $ARGUMENTS"
docker run --rm \
  -u $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd \
  -v /etc/group:/etc/group \
  -v ./:/workspace/ \
  -w /workspace/ \
  $CONTAINER_NAME \
  /bin/bash -c "nvflare provision -p $PROJECT_YML && ./_generateStartupKitArchives.sh $ARGUMENTS"|| { echo "Docker run failed"; exit 1; }

sed -i 's#'$VERSION'#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#' $PROJECT_YML
