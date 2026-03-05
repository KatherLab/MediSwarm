#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: _buildStartupKits.sh SWARM_PROJECT.yml VERSION_STRING CONTAINER_NAME"
    exit 1
fi

PROJECT_YML=$1
VERSION=$2
CONTAINER_NAME=$3

sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'$VERSION'#' $PROJECT_YML

echo "Building startup kits for project $PROJECT_YML with version $VERSION"
docker run --rm \
  -u $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd \
  -v /etc/group:/etc/group \
  -v ./:/workspace/ \
  -w /workspace/ \
  -e PROJECT_YML=$PROJECT_YML \
  -e VERSION=$VERSION \
  $CONTAINER_NAME \
  /bin/bash -c "nvflare provision -p \$PROJECT_YML && ./_generateStartupKitArchives.sh \$PROJECT_YML \$VERSION"|| { echo "Docker run failed"; exit 1; }

sed -i 's#'$VERSION'#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#' $PROJECT_YML
