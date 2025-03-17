#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: buildStartupKits.sh SWARM_PROJECT.yml"
    exit 1
fi

VERSION=`tail -n 1 odelia_image.version`
DOCKER_IMAGE=jefftud/odelia:$VERSION

sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'$VERSION'#' $1

docker run --rm -it -u $(id -u):$(id -g) -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v ./:/workspace/ -w /workspace/ $DOCKER_IMAGE /bin/bash -c "nvflare provision -p $1"

OUTPUT_FOLDER=workspace/`grep "^name: " $1 | sed 's/name: //'`

sed -i 's#'$VERSION'#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#' $1

TARGET_FOLDER=`ls -d $OUTPUT_FOLDER/prod_* | tail -n 1`

GIT_SHORT_HASH=`git rev-parse --short HEAD`
DATE=`date "+%y%m%d"`
LONG_VERSION=$VERSION-dev.$DATE.$GIT_SHORT_HASH

cd $TARGET_FOLDER
for startupkit in `ls .`; do
    zip -rq ${startupkit}_$LONG_VERSION.zip $startupkit
done
