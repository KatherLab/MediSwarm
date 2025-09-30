#!/usr/bin/env bash

set -e

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist, aborting"
   exit 1
fi

DOCKER_BUILD_ARGS="--no-cache --progress=plain";

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p)                  PROJECT_FILE="$2"; shift ;;
        -c)                  VPN_CREDENTIALS_DIR="$2"; shift ;;
        --use-docker-cache)  DOCKER_BUILD_ARGS="";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$PROJECT_FILE" || -z "$VPN_CREDENTIALS_DIR" ]]; then
    echo "Usage: buildDockerImageAndStartupKits.sh -p <swarm_project.yml> -c <VPN credentials directory> [--use-docker-cache]"
    exit 1
fi

VERSION=`./getVersionNumber.sh`
CONTAINER_VERSION_ID=`git rev-parse --short HEAD`

# prepare clean version of source code repository clone for building Docker image

CWD=`pwd`
CLEAN_SOURCE_DIR=`mktemp -d`
mkdir $CLEAN_SOURCE_DIR/MediSwarm
rsync -ax --exclude workspace . $CLEAN_SOURCE_DIR/MediSwarm/
cd $CLEAN_SOURCE_DIR/MediSwarm
git clean -x -q -f .
cd docker_config/NVFlare
git clean -x -q -f .
cd ../..
rm .git -rf
chmod a+rX . -R

# replacements in copy of source code
sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_DOCKER_IMAGE__#'$VERSION'#' docker_config/master_template.yml
sed -i 's#__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__#'$CONTAINER_VERSION_ID'#' docker_config/master_template.yml

# prepare pre-trained model weights for being included in Docker image

MODEL_WEIGHTS_FILE=$CWD'/docker_config/torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth'
MODEL_LICENSE_FILE=$CWD'/docker_config/torch_home_cache/hub/facebookresearch_dinov2_main/LICENSE'
if [[ ! -f $MODEL_WEIGHTS_FILE || ! -f $MODEL_LICENSE_FILE ]]; then
    echo "Pre-trained model not available. Attempting download"
    HUBDIR=$(dirname $(dirname $MODEL_LICENSE_FILE))
    mkdir -p $(dirname $MODEL_WEIGHTS_FILE)
    wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -O $MODEL_WEIGHTS_FILE
    wget https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip -O /tmp/dinov2.zip
    unzip /tmp/dinov2.zip -d $HUBDIR
    mv $HUBDIR/dinov2-main $HUBDIR/$(basename $(dirname $MODEL_LICENSE_FILE))
    touch $HUBDIR/trusted_list
fi

if echo 2e405cee1bad14912278296d4f42e993 $MODEL_WEIGHTS_FILE | md5sum --check - && echo 153d2db1c329326a2d9f881317ea942e $MODEL_LICENSE_FILE | md5sum --check -; then
    cp -r $CWD/docker_config/torch_home_cache $CLEAN_SOURCE_DIR/torch_home_cache
else
    exit 1
fi
chmod a+rX $CLEAN_SOURCE_DIR/torch_home_cache -R

cd $CWD

# build and print follow-up steps
CONTAINER_NAME=`grep "      docker_image: " $PROJECT_FILE | sed 's/      docker_image: //' | sed 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'$VERSION'#'`
echo $CONTAINER_NAME

docker build $DOCKER_BUILD_ARGS -t $CONTAINER_NAME $CLEAN_SOURCE_DIR -f docker_config/Dockerfile_ODELIA

echo "Docker image $CONTAINER_NAME built successfully"
echo "./_buildStartupKits.sh $PROJECT_FILE $VERSION $CONTAINER_NAME"
VPN_CREDENTIALS_DIR=$(realpath $VPN_CREDENTIALS_DIR)
./_buildStartupKits.sh $PROJECT_FILE $VERSION $CONTAINER_NAME $VPN_CREDENTIALS_DIR
echo "Startup kits built successfully"

rm -rf $CLEAN_SOURCE_DIR

echo "If you wish, manually push $CONTAINER_NAME now"
