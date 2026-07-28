#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# make sure we are building from a state without local changes
if ! git diff --quiet || ! git diff --staged --quiet ; then
   echo "Local changes exist, aborting"
   exit 1
fi

DOCKER_BUILD_ARGS="--no-cache --progress=plain";
DOCKERFILE="docker_config/Dockerfile_ODELIA"
NUM_ROUNDS_OVERRIDE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p)                  PROJECT_FILE="$2"; shift ;;
        -d|--dockerfile)     DOCKERFILE="$2"; shift ;;
        --use-docker-cache)  DOCKER_BUILD_ARGS="";;
        --num-rounds)        NUM_ROUNDS_OVERRIDE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$PROJECT_FILE" ]; then
    echo "Usage: buildDockerImageAndStartupKits.sh -p <swarm_project.yml> [-d <Dockerfile>] [--use-docker-cache] [--num-rounds N]"
    echo "  -d  Dockerfile to use (default: docker_config/Dockerfile_ODELIA)"
    echo "      For STAMP builds, use: -d docker_config/Dockerfile_STAMP"
    echo "  --num-rounds  Override num_rounds in all config_fed_server.conf (for CI/CD testing)"
    exit 1
fi

if [ ! -f "$DOCKERFILE" ]; then
    echo "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# This box also hosts the self-hosted CI runner. A multi-GB build landing on top of
# a running validate-swarm job starves it and fails unrelated PRs (#448, #388), so
# queue behind any other GPU/Docker job instead of interleaving.
. "$SCRIPT_DIR/../ci/host_gpu_lock.sh"
acquire_host_lock "image build ($(basename "$PROJECT_FILE"))" || exit 1

VERSION=`"$SCRIPT_DIR/getVersionNumber.sh"`
CONTAINER_VERSION_ID=`git rev-parse --short HEAD`

# prepare clean version of source code repository clone for building Docker image

CWD=`pwd`
CLEAN_SOURCE_DIR=""
cleanup_build_context () {
    if [[ -n "$CLEAN_SOURCE_DIR" && -d "$CLEAN_SOURCE_DIR" ]]; then
        rm -rf "$CLEAN_SOURCE_DIR"
    fi
}
trap cleanup_build_context EXIT

CLEAN_SOURCE_DIR=`mktemp -d -t mediswarm-build.XXXXXXXXXX`
mkdir "$CLEAN_SOURCE_DIR/MediSwarm"
git archive --format=tar HEAD | tar x -C "$CLEAN_SOURCE_DIR/MediSwarm/"
cd docker_config/NVFlare
git archive --format=tar HEAD | tar x -C "$CLEAN_SOURCE_DIR/MediSwarm/docker_config/NVFlare"
cd ../..

cd $CLEAN_SOURCE_DIR/MediSwarm
chmod a+rX . -R

# replacements in copy of source code
sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_DOCKER_IMAGE__#'$VERSION'#' docker_config/master_template.yml
sed -i 's#__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__#'$CONTAINER_VERSION_ID'#' docker_config/master_template.yml

# Also patch STAMP template if it exists (separate template for STAMP builds)
if [[ -f docker_config/master_template_STAMP.yml ]]; then
    sed -i 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_DOCKER_IMAGE__#'$VERSION'#' docker_config/master_template_STAMP.yml
    sed -i 's#__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__#'$CONTAINER_VERSION_ID'#' docker_config/master_template_STAMP.yml
fi

# Override num_rounds in all server configs if requested (CI/CD testing)
if [[ -n "$NUM_ROUNDS_OVERRIDE" ]]; then
    echo "Overriding num_rounds to $NUM_ROUNDS_OVERRIDE in all config_fed_server.conf files"
    find application/jobs -name "config_fed_server.conf" -exec \
        sed -i 's/num_rounds = [0-9]\+/num_rounds = '"$NUM_ROUNDS_OVERRIDE"'/' {} \;
fi

# Only cache pretrained model weights for ODELIA builds (STAMP uses pre-extracted
# H5 features and doesn't need DINOv2/challenge weights in the Docker image)
# If an environment variable MEDISWARM_BUILD_CACHE_DIR is set, it will be used as a persistent cache,
# otherwise data is stored in a temporary folder deleted after building.
if [[ "$DOCKERFILE" != *"Dockerfile_STAMP"* ]]; then
    ./scripts/build/_cacheAndCopyPretrainedModelWeights.sh "$CLEAN_SOURCE_DIR"
fi
cd $CWD

# build and print follow-up steps
CONTAINER_NAME=`grep "      docker_image: " $PROJECT_FILE | sed 's/      docker_image: //' | sed 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'$VERSION'#'`
echo $CONTAINER_NAME

docker build $DOCKER_BUILD_ARGS -t $CONTAINER_NAME $CLEAN_SOURCE_DIR -f $DOCKERFILE

# #479: the startup kits generated below get an image.conf on the ':current'
# channel (see _injectLiveSyncIntoStartupKits.sh), so docker.sh in the kits
# resolves e.g. localhost:5000/odelia:current -- but only the versioned tag was
# built. For LOCAL-registry test images, add a local ':current' alias so the
# kits' docker.sh finds the image without a running registry. Productive images
# (jefftud/*) get ':current' re-tagged on Docker Hub at release, so skip those.
if [[ "$CONTAINER_NAME" == localhost:5000/* ]]; then
    docker tag "$CONTAINER_NAME" "${CONTAINER_NAME%:*}:current"
    echo "Tagged local channel alias ${CONTAINER_NAME%:*}:current -> $CONTAINER_NAME"
fi

echo "Docker image $CONTAINER_NAME built successfully"
echo "scripts/build/_buildStartupKits.sh $PROJECT_FILE $VERSION $CONTAINER_NAME"
"$SCRIPT_DIR/_buildStartupKits.sh" $PROJECT_FILE $VERSION $CONTAINER_NAME
echo "Startup kits built successfully"

rm -rf "$CLEAN_SOURCE_DIR"
CLEAN_SOURCE_DIR=""

echo "If you wish, manually push $CONTAINER_NAME now"
