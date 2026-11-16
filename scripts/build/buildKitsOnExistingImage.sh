#!/usr/bin/env bash
#
# Provision startup kits with this checkout's kit template for a runtime image that is already
# released: a new kit version without a new image (DECADE 1.7.1 = kit 1.7.1 on the released
# jefftud/decade:1.7.0).
#
# buildDockerImageAndStartupKits.sh rebuilds the image, and with it every dependency that is not
# pinned (STAMP's and the NVFlare fork's lower bounds), so the "same" image would run other library
# versions. Here the runtime image stays exactly as released: a local template-only image on top of
# it (one layer, the version-stamped kit template where NVFlare's provisioning reads it) only
# provisions the kits. That image is never pushed; the kits run the released image.
#
# Usage (from a clean checkout, see the F9 guard in _buildStartupKits.sh for the CA):
#   scripts/build/buildKitsOnExistingImage.sh -p <project.yml> -b <image@sha256:digest> -v <kit version> [-t <template>]
#
#   -p  project YAML, e.g. application/provision/project_DECADE_allsites.yml
#   -b  the released runtime image, pinned by digest, e.g. jefftud/decade:1.7.0@sha256:51235f98...
#   -v  kit version, e.g. 1.7.1 (docker.sh's MEDISWARM_VERSION, archive names, the kits' default tag)
#   -t  kit template; default docker_config/master_template_STAMP.yml for a jefftud/decade project,
#       docker_config/master_template.yml otherwise
#
# Environment as for _buildStartupKits.sh: MEDISWARM_UPLOAD_KEYS (0 keeps the sites' own upload
# keys), MEDISWARM_KIT_IMAGE (pin image.conf, e.g. jefftud/decade:1.7.1), MEDISWARM_ALLOW_NEW_ROOT_CA.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PROJECT_FILE=""
BASE_IMAGE=""
VERSION=""
TEMPLATE=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p) PROJECT_FILE="$2"; shift ;;
        -b) BASE_IMAGE="$2"; shift ;;
        -v) VERSION="$2"; shift ;;
        -t) TEMPLATE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$PROJECT_FILE" || -z "$BASE_IMAGE" || -z "$VERSION" ]]; then
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
[ -f "$PROJECT_FILE" ] || { echo "Project file not found: $PROJECT_FILE"; exit 1; }
case "$BASE_IMAGE" in
    *@sha256:*) ;;
    *) echo "Pin the released image by digest (-b <image>@sha256:...): a tag can move between test and build."; exit 1 ;;
esac
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]] || { echo "Not a version: $VERSION"; exit 1; }

if ! git diff --quiet || ! git diff --staged --quiet; then
    echo "Local changes exist, aborting: the kits record the commit they were built from."
    exit 1
fi

PROJECT_IMAGE="$(grep -m1 -E '^[[:space:]]*docker_image:' "$PROJECT_FILE" | sed -E 's/^[[:space:]]*docker_image:[[:space:]]*//; s/[[:space:]]*$//')"
PROJECT_REPO="${PROJECT_IMAGE%:*}"
if [ -z "$TEMPLATE" ]; then
    case "$PROJECT_REPO" in
        */decade) TEMPLATE="docker_config/master_template_STAMP.yml" ;;
        *) TEMPLATE="docker_config/master_template.yml" ;;
    esac
fi
[ -f "$TEMPLATE" ] || { echo "Template not found: $TEMPLATE"; exit 1; }
case "${BASE_IMAGE%@*}" in
    "$PROJECT_REPO":*|"$PROJECT_REPO") ;;
    *) echo "The base image ${BASE_IMAGE%@*} is not of this project's repository ($PROJECT_REPO)."; exit 1 ;;
esac

CONTAINER_VERSION_ID="$(git rev-parse --short HEAD)"
KIT_IMAGE="${PROJECT_REPO}-kittemplate:${VERSION}-local"

docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
TEMPLATE_IN_IMAGE="$(docker run --rm --network none "$BASE_IMAGE" python3 -c \
    'import os, nvflare.lighter as l; print(os.path.join(os.path.dirname(l.__file__), "templates", "master_template.yml"))')"
[ -n "$TEMPLATE_IN_IMAGE" ] || { echo "Could not locate NVFlare's kit template in $BASE_IMAGE"; exit 1; }

BUILD_DIR="$(mktemp -d -t mediswarm-kittemplate.XXXXXXXXXX)"
trap 'rm -rf "$BUILD_DIR"' EXIT
sed -e "s#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_DOCKER_IMAGE__#${VERSION}#g" \
    -e "s#__REPLACED_BY_CONTAINER_VERSION_IDENTIFIER_WHEN_BUILDING_DOCKER_IMAGE__#${CONTAINER_VERSION_ID}#g" \
    "$TEMPLATE" > "$BUILD_DIR/master_template.yml"
cat > "$BUILD_DIR/Dockerfile" <<DOCKERFILE
FROM ${BASE_IMAGE}
LABEL org.mediswarm.kittemplate.version="${VERSION}" \\
      org.mediswarm.kittemplate.revision="$(git rev-parse HEAD)" \\
      org.mediswarm.kittemplate.base="${BASE_IMAGE}" \\
      org.mediswarm.kittemplate.note="local, for provisioning kits only; never push"
COPY master_template.yml ${TEMPLATE_IN_IMAGE}
DOCKERFILE
docker build -q -t "$KIT_IMAGE" "$BUILD_DIR"

echo "Kit template image ${KIT_IMAGE} (local, never pushed) on ${BASE_IMAGE}, template ${TEMPLATE} at ${CONTAINER_VERSION_ID}"
"$SCRIPT_DIR/_buildStartupKits.sh" "$PROJECT_FILE" "$VERSION" "$KIT_IMAGE"
# The docker.sh default is the project's docker_image, with the kit version where the YAML has the
# placeholder (a roster may also name a fixed tag, e.g. the released image the kits run).
case "$PROJECT_IMAGE" in
    *__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__*) KIT_DEFAULT_IMAGE="${PROJECT_REPO}:${VERSION}" ;;
    *) KIT_DEFAULT_IMAGE="$PROJECT_IMAGE" ;;
esac
echo "Kits for version ${VERSION} built. They run ${MEDISWARM_KIT_IMAGE:-${PROJECT_REPO}:current} (image.conf);"
echo "their docker.sh default is ${KIT_DEFAULT_IMAGE}."
