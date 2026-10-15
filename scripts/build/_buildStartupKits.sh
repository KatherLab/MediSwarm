#!/usr/bin/env bash
#
# Provision the startup kits of one project inside the given image, add the kit extras and
# package them. Runs from the repository root: provisioning writes ./workspace/<name>/prod_NN.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: _buildStartupKits.sh SWARM_PROJECT.yml VERSION_STRING CONTAINER_NAME"
    exit 1
fi

PROJECT_YML=$1
VERSION=$2
CONTAINER_NAME=$3

# Projects whose kits are installed at the hospitals. Their root CA must survive every
# rebuild, see the guard below.
PRODUCTION_PROJECTS="odelia_allsites decade_allsites"

# Provision from a temporary copy so the tracked project yml is never mutated.
# The previous flow sed-ed the version IN and then sed-ed it back OUT, which
# corrupts the file if the version string appears anywhere else in it.
BUILD_YML="${PROJECT_YML%.yml}.build.yml"
trap 'rm -f "$BUILD_YML"' EXIT
sed 's#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#'"$VERSION"'#g' "$PROJECT_YML" > "$BUILD_YML"

# NVFlare names the workspace after the project's name (cut to 63 characters) and keeps the
# root CA plus every participant's certificate and key in its state/cert.json. When that file
# exists, a rebuild reuses all of them, so new kits keep working with the running server and
# with every kit issued before.
PROJECT_NAME="$(sed -n 's/^name:[[:space:]]*//p' "$BUILD_YML" | head -n 1 \
  | sed 's/[[:space:]]*$//; s/^["'\'']//; s/["'\'']$//')"
PROJECT_NAME="${PROJECT_NAME:0:63}"
STATE_CERT="workspace/$PROJECT_NAME/state/cert.json"

# F9 guard. Without state/cert.json NVFlare silently mints a NEW root CA. For a production
# project every kit built from it is then rejected by the coordinator and by the kits already
# at the sites (ClientConnectorCertificateError, docs/SWARM_FAILURE_MODES.md F9). That happens
# as soon as someone builds from a fresh clone or a worktree, which has no workspace/, so
# refuse instead of building kits that can never connect.
case " $PRODUCTION_PROJECTS " in
    *" $PROJECT_NAME "*)
        if [ ! -s "$STATE_CERT" ] && [ "${MEDISWARM_ALLOW_NEW_ROOT_CA:-0}" != "1" ]; then
            cat >&2 <<EOF
REFUSING to provision the production project '$PROJECT_NAME': $STATE_CERT is missing.

That file holds the root CA the coordinator and every installed kit trust. Provisioning
without it creates a NEW root CA, and kits built from it can never connect to the running
swarm (F9 in docs/SWARM_FAILURE_MODES.md).

Build from the checkout whose workspace/$PROJECT_NAME/state/ holds the consortium's CA, or
copy that state/ folder here first. Only if a new CA is intended (every site and the server
then need new kits), rerun with MEDISWARM_ALLOW_NEW_ROOT_CA=1.
EOF
            exit 1
        fi
        ;;
esac

echo "Building startup kits for project $PROJECT_YML with version $VERSION"
if [ -s "$STATE_CERT" ]; then
    echo "Reusing the root CA in $STATE_CERT"
else
    echo "No $STATE_CERT yet: provisioning creates a new root CA for '$PROJECT_NAME'"
fi
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd \
  -v /etc/group:/etc/group \
  -v ./:/workspace/ \
  -w /workspace/ \
  -e PROJECT_YML="$BUILD_YML" \
  -e VERSION="$VERSION" \
  -e MEDISWARM_ALLOW_NEW_ROOT_CA="${MEDISWARM_ALLOW_NEW_ROOT_CA:-0}" \
  "$CONTAINER_NAME" \
  /bin/bash -c "nvflare provision --force -p \$PROJECT_YML && ./scripts/build/_injectLiveSyncIntoStartupKits.sh \$PROJECT_YML && ./scripts/build/_addToolsToStartupKits.sh \$PROJECT_YML && ./scripts/build/_generateStartupKitArchives.sh \$PROJECT_YML \$VERSION" || { echo "Docker run failed"; exit 1; }
