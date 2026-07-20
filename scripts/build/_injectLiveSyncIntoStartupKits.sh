#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: _injectLiveSyncIntoStartupKits.sh SWARM_PROJECT.yml"
  exit 1
fi

PROJECT_YML="$1"
OUTPUT_FOLDER="workspace/$(grep '^name: ' "$PROJECT_YML" | sed 's/name: //')"
TARGET_FOLDER="$(ls -d "$OUTPUT_FOLDER"/prod_* | tail -n 1)"

HELPER_SOURCE_DIR="kit_live_sync"
SYNC_CONF_SOURCE="$HELPER_SOURCE_DIR/sync.conf"
SYNC_CONF_EXAMPLE_SOURCE="$HELPER_SOURCE_DIR/sync.conf.example"
ADMIN_HELPER_SOURCE="kit_admin_tools/prepare_odelia_job.sh"

if [ ! -d "$HELPER_SOURCE_DIR" ]; then
  echo "Missing helper directory: $HELPER_SOURCE_DIR"
  exit 1
fi

if [ ! -f "$SYNC_CONF_SOURCE" ] && [ ! -f "$SYNC_CONF_EXAMPLE_SOURCE" ]; then
  echo "Missing live sync config files in $HELPER_SOURCE_DIR"
  exit 1
fi

if [ ! -f "$ADMIN_HELPER_SOURCE" ]; then
  echo "Missing admin helper: $ADMIN_HELPER_SOURCE"
  exit 1
fi

echo "Injecting live sync helpers into startup kits under $TARGET_FOLDER"

if [ ! -f "$SYNC_CONF_SOURCE" ]; then
  echo "No local live sync config found; injecting sync.conf.example only."
fi

find "$TARGET_FOLDER" -mindepth 1 -maxdepth 1 -type d | while read -r KIT_DIR; do
  STARTUP_DIR="$KIT_DIR/startup"
  ORIGINAL_DOCKER_SH="$STARTUP_DIR/docker.sh"

  if [ ! -d "$STARTUP_DIR" ]; then
    continue
  fi

  if [ ! -f "$ORIGINAL_DOCKER_SH" ]; then
    continue
  fi

  if [ -f "$SYNC_CONF_SOURCE" ]; then
    cp "$SYNC_CONF_SOURCE" "$STARTUP_DIR/sync.conf"
  else
    cp "$SYNC_CONF_EXAMPLE_SOURCE" "$STARTUP_DIR/sync.conf"
  fi
  if [ -f "$SYNC_CONF_EXAMPLE_SOURCE" ]; then
    cp "$SYNC_CONF_EXAMPLE_SOURCE" "$STARTUP_DIR/sync.conf.example"
  fi
  cp "$HELPER_SOURCE_DIR/build_heartbeat.sh" "$STARTUP_DIR/build_heartbeat.sh"
  cp "$HELPER_SOURCE_DIR/live_sync.sh" "$STARTUP_DIR/live_sync.sh"

  chmod +x "$STARTUP_DIR/build_heartbeat.sh" "$STARTUP_DIR/live_sync.sh"

  # Ship image.conf so a site follows the release channel out of the box. docker.sh
  # sources this on every run, so re-tagging :current centrally updates every node on
  # its next start -- no action at the site. (A site still chooses WHEN to run; we
  # never pull or restart under them.) Don't clobber a site's own edit on re-issue.
  if [ ! -f "$STARTUP_DIR/image.conf" ]; then
    cat > "$STARTUP_DIR/image.conf" <<'IMAGECONF'
# Which ODELIA image this node runs. Read by docker.sh on every run.
#
# Default: the release channel. When a new ODELIA release ships, we re-tag
# :current and this node picks it up the next time you start it -- nothing to do
# here. The Run Board's "Run schedule" tab always names the exact tag of a run.
#
# To pin a specific version instead, replace the line below, e.g.
#   MEDISWARM_IMAGE=jefftud/odelia:1.6.0
# For a one-off without editing this file:  ./docker.sh --image <ref> ...
MEDISWARM_IMAGE=jefftud/odelia:current
IMAGECONF
  fi

  if [ -f "$STARTUP_DIR/fl_admin.sh" ]; then
    cp "$ADMIN_HELPER_SOURCE" "$STARTUP_DIR/prepare_odelia_job.sh"
    chmod +x "$STARTUP_DIR/prepare_odelia_job.sh"
    echo "Injected admin warm-start job helper into $STARTUP_DIR"
  fi

  # Clean up legacy docker_original.sh wrapper if present from a previous build
  if [ -f "$STARTUP_DIR/docker_original.sh" ]; then
    rm -f "$STARTUP_DIR/docker_original.sh"
  fi

  echo "Injected live sync helpers into $STARTUP_DIR"
done

echo "Live sync injection finished"
