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

if [ ! -d "$HELPER_SOURCE_DIR" ]; then
  echo "Missing helper directory: $HELPER_SOURCE_DIR"
  exit 1
fi

echo "Injecting live sync helpers into startup kits under $TARGET_FOLDER"

find "$TARGET_FOLDER" -mindepth 1 -maxdepth 1 -type d | while read -r KIT_DIR; do
  STARTUP_DIR="$KIT_DIR/startup"
  ORIGINAL_DOCKER_SH="$STARTUP_DIR/docker.sh"

  if [ ! -d "$STARTUP_DIR" ]; then
    continue
  fi

  if [ ! -f "$ORIGINAL_DOCKER_SH" ]; then
    continue
  fi

  cp "$HELPER_SOURCE_DIR/sync.conf" "$STARTUP_DIR/sync.conf"
  cp "$HELPER_SOURCE_DIR/build_heartbeat.sh" "$STARTUP_DIR/build_heartbeat.sh"
  cp "$HELPER_SOURCE_DIR/live_sync.sh" "$STARTUP_DIR/live_sync.sh"

  chmod +x "$STARTUP_DIR/build_heartbeat.sh" "$STARTUP_DIR/live_sync.sh"

  if [ ! -f "$STARTUP_DIR/docker_original.sh" ]; then
    mv "$ORIGINAL_DOCKER_SH" "$STARTUP_DIR/docker_original.sh"
    chmod +x "$STARTUP_DIR/docker_original.sh"
  fi

  cat > "$STARTUP_DIR/docker.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARTUP_DIR="$SCRIPT_DIR"
KIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORIGINAL="$SCRIPT_DIR/docker_original.sh"
STATE_DIR="$STARTUP_DIR/.mediswarm_sync"
mkdir -p "$STATE_DIR"

MODE=""
SCRATCHDIR=""
SITE_NAME_FROM_ARGS=""

parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --local_training)
        MODE="local"
        shift
        ;;
      --start_client)
        MODE="swarm"
        shift
        ;;
      --scratch_dir)
        SCRATCHDIR="${2:-}"
        shift 2
        ;;
      --site_name)
        SITE_NAME_FROM_ARGS="${2:-}"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
}

SITE_NAME_FALLBACK="$(basename "$KIT_ROOT")"
if [ -n "$SITE_NAME_FROM_ARGS" ]; then
  export SITE_NAME="$SITE_NAME_FROM_ARGS"
elif [ -z "${SITE_NAME:-}" ]; then
  export SITE_NAME="$SITE_NAME_FALLBACK"
fi

parse_args "$@"

start_local_sync() {
  "$SCRIPT_DIR/live_sync.sh" \
    --mode local \
    --site-name "$SITE_NAME" \
    --kit-root "$KIT_ROOT" \
    --startup-dir "$STARTUP_DIR" \
    --scratch-dir "${SCRATCHDIR:-}" &
  SYNC_PID=$!
}

stop_local_sync() {
  if [ -n "${SYNC_PID:-}" ] && kill -0 "$SYNC_PID" >/dev/null 2>&1; then
    kill "$SYNC_PID" || true
    wait "$SYNC_PID" || true
  fi
}

start_swarm_sync_daemon() {
  local pid_file="$STATE_DIR/swarm_sync.pid"

  if [ -f "$pid_file" ]; then
    old_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" >/dev/null 2>&1; then
      echo "Live sync daemon already running with PID $old_pid"
      return 0
    fi
  fi

  nohup "$SCRIPT_DIR/live_sync.sh" \
    --mode swarm \
    --site-name "$SITE_NAME" \
    --kit-root "$KIT_ROOT" \
    --startup-dir "$STARTUP_DIR" \
    --scratch-dir "${SCRATCHDIR:-}" \
    > "$STATE_DIR/live_sync_daemon.log" 2>&1 < /dev/null &

  echo $! > "$pid_file"
  echo "Started live sync daemon with PID $(cat "$pid_file")"
}

if [ "$MODE" = "local" ]; then
  trap stop_local_sync EXIT INT TERM
  start_local_sync
  exec "$ORIGINAL" "$@"
elif [ "$MODE" = "swarm" ]; then
  start_swarm_sync_daemon
  exec "$ORIGINAL" "$@"
else
  exec "$ORIGINAL" "$@"
fi
EOF

  chmod +x "$STARTUP_DIR/docker.sh"

  echo "Patched $STARTUP_DIR/docker.sh"
done

echo "Live sync injection finished"