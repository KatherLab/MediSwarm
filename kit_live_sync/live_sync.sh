#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/sync.conf"

MODE=""
SITE_NAME=""
KIT_ROOT=""
STARTUP_DIR=""
SCRATCHDIR=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --site-name)
      SITE_NAME="${2:-}"
      shift 2
      ;;
    --kit-root)
      KIT_ROOT="${2:-}"
      shift 2
      ;;
    --startup-dir)
      STARTUP_DIR="${2:-}"
      shift 2
      ;;
    --scratch-dir)
      SCRATCHDIR="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "$MODE" ] || [ -z "$SITE_NAME" ] || [ -z "$KIT_ROOT" ] || [ -z "$STARTUP_DIR" ]; then
  echo "Missing required arguments" >&2
  exit 1
fi

STATE_DIR="$STARTUP_DIR/.mediswarm_sync"
mkdir -p "$STATE_DIR"

LAST_CKPT_SYNC_FILE="$STATE_DIR/${MODE}_last_ckpt_sync_ts"
touch "$LAST_CKPT_SYNC_FILE"

ssh_cmd() {
  ssh ${SSH_OPTS} "$@"
}

rsync_cmd() {
  rsync -az --partial --mkpath -e "ssh ${SSH_OPTS}" "$@"
}

ensure_remote_dir() {
  local remote_dir="$1"
  ssh_cmd "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${remote_dir}'"
}

find_local_run_dir() {
  local local_base="$STARTUP_DIR/runs/$SITE_NAME"
  if [ ! -d "$local_base" ]; then
    return 0
  fi
  find "$local_base" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true
}

find_swarm_job_dir() {
  find "$KIT_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name startup 2>/dev/null | while read -r d; do
    if [ -f "$d/log.txt" ] || find "$d" -maxdepth 2 -type d -name "app_${SITE_NAME}" | grep -q .; then
      printf '%s\n' "$d"
    fi
  done | sort | tail -n 1 || true
}

resolve_run_dir() {
  if [ "$MODE" = "local" ]; then
    find_local_run_dir
  else
    find_swarm_job_dir
  fi
}

sync_console_and_log() {
  local run_dir="$1"
  local remote_dir="$2"
  local hb_file="$3"

  [ -f "$hb_file" ] && rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json"

  if [ "$MODE" = "local" ]; then
    [ -f "$STARTUP_DIR/local_training_console_output.txt" ] && \
      rsync_cmd "$STARTUP_DIR/local_training_console_output.txt" \
      "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/local_training_console_output.txt"
  else
    [ -f "$STARTUP_DIR/nohup.out" ] && \
      rsync_cmd "$STARTUP_DIR/nohup.out" \
      "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/nohup.out"

    [ -f "$run_dir/log.txt" ] && \
      rsync_cmd "$run_dir/log.txt" \
      "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/log.txt"
  fi
}

sync_tensorboard() {
  local run_dir="$1"
  local remote_dir="$2"

  rsync_cmd \
    --include='*/' \
    --include='events.out.tfevents*' \
    --exclude='*' \
    "$run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/"
}

sync_checkpoints_if_due() {
  local run_dir="$1"
  local remote_dir="$2"
  local now last

  now="$(date +%s)"
  last="$(cat "$LAST_CKPT_SYNC_FILE" 2>/dev/null || echo 0)"

  if [ $((now - last)) -lt "$CKPT_SYNC_INTERVAL" ]; then
    return 0
  fi

  rsync_cmd \
    --include='*/' \
    --include='last.ckpt' \
    --include='epoch=*.ckpt' \
    --include='FL_global_model.pt' \
    --include='*_model_gt_and_classprob_*.csv' \
    --exclude='*' \
    "$run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/"

  echo "$now" > "$LAST_CKPT_SYNC_FILE"
}

final_sync() {
  local run_dir
  run_dir="$(resolve_run_dir || true)"

  if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    exit 0
  fi

  local remote_dir
  remote_dir="${REMOTE_BASE}/${SITE_NAME}/${MODE}/$(basename "$run_dir")"
  ensure_remote_dir "$remote_dir"

  local hb_file="$STATE_DIR/${MODE}_heartbeat_final.json"
  "$SCRIPT_DIR/build_heartbeat.sh" "$SITE_NAME" "$MODE" "$run_dir" "finished" "$hb_file" >/dev/null

  rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat_final.json" || true
  sync_console_and_log "$run_dir" "$remote_dir" "$hb_file" || true
  rsync_cmd "$run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
}

trap final_sync EXIT

while true; do
  run_dir="$(resolve_run_dir || true)"

  if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    remote_dir="${REMOTE_BASE}/${SITE_NAME}/${MODE}/$(basename "$run_dir")"
    ensure_remote_dir "$remote_dir"

    hb_file="$STATE_DIR/${MODE}_heartbeat.json"
    "$SCRIPT_DIR/build_heartbeat.sh" "$SITE_NAME" "$MODE" "$run_dir" "running" "$hb_file" >/dev/null

    sync_console_and_log "$run_dir" "$remote_dir" "$hb_file" || true
    sync_tensorboard "$run_dir" "$remote_dir" || true
    sync_checkpoints_if_due "$run_dir" "$remote_dir" || true
  fi

  sleep "$LOG_SYNC_INTERVAL"
done