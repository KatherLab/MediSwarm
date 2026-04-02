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
    --mode) MODE="${2:-}"; shift 2 ;;
    --site-name) SITE_NAME="${2:-}"; shift 2 ;;
    --kit-root) KIT_ROOT="${2:-}"; shift 2 ;;
    --startup-dir) STARTUP_DIR="${2:-}"; shift 2 ;;
    --scratch-dir) SCRATCHDIR="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -n "$MODE" ] || { echo "MODE missing" >&2; exit 1; }
[ -n "$SITE_NAME" ] || { echo "SITE_NAME missing" >&2; exit 1; }
[ -n "$KIT_ROOT" ] || { echo "KIT_ROOT missing" >&2; exit 1; }
[ -n "$STARTUP_DIR" ] || { echo "STARTUP_DIR missing" >&2; exit 1; }

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

find_latest_job_id() {
  find "$KIT_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name startup ! -name local ! -name transfer 2>/dev/null | while read -r d; do
    b="$(basename "$d")"
    case "$b" in
      *-*-*-*-*) printf '%s\n' "$b" ;;
    esac
  done | sort | tail -n 1 || true
}

extract_run_name_from_nohup() {
  local nohup_file="$STARTUP_DIR/nohup.out"
  [ -f "$nohup_file" ] || return 0

  grep -E 'Run name: |Run directory:' "$nohup_file" 2>/dev/null | tail -n 1 | \
    sed -E 's#.*Run name: ##; s#.*Run directory: .*/##' || true
}

find_latest_local_run_name() {
  local base="$STARTUP_DIR/runs/$SITE_NAME"
  [ -d "$base" ] || return 0
  find "$base" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sed 's#.*/##' | sort | tail -n 1 || true
}

build_remote_dir() {
  local run_id="$1"
  printf '%s/%s/%s/%s' "$REMOTE_BASE" "$SITE_NAME" "$MODE" "$run_id"
}

sync_local() {
  local run_name run_dir remote_dir hb_file now last

  run_name="$(find_latest_local_run_name || true)"
  [ -n "$run_name" ] || return 0
  run_dir="$STARTUP_DIR/runs/$SITE_NAME/$run_name"
  [ -d "$run_dir" ] || return 0

  remote_dir="$(build_remote_dir "$run_name")"
  ensure_remote_dir "$remote_dir"

  export SCRATCHDIR=""
  hb_file="$STATE_DIR/local_heartbeat.json"
  "$SCRIPT_DIR/build_heartbeat.sh" "$SITE_NAME" "local" "$KIT_ROOT" "" "$run_name" "running" "$hb_file" >/dev/null

  rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json" || true

  [ -f "$STARTUP_DIR/local_training_console_output.txt" ] && \
    rsync_cmd "$STARTUP_DIR/local_training_console_output.txt" \
      "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/local_training_console_output.txt" || true

  now="$(date +%s)"
  last="$(cat "$LAST_CKPT_SYNC_FILE" 2>/dev/null || echo 0)"

  rsync_cmd \
    --include='*/' \
    --include='events.out.tfevents*' \
    --exclude='*' \
    "$run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true

  if [ $((now - last)) -ge "$CKPT_SYNC_INTERVAL" ]; then
    rsync_cmd \
      --include='*/' \
      --include='last.ckpt' \
      --include='epoch=*.ckpt' \
      --include='*_model_gt_and_classprob_*.csv' \
      --exclude='*' \
      "$run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
    echo "$now" > "$LAST_CKPT_SYNC_FILE"
  fi
}

sync_swarm() {
  local job_id run_name remote_dir hb_file now last scratch_run_dir

  job_id="$(find_latest_job_id || true)"
  [ -n "$job_id" ] || return 0

  run_name="$(extract_run_name_from_nohup || true)"
  remote_dir="$(build_remote_dir "$job_id")"
  ensure_remote_dir "$remote_dir"

  export SCRATCHDIR
  hb_file="$STATE_DIR/swarm_heartbeat.json"
  "$SCRIPT_DIR/build_heartbeat.sh" "$SITE_NAME" "swarm" "$KIT_ROOT" "$job_id" "$run_name" "running" "$hb_file" >/dev/null

  rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json" || true

  [ -f "$STARTUP_DIR/nohup.out" ] && \
    rsync_cmd "$STARTUP_DIR/nohup.out" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/nohup.out" || true

  [ -f "$KIT_ROOT/log.txt" ] && \
    rsync_cmd "$KIT_ROOT/log.txt" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/log.txt" || true

  [ -f "$KIT_ROOT/$job_id/app_${SITE_NAME}/FL_global_model.pt" ] && \
    rsync_cmd "$KIT_ROOT/$job_id/app_${SITE_NAME}/FL_global_model.pt" \
      "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/FL_global_model.pt" || true

  [ -f "$KIT_ROOT/$job_id/app_${SITE_NAME}/best_FL_global_model.pt" ] && \
    rsync_cmd "$KIT_ROOT/$job_id/app_${SITE_NAME}/best_FL_global_model.pt" \
      "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/best_FL_global_model.pt" || true

  if [ -n "$SCRATCHDIR" ] && [ -n "$run_name" ]; then
    scratch_run_dir="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
  else
    scratch_run_dir=""
  fi

  if [ -n "$scratch_run_dir" ] && [ -d "$scratch_run_dir" ]; then
    rsync_cmd \
      --include='*/' \
      --include='events.out.tfevents*' \
      --exclude='*' \
      "$scratch_run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true

    now="$(date +%s)"
    last="$(cat "$LAST_CKPT_SYNC_FILE" 2>/dev/null || echo 0)"
    if [ $((now - last)) -ge "$CKPT_SYNC_INTERVAL" ]; then
      rsync_cmd \
        --include='*/' \
        --include='last.ckpt' \
        --include='epoch=*.ckpt' \
        --include='*_model_gt_and_classprob_*.csv' \
        --exclude='*' \
        "$scratch_run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
      echo "$now" > "$LAST_CKPT_SYNC_FILE"
    fi
  fi
}

final_sync() {
  if [ "$MODE" = "local" ]; then
    sync_local || true
    local run_name
    run_name="$(find_latest_local_run_name || true)"
    if [ -n "$run_name" ]; then
      remote_dir="$(build_remote_dir "$run_name")"
      hb_file="$STATE_DIR/local_heartbeat_final.json"
      export SCRATCHDIR=""
      "$SCRIPT_DIR/build_heartbeat.sh" "$SITE_NAME" "local" "$KIT_ROOT" "" "$run_name" "finished" "$hb_file" >/dev/null
      rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat_final.json" || true
      rsync_cmd "$STARTUP_DIR/runs/$SITE_NAME/$run_name/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
    fi
  else
    sync_swarm || true
    local job_id run_name remote_dir hb_file scratch_run_dir
    job_id="$(find_latest_job_id || true)"
    run_name="$(extract_run_name_from_nohup || true)"
    if [ -n "$job_id" ]; then
      remote_dir="$(build_remote_dir "$job_id")"
      hb_file="$STATE_DIR/swarm_heartbeat_final.json"
      export SCRATCHDIR
      "$SCRIPT_DIR/build_heartbeat.sh" "$SITE_NAME" "swarm" "$KIT_ROOT" "$job_id" "$run_name" "finished" "$hb_file" >/dev/null
      rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat_final.json" || true

      [ -f "$STARTUP_DIR/nohup.out" ] && \
        rsync_cmd "$STARTUP_DIR/nohup.out" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/nohup.out" || true

      [ -f "$KIT_ROOT/log.txt" ] && \
        rsync_cmd "$KIT_ROOT/log.txt" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/log.txt" || true

      [ -f "$KIT_ROOT/$job_id/app_${SITE_NAME}/FL_global_model.pt" ] && \
        rsync_cmd "$KIT_ROOT/$job_id/app_${SITE_NAME}/FL_global_model.pt" \
          "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/FL_global_model.pt" || true

      [ -f "$KIT_ROOT/$job_id/app_${SITE_NAME}/best_FL_global_model.pt" ] && \
        rsync_cmd "$KIT_ROOT/$job_id/app_${SITE_NAME}/best_FL_global_model.pt" \
          "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/best_FL_global_model.pt" || true

      if [ -n "$SCRATCHDIR" ] && [ -n "$run_name" ]; then
        scratch_run_dir="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
        [ -d "$scratch_run_dir" ] && \
          rsync_cmd "$scratch_run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
      fi
    fi
  fi
}

trap final_sync EXIT

while true; do
  if [ "$MODE" = "local" ]; then
    sync_local || true
  else
    sync_swarm || true
  fi
  sleep "$LOG_SYNC_INTERVAL"
done