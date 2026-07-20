#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_CONF="$SCRIPT_DIR/sync.conf"

_ask_continue_without_sync() {
    # In non-interactive environments (no TTY), default to continuing without sync
    if [ ! -t 0 ]; then
        echo "[INFO] Non-interactive mode — continuing without live sync."
        return 0
    fi
    read -r -p "Continue training without live sync? [y/N] " ans
    case "$ans" in
        [yY]*) return 0 ;;
        *) echo "Aborted."; return 1 ;;
    esac
}

if [ ! -f "$SYNC_CONF" ]; then
    echo ""
    echo "[WARN] Live-sync: sync.conf not found — training artifacts will not be synced."
    echo ""
    _ask_continue_without_sync && exit 0 || exit 1
fi

# shellcheck source=/dev/null
source "$SYNC_CONF"

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

# Ensure a writable state dir. A client previously started via `sudo` can leave
# .mediswarm_sync (and its files) root-owned; under `set -euo pipefail` the
# mkdir/touch below would then EACCES and silently kill the whole daemon so the
# site stops reporting (issue #398). Reclaim ownership if we can (passwordless
# sudo), otherwise fall back to a user-writable dir so live-sync keeps running.
_state_dir_writable() { [ -d "$STATE_DIR" ] && ( : > "$STATE_DIR/.wtest" ) 2>/dev/null && rm -f "$STATE_DIR/.wtest" 2>/dev/null; }
_ensure_writable_state_dir() {
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  _state_dir_writable && return 0
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo -n chown -R "$(id -u):$(id -g)" "$STATE_DIR" 2>/dev/null || true
  fi
  _state_dir_writable && return 0
  local fallback="${TMPDIR:-/tmp}/mediswarm_sync_$(id -u)_$(printf '%s' "$STARTUP_DIR" | sha256sum | cut -c1-8)"
  mkdir -p "$fallback" 2>/dev/null || true
  echo "[WARN] $STATE_DIR not writable (likely root-owned from a sudo-launched client); using $fallback. Fix: sudo chown -R \$(id -un): '$STATE_DIR'" >&2
  STATE_DIR="$fallback"
}
_ensure_writable_state_dir

LIVE_SYNC_LOG="$STATE_DIR/live_sync_daemon.log"
touch "$LIVE_SYNC_LOG" 2>/dev/null || true
if [ -w "$LIVE_SYNC_LOG" ]; then
  exec > >(tee -a "$LIVE_SYNC_LOG") 2>&1
else
  echo "[WARN] cannot write $LIVE_SYNC_LOG; logging to stdout only" >&2
fi

SYNC_STATUS="ok"
SYNC_ERROR=""

record_sync_error() {
  local msg="$1"
  SYNC_STATUS="error"
  SYNC_ERROR="$msg"
  echo "[WARN] $msg"
}

export_sync_env() {
  export MEDISWARM_SYNC_STATUS="$SYNC_STATUS"
  export MEDISWARM_SYNC_ERROR="$SYNC_ERROR"
  export MEDISWARM_LIVE_SYNC_VERSION="${MEDISWARM_LIVE_SYNC_VERSION:-2}"
}

# ── SSH connectivity check ──────────────────────────────────────────
# Verify we can reach the monitoring server before entering the sync
# loop.  With BatchMode=yes the connection will fail immediately if
# key-based auth is not set up instead of prompting for a password.
echo "Testing SSH connectivity to ${REMOTE_USER}@${REMOTE_HOST}..."
if ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" 'echo ok' >/dev/null 2>&1; then
    echo "SSH OK — live sync enabled."
else
    record_sync_error "Live-sync cannot connect to ${REMOTE_USER}@${REMOTE_HOST}"
    echo ""
    echo "============================================================"
    echo "[WARN] Live-sync: cannot connect to ${REMOTE_USER}@${REMOTE_HOST}"
    echo ""
    echo "  To enable live sync, share your SSH public key with your"
    echo "  swarm operator.  Generate one (if you haven't already) with:"
    echo ""
    echo "    ssh-keygen -t ed25519 -C \"\$(hostname)@mediswarm\""
    echo "    cat ~/.ssh/id_ed25519.pub  # send this to the swarm operator"
    echo ""
    echo "  Test connectivity manually with:"
    echo "    ssh ${SSH_OPTS} ${REMOTE_USER}@${REMOTE_HOST} 'echo ok'"
    echo "============================================================"
    echo ""
    _ask_continue_without_sync && exit 0 || exit 1
fi

LAST_CKPT_SYNC_FILE="$STATE_DIR/${MODE}_last_ckpt_sync_ts"
touch "$LAST_CKPT_SYNC_FILE" 2>/dev/null || true

# Track the run we are currently syncing so we can finalize it if a new run appears
CURRENT_LOCAL_RUN=""

ssh_cmd() {
  ssh ${SSH_OPTS} "$@" || {
    record_sync_error "ssh failed: $*"
    return 1
  }
}

rsync_cmd() {
  rsync -az --partial --mkpath -e "ssh ${SSH_OPTS}" "$@" || {
    record_sync_error "rsync failed: $*"
    return 1
  }
}

ensure_remote_dir() {
  local remote_dir="$1"
  ssh_cmd "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${remote_dir}'"
}

sync_daemon_log() {
  local remote_dir="$1"
  [ -f "$LIVE_SYNC_LOG" ] || return 0
  rsync_cmd "$LIVE_SYNC_LOG" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/live_sync_daemon.log" || true
}

build_heartbeat() {
  export_sync_env
  "$SCRIPT_DIR/build_heartbeat.sh" "$@"
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
  local base result=""
  # Check scratch dir first (primary location since all jobs now use SCRATCH_DIR)
  if [ -n "$SCRATCHDIR" ]; then
    base="$SCRATCHDIR/runs/$SITE_NAME"
    if [ -d "$base" ]; then
      result="$(find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null | \
        awk -v min_ts="${MEDISWARM_LOCAL_SYNC_START_TS:-0}" '$1 >= min_ts {print $2}' | \
        sort | tail -n 1 || true)"
    fi
  fi
  # Fall back to startup dir (backward compatibility with older builds)
  if [ -z "$result" ]; then
    base="$STARTUP_DIR/runs/$SITE_NAME"
    if [ -d "$base" ]; then
      result="$(find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null | \
        awk -v min_ts="${MEDISWARM_LOCAL_SYNC_START_TS:-0}" '$1 >= min_ts {print $2}' | \
        sort | tail -n 1 || true)"
    fi
  fi
  printf '%s' "$result"
}

build_remote_dir() {
  local run_id="$1"
  printf '%s/%s/%s/%s' "$REMOTE_BASE" "$SITE_NAME" "$MODE" "$run_id"
}

_finalize_local_run() {
  # Write a heartbeat_final.json for a given run_name and mark it "finished"
  local old_run="$1"
  [ -n "$old_run" ] || return 0

  local old_remote_dir old_hb_final old_run_dir
  old_remote_dir="$(build_remote_dir "$old_run")"
  old_hb_final="$STATE_DIR/local_heartbeat_final_${old_run}.json"
  export SCRATCHDIR
  build_heartbeat "$SITE_NAME" "local" "$KIT_ROOT" "" "$old_run" "finished" "$old_hb_final" >/dev/null
  rsync_cmd "$old_hb_final" "${REMOTE_USER}@${REMOTE_HOST}:${old_remote_dir}/heartbeat_final.json" || true
  sync_daemon_log "$old_remote_dir"

  # Final sync of the old run's artifacts
  old_run_dir=""
  if [ -n "$SCRATCHDIR" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$old_run" ]; then
    old_run_dir="$SCRATCHDIR/runs/$SITE_NAME/$old_run"
  elif [ -d "$STARTUP_DIR/runs/$SITE_NAME/$old_run" ]; then
    old_run_dir="$STARTUP_DIR/runs/$SITE_NAME/$old_run"
  fi
  if [ -n "$old_run_dir" ]; then
    rsync_cmd "$old_run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${old_remote_dir}/run_dir/" || true
  fi
}

sync_local() {
  local run_name run_dir remote_dir hb_file now last

  run_name="$(find_latest_local_run_name || true)"
  [ -n "$run_name" ] || return 0

  # If the run changed (new training started), finalize the old one
  if [ -n "$CURRENT_LOCAL_RUN" ] && [ "$CURRENT_LOCAL_RUN" != "$run_name" ]; then
    _finalize_local_run "$CURRENT_LOCAL_RUN" || true
  fi
  CURRENT_LOCAL_RUN="$run_name"

  # Determine run_dir: check scratch dir first, fall back to startup dir
  run_dir=""
  if [ -n "$SCRATCHDIR" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$run_name" ]; then
    run_dir="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
  elif [ -d "$STARTUP_DIR/runs/$SITE_NAME/$run_name" ]; then
    run_dir="$STARTUP_DIR/runs/$SITE_NAME/$run_name"
  fi
  [ -n "$run_dir" ] || return 0

  remote_dir="$(build_remote_dir "$run_name")"
  ensure_remote_dir "$remote_dir"

  hb_file="$STATE_DIR/local_heartbeat.json"
  export SCRATCHDIR
  build_heartbeat "$SITE_NAME" "local" "$KIT_ROOT" "" "$run_name" "running" "$hb_file" >/dev/null

  rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json" || true
  sync_daemon_log "$remote_dir"

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
    echo "$now" > "$LAST_CKPT_SYNC_FILE" 2>/dev/null || true
  fi
}

sync_swarm() {
  local job_id run_name remote_dir hb_file now last

  job_id="$(find_latest_job_id || true)"
  if [ -z "$job_id" ]; then
    remote_dir="$(build_remote_dir "_active")"
    ensure_remote_dir "$remote_dir" || return 0
    hb_file="$STATE_DIR/swarm_active_heartbeat.json"
    export SCRATCHDIR
    build_heartbeat "$SITE_NAME" "swarm" "$KIT_ROOT" "" "" "waiting" "$hb_file" >/dev/null
    rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json" || true
    [ -f "$STARTUP_DIR/nohup.out" ] && \
      rsync_cmd "$STARTUP_DIR/nohup.out" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/nohup.out" || true
    [ -f "$KIT_ROOT/log.txt" ] && \
      rsync_cmd "$KIT_ROOT/log.txt" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/log.txt" || true
    sync_daemon_log "$remote_dir"
    return 0
  fi

  ssh_cmd "${REMOTE_USER}@${REMOTE_HOST}" "rm -rf '$(build_remote_dir "_active")'" || true

  run_name="$(extract_run_name_from_nohup || true)"
  remote_dir="$(build_remote_dir "$job_id")"
  ensure_remote_dir "$remote_dir"

  export SCRATCHDIR
  hb_file="$STATE_DIR/swarm_heartbeat.json"
  build_heartbeat "$SITE_NAME" "swarm" "$KIT_ROOT" "$job_id" "$run_name" "running" "$hb_file" >/dev/null

  rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json" || true
  sync_daemon_log "$remote_dir"

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

  # Determine run dir: check scratch dir first, fall back to startup dir
  local run_dir_for_sync=""
  if [ -n "$SCRATCHDIR" ] && [ -n "$run_name" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$run_name" ]; then
    run_dir_for_sync="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
  elif [ -n "$run_name" ] && [ -d "$STARTUP_DIR/runs/$SITE_NAME/$run_name" ]; then
    run_dir_for_sync="$STARTUP_DIR/runs/$SITE_NAME/$run_name"
  fi

  if [ -n "$run_dir_for_sync" ]; then
    rsync_cmd \
      --include='*/' \
      --include='events.out.tfevents*' \
      --exclude='*' \
      "$run_dir_for_sync/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true

    now="$(date +%s)"
    last="$(cat "$LAST_CKPT_SYNC_FILE" 2>/dev/null || echo 0)"
    if [ $((now - last)) -ge "$CKPT_SYNC_INTERVAL" ]; then
      rsync_cmd \
        --include='*/' \
        --include='last.ckpt' \
        --include='epoch=*.ckpt' \
        --include='*_model_gt_and_classprob_*.csv' \
        --exclude='*' \
        "$run_dir_for_sync/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
      echo "$now" > "$LAST_CKPT_SYNC_FILE" 2>/dev/null || true
    fi
  fi
}

final_sync() {
  if [ "$MODE" = "local" ]; then
    sync_local || true
    local run_name run_dir
    run_name="$(find_latest_local_run_name || true)"
    if [ -n "$run_name" ]; then
      # Determine run_dir: check scratch dir first, fall back to startup dir
      run_dir=""
      if [ -n "$SCRATCHDIR" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$run_name" ]; then
        run_dir="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
      elif [ -d "$STARTUP_DIR/runs/$SITE_NAME/$run_name" ]; then
        run_dir="$STARTUP_DIR/runs/$SITE_NAME/$run_name"
      fi

      remote_dir="$(build_remote_dir "$run_name")"
      hb_file="$STATE_DIR/local_heartbeat_final.json"
      export SCRATCHDIR
      build_heartbeat "$SITE_NAME" "local" "$KIT_ROOT" "" "$run_name" "finished" "$hb_file" >/dev/null
      rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat_final.json" || true
      sync_daemon_log "$remote_dir"
      if [ -n "$run_dir" ]; then
        rsync_cmd "$run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
      fi
    fi
  else
    sync_swarm || true
    local job_id run_name remote_dir hb_file
    job_id="$(find_latest_job_id || true)"
    run_name="$(extract_run_name_from_nohup || true)"
    if [ -n "$job_id" ]; then
      remote_dir="$(build_remote_dir "$job_id")"
      hb_file="$STATE_DIR/swarm_heartbeat_final.json"
      export SCRATCHDIR
      build_heartbeat "$SITE_NAME" "swarm" "$KIT_ROOT" "$job_id" "$run_name" "finished" "$hb_file" >/dev/null
      rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat_final.json" || true
      sync_daemon_log "$remote_dir"

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

      # Sync run dir: check scratch dir first, fall back to startup dir
      local final_run_dir=""
      if [ -n "$SCRATCHDIR" ] && [ -n "$run_name" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$run_name" ]; then
        final_run_dir="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
      elif [ -n "$run_name" ] && [ -d "$STARTUP_DIR/runs/$SITE_NAME/$run_name" ]; then
        final_run_dir="$STARTUP_DIR/runs/$SITE_NAME/$run_name"
      fi
      if [ -n "$final_run_dir" ]; then
        rsync_cmd "$final_run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
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
