#!/usr/bin/env bash
set -euo pipefail

SITE_NAME="${1:-unknown}"
MODE="${2:-unknown}"
KIT_ROOT="${3:-}"
JOB_ID="${4:-}"
RUN_NAME="${5:-}"
STATUS="${6:-running}"
OUT_FILE="${7:-/tmp/mediswarm_heartbeat.json}"

timestamp="$(date -u +%FT%TZ)"
sync_version="${MEDISWARM_LIVE_SYNC_VERSION:-2}"
sync_status="${MEDISWARM_SYNC_STATUS:-ok}"
sync_error="${MEDISWARM_SYNC_ERROR:-}"

# Extract kit version from docker.sh (baked in at build time)
kit_version=""
docker_sh="${KIT_ROOT:+$KIT_ROOT/startup/docker.sh}"
if [ -n "$docker_sh" ] && [ -f "$docker_sh" ]; then
  kit_version="$(grep -oP '(?<=MEDISWARM_VERSION=)\S+' "$docker_sh" 2>/dev/null | head -1 || true)"
  [ -n "$kit_version" ] || kit_version="$(grep -oP '(?<=jefftud/odelia:)\S+' "$docker_sh" 2>/dev/null | head -1 || true)"
  # Strip trailing quotes/whitespace that may be captured from shell strings
  kit_version="${kit_version%\"}"
  kit_version="${kit_version#\"}"
fi

# Strip ANSI escape codes from RUN_NAME (may come from colored terminal output)
RUN_NAME="$(printf '%s' "$RUN_NAME" | sed 's/\x1b\[[0-9;]*m//g')"

# Capture hostname for the webviewer dashboard
hb_hostname="$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo "unknown")"
hb_ip="${MEDISWARM_SITE_IP:-}"
if [ -z "$hb_ip" ]; then
  hb_ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | awk 'NF && $1 !~ /^127\./ && $1 != "::1" {print $1; exit}' || true)"
fi
if [ -z "$hb_ip" ] && command -v ip >/dev/null 2>&1; then
  hb_ip="$(ip -o -4 addr show scope global 2>/dev/null | awk '{split($4, a, "/"); print a[1]; exit}' || true)"
fi

log_file=""
console_file=""
global_model=""
best_global_model=""
run_dir=""
last_ckpt=""
epoch_ckpt=""
tb_file=""
job_dir_seen="false"
run_dir_seen="false"

if [ -n "$KIT_ROOT" ] && [ -d "$KIT_ROOT" ]; then
  if [ -f "$KIT_ROOT/log.txt" ]; then
    log_file="$KIT_ROOT/log.txt"
  fi
  if [ -f "$KIT_ROOT/startup/nohup.out" ]; then
    console_file="$KIT_ROOT/startup/nohup.out"
  fi
fi

if [ -n "$JOB_ID" ] && [ -d "$KIT_ROOT/$JOB_ID/app_${SITE_NAME}" ]; then
  job_dir_seen="true"
  global_model="$KIT_ROOT/$JOB_ID/app_${SITE_NAME}/FL_global_model.pt"
  best_global_model="$KIT_ROOT/$JOB_ID/app_${SITE_NAME}/best_FL_global_model.pt"
fi

if [ -n "${SCRATCHDIR:-}" ] && [ -n "$RUN_NAME" ] && [ -d "${SCRATCHDIR}/runs/${SITE_NAME}/${RUN_NAME}" ]; then
  run_dir="${SCRATCHDIR}/runs/${SITE_NAME}/${RUN_NAME}"
  run_dir_seen="true"
  last_ckpt="$(find "$run_dir" -type f -name 'last.ckpt' 2>/dev/null | sort | tail -n 1 || true)"
  epoch_ckpt="$(find "$run_dir" -type f -name 'epoch=*.ckpt' 2>/dev/null | sort | tail -n 1 || true)"
  tb_file="$(find "$run_dir" -type f -name 'events.out.tfevents*' 2>/dev/null | sort | tail -n 1 || true)"
fi

export HB_SITE_NAME="$SITE_NAME"
export HB_MODE="$MODE"
export HB_JOB_ID="$JOB_ID"
export HB_RUN_NAME="$RUN_NAME"
export HB_TIMESTAMP="$timestamp"
export HB_STATUS="$STATUS"
export HB_KIT_VERSION="$kit_version"
export HB_KIT_ROOT="$KIT_ROOT"
export HB_LOG_FILE="$log_file"
export HB_CONSOLE_FILE="$console_file"
export HB_GLOBAL_MODEL="$global_model"
export HB_BEST_GLOBAL_MODEL="$best_global_model"
export HB_RUN_DIR="$run_dir"
export HB_LAST_CKPT="$last_ckpt"
export HB_EPOCH_CKPT="$epoch_ckpt"
export HB_TB_FILE="$tb_file"
export HB_HOSTNAME="$hb_hostname"
export HB_IP_ADDRESS="$hb_ip"
export HB_SYNC_VERSION="$sync_version"
export HB_SYNC_STATUS="$sync_status"
export HB_SYNC_ERROR="$sync_error"
export HB_JOB_DIR_SEEN="$job_dir_seen"
export HB_RUN_DIR_SEEN="$run_dir_seen"

python3 - "$OUT_FILE" <<'PY'
import json
import os
import sys

out_file = sys.argv[1]
data = {
    "site_name": os.environ.get("HB_SITE_NAME", ""),
    "mode": os.environ.get("HB_MODE", ""),
    "job_id": os.environ.get("HB_JOB_ID", ""),
    "run_name": os.environ.get("HB_RUN_NAME", ""),
    "timestamp": os.environ.get("HB_TIMESTAMP", ""),
    "status": os.environ.get("HB_STATUS", ""),
    "kit_version": os.environ.get("HB_KIT_VERSION", ""),
    "kit_root": os.environ.get("HB_KIT_ROOT", ""),
    "log_file": os.environ.get("HB_LOG_FILE", ""),
    "console_file": os.environ.get("HB_CONSOLE_FILE", ""),
    "global_model": os.environ.get("HB_GLOBAL_MODEL", ""),
    "best_global_model": os.environ.get("HB_BEST_GLOBAL_MODEL", ""),
    "run_dir": os.environ.get("HB_RUN_DIR", ""),
    "last_ckpt": os.environ.get("HB_LAST_CKPT", ""),
    "epoch_ckpt": os.environ.get("HB_EPOCH_CKPT", ""),
    "tb_file": os.environ.get("HB_TB_FILE", ""),
    "hostname": os.environ.get("HB_HOSTNAME", ""),
    "ip_address": os.environ.get("HB_IP_ADDRESS", ""),
    "sync_version": os.environ.get("HB_SYNC_VERSION", ""),
    "sync_status": os.environ.get("HB_SYNC_STATUS", ""),
    "sync_error": os.environ.get("HB_SYNC_ERROR", ""),
    "job_dir_seen": os.environ.get("HB_JOB_DIR_SEEN", "false") == "true",
    "run_dir_seen": os.environ.get("HB_RUN_DIR_SEEN", "false") == "true",
}
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY

echo "$OUT_FILE"
