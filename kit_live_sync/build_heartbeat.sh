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

log_file=""
console_file=""
global_model=""
best_global_model=""
run_dir=""
last_ckpt=""
epoch_ckpt=""
tb_file=""

if [ -n "$KIT_ROOT" ] && [ -d "$KIT_ROOT" ]; then
  if [ -f "$KIT_ROOT/log.txt" ]; then
    log_file="$KIT_ROOT/log.txt"
  fi
  if [ -f "$KIT_ROOT/startup/nohup.out" ]; then
    console_file="$KIT_ROOT/startup/nohup.out"
  fi
fi

if [ -n "$JOB_ID" ] && [ -d "$KIT_ROOT/$JOB_ID/app_${SITE_NAME}" ]; then
  global_model="$KIT_ROOT/$JOB_ID/app_${SITE_NAME}/FL_global_model.pt"
  best_global_model="$KIT_ROOT/$JOB_ID/app_${SITE_NAME}/best_FL_global_model.pt"
fi

if [ -n "${SCRATCHDIR:-}" ] && [ -n "$RUN_NAME" ] && [ -d "${SCRATCHDIR}/runs/${SITE_NAME}/${RUN_NAME}" ]; then
  run_dir="${SCRATCHDIR}/runs/${SITE_NAME}/${RUN_NAME}"
  last_ckpt="$(find "$run_dir" -type f -name 'last.ckpt' 2>/dev/null | sort | tail -n 1 || true)"
  epoch_ckpt="$(find "$run_dir" -type f -name 'epoch=*.ckpt' 2>/dev/null | sort | tail -n 1 || true)"
  tb_file="$(find "$run_dir" -type f -name 'events.out.tfevents*' 2>/dev/null | sort | tail -n 1 || true)"
fi

cat > "$OUT_FILE" <<EOF
{
  "site_name": "$SITE_NAME",
  "mode": "$MODE",
  "job_id": "$JOB_ID",
  "run_name": "$RUN_NAME",
  "timestamp": "$timestamp",
  "status": "$STATUS",
  "kit_root": "$KIT_ROOT",
  "log_file": "$log_file",
  "console_file": "$console_file",
  "global_model": "$global_model",
  "best_global_model": "$best_global_model",
  "run_dir": "$run_dir",
  "last_ckpt": "$last_ckpt",
  "epoch_ckpt": "$epoch_ckpt",
  "tb_file": "$tb_file"
}
EOF

echo "$OUT_FILE"