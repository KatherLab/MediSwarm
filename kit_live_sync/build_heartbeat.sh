#!/usr/bin/env bash
set -euo pipefail

SITE_NAME="${1:-unknown}"
MODE="${2:-unknown}"
RUN_DIR="${3:-}"
STATUS="${4:-running}"
OUT_FILE="${5:-/tmp/mediswarm_heartbeat.json}"

timestamp="$(date -u +%FT%TZ)"
run_id=""
log_file=""
console_file=""
last_ckpt=""
epoch_ckpt=""
tb_file=""
global_model=""

if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
  run_id="$(basename "$RUN_DIR")"

  if [ "$MODE" = "local" ]; then
    if [ -f "$(dirname "$RUN_DIR")/../local_training_console_output.txt" ]; then
      console_file="$(cd "$(dirname "$RUN_DIR")/../" && pwd)/local_training_console_output.txt"
    fi
  fi

  if [ "$MODE" = "swarm" ]; then
    if [ -f "$RUN_DIR/log.txt" ]; then
      log_file="$RUN_DIR/log.txt"
    fi
    if [ -f "$(cd "$(dirname "$RUN_DIR")/startup" 2>/dev/null && pwd)/nohup.out" ]; then
      console_file="$(cd "$(dirname "$RUN_DIR")/startup" && pwd)/nohup.out"
    fi
  fi

  last_ckpt="$(find "$RUN_DIR" -type f -name 'last.ckpt' 2>/dev/null | sort | tail -n 1 || true)"
  epoch_ckpt="$(find "$RUN_DIR" -type f -name 'epoch=*.ckpt' 2>/dev/null | sort | tail -n 1 || true)"
  tb_file="$(find "$RUN_DIR" -type f -name 'events.out.tfevents*' 2>/dev/null | sort | tail -n 1 || true)"
  global_model="$(find "$RUN_DIR" -type f -name 'FL_global_model.pt' 2>/dev/null | sort | tail -n 1 || true)"
fi

cat > "$OUT_FILE" <<EOF
{
  "site_name": "$SITE_NAME",
  "mode": "$MODE",
  "run_id": "$run_id",
  "timestamp": "$timestamp",
  "status": "$STATUS",
  "run_dir": "$RUN_DIR",
  "log_file": "$log_file",
  "console_file": "$console_file",
  "last_ckpt": "$last_ckpt",
  "epoch_ckpt": "$epoch_ckpt",
  "tb_file": "$tb_file",
  "global_model": "$global_model"
}
EOF

echo "$OUT_FILE"