#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WATCH_LOG="${WATCH_LOG:-$PROJECT_ROOT/duke_results/duke_split_local_matrix_260417/logs/reorder_after_node_a.log}"
MATRIX_LOG="${MATRIX_LOG:-$PROJECT_ROOT/duke_results/duke_split_local_matrix_260417/logs/matrix_resume_after_node_a_to_c.log}"
RESULT_JSON="${RESULT_JSON:-$PROJECT_ROOT/duke_results/duke_split_local_matrix_260417/cells/full/node_A/5Pimed/result.json}"

REMOTE_HOST="${REMOTE_HOST:-dd-dl0}"
REMOTE_USER="${REMOTE_USER:-swarm}"
REMOTE_PASS="${REMOTE_PASS:?REMOTE_PASS must be set (export REMOTE_PASS=<password>)}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR}"

mkdir -p "$(dirname "$WATCH_LOG")"
: > "$WATCH_LOG"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "$WATCH_LOG"
}

cd "$PROJECT_ROOT"
log "waiting for $RESULT_JSON"

while [[ ! -f "$RESULT_JSON" ]]; do
    sleep 5
done

log "detected node_A/5Pimed completion"
pkill -f "run_duke_split_local_matrix\\.sh --matrix-only" || true
sleep 3

log "clearing node_B partial state"
rm -rf "$PROJECT_ROOT/duke_results/duke_split_local_matrix_260417/cells/full/node_B" || true
sshpass -p "$REMOTE_PASS" ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
    "rm -rf /mnt/dlhd0/duke_local_matrix_260417/full/node_B" || true

log "restarting matrix with node_C ahead of node_B"
setsid -f bash -lc "cd '$PROJECT_ROOT' && ./scripts/evaluation/run_duke_split_local_matrix.sh --matrix-only >> '$MATRIX_LOG' 2>&1"
log "restart dispatched"
