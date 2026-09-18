#!/usr/bin/env bash
# =============================================================================
# Collect swarm global checkpoints + client logs from all sites (#160, #347)
# =============================================================================
# Pulls, from each swarm site:
#   - the latest aggregated global mirrored by WarmStartablePTFileModelPersistor
#     (default /scratch/mediswarm_latest_global.pt inside the client container)
#   - the NVFlare client log (log.txt at the startup-kit root)
# into a local directory, so the operator has a central copy of progress that is
# never stranded on a crashed node, and a checkpoint to warm-start the next run.
#
# Sites are read from a config file (default: ./swarm_sites.conf), one per line:
#     <SITE_NAME> <SSH_HOST> [<CLIENT_CONTAINER_NAME>]
# e.g.:
#     UKA   swarm@<SITE_IP>     odelia_swarm_client_UKA_1
# SSH auth uses your ssh config / agent (no credentials are stored here).
# If <CLIENT_CONTAINER_NAME> is omitted, the first container matching 'client'
# is used. Prefix the ssh command with sudo on the remote via your ssh config if
# docker requires it there.
#
# Usage:
#   ./collect_swarm_globals.sh [-f sites.conf] [-o out_dir] [-p container_global_path]
# =============================================================================

set -uo pipefail

SITES_FILE="./swarm_sites.conf"
OUT_DIR="./swarm_artifacts_$(date -u +%Y%m%dT%H%M%SZ)"
GLOBAL_PATH="/scratch/mediswarm_latest_global.pt"

while getopts "f:o:p:h" opt; do
    case "$opt" in
        f) SITES_FILE="$OPTARG" ;;
        o) OUT_DIR="$OPTARG" ;;
        p) GLOBAL_PATH="$OPTARG" ;;
        h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option"; exit 1 ;;
    esac
done

if [[ ! -f "$SITES_FILE" ]]; then
    echo "Sites file '$SITES_FILE' not found. See the header of this script for the format." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Collecting to $OUT_DIR (global mirror: $GLOBAL_PATH)"

while read -r SITE HOST CONTAINER _rest; do
    [[ -z "${SITE:-}" || "${SITE:0:1}" == "#" ]] && continue
    echo "=== $SITE ($HOST) ==="
    # resolve the client container name if not given
    if [[ -z "${CONTAINER:-}" ]]; then
        CONTAINER=$(ssh -o ConnectTimeout=15 "$HOST" "docker ps --format '{{.Names}}' | grep -i client | head -1" 2>/dev/null)
    fi
    if [[ -z "${CONTAINER:-}" ]]; then
        echo "  ! no client container found on $HOST"; continue
    fi

    # 1) latest global mirror
    if ssh -o ConnectTimeout=15 "$HOST" "docker exec '$CONTAINER' test -f '$GLOBAL_PATH'" 2>/dev/null; then
        ssh -o ConnectTimeout=30 "$HOST" "docker exec '$CONTAINER' cat '$GLOBAL_PATH'" > "$OUT_DIR/${SITE}_latest_global.pt" 2>/dev/null \
            && echo "  global  -> ${SITE}_latest_global.pt ($(du -h "$OUT_DIR/${SITE}_latest_global.pt" 2>/dev/null | cut -f1))" \
            || echo "  ! failed to pull global from $SITE"
    else
        echo "  - no global mirror at $GLOBAL_PATH yet"
    fi

    # 2) client log (NVFlare log.txt at the startup-kit root)
    ssh -o ConnectTimeout=20 "$HOST" "docker exec '$CONTAINER' bash -lc 'cat \$(ls -t /startupkit/log.txt /*/log.txt 2>/dev/null | head -1)'" \
        > "$OUT_DIR/${SITE}_log.txt" 2>/dev/null \
        && echo "  log     -> ${SITE}_log.txt" \
        || echo "  - no client log collected for $SITE"
done < "$SITES_FILE"

echo "Done. Artifacts in $OUT_DIR"
