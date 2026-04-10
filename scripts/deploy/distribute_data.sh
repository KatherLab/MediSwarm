#!/usr/bin/env bash
# ============================================================================
# distribute_data.sh — One-time data distribution for 4-client ODELIA deploy test
#
# Copies institution data from dl3 (source of truth) to dl0 and dl2.
# dl3 already has all 5 institution datasets locally, including UMCU_1 and CAM_1
# which it trains on (two clients on one machine).
#
# Data source: dl3:/mnt/swarm_alpha/Odelia_challange/ODELIA_Challenge_unilateral/
# Layout per institution: {INSTITUTION}/data_unilateral/ + metadata_unilateral/
#
# Target layout after distribution:
#   Cosmos: /mnt/sda1/ODELIA_Challenge_unilateral/{UKA_1}  (eval data, pre-existing)
#   dl0:    /mnt/dlhd0/odelia_data/RUMC_1/                 (copied from dl3)
#   dl2:    /mnt/sda1/odelia_data/MHA_1/                   (copied from dl3)
#   dl3:    /mnt/swarm_alpha/.../CAM_1/ + UMCU_1/          (source, pre-existing)
#
# Usage:
#   ./scripts/deploy/distribute_data.sh [--dry-run]
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${DISTRIBUTE_DATA_CONFIG:-$SCRIPT_DIR/distribute_data.conf}"

# ── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}=== $* ===${NC}"; }

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    warn "DRY RUN MODE — no files will be copied"
fi

# ── Configuration ──────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG_FILE" ]]; then
    err "Missing config file: $CONFIG_FILE"
    err "Copy $SCRIPT_DIR/distribute_data.conf.example to $CONFIG_FILE and fill in the host, user, and password values."
    exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

for required_var in \
    DL3_HOST DL3_USER DL3_PASS DL3_DATA_ROOT \
    DL0_HOST DL0_USER DL0_PASS DL0_DATA_DIR \
    DL2_HOST DL2_USER DL2_PASS DL2_DATA_DIR \
    COSMOS_EVAL_DIR SSH_OPTS
do
    if [[ -z "${!required_var:-}" ]] || [[ "${!required_var}" == "CHANGEME" ]]; then
        err "Required config value $required_var is missing in $CONFIG_FILE"
        exit 1
    fi
done

# ── Check dependencies ─────────────────────────────────────────────────────
if ! command -v sshpass &>/dev/null; then
    err "sshpass is required. Install with: sudo apt-get install sshpass"
    exit 1
fi

# ── Verify source data exists on dl3 ──────────────────────────────────────
step "Verifying source data on dl3 ($DL3_HOST)"

for inst in CAM_1 MHA_1 RUMC_1 UKA_1 UMCU_1; do
    if sshpass -p "$DL3_PASS" ssh $SSH_OPTS "$DL3_USER@$DL3_HOST" \
        "test -d '$DL3_DATA_ROOT/$inst/data_unilateral'"; then
        ok "  $inst/data_unilateral exists on dl3"
    else
        err "  $inst/data_unilateral NOT FOUND on dl3"
        exit 1
    fi
done

# ── Verify Cosmos evaluation data ─────────────────────────────────────────
step "Verifying evaluation data on Cosmos"

if [[ -d "$COSMOS_EVAL_DIR/UKA_1/data_unilateral" ]]; then
    ok "  UKA_1/data_unilateral exists locally (for evaluation)"
else
    err "  UKA_1/data_unilateral NOT FOUND at $COSMOS_EVAL_DIR/UKA_1/"
    exit 1
fi

# ── Helper: copy data from dl3 to a target machine ──────────────────────
# Strategy: SSH into dl3 and run scp FROM dl3 TO the target machine directly.
# This avoids downloading to Cosmos and re-uploading (which would be slow for
# large NIfTI datasets).
copy_data_via_dl3() {
    local institution=$1
    local target_host=$2
    local target_user=$3
    local target_pass=$4
    local target_dir=$5

    info "Copying $institution from dl3 → $target_host:$target_dir/$institution/"

    if $DRY_RUN; then
        info "  [DRY RUN] Would create $target_dir on $target_host"
        info "  [DRY RUN] Would scp $DL3_DATA_ROOT/$institution → $target_host:$target_dir/"
        return
    fi

    # Create target directory on destination machine
    sshpass -p "$target_pass" ssh $SSH_OPTS "$target_user@$target_host" \
        "mkdir -p '$target_dir'"

    # Check if data already exists on target
    if sshpass -p "$target_pass" ssh $SSH_OPTS "$target_user@$target_host" \
        "test -d '$target_dir/$institution/data_unilateral'"; then
        warn "  $institution already exists on $target_host — skipping (delete manually to re-copy)"
        return
    fi

    # SSH into dl3 and scp from there to the target machine
    # Note: dl3 must be able to reach the target via Tailscale IPs
    sshpass -p "$DL3_PASS" ssh $SSH_OPTS "$DL3_USER@$DL3_HOST" \
        "sshpass -p '$target_pass' scp -r $SSH_OPTS '$DL3_DATA_ROOT/$institution' '$target_user@$target_host:$target_dir/'"

    # Verify the copy
    if sshpass -p "$target_pass" ssh $SSH_OPTS "$target_user@$target_host" \
        "test -d '$target_dir/$institution/data_unilateral'"; then
        ok "  $institution copied successfully to $target_host"
    else
        err "  Failed to copy $institution to $target_host"
        exit 1
    fi
}

# ── Copy MHA_1 → dl2 ─────────────────────────────────────────────────────
step "Distributing MHA_1 to dl2 ($DL2_HOST)"
copy_data_via_dl3 "MHA_1" "$DL2_HOST" "$DL2_USER" "$DL2_PASS" "$DL2_DATA_DIR"

# ── Copy RUMC_1 → dl0 ────────────────────────────────────────────────────
step "Distributing RUMC_1 to dl0 ($DL0_HOST)"
copy_data_via_dl3 "RUMC_1" "$DL0_HOST" "$DL0_USER" "$DL0_PASS" "$DL0_DATA_DIR"

# ── Summary ───────────────────────────────────────────────────────────────
step "Data Distribution Summary"
echo ""
echo "  Cosmos (localhost):  $COSMOS_EVAL_DIR/UKA_1  [evaluation data, pre-existing]"
echo "  dl0 ($DL0_HOST):     $DL0_DATA_DIR/RUMC_1    [copied from dl3]"
echo "  dl2 ($DL2_HOST):     $DL2_DATA_DIR/MHA_1     [copied from dl3]"
echo "  dl3 ($DL3_HOST):     $DL3_DATA_ROOT/{CAM_1,UMCU_1}  [source, pre-existing]"
echo ""

if $DRY_RUN; then
    warn "DRY RUN complete — no files were copied"
else
    ok "Data distribution complete!"
fi
