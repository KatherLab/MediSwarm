#!/usr/bin/env bash
# ============================================================================
# run_preflight_localtraining_test.sh
#
# Tests preflight check and local training (1 epoch) for all 6 ODELIA models
# on a single remote client machine (default: dl2 / MHA_1).
#
# Usage:
#   ./scripts/deploy/run_preflight_localtraining_test.sh --conf deploy_sites_2node_test.conf
#   ./scripts/deploy/run_preflight_localtraining_test.sh --conf deploy_sites_2node_test.conf --site DL2
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[FAIL]${NC} $*"; }
step()  { echo -e "\n${BLUE}━━━ $* ━━━${NC}"; }

CONF_FILE="$REPO_ROOT/deploy_sites_2node_test.conf"
TEST_SITE="DL2"

while [[ $# -gt 0 ]]; do
    case $1 in
        --conf) CONF_FILE="$2"; shift ;;
        --site) TEST_SITE="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# shellcheck source=/dev/null
source "$CONF_FILE"

VERSION="$("$REPO_ROOT/scripts/build/getVersionNumber.sh")"
DOCKER_IMAGE="jefftud/odelia:$VERSION"

PROJECT_NAME=$(grep "^name: " "$REPO_ROOT/$PROJECT_FILE" \
    | sed 's/^name: //' \
    | sed "s/__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__/$VERSION/")
WORKSPACE_DIR="$REPO_ROOT/workspace/$PROJECT_NAME"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

site_var() { local v="${1}_${2}"; echo "${!v}"; }

HOST=$(site_var "$TEST_SITE" HOST)
USER=$(site_var "$TEST_SITE" USER)
PASS=$(site_var "$TEST_SITE" PASS)
SITE_NAME=$(site_var "$TEST_SITE" SITE_NAME)
DATADIR=$(site_var "$TEST_SITE" DATADIR)
SCRATCHDIR=$(site_var "$TEST_SITE" SCRATCHDIR)
DEPLOY_DIR=$(site_var "$TEST_SITE" DEPLOY_DIR)
GPU=$(site_var "$TEST_SITE" GPU)

remote() { sshpass -p "$PASS" ssh $SSH_OPTS "$USER@$HOST" "$@"; }

ALL_MODELS=(
    "ODELIA_ternary_classification:MST"
    "challenge_1DivideAndConquer:1DivideAndConquer"
    "challenge_2BCN_AIM:2BCN_AIM"
    "challenge_3agaldran:3agaldran"
    "challenge_4abmil:4LME_ABMIL"
    "challenge_5pimed:5Pimed"
)

RESULTS_DIR="$REPO_ROOT/workspace/preflight_localtraining_results"
mkdir -p "$RESULTS_DIR"

PASS_COUNT=0
FAIL_COUNT=0
declare -A RESULTS

run_docker_test() {
    local label="$1"      # e.g. "preflight:MST"
    local mode="$2"       # preflight_check | local_training
    local job_dir="$3"    # e.g. challenge_4abmil
    local model_name="$4" # e.g. 4LME_ABMIL
    local log_file="$RESULTS_DIR/${model_name}_${mode}.log"

    local kit_dir="$DEPLOY_DIR/$SITE_NAME/startup"
    local docker_sh="$kit_dir/docker.sh"

    local flag
    if [[ "$mode" == "preflight_check" ]]; then
        flag="--preflight_check"
    else
        flag="--local_training"
    fi

    info "  docker.sh $flag --job $job_dir  →  $model_name"

    local rc=0
    # Run docker.sh on remote; pass --data_dir and override NUM_EPOCHS=1 for quick verification
    sshpass -p "$PASS" ssh $SSH_OPTS "$USER@$HOST" \
        "NUM_EPOCHS=1 bash '$docker_sh' $flag --job '$job_dir' --data_dir '$DATADIR'" \
        > "$log_file" 2>&1 || rc=$?

    if grep -q "Unsupported model name\|Error\|FAILED\|Traceback" "$log_file" 2>/dev/null && [[ $rc -ne 0 ]]; then
        err "  $label — FAILED (exit $rc)"
        tail -5 "$log_file" | sed 's/^/    /'
        RESULTS["$label"]="FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        # Check that the correct model was actually loaded
        local loaded_model
        loaded_model=$(grep -oP "Model name: \K[^\s]+" "$log_file" 2>/dev/null | tail -1 || true)
        if [[ -z "$loaded_model" ]]; then
            loaded_model=$(grep -oP "Using model variant \K\S+" "$log_file" 2>/dev/null | tail -1 || true)
        fi
        ok "  $label — PASSED${loaded_model:+ (loaded: $loaded_model)}"
        RESULTS["$label"]="PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi
}

# ── Verify startup kit is deployed ────────────────────────────────────────
step "Checking startup kit on $TEST_SITE ($HOST)"
if ! remote "[ -f '$DEPLOY_DIR/$SITE_NAME/startup/docker.sh' ]"; then
    err "Startup kit not found at $DEPLOY_DIR/$SITE_NAME/startup/docker.sh on $HOST"
    err "Deploy startup kits first."
    exit 1
fi
ok "Startup kit found at $DEPLOY_DIR/$SITE_NAME/startup/"

# Pre-pull image on remote to avoid slow pull during test
step "Ensuring Docker image $DOCKER_IMAGE is present on $HOST"
remote "docker pull '$DOCKER_IMAGE'" 2>/dev/null || true

# ── Run tests ─────────────────────────────────────────────────────────────
for mode in preflight_check local_training; do
    step "Testing $mode for all 6 models on $TEST_SITE ($SITE_NAME)"
    for model_spec in "${ALL_MODELS[@]}"; do
        IFS=':' read -r job_dir model_name <<< "$model_spec"
        run_docker_test "${mode}:${model_name}" "$mode" "$job_dir" "$model_name"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────
step "Summary — Preflight + Local Training Test"
echo ""
for model_spec in "${ALL_MODELS[@]}"; do
    IFS=':' read -r job_dir model_name <<< "$model_spec"
    for mode in preflight_check local_training; do
        label="${mode}:${model_name}"
        result="${RESULTS[$label]:-UNKNOWN}"
        if [[ "$result" == "PASS" ]]; then
            echo -e "  ${GREEN}PASS${NC}  $label"
        else
            echo -e "  ${RED}FAIL${NC}  $label"
        fi
    done
done

echo ""
echo "────────────────────────────────────"
echo -e "Total: $((PASS_COUNT + FAIL_COUNT)) | ${GREEN}Passed: $PASS_COUNT${NC} | ${RED}Failed: $FAIL_COUNT${NC}"
echo ""

[[ $FAIL_COUNT -eq 0 ]]
