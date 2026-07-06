#!/usr/bin/env bash
# ============================================================================
# run_stamp_startupkit_smoke.sh — Single-machine STAMP startup-kit smoke test
#
# Exercises the EXACT participant-facing commands from
# assets/readme/README.participant.STAMP.md against a freshly built STAMP
# startup kit's generated docker.sh, using synthetic data:
#
#   1. ./docker.sh --dummy_training                       (Docker + GPU sanity)
#   2. ./docker.sh --preflight_check                      (data loads, 1 epoch)
#   3. ./docker.sh --preflight_check --log_dataset_details (plausibility check)
#   4. ./docker.sh --local_training                       (local baseline)
#
# Unlike scripts/ci/runIntegrationTests.sh (which drives the Python entrypoint
# directly), this validates the generated docker.sh wrapper that partners
# actually run — the env-var forwarding, --data_dir mount, and mode dispatch.
#
# The STAMP image + startup kits must be built BEFORE running this:
#   ./scripts/build/buildDockerImageAndStartupKits.sh \
#       -p application/provision/project_deploy_test_2site.yml \
#       -d docker_config/Dockerfile_STAMP --use-docker-cache --num-rounds 2
#
# Usage:
#   ./scripts/deploy/run_stamp_startupkit_smoke.sh --kit-dir workspace/<project>/prod_00/RUMC_1
#   ./scripts/deploy/run_stamp_startupkit_smoke.sh --kit-dir <dir> --task classification --gpu device=0
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*" >&2; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
err()   { echo -e "${RED}[FAIL]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}=== $* ===${NC}" >&2; }

# ── Defaults ────────────────────────────────────────────────────────────────
KIT_DIR=""
TASK="classification"
GPU="device=0"
DIM_INPUT="1024"
NUM_CLASSES="3"
GROUND_TRUTH_LABEL="Diagnosis"
MODEL_NAME="vit"
DATA_DIR=""            # generated synthetic data root (auto if empty)
SCRATCH_DIR=""         # auto if empty
KEEP_DATA=false        # keep the synthetic data dir on exit

usage() {
    cat <<EOF
Usage: $0 --kit-dir DIR [options]

Required:
  --kit-dir DIR          Site startup-kit root (the dir containing startup/docker.sh),
                         e.g. workspace/<project>/prod_00/RUMC_1

Options:
  --task TASK            classification | survival | regression (default: classification)
  --gpu SPEC             GPU spec passed to docker.sh --GPU (default: device=0)
  --dim N                STAMP_DIM_INPUT (default: 1024)
  --num-classes N        STAMP_NUM_CLASSES (default: 3)
  --ground-truth LABEL   STAMP_GROUND_TRUTH_LABEL (default: Diagnosis; survival/regression differ)
  --model NAME           STAMP_MODEL_NAME (default: vit)
  --data-dir DIR         Reuse an existing synthetic data root instead of generating
  --scratch-dir DIR      Scratch dir (default: a fresh temp dir)
  --keep-data            Do not delete the generated synthetic data on exit
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kit-dir)      KIT_DIR="$2"; shift ;;
        --task)         TASK="$2"; shift ;;
        --gpu)          GPU="$2"; shift ;;
        --dim)          DIM_INPUT="$2"; shift ;;
        --num-classes)  NUM_CLASSES="$2"; shift ;;
        --ground-truth) GROUND_TRUTH_LABEL="$2"; shift ;;
        --model)        MODEL_NAME="$2"; shift ;;
        --data-dir)     DATA_DIR="$2"; shift ;;
        --scratch-dir)  SCRATCH_DIR="$2"; shift ;;
        --keep-data)    KEEP_DATA=true ;;
        -h|--help)      usage; exit 0 ;;
        *) err "Unknown argument: $1"; usage; exit 1 ;;
    esac
    shift
done

[[ -n "$KIT_DIR" ]] || { err "Must specify --kit-dir"; usage; exit 1; }
[[ "$KIT_DIR" = /* ]] || KIT_DIR="$REPO_ROOT/$KIT_DIR"
DOCKER_SH="$KIT_DIR/startup/docker.sh"
[[ -f "$DOCKER_SH" ]] || { err "docker.sh not found at $DOCKER_SH — build the startup kits first"; exit 1; }

# Site name is the kit dir basename (e.g. RUMC_1); the synthetic data dir must
# match so /data/<SITE_NAME>/{features,clini_table.csv} resolve inside Docker.
SITE_NAME="$(basename "$KIT_DIR")"

# Ground-truth label defaults per task (matches create_synthetic_stamp_dataset.py).
if [[ "$GROUND_TRUTH_LABEL" == "Diagnosis" ]]; then
    case "$TASK" in
        regression) GROUND_TRUTH_LABEL="Target" ;;
        survival)   GROUND_TRUTH_LABEL="" ;;   # survival uses Time/Event, not a single label
    esac
fi

VERSION="$("$REPO_ROOT/scripts/build/getVersionNumber.sh")"
DOCKER_IMAGE="jefftud/odelia:$VERSION"

# ── Scratch + synthetic data ────────────────────────────────────────────────
CLEANUP_DIRS=()
cleanup() { for d in "${CLEANUP_DIRS[@]:-}"; do [[ -n "$d" && -d "$d" ]] && rm -rf "$d"; done; }
trap cleanup EXIT

if [[ -z "$SCRATCH_DIR" ]]; then
    SCRATCH_DIR="$(mktemp -d -t stamp-smoke-scratch.XXXXXX)"
    CLEANUP_DIRS+=("$SCRATCH_DIR")
fi
mkdir -p "$SCRATCH_DIR"

if [[ -z "$DATA_DIR" ]]; then
    DATA_DIR="$(mktemp -d -t stamp-smoke-data.XXXXXX)"
    [[ "$KEEP_DATA" == true ]] || CLEANUP_DIRS+=("$DATA_DIR")
    step "Generating synthetic $TASK data (site: $SITE_NAME)"
    docker run --rm -v "$DATA_DIR":/data "$DOCKER_IMAGE" \
        python3 application/jobs/STAMP_classification/app/scripts/create_synthetic_dataset/create_synthetic_stamp_dataset.py \
        /data --task "$TASK"
    # Remap generated client_A -> the kit's site name so paths resolve.
    if [[ ! -d "$DATA_DIR/$SITE_NAME" ]]; then
        cp -r "$DATA_DIR/client_A" "$DATA_DIR/$SITE_NAME"
    fi
    ok "Synthetic data at $DATA_DIR (site dir: $SITE_NAME)"
fi

# ── STAMP env vars (forwarded into the container by docker.sh) ───────────────
export STAMP_CLINI_TABLE="/data/$SITE_NAME/clini_table.csv"
export STAMP_FEATURE_DIR="/data/$SITE_NAME/features"
export STAMP_TASK="$TASK"
export STAMP_MODEL_NAME="$MODEL_NAME"
export STAMP_DIM_INPUT="$DIM_INPUT"
export STAMP_NUM_CLASSES="$NUM_CLASSES"
export STAMP_PATIENT_LABEL="PATIENT"
[[ -n "$GROUND_TRUTH_LABEL" ]] && export STAMP_GROUND_TRUTH_LABEL="$GROUND_TRUTH_LABEL"
if [[ "$TASK" == "survival" ]]; then
    export STAMP_TIME_LABEL="Time"
    export STAMP_STATUS_LABEL="Event"
fi
# Keep the smoke fast.
export STAMP_BAG_SIZE="64"
export STAMP_BATCH_SIZE="8"
export STAMP_MAX_EPOCHS="1"
export STAMP_PATIENCE="1"
export STAMP_NUM_WORKERS="0"

RESULTS_DIR="$REPO_ROOT/workspace/stamp_startupkit_smoke_results"
mkdir -p "$RESULTS_DIR"

declare -A RESULTS
PASS_COUNT=0; FAIL_COUNT=0

run_step() {
    local label="$1"; shift
    local log="$RESULTS_DIR/${SITE_NAME}_${label}.log"
    info "$label  →  $*"
    local rc=0
    ( cd "$KIT_DIR/startup" && "$@" ) >"$log" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]] || grep -qE "Traceback|FATAL|Unsupported|Error:" "$log"; then
        err "$label — FAILED (exit $rc), see $log"
        tail -8 "$log" | sed 's/^/    /' >&2
        RESULTS["$label"]="FAIL"; FAIL_COUNT=$((FAIL_COUNT+1))
    else
        ok "$label — PASSED"
        RESULTS["$label"]="PASS"; PASS_COUNT=$((PASS_COUNT+1))
    fi
}

step "STAMP startup-kit smoke — image $DOCKER_IMAGE, site $SITE_NAME, task $TASK"

# 1. Dummy training (no data)
run_step dummy_training \
    ./docker.sh --scratch_dir "$SCRATCH_DIR" --GPU "$GPU" --dummy_training

# 2. Preflight check
run_step preflight_check \
    ./docker.sh --data_dir "$DATA_DIR" --scratch_dir "$SCRATCH_DIR" --GPU "$GPU" --preflight_check

# 3. Plausibility check (detailed dataset inspection)
run_step plausibility_check \
    ./docker.sh --data_dir "$DATA_DIR" --scratch_dir "$SCRATCH_DIR" --GPU "$GPU" --preflight_check --log_dataset_details

# 4. Local training (baseline)
run_step local_training \
    ./docker.sh --data_dir "$DATA_DIR" --scratch_dir "$SCRATCH_DIR" --GPU "$GPU" --local_training

# ── Summary ─────────────────────────────────────────────────────────────────
step "Summary — STAMP startup-kit smoke ($SITE_NAME / $TASK)"
for label in dummy_training preflight_check plausibility_check local_training; do
    result="${RESULTS[$label]:-UNKNOWN}"
    if [[ "$result" == "PASS" ]]; then echo -e "  ${GREEN}PASS${NC}  $label" >&2
    else echo -e "  ${RED}FAIL${NC}  $label" >&2; fi
done
echo "" >&2
echo -e "Total: $((PASS_COUNT+FAIL_COUNT)) | ${GREEN}Passed: $PASS_COUNT${NC} | ${RED}Failed: $FAIL_COUNT${NC}" >&2
echo "Logs: $RESULTS_DIR" >&2

[[ $FAIL_COUNT -eq 0 ]]
