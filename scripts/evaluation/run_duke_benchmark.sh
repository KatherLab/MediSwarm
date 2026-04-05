#!/usr/bin/env bash
# ============================================================================
# run_duke_benchmark.sh — End-to-end Duke Breast MRI benchmark on dl0/dl2/dl3
#
# Orchestrates: build -> push -> deploy -> train (swarm) -> collect results ->
#               run benchmark_models.py -> summarise
#
# Usage:
#   ./run_duke_benchmark.sh [options]
#
# Options:
#   --project FILE     Project provision YAML (default: project_DUKE_test.yml)
#   --job JOB          Job to submit (default: ODELIA_ternary_classification)
#   --models M1 M2..   Models for local benchmark (default: all built-in)
#   --local-epochs N   Epochs for local benchmark_models.py (default: 5)
#   --skip-build       Skip Docker build (use existing image)
#   --skip-deploy      Skip deployment (already deployed)
#   --skip-swarm       Skip swarm training (only run local benchmark)
#   --skip-local       Skip local benchmark (only run swarm)
#   --collect-only     Only collect results from a previous swarm run
#   --output-dir DIR   Directory for results (default: ./duke_results)
#   --dry-run          Print what would be done without executing
#   -h, --help         Show this help
#
# Prerequisites:
#   - deploy_sites.conf configured with DL0/DL2/DL3 entries and
#     SITES=(DL0 DL2 DL3)
#   - sshpass, expect installed
#   - Duke dataset available on each site at the configured DATADIR paths
#   - GPU available on each site
#
# The script uses deploy_and_test.sh for the swarm pipeline and
# benchmark_models.py for local (single-site) model comparison.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_SCRIPT="$PROJECT_ROOT/deploy_and_test.sh"

# ── Defaults ───────────────────────────────────────────────────────────────
PROJECT_FILE="application/provision/project_DUKE_test.yml"
JOB_NAME="ODELIA_ternary_classification"
LOCAL_EPOCHS=5
OUTPUT_DIR="$PROJECT_ROOT/duke_results"
SKIP_BUILD=false
SKIP_DEPLOY=false
SKIP_SWARM=false
SKIP_LOCAL=false
COLLECT_ONLY=false
DRY_RUN=false
BENCHMARK_MODELS=()

# ── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[DUKE]${NC} $*"; }
ok()    { echo -e "${GREEN}[DUKE]${NC} $*"; }
warn()  { echo -e "${YELLOW}[DUKE]${NC} $*"; }
err()   { echo -e "${RED}[DUKE]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}═══ DUKE BENCHMARK: $* ═══${NC}"; }

# ── Parse Arguments ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)      PROJECT_FILE="$2"; shift 2 ;;
        --job)          JOB_NAME="$2"; shift 2 ;;
        --models)       shift; while [[ $# -gt 0 && "$1" != --* ]]; do
                            BENCHMARK_MODELS+=("$1"); shift
                        done ;;
        --local-epochs) LOCAL_EPOCHS="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=true; shift ;;
        --skip-deploy)  SKIP_DEPLOY=true; shift ;;
        --skip-swarm)   SKIP_SWARM=true; shift ;;
        --skip-local)   SKIP_LOCAL=true; shift ;;
        --collect-only) COLLECT_ONLY=true; SKIP_BUILD=true; SKIP_DEPLOY=true; shift ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)
            head -n 35 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Sanity Checks ──────────────────────────────────────────────────────────
if [[ ! -f "$DEPLOY_SCRIPT" ]]; then
    err "deploy_and_test.sh not found at: $DEPLOY_SCRIPT"
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/$PROJECT_FILE" ]]; then
    err "Project file not found: $PROJECT_ROOT/$PROJECT_FILE"
    exit 1
fi

# ── Summary ────────────────────────────────────────────────────────────────
step "Configuration"
info "Project file:   $PROJECT_FILE"
info "Job:            $JOB_NAME"
info "Local epochs:   $LOCAL_EPOCHS"
info "Output dir:     $OUTPUT_DIR"
info "Skip build:     $SKIP_BUILD"
info "Skip deploy:    $SKIP_DEPLOY"
info "Skip swarm:     $SKIP_SWARM"
info "Skip local:     $SKIP_LOCAL"
info "Collect only:   $COLLECT_ONLY"
if [[ ${#BENCHMARK_MODELS[@]} -gt 0 ]]; then
    info "Models:         ${BENCHMARK_MODELS[*]}"
else
    info "Models:         (all built-in)"
fi

if [[ "$DRY_RUN" == true ]]; then
    ok "Dry run — exiting."
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

# ── Timestamp ──────────────────────────────────────────────────────────────
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RESULTS_DIR="$OUTPUT_DIR/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"
info "Results directory: $RESULTS_DIR"

# Save configuration for reproducibility
cat > "$RESULTS_DIR/benchmark_config.json" <<CONFEOF
{
    "timestamp": "$TIMESTAMP",
    "project_file": "$PROJECT_FILE",
    "job_name": "$JOB_NAME",
    "local_epochs": $LOCAL_EPOCHS,
    "git_commit": "$(git -C "$PROJECT_ROOT" rev-parse HEAD)",
    "git_branch": "$(git -C "$PROJECT_ROOT" branch --show-current)",
    "version": "$("$PROJECT_ROOT/scripts/build/getVersionNumber.sh")"
}
CONFEOF

# ── Phase 1: Build & Push ─────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false && "$COLLECT_ONLY" == false ]]; then
    step "Phase 1: Build Docker image"
    (
        export PROJECT_FILE
        "$DEPLOY_SCRIPT" build
    )
    "$DEPLOY_SCRIPT" push
else
    info "Skipping build/push"
fi

# ── Phase 2: Deploy ───────────────────────────────────────────────────────
if [[ "$SKIP_DEPLOY" == false && "$COLLECT_ONLY" == false ]]; then
    step "Phase 2: Deploy to sites"
    (
        export PROJECT_FILE
        "$DEPLOY_SCRIPT" deploy
    )
else
    info "Skipping deploy"
fi

# ── Phase 3: Swarm Training ───────────────────────────────────────────────
if [[ "$SKIP_SWARM" == false && "$COLLECT_ONLY" == false ]]; then
    step "Phase 3: Swarm training"

    info "Starting server..."
    (export PROJECT_FILE; "$DEPLOY_SCRIPT" start-server)

    info "Starting clients..."
    (export PROJECT_FILE; "$DEPLOY_SCRIPT" start-clients)

    info "Waiting 30s for clients to register..."
    sleep 30

    info "Submitting job: $JOB_NAME"
    (export PROJECT_FILE; "$DEPLOY_SCRIPT" submit "$JOB_NAME")

    ok "Swarm training submitted. Monitor with:"
    info "  ./deploy_and_test.sh status"
    info "  ./deploy_and_test.sh logs <SITE>"
    echo ""
    warn "Wait for training to complete before running --collect-only."
    warn "Training typically takes 2-8 hours depending on dataset size and rounds."
else
    info "Skipping swarm training"
fi

# ── Phase 4: Collect Swarm Results ─────────────────────────────────────────
if [[ "$COLLECT_ONLY" == true || "$SKIP_SWARM" == false ]]; then
    step "Phase 4: Collect swarm results"

    SWARM_RESULTS_DIR="$RESULTS_DIR/swarm"
    mkdir -p "$SWARM_RESULTS_DIR"

    # Find workspace with checkpoints
    VERSION=$("$PROJECT_ROOT/scripts/build/getVersionNumber.sh")
    PROJECT_NAME=$(grep "^name: " "$PROJECT_ROOT/$PROJECT_FILE" \
        | sed 's/^name: //' \
        | sed "s/__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__/$VERSION/")
    WORKSPACE_DIR="$PROJECT_ROOT/workspace/$PROJECT_NAME"

    if [[ -d "$WORKSPACE_DIR" ]]; then
        # Find latest prod directory
        PROD_DIR=$(ls -d "$WORKSPACE_DIR"/prod_* 2>/dev/null | sort -V | tail -n 1)
        if [[ -n "$PROD_DIR" ]]; then
            info "Collecting from: $PROD_DIR"

            # Copy checkpoint files
            find "$PROD_DIR" -name "FL_global_model.pt" -o -name "best_FL_global_model.pt" | while read -r ckpt; do
                rel_path="${ckpt#"$PROD_DIR/"}"
                dest_dir="$SWARM_RESULTS_DIR/$(dirname "$rel_path")"
                mkdir -p "$dest_dir"
                cp "$ckpt" "$dest_dir/"
                info "  Collected: $rel_path"
            done

            # Copy CSV outputs
            find "$PROD_DIR" -name "*.csv" | while read -r csv; do
                rel_path="${csv#"$PROD_DIR/"}"
                dest_dir="$SWARM_RESULTS_DIR/$(dirname "$rel_path")"
                mkdir -p "$dest_dir"
                cp "$csv" "$dest_dir/"
            done

            # Run predict.py on collected checkpoints if available
            if [[ -f "$SCRIPT_DIR/predict.py" ]]; then
                info "Running prediction evaluation on swarm checkpoints..."
                python3 "$SCRIPT_DIR/predict.py" \
                    --workspace "$PROD_DIR" \
                    --best-only \
                    --ensemble \
                    --output-dir "$SWARM_RESULTS_DIR/predictions" \
                    2>&1 | tee "$SWARM_RESULTS_DIR/predict_output.log" || \
                    warn "predict.py failed — check $SWARM_RESULTS_DIR/predict_output.log"
            fi

            ok "Swarm results collected at: $SWARM_RESULTS_DIR"
        else
            warn "No prod_* directories found in $WORKSPACE_DIR"
        fi
    else
        warn "Workspace not found: $WORKSPACE_DIR"
        warn "Swarm training may not have completed yet."
    fi
fi

# ── Phase 5: Local Benchmark ──────────────────────────────────────────────
if [[ "$SKIP_LOCAL" == false ]]; then
    step "Phase 5: Local model benchmark"

    LOCAL_RESULTS_DIR="$RESULTS_DIR/local"
    mkdir -p "$LOCAL_RESULTS_DIR"

    BENCHMARK_ARGS=(
        --max-epochs "$LOCAL_EPOCHS"
        --output "$LOCAL_RESULTS_DIR/benchmark_results.json"
    )

    if [[ ${#BENCHMARK_MODELS[@]} -gt 0 ]]; then
        BENCHMARK_ARGS+=(--models "${BENCHMARK_MODELS[@]}")
    fi

    info "Running benchmark_models.py with args: ${BENCHMARK_ARGS[*]}"

    python3 "$SCRIPT_DIR/benchmark_models.py" "${BENCHMARK_ARGS[@]}" \
        2>&1 | tee "$LOCAL_RESULTS_DIR/benchmark_output.log" || \
        warn "benchmark_models.py failed — check $LOCAL_RESULTS_DIR/benchmark_output.log"

    ok "Local benchmark results at: $LOCAL_RESULTS_DIR"
else
    info "Skipping local benchmark"
fi

# ── Phase 6: Summary ──────────────────────────────────────────────────────
step "Summary"

echo ""
info "Results directory: $RESULTS_DIR"
info "Contents:"
find "$RESULTS_DIR" -type f | sort | while read -r f; do
    size=$(du -h "$f" | cut -f1)
    echo "  $size  ${f#"$RESULTS_DIR/"}"
done

echo ""
ok "Duke benchmark pipeline complete!"
info "Next steps:"
info "  1. Review results in $RESULTS_DIR/"
info "  2. Compare swarm vs local performance"
info "  3. Update docs/DUKE_BENCHMARK_RESULTS.md with findings"
