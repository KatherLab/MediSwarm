#!/usr/bin/env bash
# run_duke_iid_startupkit_smoke.sh — End-to-end startup-kit smoke test for the
# Duke IID 3-client topology.
#
# This validates workflow mechanics, not model quality:
#   build image + startup kits with num_rounds=2
#   push image
#   deploy server/admin/client kits
#   submit all six ODELIA jobs through the admin kit
#   collect final global checkpoints
#   run dl0 prediction smoke tests on Duke + ODELIA data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC} $*" >&2; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}=== $* ===${NC}" >&2; }

CONF_FILE="$REPO_ROOT/deploy_sites_duke_iid.conf"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_ROOT="$REPO_ROOT/workspace/deploy_test_results/startupkit_smoke_$RUN_ID"
TIMEOUT_MINUTES=10080
NUM_ROUNDS=2
SKIP_BUILD=false
SKIP_PUSH=false
USE_DOCKER_CACHE=false
ALLOW_NON_MAIN=false

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --conf FILE            Deploy-site configuration file (default: deploy_sites_duke_iid.conf)
  --results-root DIR     Top-level output directory
  --timeout MINUTES      Per-model training timeout for run_deploy_test.sh
  --num-rounds N         Federated rounds to inject into all jobs (default: 2)
  --skip-build           Reuse existing image/startup kits
  --skip-push            Do not docker push the image
  --use-docker-cache     Pass --use-docker-cache to the build script
  --allow-non-main       Do not fail if current branch is not main
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conf)             CONF_FILE="$2"; shift ;;
        --results-root)     RESULTS_ROOT="$2"; shift ;;
        --timeout)          TIMEOUT_MINUTES="$2"; shift ;;
        --num-rounds)       NUM_ROUNDS="$2"; shift ;;
        --skip-build)       SKIP_BUILD=true ;;
        --skip-push)        SKIP_PUSH=true ;;
        --use-docker-cache) USE_DOCKER_CACHE=true ;;
        --allow-non-main)   ALLOW_NON_MAIN=true ;;
        -h|--help)          usage; exit 0 ;;
        *)                  err "Unknown argument: $1"; usage; exit 1 ;;
    esac
    shift
done

resolve_path() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        printf '%s\n' "$p"
    else
        printf '%s/%s\n' "$REPO_ROOT" "$p"
    fi
}

CONF_FILE="$(resolve_path "$CONF_FILE")"
RESULTS_ROOT="$(resolve_path "$RESULTS_ROOT")"

if [[ ! -f "$CONF_FILE" ]]; then
    err "Configuration file not found: $CONF_FILE"
    exit 1
fi

# shellcheck source=/dev/null
source "$CONF_FILE"

VERSION="$("$REPO_ROOT/scripts/build/getVersionNumber.sh")"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
DOCKER_IMAGE="jefftud/odelia:$VERSION"
PROJECT_FILE="${PROJECT_FILE:?PROJECT_FILE must be set in $CONF_FILE}"

TRAINING_RESULTS_DIR="$RESULTS_ROOT/training"
EVAL_RESULTS_DIR="$RESULTS_ROOT/eval"
mkdir -p "$TRAINING_RESULTS_DIR" "$EVAL_RESULTS_DIR"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

site_var() {
    local site="$1" var="$2"
    local full_var="${site}_${var}"
    echo "${!full_var}"
}

remote_exec_site() {
    local site="$1"; shift
    local host user pass
    host="$(site_var "$site" HOST)"
    user="$(site_var "$site" USER)"
    pass="$(site_var "$site" PASS)"
    if [[ -n "$pass" ]]; then
        sshpass -p "$pass" ssh $SSH_OPTS "$user@$host" "$@"
    else
        ssh $SSH_OPTS "$user@$host" "$@"
    fi
}

step "Preflight"
if [[ "$BRANCH" != "main" && "$ALLOW_NON_MAIN" != true ]]; then
    err "Current branch is '$BRANCH', expected 'main'. Use --allow-non-main to override."
    exit 1
fi
ok "Branch: $BRANCH ($GIT_SHA)"

for cmd in docker sshpass jq expect; do
    command -v "$cmd" >/dev/null || { err "Missing required command: $cmd"; exit 1; }
done
docker info >/dev/null
ok "Local Docker is available"

for site in "${CLIENT_SITES[@]}"; do
    site_name="$(site_var "$site" SITE_NAME)"
    datadir="$(site_var "$site" DATADIR)"
    scratchdir="$(site_var "$site" SCRATCHDIR)"
    info "Checking $site_name data/GPU/Docker"
    remote_exec_site "$site" "command -v docker >/dev/null && nvidia-smi >/dev/null && test -d '$datadir' && mkdir -p '$scratchdir'"
    ok "$site_name preflight passed"
done

remote_exec_site "DL0" "test -d '${DL0_DUKE_DATA:-/mnt/dlhd0/DUKE_iid}' && test -d '${DL0_ODELIA_DATA:-/mnt/dlhd0/medswarmdata}'"
ok "dl0 evaluation data paths are present"

if [[ "$SKIP_BUILD" != true ]]; then
    step "Build Docker image and startup kits"
    build_args=("-p" "$PROJECT_FILE" "--num-rounds" "$NUM_ROUNDS")
    if [[ "$USE_DOCKER_CACHE" == true ]]; then
        build_args+=("--use-docker-cache")
    fi
    (cd "$REPO_ROOT" && bash scripts/build/buildDockerImageAndStartupKits.sh "${build_args[@]}")
else
    warn "Skipping build; reusing existing startup kits/image for $DOCKER_IMAGE"
fi

if [[ "$SKIP_PUSH" != true ]]; then
    step "Push Docker image"
    docker push "$DOCKER_IMAGE"
    ok "Pushed $DOCKER_IMAGE"
else
    warn "Skipping docker push for $DOCKER_IMAGE"
fi

step "Train all six models through distributed startup kits"
"$REPO_ROOT/scripts/deploy/run_deploy_test.sh" \
    --all \
    --conf "$CONF_FILE" \
    --skip-eval \
    --results-dir "$TRAINING_RESULTS_DIR" \
    --timeout "$TIMEOUT_MINUTES"

step "Run dl0 prediction smoke evaluation"
"$REPO_ROOT/scripts/deploy/run_startupkit_smoke_eval_dl0.sh" \
    --conf "$CONF_FILE" \
    --checkpoint-root "$TRAINING_RESULTS_DIR" \
    --output-dir "$EVAL_RESULTS_DIR" \
    --image "$DOCKER_IMAGE"

{
    echo "# Duke IID Startup Kit Full Deploy Smoke Test"
    echo ""
    echo "- Run ID: \`$RUN_ID\`"
    echo "- Status: \`pass\`"
    echo "- Branch: \`$BRANCH\`"
    echo "- Git SHA: \`$GIT_SHA\`"
    echo "- Docker image: \`$DOCKER_IMAGE\`"
    echo "- Federated rounds: \`$NUM_ROUNDS\`"
    echo "- Config: \`$CONF_FILE\`"
    echo "- Training summary: \`$TRAINING_RESULTS_DIR/summary.json\`"
    echo "- Eval summary: \`$EVAL_RESULTS_DIR/summary.json\`"
    echo "- Eval report: \`$EVAL_RESULTS_DIR/REPORT.md\`"
} > "$RESULTS_ROOT/RUN.md"

ok "Full startup-kit smoke test passed. Report: $RESULTS_ROOT/RUN.md"
