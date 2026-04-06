#!/usr/bin/env bash
# ============================================================================
# run_deploy_test.sh — Orchestrate multi-site ODELIA deploy tests
#
# Runs all 6 ODELIA models through a full federated training + evaluation
# cycle across 4 physical machines connected via Tailscale VPN.
#
# For each model:
#   1. (Re)start the NVFlare server on Cosmos
#   2. Start clients on all 4 sites with the correct MODEL_NAME
#   3. Submit the corresponding training job
#   4. Poll for training completion (Server runner finished)
#   5. Stop all containers
#   6. Evaluate the final global model on UKA_1 (held-out test site)
#   7. Record pass/fail + metrics
#
# Usage:
#   ./scripts/deploy/run_deploy_test.sh --all --conf deploy_sites_4node_test.conf
#   ./scripts/deploy/run_deploy_test.sh --model MST --job ODELIA_ternary_classification --conf deploy_sites_4node_test.conf
#
# The Docker image and startup kits must be built BEFORE running this script:
#   ./scripts/build/buildDockerImageAndStartupKits.sh -p application/provision/project_deploy_test_4site.yml
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

# ── Parse arguments ────────────────────────────────────────────────────────
RUN_ALL=false
SINGLE_MODEL=""
SINGLE_JOB=""
CONF_FILE=""
SKIP_BUILD=false
TIMEOUT_MINUTES=240   # Per-model training timeout (4 hours)

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)          RUN_ALL=true ;;
        --model)        SINGLE_MODEL="$2"; shift ;;
        --job)          SINGLE_JOB="$2"; shift ;;
        --conf)         CONF_FILE="$2"; shift ;;
        --skip-build)   SKIP_BUILD=true ;;
        --timeout)      TIMEOUT_MINUTES="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--all | --model NAME --job JOB_DIR] --conf CONF_FILE [--skip-build] [--timeout MINUTES]"
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

# Resolve conf file path
if [[ -z "$CONF_FILE" ]]; then
    # Default to deploy_sites_4node_test.conf in repo root
    CONF_FILE="$REPO_ROOT/deploy_sites_4node_test.conf"
fi
if [[ ! -f "$CONF_FILE" ]]; then
    # Try relative to repo root
    if [[ -f "$REPO_ROOT/$CONF_FILE" ]]; then
        CONF_FILE="$REPO_ROOT/$CONF_FILE"
    else
        err "Configuration file not found: $CONF_FILE"
        exit 1
    fi
fi

# Validate arguments
if [[ "$RUN_ALL" == false && -z "$SINGLE_MODEL" ]]; then
    err "Must specify --all or --model NAME --job JOB_DIR"
    exit 1
fi

# ── Load configuration ─────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$CONF_FILE"

VERSION=$("$REPO_ROOT/scripts/build/getVersionNumber.sh")
DOCKER_IMAGE="jefftud/odelia:$VERSION"

PROJECT_NAME=$(grep "^name: " "$REPO_ROOT/$PROJECT_FILE" \
    | sed 's/^name: //' \
    | sed "s/__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__/$VERSION/")
WORKSPACE_DIR="$REPO_ROOT/workspace/$PROJECT_NAME"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Results directory
RESULTS_DIR="$REPO_ROOT/workspace/deploy_test_results"
mkdir -p "$RESULTS_DIR"

# ── All 6 ODELIA models ───────────────────────────────────────────────────
# Format: JOB_DIR:MODEL_NAME
ALL_MODELS=(
    "ODELIA_ternary_classification:MST"
    "challenge_1DivideAndConquer:1DivideAndConquer"
    "challenge_2BCN_AIM:2BCN_AIM"
    "challenge_3agaldran:3agaldran"
    "challenge_4abmil:4LME_ABMIL"
    "challenge_5pimed:5Pimed"
)

# ── Evaluation configuration ──────────────────────────────────────────────
# UKA_1 is the held-out test site — data is on Cosmos
EVAL_SITE_NAME="UKA_1"
EVAL_DATA_DIR="/mnt/sda1/ODELIA_Challenge_unilateral"
EVAL_SCRATCH_DIR="/mnt/scratch/deploy_test_eval"

# ── Helper functions (same pattern as deploy_and_test.sh) ──────────────────

site_var() {
    local site=$1 var=$2
    local full_var="${site}_${var}"
    echo "${!full_var}"
}

remote_exec() {
    local site=$1; shift
    local host user pass
    host=$(site_var "$site" HOST)
    user=$(site_var "$site" USER)
    pass=$(site_var "$site" PASS)

    if [[ "$host" == "localhost" ]]; then
        # Local execution (for Cosmos)
        eval "$@"
    else
        sshpass -p "$pass" ssh $SSH_OPTS "$user@$host" "$@"
    fi
}

remote_copy() {
    local site=$1 src=$2 dst=$3
    local host user pass
    host=$(site_var "$site" HOST)
    user=$(site_var "$site" USER)
    pass=$(site_var "$site" PASS)

    if [[ "$host" == "localhost" ]]; then
        cp "$src" "$dst"
    else
        sshpass -p "$pass" scp $SSH_OPTS "$src" "$user@$host:$dst"
    fi
}

find_latest_prod() {
    if [[ ! -d "$WORKSPACE_DIR" ]]; then
        err "Workspace not found: $WORKSPACE_DIR"
        err "Run buildDockerImageAndStartupKits.sh first."
        exit 1
    fi
    ls -d "$WORKSPACE_DIR"/prod_* 2>/dev/null | sort -V | tail -n 1
}

# ── Stop all containers ───────────────────────────────────────────────────
stop_all() {
    info "Stopping all NVFlare containers..."

    # Stop local containers
    local local_containers
    local_containers=$(docker ps --format '{{.Names}}' | grep -E "odelia_swarm|nvflare" || true)
    if [[ -n "$local_containers" ]]; then
        echo "$local_containers" | xargs docker kill 2>/dev/null || true
        echo "$local_containers" | xargs docker rm -f 2>/dev/null || true
    fi

    # Stop remote containers
    for site in "${SITES[@]}"; do
        local host
        host=$(site_var "$site" HOST)
        if [[ "$host" == "localhost" ]]; then
            continue  # Already handled above
        fi
        remote_exec "$site" \
            "docker ps --format '{{.Names}}' | grep -E 'odelia_swarm|nvflare' | xargs -r docker kill 2>/dev/null; \
             docker ps -a --format '{{.Names}}' | grep -E 'odelia_swarm|nvflare' | xargs -r docker rm -f 2>/dev/null" \
            2>/dev/null || warn "  Could not stop containers on $host"
    done

    # Wait for containers to fully stop
    sleep 5
    ok "All containers stopped"
}

# ── Deploy startup kits ───────────────────────────────────────────────────
deploy_kits() {
    local prod_dir
    prod_dir=$(find_latest_prod)
    info "Deploying startup kits from: $prod_dir"

    for site in "${SITES[@]}"; do
        local site_name host deploy_dir
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)

        local zip_file="$prod_dir/${site_name}_${VERSION}.zip"
        if [[ ! -f "$zip_file" ]]; then
            # Some site names in the startup kit may differ — try the zip
            zip_file=$(ls "$prod_dir"/${site_name}*.zip 2>/dev/null | head -1 || true)
            if [[ -z "$zip_file" ]]; then
                err "Startup kit not found for $site_name in $prod_dir"
                exit 1
            fi
        fi

        if [[ "$host" == "localhost" ]]; then
            mkdir -p "$deploy_dir"
            cp "$zip_file" "$deploy_dir/"
            cd "$deploy_dir" && rm -rf "${site_name}" && unzip -qo "$(basename "$zip_file")"
            cd "$REPO_ROOT"
        else
            remote_exec "$site" "mkdir -p '$deploy_dir'"
            remote_copy "$site" "$zip_file" "$deploy_dir/"
            remote_exec "$site" "cd '$deploy_dir' && rm -rf '${site_name}' && unzip -qo '$(basename "$zip_file")'"
        fi
        ok "  Deployed $site_name to $host:$deploy_dir/"
    done

    # Also deploy the server and admin kits locally
    local server_name="${SERVER_NAME:-dl3.tud.de}"
    local server_zip="$prod_dir/${server_name}_${VERSION}.zip"
    if [[ ! -f "$server_zip" ]]; then
        server_zip=$(ls "$prod_dir"/${server_name}*.zip 2>/dev/null | head -1 || true)
    fi
    if [[ -n "$server_zip" && -f "$server_zip" ]]; then
        local deploy_base
        deploy_base=$(site_var "$(echo "${SITES[0]}")" DEPLOY_DIR)
        mkdir -p "$deploy_base"
        cp "$server_zip" "$deploy_base/"
        cd "$deploy_base" && rm -rf "$server_name" && unzip -qo "$(basename "$server_zip")"
        cd "$REPO_ROOT"
        ok "  Deployed server kit ($server_name) locally"
    fi

    local admin_zip="$prod_dir/${ADMIN_USER}_${VERSION}.zip"
    if [[ ! -f "$admin_zip" ]]; then
        admin_zip=$(ls "$prod_dir"/${ADMIN_USER}*.zip 2>/dev/null | head -1 || true)
    fi
    if [[ -n "$admin_zip" && -f "$admin_zip" ]]; then
        local deploy_base
        deploy_base=$(site_var "$(echo "${SITES[0]}")" DEPLOY_DIR)
        cp "$admin_zip" "$deploy_base/"
        cd "$deploy_base" && rm -rf "$ADMIN_USER" && unzip -qo "$(basename "$admin_zip")"
        cd "$REPO_ROOT"
        ok "  Deployed admin kit ($ADMIN_USER) locally"
    fi
}

# ── Start server ──────────────────────────────────────────────────────────
start_server() {
    local prod_dir deploy_base server_name server_startup
    prod_dir=$(find_latest_prod)
    deploy_base=$(site_var "$(echo "${SITES[0]}")" DEPLOY_DIR)
    server_name="${SERVER_NAME:-dl3.tud.de}"
    server_startup="$deploy_base/$server_name/startup"

    if [[ ! -d "$server_startup" ]]; then
        # Fall back to prod_dir
        server_startup="$prod_dir/$server_name/startup"
    fi

    if [[ ! -d "$server_startup" ]]; then
        err "Server startup kit not found at $server_startup"
        exit 1
    fi

    info "Starting server from: $server_startup"
    cd "$server_startup"
    ./docker.sh --no_pull --start_server
    cd "$REPO_ROOT"

    info "Waiting 15s for server to initialize..."
    sleep 15

    if docker ps --format '{{.Names}}' | grep -qE "odelia_swarm|nvflare"; then
        ok "Server container is running"
    else
        warn "Server container not detected — it may still be starting"
    fi
}

# ── Start clients ─────────────────────────────────────────────────────────
start_clients() {
    local model_name="${1:-}"
    local model_flag=""
    if [[ -n "$model_name" ]]; then
        model_flag="--model_name '$model_name'"
        info "Starting clients with MODEL_NAME=$model_name"
    fi

    for site in "${SITES[@]}"; do
        local site_name host deploy_dir datadir scratchdir gpu
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)
        datadir=$(site_var "$site" DATADIR)
        scratchdir=$(site_var "$site" SCRATCHDIR)
        gpu=$(site_var "$site" GPU)

        info "Starting client: $site_name @ $host"

        if [[ "$host" == "localhost" ]]; then
            cd "$deploy_dir/$site_name/startup"
            export SITE_NAME="$site_name"
            export DATADIR="$datadir"
            export SCRATCHDIR="$scratchdir"
            eval "./docker.sh --data_dir '$datadir' --scratch_dir '$scratchdir' --GPU '$gpu' $model_flag --start_client"
            cd "$REPO_ROOT"
        else
            remote_exec "$site" \
                "cd '$deploy_dir/$site_name/startup' && \
                 export SITE_NAME='$site_name' && \
                 export DATADIR='$datadir' && \
                 export SCRATCHDIR='$scratchdir' && \
                 ./docker.sh --data_dir '$datadir' --scratch_dir '$scratchdir' --GPU '$gpu' $model_flag --start_client"
        fi

        ok "  Client started: $site_name"
    done

    ok "All clients started"
}

# ── Submit job ────────────────────────────────────────────────────────────
submit_job() {
    local job_name="$1"
    local deploy_base
    deploy_base=$(site_var "$(echo "${SITES[0]}")" DEPLOY_DIR)
    local admin_startup="$deploy_base/$ADMIN_USER/startup"

    if [[ ! -d "$admin_startup" ]]; then
        local prod_dir
        prod_dir=$(find_latest_prod)
        admin_startup="$prod_dir/$ADMIN_USER/startup"
    fi

    if [[ ! -d "$admin_startup" ]]; then
        err "Admin startup kit not found"
        exit 1
    fi

    local job_path="MediSwarm/application/jobs/$job_name"
    info "Submitting job: $job_name (path: $job_path)"

    # Generate expect script
    local expect_script
    expect_script=$(mktemp /tmp/mediswarm_deploy_test_XXXXXX.exp)
    cat > "$expect_script" <<EXPECT_EOF
#!/usr/bin/env expect
set timeout 120
spawn ./docker.sh --no_pull
expect "User Name: "
send "$ADMIN_USER\r"
expect "> "
send "submit_job $job_path\r"
expect "> "
send "list_jobs\r"
expect "> "
send "bye\r"
expect eof
EXPECT_EOF
    chmod +x "$expect_script"

    cd "$admin_startup"
    expect -f "$expect_script" || true
    cd "$REPO_ROOT"

    rm -f "$expect_script"
    ok "Job submitted: $job_name"
}

# ── Wait for training completion ──────────────────────────────────────────
wait_for_completion() {
    local model_name="$1"
    local timeout_minutes="${2:-$TIMEOUT_MINUTES}"
    local deploy_base
    deploy_base=$(site_var "$(echo "${SITES[0]}")" DEPLOY_DIR)
    local server_name="${SERVER_NAME:-dl3.tud.de}"
    local server_log="$deploy_base/$server_name/startup/nohup.out"

    if [[ ! -f "$server_log" ]]; then
        # Fall back to prod dir
        local prod_dir
        prod_dir=$(find_latest_prod)
        server_log="$prod_dir/$server_name/startup/nohup.out"
    fi

    local max_attempts=$(( timeout_minutes * 2 ))  # Check every 30 seconds
    local attempt=0

    info "Waiting for training to complete: $model_name (timeout: ${timeout_minutes}min, checking every 30s)"
    info "Server log: $server_log"

    while [[ $attempt -lt $max_attempts ]]; do
        if [[ -f "$server_log" ]] && grep -q 'Server runner finished\.' "$server_log" 2>/dev/null; then
            ok "Training completed for $model_name! (after $((attempt * 30))s)"
            return 0
        fi

        # Also check if the server container died
        if ! docker ps --format '{{.Names}}' | grep -qE "odelia_swarm|nvflare"; then
            # Container is gone — check if it completed
            if [[ -f "$server_log" ]] && grep -q 'Server runner finished\.' "$server_log" 2>/dev/null; then
                ok "Training completed for $model_name (container exited cleanly)"
                return 0
            fi
            warn "Server container is no longer running — training may have failed"
            return 1
        fi

        attempt=$((attempt + 1))
        sleep 30
    done

    err "Timeout after ${timeout_minutes}min waiting for $model_name training to complete"
    return 1
}

# ── Evaluate model ────────────────────────────────────────────────────────
evaluate_model() {
    local model_name="$1"
    local job_name="$2"

    step "Evaluating $model_name on $EVAL_SITE_NAME"

    # Find the workspace with checkpoints
    # After swarm training, checkpoints are in the client's scratch dir
    # The NVFlare global model is saved as FL_global_model.pt and best_FL_global_model.pt
    local prod_dir
    prod_dir=$(find_latest_prod)

    mkdir -p "$EVAL_SCRATCH_DIR"

    info "Running predict.py with --workspace on $prod_dir"
    info "  MODEL_NAME=$model_name"
    info "  SITE_NAME=$EVAL_SITE_NAME"
    info "  DATA_DIR=$EVAL_DATA_DIR"

    # Run evaluation inside Docker container using the same image
    local eval_output_dir="$RESULTS_DIR/${model_name}_evaluation"
    mkdir -p "$eval_output_dir"

    # Use predict.py with the workspace to discover checkpoints
    # The predict.py script runs natively (not in Docker) since it imports from the repo
    export DATA_DIR="$EVAL_DATA_DIR"
    export SITE_NAME="$EVAL_SITE_NAME"
    export SCRATCH_DIR="$EVAL_SCRATCH_DIR"
    export MODEL_NAME="$model_name"

    local eval_result=0
    python3 "$REPO_ROOT/scripts/evaluation/predict.py" \
        --workspace "$prod_dir" \
        --model-name "$model_name" \
        --output-dir "$eval_output_dir" \
        --best-only \
        --split test \
        2>&1 | tee "$eval_output_dir/predict_stdout.log" || eval_result=$?

    if [[ $eval_result -eq 0 ]]; then
        ok "Evaluation completed for $model_name"
        return 0
    else
        err "Evaluation failed for $model_name (exit code: $eval_result)"
        return 1
    fi
}

# ── Record result ─────────────────────────────────────────────────────────
record_result() {
    local model_name="$1"
    local job_name="$2"
    local train_status="$3"   # pass or fail
    local eval_status="$4"    # pass, fail, or skipped
    local duration_seconds="$5"

    local result_file="$RESULTS_DIR/deploy_test_${model_name}.json"
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    cat > "$result_file" <<EOF
{
    "model_name": "$model_name",
    "job_name": "$job_name",
    "training_status": "$train_status",
    "evaluation_status": "$eval_status",
    "duration_seconds": $duration_seconds,
    "timestamp": "$timestamp",
    "docker_image": "$DOCKER_IMAGE",
    "version": "$VERSION",
    "sites": $(printf '%s\n' "${SITES[@]}" | jq -R . | jq -s .)
}
EOF

    info "Result recorded: $result_file"
}

# ── Run one model through the full cycle ──────────────────────────────────
run_single_model() {
    local job_name="$1"
    local model_name="$2"
    local start_time
    start_time=$(date +%s)

    step "Running deploy test: $model_name (job: $job_name)"
    echo ""
    info "Docker image: $DOCKER_IMAGE"
    info "Workspace: $WORKSPACE_DIR"
    echo ""

    local train_status="fail"
    local eval_status="skipped"

    # 1. Stop any existing containers
    stop_all

    # 2. Start server
    start_server

    # 3. Start clients with MODEL_NAME
    start_clients "$model_name"

    # 4. Wait for clients to register
    info "Waiting 30s for clients to register with server..."
    sleep 30

    # 5. Submit job
    submit_job "$job_name"

    # 6. Wait for training completion
    if wait_for_completion "$model_name"; then
        train_status="pass"
    else
        train_status="fail"
    fi

    # 7. Stop containers
    stop_all

    # 8. Evaluate on UKA_1 (only if training passed)
    if [[ "$train_status" == "pass" ]]; then
        if evaluate_model "$model_name" "$job_name"; then
            eval_status="pass"
        else
            eval_status="fail"
        fi
    else
        warn "Skipping evaluation for $model_name — training did not complete"
        eval_status="skipped"
    fi

    # 9. Record result
    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))

    record_result "$model_name" "$job_name" "$train_status" "$eval_status" "$duration"

    echo ""
    if [[ "$train_status" == "pass" && "$eval_status" == "pass" ]]; then
        ok "PASSED: $model_name (training + evaluation) in ${duration}s"
    elif [[ "$train_status" == "pass" ]]; then
        warn "PARTIAL: $model_name (training passed, evaluation $eval_status) in ${duration}s"
    else
        err "FAILED: $model_name (training $train_status) in ${duration}s"
    fi

    return 0  # Don't fail the entire script on individual model failure
}

# ── Generate summary ──────────────────────────────────────────────────────
generate_summary() {
    step "Deploy Test Summary"

    local summary_file="$RESULTS_DIR/summary.json"
    local total=0
    local passed=0
    local failed=0

    echo "{"
    echo '  "results": ['

    local first=true
    for result_file in "$RESULTS_DIR"/deploy_test_*.json; do
        [[ -f "$result_file" ]] || continue
        total=$((total + 1))

        local model train eval
        model=$(jq -r '.model_name' "$result_file")
        train=$(jq -r '.training_status' "$result_file")
        eval=$(jq -r '.evaluation_status' "$result_file")
        local duration
        duration=$(jq -r '.duration_seconds' "$result_file")

        if [[ "$train" == "pass" && "$eval" == "pass" ]]; then
            passed=$((passed + 1))
            echo -e "  ${GREEN}PASS${NC}  $model (${duration}s)"
        else
            failed=$((failed + 1))
            echo -e "  ${RED}FAIL${NC}  $model (train=$train, eval=$eval, ${duration}s)"
        fi

        if [[ "$first" == true ]]; then
            first=false
        else
            echo ","
        fi
        cat "$result_file"
    done > /dev/null  # summary prints go to stdout

    echo ""
    echo "────────────────────────────────────────"
    echo -e "Total: $total | ${GREEN}Passed: $passed${NC} | ${RED}Failed: $failed${NC}"
    echo ""

    # Write machine-readable summary
    cat > "$summary_file" <<EOF
{
    "total": $total,
    "passed": $passed,
    "failed": $failed,
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "version": "$VERSION"
}
EOF

    # Print individual results
    for result_file in "$RESULTS_DIR"/deploy_test_*.json; do
        [[ -f "$result_file" ]] || continue
        local model train eval duration
        model=$(jq -r '.model_name' "$result_file")
        train=$(jq -r '.training_status' "$result_file")
        eval_stat=$(jq -r '.evaluation_status' "$result_file")
        duration=$(jq -r '.duration_seconds' "$result_file")

        if [[ "$train" == "pass" && "$eval_stat" == "pass" ]]; then
            echo -e "  ${GREEN}PASS${NC}  $model (${duration}s)"
        else
            echo -e "  ${RED}FAIL${NC}  $model (train=$train, eval=$eval_stat, ${duration}s)"
        fi
    done

    echo ""
    if [[ $failed -gt 0 ]]; then
        err "$failed model(s) failed the deploy test"
        return 1
    else
        ok "All $total models passed the deploy test!"
        return 0
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────

# Check dependencies
for cmd in sshpass expect jq docker; do
    if ! command -v "$cmd" &>/dev/null; then
        err "Missing required tool: $cmd"
        exit 1
    fi
done

# Verify workspace exists
if [[ ! -d "$WORKSPACE_DIR" ]]; then
    err "Workspace not found: $WORKSPACE_DIR"
    err "Build startup kits first:"
    err "  ./scripts/build/buildDockerImageAndStartupKits.sh -p $PROJECT_FILE"
    exit 1
fi

# Deploy startup kits to all sites (once — shared across all models)
step "Deploying startup kits to all sites"
deploy_kits

if [[ "$RUN_ALL" == true ]]; then
    # Run all 6 models sequentially
    step "Running deploy test for all 6 ODELIA models"
    echo ""

    for model_spec in "${ALL_MODELS[@]}"; do
        IFS=':' read -r job_name model_name <<< "$model_spec"
        run_single_model "$job_name" "$model_name"
    done

    # Generate summary
    generate_summary
else
    # Run a single model
    if [[ -z "$SINGLE_JOB" ]]; then
        err "Must specify --job JOB_DIR when using --model"
        exit 1
    fi
    run_single_model "$SINGLE_JOB" "$SINGLE_MODEL"
fi
