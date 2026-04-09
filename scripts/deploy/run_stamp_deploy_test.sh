#!/usr/bin/env bash
# ============================================================================
# run_stamp_deploy_test.sh — Orchestrate 2-node STAMP deploy test
#
# Runs STAMP_classification through a full federated training cycle across
# 2 physical machines connected via Tailscale VPN, with Cosmos as the
# server/admin node (no training client on Cosmos).
#
# Architecture:
#   Cosmos (localhost) — NVFlare server + admin only
#   dl0                — RUMC_1 client
#   dl2                — MHA_1 client
#
# Workflow:
#   1. Stop any lingering containers (local + remote)
#   2. Deploy startup kits to all sites
#   3. Fix DNS on remote machines (dl3.tud.de → Cosmos Tailscale IP)
#   4. Pre-pull Docker image on all remote machines
#   5. Start NVFlare server on Cosmos
#   6. Start NVFlare clients on DL0 + DL2 (with STAMP_* env vars)
#   7. Wait for client registration
#   8. Submit STAMP_classification job via admin
#   9. Wait for training completion ("Server runner finished.")
#  10. Record pass/fail result
#
# Unlike the ODELIA deploy test (run_deploy_test.sh), this script:
#   - Runs ONE model only (STAMP_classification)
#   - Exports STAMP_* environment variables on remote hosts before docker.sh
#   - Uses stamp_swarm container name prefix (not odelia_swarm)
#   - Has no evaluation step (STAMP has no predict.py equivalent yet)
#   - Has no retry loop (simpler for initial testing)
#
# Usage:
#   ./scripts/deploy/run_stamp_deploy_test.sh --conf deploy_sites_2node_stamp_test.conf
#   ./scripts/deploy/run_stamp_deploy_test.sh --conf deploy_sites_2node_stamp_test.conf --timeout 60
#
# The Docker image and startup kits must be built BEFORE running this script:
#   ./scripts/build/buildDockerImageAndStartupKits.sh \
#       -p application/provision/project_deploy_test_2site.yml \
#       -d docker_config/Dockerfile_STAMP --num-rounds 2
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

info()  { echo -e "${BLUE}[INFO]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC} $*" >&2; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}=== $* ===${NC}" >&2; }

# ── Parse arguments ────────────────────────────────────────────────────────
CONF_FILE=""
TIMEOUT_MINUTES=120   # 2 hours for a quick 2-round test

while [[ $# -gt 0 ]]; do
    case $1 in
        --conf)         CONF_FILE="$2"; shift ;;
        --timeout)      TIMEOUT_MINUTES="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 --conf CONF_FILE [--timeout MINUTES]"
            echo ""
            echo "  --conf      Path to site configuration file (required)"
            echo "  --timeout   Per-model training timeout in minutes (default: 120)"
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
    err "Must specify --conf CONF_FILE"
    exit 1
fi
if [[ ! -f "$CONF_FILE" ]]; then
    if [[ -f "$REPO_ROOT/$CONF_FILE" ]]; then
        CONF_FILE="$REPO_ROOT/$CONF_FILE"
    else
        err "Configuration file not found: $CONF_FILE"
        exit 1
    fi
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

# Cosmos deploy dir (server + admin live here)
DEPLOY_BASE="${COSMOS_DEPLOY_DIR:-/home/jeff/deploy_test_stamp}"

# Results directory
RESULTS_DIR="$REPO_ROOT/workspace/deploy_test_stamp_results"
mkdir -p "$RESULTS_DIR"

# Job to run
JOB_NAME="${DEFAULT_JOB:-STAMP_classification}"

# ── STAMP environment variables ──────────────────────────────────────────
# These are exported on each remote host BEFORE calling docker.sh.
# The STAMP master template's env var forwarding loop (for _var in STAMP_*)
# picks them up and passes them into the Docker container.

setup_stamp_env() {
    local site_name="$1"
    local data_dir="$2"   # host path — only used by caller for --data_dir flag
    # Inside the container, --data_dir is mounted at /data/ (read-only).
    # STAMP env vars must reference the container-internal path, not the host path.
    cat <<EOF
export STAMP_CLINI_TABLE="/data/${site_name}/clini_table.csv"
export STAMP_FEATURE_DIR="/data/${site_name}/features"
export STAMP_GROUND_TRUTH_LABEL="Diagnosis"
export STAMP_PATIENT_LABEL="PATIENT"
export STAMP_TASK="classification"
export STAMP_MODEL_NAME="vit"
export STAMP_DIM_INPUT="1024"
export STAMP_NUM_CLASSES="3"
export STAMP_BAG_SIZE="64"
export STAMP_BATCH_SIZE="8"
export STAMP_MAX_EPOCHS="2"
export STAMP_PATIENCE="2"
export STAMP_NUM_WORKERS="0"
export STAMP_SEED="42"
export STAMP_NUM_ROUNDS="2"
export STAMP_EPOCHS_PER_ROUND="2"
export STAMP_EPOCHS_REFERENCE_DATASET_SIZE="15"
export STAMP_EPOCHS_MAX_CAP="4"
EOF
}

# ── Helper functions ──────────────────────────────────────────────────────

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

    sshpass -p "$pass" ssh $SSH_OPTS "$user@$host" "$@"
}

remote_copy() {
    local site=$1 src=$2 dst=$3
    local host user pass
    host=$(site_var "$site" HOST)
    user=$(site_var "$site" USER)
    pass=$(site_var "$site" PASS)

    sshpass -p "$pass" scp $SSH_OPTS "$src" "$user@$host:$dst"
}

find_latest_prod() {
    if [[ ! -d "$WORKSPACE_DIR" ]]; then
        err "Workspace not found: $WORKSPACE_DIR"
        err "Run buildDockerImageAndStartupKits.sh first."
        exit 1
    fi
    ls -d "$WORKSPACE_DIR"/prod_* 2>/dev/null | sort -V | tail -n 1
}

# ── Resolve server startup directory ───────────────────────────────────
_server_startup_dir=""

resolve_server_startup_dir() {
    local server_name="${SERVER_NAME:-dl3.tud.de}"
    local candidate="$DEPLOY_BASE/$server_name/startup"
    if [[ -d "$candidate" ]]; then
        _server_startup_dir="$candidate"
    else
        local prod_dir
        prod_dir=$(find_latest_prod)
        candidate="$prod_dir/$server_name/startup"
        if [[ -d "$candidate" ]]; then
            _server_startup_dir="$candidate"
        else
            _server_startup_dir=""
        fi
    fi
}

# ── Fix DNS: ensure remote clients can reach the NVFlare server ──────────
fix_remote_dns() {
    local server_fqdn="${SERVER_NAME:-dl3.tud.de}"
    local cosmos_ip
    cosmos_ip=$(tailscale ip -4 2>/dev/null) || {
        err "Cannot determine Cosmos Tailscale IP (is tailscale running?)"
        return 1
    }

    step "Ensuring $server_fqdn resolves to $cosmos_ip on all remote machines"

    local -A visited_hosts=()
    for site in "${CLIENT_SITES[@]}"; do
        local host
        host=$(site_var "$site" HOST)
        [[ -n "${visited_hosts[$host]:-}" ]] && continue
        visited_hosts[$host]=1

        # Skip if this machine IS Cosmos
        [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$cosmos_ip" ]] && continue

        info "Checking /etc/hosts on $host for $server_fqdn ..."

        local current_ip
        current_ip=$(remote_exec "$site" \
            "grep -E '\\b${server_fqdn}\\b' /etc/hosts 2>/dev/null | awk '{print \$1}' | head -1" \
            2>/dev/null || true)

        if [[ "$current_ip" == "$cosmos_ip" ]]; then
            ok "  $host already maps $server_fqdn → $cosmos_ip"
            continue
        fi

        if [[ -n "$current_ip" ]]; then
            info "  $host maps $server_fqdn → $current_ip (wrong, updating to $cosmos_ip)"
            remote_exec "$site" \
                "echo '$(site_var "$site" PASS)' | sudo -S sed -i 's|^.*\\b${server_fqdn}\\b.*\$|${cosmos_ip} ${server_fqdn}|' /etc/hosts" \
                2>/dev/null
        else
            info "  $host has no entry for $server_fqdn (adding)"
            remote_exec "$site" \
                "echo '$(site_var "$site" PASS)' | sudo -S bash -c 'echo \"${cosmos_ip} ${server_fqdn}\" >> /etc/hosts'" \
                2>/dev/null
        fi

        # Verify
        local verify_ip
        verify_ip=$(remote_exec "$site" \
            "grep -E '\\b${server_fqdn}\\b' /etc/hosts | awk '{print \$1}' | head -1" \
            2>/dev/null || true)
        if [[ "$verify_ip" == "$cosmos_ip" ]]; then
            ok "  $host now maps $server_fqdn → $cosmos_ip"
        else
            err "  Failed to update /etc/hosts on $host (got: $verify_ip)"
            return 1
        fi
    done

    ok "DNS resolution verified on all remote machines"
}

# ── Stop all containers ───────────────────────────────────────────────────
stop_all() {
    info "Stopping all STAMP/NVFlare containers..."

    # Stop local containers on Cosmos (server + admin)
    local local_containers
    local_containers=$(docker ps --format '{{.Names}}' | grep -E "stamp_swarm|nvflare" || true)
    if [[ -n "$local_containers" ]]; then
        echo "$local_containers" | xargs docker kill 2>/dev/null || true
        echo "$local_containers" | xargs docker rm -f 2>/dev/null || true
    fi

    # Stop remote client containers
    local -A visited_hosts=()
    for site in "${CLIENT_SITES[@]}"; do
        local host
        host=$(site_var "$site" HOST)
        if [[ -n "${visited_hosts[$host]:-}" ]]; then
            continue
        fi
        visited_hosts[$host]=1
        remote_exec "$site" \
            "docker ps --format '{{.Names}}' | grep -E 'stamp_swarm|nvflare' | xargs -r docker kill 2>/dev/null; \
             docker ps -a --format '{{.Names}}' | grep -E 'stamp_swarm|nvflare' | xargs -r docker rm -f 2>/dev/null" \
            2>/dev/null || warn "  Could not stop containers on $host"
    done

    # Clean up root-owned files left by NVFlare server container
    local server_name="${SERVER_NAME:-dl3.tud.de}"
    local server_dir="$DEPLOY_BASE/$server_name"
    if [[ -d "$server_dir" ]]; then
        sudo rm -rf "$server_dir" 2>/dev/null \
            || docker run --rm -v "$DEPLOY_BASE:/cleanup" alpine \
                rm -rf "/cleanup/$server_name" 2>/dev/null \
            || warn "Could not fully clean $server_dir (root-owned files may remain)"
    fi

    sleep 5
    ok "All containers stopped"
}

# ── Pre-pull Docker image on all remote machines ─────────────────────────
pre_pull_images() {
    step "Pre-pulling Docker image on remote machines"
    info "Image: $DOCKER_IMAGE"

    local -A visited_hosts=()
    for site in "${CLIENT_SITES[@]}"; do
        local host
        host=$(site_var "$site" HOST)
        if [[ -n "${visited_hosts[$host]:-}" ]]; then
            continue
        fi
        visited_hosts[$host]=1

        info "Pulling $DOCKER_IMAGE on $host ..."
        remote_exec "$site" "docker pull '$DOCKER_IMAGE'" || {
            err "Failed to pull image on $host"
            exit 1
        }
        ok "  Image pulled on $host"
    done

    ok "Docker image available on all remote machines"
}

# ── Deploy startup kits ───────────────────────────────────────────────────
deploy_kits() {
    local prod_dir
    prod_dir=$(find_latest_prod)
    info "Deploying startup kits from: $prod_dir"

    # Deploy client kits to remote machines
    for site in "${CLIENT_SITES[@]}"; do
        local site_name host deploy_dir
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)

        local zip_file="$prod_dir/${site_name}_${VERSION}.zip"
        if [[ ! -f "$zip_file" ]]; then
            zip_file=$(ls "$prod_dir"/${site_name}*.zip 2>/dev/null | head -1 || true)
            if [[ -z "$zip_file" ]]; then
                err "Startup kit not found for $site_name in $prod_dir"
                exit 1
            fi
        fi

        remote_exec "$site" "mkdir -p '$deploy_dir'"
        remote_copy "$site" "$zip_file" "$deploy_dir/"
        remote_exec "$site" "cd '$deploy_dir' && rm -rf '${site_name}' && unzip -qo '$(basename "$zip_file")'"
        ok "  Deployed $site_name to $host:$deploy_dir/"
    done

    # Deploy server kit locally (on Cosmos)
    local server_name="${SERVER_NAME:-dl3.tud.de}"
    local server_zip="$prod_dir/${server_name}_${VERSION}.zip"
    if [[ ! -f "$server_zip" ]]; then
        server_zip=$(ls "$prod_dir"/${server_name}*.zip 2>/dev/null | head -1 || true)
    fi
    if [[ -n "$server_zip" && -f "$server_zip" ]]; then
        mkdir -p "$DEPLOY_BASE"
        cp "$server_zip" "$DEPLOY_BASE/"
        cd "$DEPLOY_BASE" && rm -rf "$server_name" && unzip -qo "$(basename "$server_zip")"
        cd "$REPO_ROOT"
        ok "  Deployed server kit ($server_name) locally on Cosmos"
    fi

    # Deploy admin kit locally (on Cosmos)
    local admin_zip="$prod_dir/${ADMIN_USER}_${VERSION}.zip"
    if [[ ! -f "$admin_zip" ]]; then
        admin_zip=$(ls "$prod_dir"/${ADMIN_USER}*.zip 2>/dev/null | head -1 || true)
    fi
    if [[ -n "$admin_zip" && -f "$admin_zip" ]]; then
        cp "$admin_zip" "$DEPLOY_BASE/"
        cd "$DEPLOY_BASE" && rm -rf "$ADMIN_USER" && unzip -qo "$(basename "$admin_zip")"
        cd "$REPO_ROOT"
        ok "  Deployed admin kit ($ADMIN_USER) locally on Cosmos"
    fi
}

# ── Start server (on Cosmos — always local) ──────────────────────────────
start_server() {
    local server_name="${SERVER_NAME:-dl3.tud.de}"
    local server_startup="$DEPLOY_BASE/$server_name/startup"

    if [[ ! -d "$server_startup" ]]; then
        local prod_dir
        prod_dir=$(find_latest_prod)
        server_startup="$prod_dir/$server_name/startup"
    fi

    if [[ ! -d "$server_startup" ]]; then
        err "Server startup kit not found at $server_startup"
        exit 1
    fi

    _server_startup_dir="$server_startup"

    info "Starting server from: $server_startup"
    cd "$server_startup"
    ./docker.sh --no_pull --start_server
    cd "$REPO_ROOT"

    info "Waiting 15s for server to initialize..."
    sleep 15

    if docker ps --format '{{.Names}}' | grep -qE "stamp_swarm|nvflare"; then
        ok "Server container is running"
    else
        warn "Server container not detected — it may still be starting"
    fi
}

# ── Start clients (all remote) ───────────────────────────────────────────
# KEY DIFFERENCE from ODELIA: exports STAMP_* env vars on each remote host
# before calling docker.sh.  The STAMP master template's forwarding loop
# (for _var in $(env | grep '^STAMP_')) picks them up.
start_clients() {
    for site in "${CLIENT_SITES[@]}"; do
        local site_name host deploy_dir datadir scratchdir gpu
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)
        datadir=$(site_var "$site" DATADIR)
        scratchdir=$(site_var "$site" SCRATCHDIR)
        gpu=$(site_var "$site" GPU)

        info "Starting client: $site_name @ $host (with STAMP_* env vars)"

        # Generate STAMP env exports for this site
        local stamp_exports
        stamp_exports=$(setup_stamp_env "$site_name" "$datadir")

        remote_exec "$site" \
            "$stamp_exports && \
             cd '$deploy_dir/$site_name/startup' && \
             ./docker.sh --no_pull --data_dir '$datadir' --scratch_dir '$scratchdir' --GPU '$gpu' --start_client"

        ok "  Client started: $site_name"
    done

    ok "All clients started"
}

# ── Wait for all clients to register with the server ─────────────────────
wait_for_client_registration() {
    resolve_server_startup_dir
    local server_log="${_server_startup_dir}/nohup.out"
    local max_wait=600  # 10 minutes
    local elapsed=0
    local expected_clients=("${CLIENT_SITES[@]}")

    info "Waiting for all ${#expected_clients[@]} clients to register (timeout: ${max_wait}s)"
    info "Server log: $server_log"

    while [[ $elapsed -lt $max_wait ]]; do
        local all_registered=true
        for site in "${expected_clients[@]}"; do
            local site_name
            site_name=$(site_var "$site" SITE_NAME)
            if [[ -f "$server_log" ]] && grep -q "New client ${site_name}@" "$server_log" 2>/dev/null; then
                :  # Client registered
            else
                all_registered=false
                break
            fi
        done

        if $all_registered; then
            ok "All ${#expected_clients[@]} clients registered with the server"
            sleep 5
            return 0
        fi

        sleep 10
        elapsed=$((elapsed + 10))

        if (( elapsed % 60 == 0 )); then
            local registered=0
            for site in "${expected_clients[@]}"; do
                local site_name
                site_name=$(site_var "$site" SITE_NAME)
                if [[ -f "$server_log" ]] && grep -q "New client ${site_name}@" "$server_log" 2>/dev/null; then
                    registered=$((registered + 1))
                fi
            done
            info "  ${registered}/${#expected_clients[@]} clients registered (${elapsed}s elapsed)"
        fi
    done

    # Timeout: show which clients are missing
    warn "Timed out waiting for client registration. Status:"
    for site in "${expected_clients[@]}"; do
        local site_name
        site_name=$(site_var "$site" SITE_NAME)
        if [[ -f "$server_log" ]] && grep -q "New client ${site_name}@" "$server_log" 2>/dev/null; then
            ok "    $site_name — registered"
        else
            err "    $site_name — NOT registered"
        fi
    done

    err "Not all clients registered within ${max_wait}s"
    return 1
}

# ── Submit job (via admin on Cosmos — always local) ──────────────────────
submit_job() {
    local job_name="$1"
    local admin_startup="$DEPLOY_BASE/$ADMIN_USER/startup"

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

    local expect_script
    expect_script=$(mktemp /tmp/mediswarm_stamp_deploy_test_XXXXXX.exp)
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
    resolve_server_startup_dir
    local server_log="${_server_startup_dir}/nohup.out"

    # Record current line count so we only check NEW lines
    local start_line=0
    if [[ -f "$server_log" ]]; then
        start_line=$(wc -l < "$server_log" 2>/dev/null || echo 0)
    fi

    local max_attempts=$(( timeout_minutes * 2 ))  # Check every 30 seconds
    local attempt=0

    info "Waiting for training to complete: $model_name (timeout: ${timeout_minutes}min, checking every 30s)"
    info "Server log: $server_log (scanning from line $((start_line + 1)))"

    while [[ $attempt -lt $max_attempts ]]; do
        if [[ -f "$server_log" ]]; then
            local new_lines
            new_lines=$(tail -n +"$((start_line + 1))" "$server_log" 2>/dev/null || true)

            # Check for clean completion
            if echo "$new_lines" | grep -q 'Server runner finished\.' 2>/dev/null; then
                # Check if it was a fatal finish
                local finish_lineno
                finish_lineno=$(echo "$new_lines" | grep -n 'Server runner finished\.' | tail -1 | cut -d: -f1)
                local context_start=$(( finish_lineno - 50 ))
                [[ $context_start -lt 1 ]] && context_start=1
                local context
                context=$(echo "$new_lines" | sed -n "${context_start},${finish_lineno}p")

                if echo "$context" | grep -q 'FATAL_SYSTEM_ERROR\|ABORT_RUN\|EXECUTION_EXCEPTION.*abort' 2>/dev/null; then
                    err "Training ABORTED for $model_name — fatal error detected (after $((attempt * 30))s)"
                    echo "$new_lines" | grep -i 'FATAL\|ABORT\|EXCEPTION\|ERROR' | tail -20 >&2
                    return 1
                else
                    ok "Training completed for $model_name! (after $((attempt * 30))s)"
                    return 0
                fi
            fi
        fi

        # Check if the server container died
        if ! docker ps --format '{{.Names}}' | grep -qE "stamp_swarm|nvflare"; then
            if [[ -f "$server_log" ]]; then
                local new_lines
                new_lines=$(tail -n +"$((start_line + 1))" "$server_log" 2>/dev/null || true)
                if echo "$new_lines" | grep -q 'Server runner finished\.' 2>/dev/null; then
                    ok "Training completed for $model_name (container exited cleanly)"
                    return 0
                fi
            fi
            warn "Server container is no longer running — training may have failed"
            return 1
        fi

        attempt=$((attempt + 1))
        sleep 30

        if (( attempt % 4 == 0 )); then
            info "  Still waiting for $model_name... ($((attempt * 30))s elapsed)"
        fi
    done

    err "Timeout after ${timeout_minutes}min waiting for $model_name training to complete"
    return 1
}

# ── Record result ─────────────────────────────────────────────────────────
record_result() {
    local model_name="$1"
    local job_name="$2"
    local train_status="$3"   # pass or fail
    local duration_seconds="$4"

    local result_file="$RESULTS_DIR/deploy_test_stamp_${model_name}.json"
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    cat > "$result_file" <<EOF
{
    "model_name": "$model_name",
    "job_name": "$job_name",
    "training_status": "$train_status",
    "duration_seconds": $duration_seconds,
    "timestamp": "$timestamp",
    "docker_image": "$DOCKER_IMAGE",
    "version": "$VERSION",
    "client_sites": $(printf '%s\n' "${CLIENT_SITES[@]}" | jq -R . | jq -s .)
}
EOF

    info "Result recorded: $result_file"
}

# ── Run one training attempt ─────────────────────────────────────────────
_run_training_attempt() {
    local job_name="$1"
    local model_name="$2"

    # 1. Stop any existing containers
    stop_all

    # 2. Start server on Cosmos
    start_server

    # 3. Start clients with STAMP_* env vars
    start_clients

    # 4. Wait for all clients to register
    wait_for_client_registration

    # 5. Submit job via admin
    submit_job "$job_name"

    # 6. Wait for training completion
    if wait_for_completion "$model_name"; then
        return 0
    else
        return 1
    fi
}

# ── Run the deploy test ──────────────────────────────────────────────────
MAX_RETRIES=2  # Total attempts = 1 + MAX_RETRIES

run_single_model() {
    local job_name="$1"
    local model_name="$2"
    local start_time
    start_time=$(date +%s)

    step "Running STAMP deploy test: $model_name (job: $job_name)"
    echo ""
    info "Docker image: $DOCKER_IMAGE"
    info "Workspace: $WORKSPACE_DIR"
    info "Server: Cosmos (localhost)"
    info "Clients: ${CLIENT_SITES[*]}"
    echo ""

    local train_status="fail"

    local attempt=1
    while [[ $attempt -le $((MAX_RETRIES + 1)) ]]; do
        if [[ $attempt -gt 1 ]]; then
            warn "Retry $((attempt - 1))/$MAX_RETRIES for $model_name"
        fi

        if _run_training_attempt "$job_name" "$model_name"; then
            train_status="pass"
            break
        fi

        attempt=$((attempt + 1))
        if [[ $attempt -le $((MAX_RETRIES + 1)) ]]; then
            info "Waiting 30s before next attempt for $model_name"
            sleep 30
        fi
    done

    # 7. Stop all containers
    stop_all

    # 8. Record result
    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))

    record_result "$model_name" "$job_name" "$train_status" "$duration"

    echo ""
    if [[ "$train_status" == "pass" ]]; then
        ok "PASSED: $model_name in ${duration}s"
    else
        err "FAILED: $model_name in ${duration}s"
    fi

    return 0
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
    err "  ./scripts/build/buildDockerImageAndStartupKits.sh -p $PROJECT_FILE -d docker_config/Dockerfile_STAMP --num-rounds 2"
    exit 1
fi

# Clean up any leftover containers from previous runs
stop_all

# Deploy startup kits to all sites
step "Deploying startup kits to all sites"
deploy_kits

# Fix DNS on remote machines
fix_remote_dns

# Pre-pull Docker image on all remote machines
pre_pull_images

# Run the STAMP deploy test
run_single_model "$JOB_NAME" "vit"
