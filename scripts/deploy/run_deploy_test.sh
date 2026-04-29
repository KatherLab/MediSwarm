#!/usr/bin/env bash
# ============================================================================
# run_deploy_test.sh — Orchestrate multi-site ODELIA deploy tests
#
# Runs all 6 ODELIA models through a federated startup-kit deploy test across
# physical machines connected via Tailscale VPN, with Cosmos as the server/admin
# node (no training client on Cosmos).
#
# Architecture:
#   Cosmos (localhost) — NVFlare server + admin only
#   clients            — configured by CLIENT_SITES in the deploy_sites*.conf
#
# For each model:
#   1. Stop any lingering containers (local + remote)
#   2. Start the NVFlare server on Cosmos
#   3. Start configured training clients on the remote machines
#   4. Submit the corresponding training job via admin
#   5. Poll for training completion (Server runner finished)
#   6. Stop all containers
#   7. Optionally evaluate the final global model on the configured eval site
#   8. Record pass/fail + metrics
#
# Usage:
#   ./scripts/deploy/run_deploy_test.sh --all --conf deploy_sites_4node_test.conf
#   ./scripts/deploy/run_deploy_test.sh --model MST --job ODELIA_ternary_classification --conf deploy_sites_4node_test.conf
#   ./scripts/deploy/run_deploy_test.sh --all --conf deploy_sites_duke_iid.conf --skip-eval --results-dir workspace/deploy_test_results/startupkit_smoke_<ts>/training
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

info()  { echo -e "${BLUE}[INFO]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC} $*" >&2; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}=== $* ===${NC}" >&2; }

# ── Parse arguments ────────────────────────────────────────────────────────
RUN_ALL=false
SINGLE_MODEL=""
SINGLE_JOB=""
CONF_FILE=""
SKIP_BUILD=false
SKIP_EVAL=false
RESULTS_DIR_OVERRIDE=""
TIMEOUT_MINUTES=10080   # Per-model training timeout (7 days)

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)          RUN_ALL=true ;;
        --model)        SINGLE_MODEL="$2"; shift ;;
        --job)          SINGLE_JOB="$2"; shift ;;
        --conf)         CONF_FILE="$2"; shift ;;
        --skip-build)   SKIP_BUILD=true ;;
        --skip-eval)    SKIP_EVAL=true ;;
        --results-dir)  RESULTS_DIR_OVERRIDE="$2"; shift ;;
        --timeout)      TIMEOUT_MINUTES="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--all | --model NAME --job JOB_DIR] --conf CONF_FILE [--skip-build] [--skip-eval] [--results-dir DIR] [--timeout MINUTES]"
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
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
DOCKER_IMAGE="jefftud/odelia:$VERSION"

PROJECT_NAME=$(grep "^name: " "$REPO_ROOT/$PROJECT_FILE" \
    | sed 's/^name: //' \
    | sed "s/__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__/$VERSION/")
WORKSPACE_DIR="$REPO_ROOT/workspace/$PROJECT_NAME"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Cosmos deploy dir (server + admin live here)
DEPLOY_BASE="${COSMOS_DEPLOY_DIR:-/home/jeff/deploy_test}"

# Results directory
if [[ -n "$RESULTS_DIR_OVERRIDE" ]]; then
    if [[ "$RESULTS_DIR_OVERRIDE" = /* ]]; then
        RESULTS_DIR="$RESULTS_DIR_OVERRIDE"
    else
        RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR_OVERRIDE"
    fi
else
    RESULTS_DIR="$REPO_ROOT/workspace/deploy_test_results"
fi
mkdir -p "$RESULTS_DIR"

# Updated by collect_checkpoints() so result JSON can record the run that was
# actually evaluated/collected.
LAST_JOB_ID=""

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

# ── Resolve the actual server startup directory ─────────────────────────
# start_server() may use either DEPLOY_BASE or prod_dir.  The wait functions
# need the same path to find nohup.out.  Call this after start_server().
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
# The NVFlare server name (dl3.tud.de) must resolve to Cosmos's Tailscale IP
# on all remote machines.  Some machines have stale /etc/hosts entries pointing
# to a LAN IP (192.168.33.195) that isn't reachable via Tailscale.
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

        # Read the current mapping (if any)
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
            # Replace the existing entry
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
    info "Stopping all NVFlare containers..."

    # Stop local containers on Cosmos (server + admin)
    local local_containers
    local_containers=$(docker ps --format '{{.Names}}' | grep -E "odelia_swarm|nvflare" || true)
    if [[ -n "$local_containers" ]]; then
        echo "$local_containers" | xargs docker kill 2>/dev/null || true
        echo "$local_containers" | xargs docker rm -f 2>/dev/null || true
    fi

    # Stop remote client containers
    # Track unique hosts to avoid stopping the same machine twice
    local -A visited_hosts=()
    for site in "${CLIENT_SITES[@]}"; do
        local host
        host=$(site_var "$site" HOST)
        if [[ -n "${visited_hosts[$host]:-}" ]]; then
            continue  # Already cleaned this host
        fi
        visited_hosts[$host]=1
        remote_exec "$site" \
            "docker ps --format '{{.Names}}' | grep -E 'odelia_swarm|nvflare' | xargs -r docker kill 2>/dev/null; \
             docker ps -a --format '{{.Names}}' | grep -E 'odelia_swarm|nvflare' | xargs -r docker rm -f 2>/dev/null" \
            2>/dev/null || warn "  Could not stop containers on $host"
    done

    # Clean up root-owned files left by NVFlare server container in DEPLOY_BASE.
    # The Docker container runs as root, creating workspace files (UUID dirs)
    # that non-root users can't delete.  This blocks the next deploy_kits() run.
    local server_name="${SERVER_NAME:-dl3.tud.de}"
    local server_dir="$DEPLOY_BASE/$server_name"
    if [[ -d "$server_dir" ]]; then
        # Remove the entire server dir (deploy_kits will re-create it).
        # Needs sudo/docker because NVFlare creates root-owned workspace files.
        sudo rm -rf "$server_dir" 2>/dev/null \
            || docker run --rm -v "$DEPLOY_BASE:/cleanup" alpine \
                rm -rf "/cleanup/$server_name" 2>/dev/null \
            || warn "Could not fully clean $server_dir (root-owned files may remain)"
    fi

    # Wait for containers to fully stop
    sleep 5
    ok "All containers stopped"
}

# ── Pre-pull Docker image on all remote machines ─────────────────────────
pre_pull_images() {
    step "Pre-pulling Docker image on remote machines"
    info "Image: $DOCKER_IMAGE"

    # Track unique hosts to avoid pulling on the same machine twice
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
        # Fall back to prod_dir
        local prod_dir
        prod_dir=$(find_latest_prod)
        server_startup="$prod_dir/$server_name/startup"
    fi

    if [[ ! -d "$server_startup" ]]; then
        err "Server startup kit not found at $server_startup"
        exit 1
    fi

    # Record which dir was actually used so wait functions can find nohup.out
    _server_startup_dir="$server_startup"

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

# ── Start clients (all remote) ───────────────────────────────────────────
start_clients() {
    local model_name="${1:-}"
    local model_flag=""
    if [[ -n "$model_name" ]]; then
        model_flag="--model_name '$model_name'"
        info "Starting clients with MODEL_NAME=$model_name"
    fi

    for site in "${CLIENT_SITES[@]}"; do
        local site_name host deploy_dir datadir scratchdir gpu
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)
        datadir=$(site_var "$site" DATADIR)
        scratchdir=$(site_var "$site" SCRATCHDIR)
        gpu=$(site_var "$site" GPU)

        info "Starting client: $site_name @ $host"

        remote_exec "$site" \
            "cd '$deploy_dir/$site_name/startup' && \
             export SITE_NAME='$site_name' && \
             export DATADIR='$datadir' && \
             export SCRATCHDIR='$scratchdir' && \
             ./docker.sh --no_pull --data_dir '$datadir' --scratch_dir '$scratchdir' --GPU '$gpu' $model_flag --start_client"

        ok "  Client started: $site_name"
    done

    ok "All clients started"
}

# ── Verify model selection made it into each client container ─────────────
verify_client_model_env() {
    local model_name="$1"
    [[ -z "$model_name" ]] && return 0

    info "Verifying client containers use MODEL_NAME=$model_name"
    for site in "${CLIENT_SITES[@]}"; do
        local site_name host
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)

        remote_exec "$site" "
            container=\$(docker ps --filter 'name=odelia_swarm_client_${site_name}_' --format '{{.Names}}' | head -1)
            test -n \"\$container\"
            docker inspect \"\$container\" --format '{{range .Config.Env}}{{println .}}{{end}}' \
                | grep -Fx 'MODEL_NAME=$model_name' >/dev/null
        " || {
            err "MODEL_NAME=$model_name not found in running client container for $site_name @ $host"
            return 1
        }
        ok "  $site_name container has MODEL_NAME=$model_name"
    done
}

# ── Wait for all clients to register with the server ─────────────────────
wait_for_client_registration() {
    resolve_server_startup_dir
    local server_log="${_server_startup_dir}/nohup.out"
    local max_wait=600  # 10 minutes — image pull on slow links can take a while
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
            # Give a few extra seconds for clients to settle
            sleep 5
            return 0
        fi

        sleep 10
        elapsed=$((elapsed + 10))

        if (( elapsed % 60 == 0 )); then
            # Show which clients are registered so far
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
    resolve_server_startup_dir
    local server_log="${_server_startup_dir}/nohup.out"

    # IMPORTANT: nohup.out is APPENDED across server restarts.  If we grep
    # the whole file we'll match "Server runner finished." left over from a
    # previous model's run (possibly an abort).  Record how many lines exist
    # RIGHT NOW so we only search NEW lines added during this run.
    local start_line=0
    if [[ -f "$server_log" ]]; then
        start_line=$(wc -l < "$server_log" 2>/dev/null || echo 0)
    fi

    local max_attempts=$(( timeout_minutes * 2 ))  # Check every 30 seconds
    local attempt=0

    info "Waiting for training to complete: $model_name (timeout: ${timeout_minutes}min, checking every 30s)"
    info "Server log: $server_log (scanning from line $((start_line + 1)))"

    # ── Helper: check if the LAST "Server runner finished" was clean ──────
    # NVFlare auto-retries failed jobs (up to 3 times) with new run IDs.
    # Each failed attempt produces FATAL_SYSTEM_ERROR + "Server runner finished."
    # followed by a new job launch.  We must NOT declare failure when a
    # subsequent retry succeeds.
    #
    # Strategy: find ALL "Server runner finished." lines.  For each one,
    # check whether it was immediately preceded by a FATAL/ABORT within
    # the same job run (within the last 50 lines before it).  Only the
    # LAST such block matters — if the last finish is clean, training
    # succeeded despite earlier failed attempts.
    _check_last_finish() {
        local new_lines="$1"
        local finish_count
        finish_count=$(echo "$new_lines" | { grep -c 'Server runner finished\.' 2>/dev/null || true; })

        if [[ "$finish_count" -eq 0 ]]; then
            echo "none"
            return
        fi

        # Get the line number (within new_lines) of the LAST "Server runner finished."
        local last_finish_lineno
        last_finish_lineno=$(echo "$new_lines" | grep -n 'Server runner finished\.' | tail -1 | cut -d: -f1)

        # Check the 50 lines before the last finish for fatal errors
        local context_start=$(( last_finish_lineno - 50 ))
        [[ $context_start -lt 1 ]] && context_start=1
        local context
        context=$(echo "$new_lines" | sed -n "${context_start},${last_finish_lineno}p")

        if echo "$context" | grep -q 'FATAL_SYSTEM_ERROR\|ABORT_RUN\|EXECUTION_EXCEPTION.*abort' 2>/dev/null; then
            # Last finish was a failure.  But check if there are signs of a
            # new job being launched AFTER this failure (auto-retry in progress).
            local after_finish
            after_finish=$(echo "$new_lines" | tail -n +"$((last_finish_lineno + 1))")
            if echo "$after_finish" | grep -q 'Launch job_id:\|JobRunner.*Got the job\|ServerRunner.*starting' 2>/dev/null; then
                echo "retry_in_progress"
            else
                echo "fatal"
            fi
        else
            echo "clean"
        fi
    }

    while [[ $attempt -lt $max_attempts ]]; do
        if [[ -f "$server_log" ]]; then
            local new_lines
            new_lines=$(tail -n +"$((start_line + 1))" "$server_log" 2>/dev/null || true)

            local status
            status=$(_check_last_finish "$new_lines")

            case "$status" in
                clean)
                    ok "Training completed for $model_name! (after $((attempt * 30))s)"
                    return 0
                    ;;
                fatal)
                    err "Training ABORTED for $model_name — fatal error detected (after $((attempt * 30))s)"
                    echo "$new_lines" | grep -i 'FATAL\|ABORT\|EXCEPTION\|ERROR' | tail -20 >&2
                    return 1
                    ;;
                retry_in_progress)
                    # NVFlare is auto-retrying — keep waiting
                    if (( attempt % 4 == 0 )); then
                        info "  NVFlare auto-retry in progress for $model_name, still waiting..."
                    fi
                    ;;
                none)
                    # No finish line yet — still running
                    ;;
            esac
        fi

        # Periodically show client nohup.out status so we can detect hangs early
        if (( attempt % 4 == 0 && attempt > 0 )); then
            for site in "${CLIENT_SITES[@]}"; do
                local site_name client_host client_user client_pass client_deploy_dir
                site_name=$(site_var "$site" SITE_NAME)
                client_host=$(site_var "$site" HOST)
                client_user=$(site_var "$site" USER)
                client_pass=$(site_var "$site" PASS)
                client_deploy_dir=$(site_var "$site" DEPLOY_DIR)
                local client_nohup="$client_deploy_dir/$site_name/startup/nohup.out"
                local last_line
                last_line=$(sshpass -p "$client_pass" ssh $SSH_OPTS "$client_user@$client_host" \
                    "tail -1 '$client_nohup' 2>/dev/null || echo '(no nohup.out yet)'" 2>/dev/null || echo "(SSH failed)")
                info "  Client $site_name last log: $last_line"
            done
        fi

        # Also check if the server container died
        if ! docker ps --format '{{.Names}}' | grep -qE "odelia_swarm|nvflare"; then
            # Container is gone — do a final check on the log
            if [[ -f "$server_log" ]]; then
                local new_lines
                new_lines=$(tail -n +"$((start_line + 1))" "$server_log" 2>/dev/null || true)
                local status
                status=$(_check_last_finish "$new_lines")
                if [[ "$status" == "clean" ]]; then
                    ok "Training completed for $model_name (container exited cleanly)"
                    return 0
                fi
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

# ── Collect checkpoints from client machines ─────────────────────────────
# After swarm training, FL_global_model.pt lives inside each client's Docker
# workspace on the remote machine at:
#   $deploy_dir/$site_name/<job_id>/app_$site_name/FL_global_model.pt
# This function SSHes into each client, finds the checkpoint, and SCPs it
# back to a local staging directory so predict.py can discover it.
collect_checkpoints() {
    local model_name="$1"

    step "Collecting checkpoints from client machines for $model_name"

    # Find the latest job ID from the server log (the one that actually succeeded).
    # Try the live log first, then the saved copy (stop_all deletes the server dir).
    resolve_server_startup_dir
    local server_log="${_server_startup_dir}/nohup.out"
    local saved_log="$RESULTS_DIR/${model_name}_server_nohup.out"
    local job_id=""
    for logfile in "$server_log" "$saved_log"; do
        if [[ -f "$logfile" ]]; then
            # Get the LAST "Server runner finished" and extract its job ID
            # The format in the log: [run=<job_id>] ... Server runner finished.
            job_id=$(grep 'Server runner finished\.' "$logfile" \
                | tail -1 \
                | grep -oP 'run=\K[0-9a-f-]+' \
                || true)
            [[ -n "$job_id" ]] && break
        fi
    done

    if [[ -z "$job_id" ]]; then
        warn "Could not determine job ID from server log — trying to find checkpoints by glob"
    else
        info "Job ID from server log: $job_id"
    fi
    LAST_JOB_ID="$job_id"

    # Staging directory on Cosmos for collected checkpoints
    local staging_dir="$RESULTS_DIR/${model_name}_checkpoints"
    rm -rf "$staging_dir"
    mkdir -p "$staging_dir"

    local found_any=false

    for site in "${CLIENT_SITES[@]}"; do
        local site_name host deploy_dir user pass
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)
        user=$(site_var "$site" USER)
        pass=$(site_var "$site" PASS)

        # Build the expected path on the remote machine
        local remote_base="$deploy_dir/$site_name"

        info "  Checking $site_name @ $host for checkpoints..."

        # Try the specific job_id first, then fall back to any UUID dir
        local search_cmd
        if [[ -n "$job_id" ]]; then
            search_cmd="ls -1 '$remote_base/$job_id/app_${site_name}/FL_global_model.pt' '$remote_base/$job_id/app_${site_name}/best_FL_global_model.pt' 2>/dev/null || true"
        else
            search_cmd="find '$remote_base' -maxdepth 3 \( -name 'FL_global_model.pt' -o -name 'best_FL_global_model.pt' \) 2>/dev/null || true"
        fi

        local remote_files
        if [[ "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
            remote_files=$(eval "$search_cmd" 2>/dev/null || true)
        elif [[ -n "$pass" ]]; then
            remote_files=$(sshpass -p "$pass" ssh $SSH_OPTS "$user@$host" "$search_cmd" 2>/dev/null || true)
        else
            remote_files=$(ssh $SSH_OPTS "$user@$host" "$search_cmd" 2>/dev/null || true)
        fi

        if [[ -z "$remote_files" ]]; then
            info "    No checkpoints found on $site_name"
            continue
        fi

        # Create a site-specific subdir matching what predict.py expects:
        # staging_dir/app_<site_name>/FL_global_model.pt
        local local_app_dir="$staging_dir/app_${site_name}"
        mkdir -p "$local_app_dir"

        while IFS= read -r remote_file; do
            [[ -z "$remote_file" ]] && continue
            local basename_file
            basename_file=$(basename "$remote_file")

            info "    Copying $basename_file from $site_name..."
            if [[ "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
                cp "$remote_file" "$local_app_dir/$basename_file" 2>/dev/null || {
                    # Might be root-owned inside Docker workspace
                    sudo cp "$remote_file" "$local_app_dir/$basename_file" 2>/dev/null || {
                        warn "    Could not copy $remote_file (permission denied)"
                        continue
                    }
                }
            elif [[ -n "$pass" ]]; then
                sshpass -p "$pass" scp $SSH_OPTS "$user@$host:$remote_file" "$local_app_dir/$basename_file" || {
                    warn "    Failed to SCP $remote_file from $host"
                    continue
                }
            else
                scp $SSH_OPTS "$user@$host:$remote_file" "$local_app_dir/$basename_file" || {
                    warn "    Failed to SCP $remote_file from $host"
                    continue
                }
            fi
            ok "    Collected $basename_file from $site_name"
            found_any=true
        done <<< "$remote_files"
    done

    if [[ "$found_any" == true ]]; then
        info "Checkpoints staged at: $staging_dir"
        info "Staged files:"
        find "$staging_dir" -name '*.pt' -exec ls -lh {} \; 2>/dev/null | while read -r line; do
            info "  $line"
        done
        echo "$staging_dir"
        return 0
    else
        err "No checkpoints found on any client machine"
        return 1
    fi
}

# ── Evaluate model ────────────────────────────────────────────────────────
evaluate_model() {
    local model_name="$1"
    local job_name="$2"

    step "Evaluating $model_name on $EVAL_SITE_NAME"

    # Collect checkpoints from remote client machines
    local checkpoint_dir
    checkpoint_dir=$(collect_checkpoints "$model_name") || {
        err "Cannot evaluate $model_name — no checkpoints collected"
        return 1
    }

    mkdir -p "$EVAL_SCRATCH_DIR"

    local eval_output_dir="$RESULTS_DIR/${model_name}_evaluation"
    mkdir -p "$eval_output_dir"

    info "Running predict.py INSIDE Docker container"
    info "  Docker image:  $DOCKER_IMAGE"
    info "  MODEL_NAME:    $model_name"
    info "  SITE_NAME:     $EVAL_SITE_NAME"
    info "  DATA_DIR:      $EVAL_DATA_DIR → /data"
    info "  SCRATCH_DIR:   $EVAL_SCRATCH_DIR → /scratch"
    info "  Checkpoints:   $checkpoint_dir → /workspace"
    info "  Output:        $eval_output_dir → /output"

    local eval_result=0
    docker run --rm \
        --gpus=device=0 \
        --net=host \
        --ipc=host \
        -v "$EVAL_DATA_DIR:/data/:ro" \
        -v "$EVAL_SCRATCH_DIR:/scratch/" \
        -v "$checkpoint_dir:/workspace/:ro" \
        -v "$eval_output_dir:/output/" \
        --env SITE_NAME="$EVAL_SITE_NAME" \
        --env DATA_DIR=/data \
        --env SCRATCH_DIR=/scratch \
        --env MODEL_NAME="$model_name" \
        --env TORCH_HOME=/torch_home \
        --env CONFIG=unilateral \
        "$DOCKER_IMAGE" \
        python3 /MediSwarm/scripts/evaluation/predict.py \
            --workspace /workspace \
            --model-name "$model_name" \
            --output-dir /output \
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
    local checkpoint_status="${6:-skipped}"  # pass, fail, or skipped
    local job_id="${7:-$LAST_JOB_ID}"

    local result_file="$RESULTS_DIR/deploy_test_${model_name}.json"
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    cat > "$result_file" <<EOF
{
    "model_name": "$model_name",
    "job_name": "$job_name",
    "job_id": "$job_id",
    "training_status": "$train_status",
    "evaluation_status": "$eval_status",
    "checkpoint_status": "$checkpoint_status",
    "duration_seconds": $duration_seconds,
    "timestamp": "$timestamp",
    "docker_image": "$DOCKER_IMAGE",
    "version": "$VERSION",
    "git_sha": "$GIT_SHA",
    "project_file": "$PROJECT_FILE",
    "workspace_dir": "$WORKSPACE_DIR",
    "client_sites": $(printf '%s\n' "${CLIENT_SITES[@]}" | jq -R . | jq -s .)
}
EOF

    info "Result recorded: $result_file"
}

# ── Run one training attempt ──────────────────────────────────────────────
_run_training_attempt() {
    local job_name="$1"
    local model_name="$2"

    # 1. Stop any existing containers
    stop_all

    # 2. Start server on Cosmos
    start_server

    # 3. Start clients on remote machines
    start_clients "$model_name"

    # 4. Verify the selected model was propagated to every client container
    verify_client_model_env "$model_name"

    # 5. Wait for ALL clients to register with the server
    wait_for_client_registration

    # 6. Submit job via admin on Cosmos
    submit_job "$job_name"

    # 7. Wait for training completion
    if wait_for_completion "$model_name"; then
        return 0
    else
        return 1
    fi
}

# ── Run one model through the full cycle ──────────────────────────────────
MAX_RETRIES=2  # Total attempts = 1 + MAX_RETRIES

run_single_model() {
    local job_name="$1"
    local model_name="$2"
    local start_time
    start_time=$(date +%s)

    step "Running deploy test: $model_name (job: $job_name)"
    echo ""
    info "Docker image: $DOCKER_IMAGE"
    info "Workspace: $WORKSPACE_DIR"
    info "Server: Cosmos (localhost)"
    info "Clients: ${CLIENT_SITES[*]}"
    echo ""

    local train_status="fail"
    local eval_status="skipped"
    local checkpoint_status="skipped"
    LAST_JOB_ID=""

    # Retry loop: transient failures (e.g. worker process race on dl3
    # when two NVFlare clients share the same machine) resolve on restart.
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
    done

    # Save the server log before stop_all deletes the server directory.
    # collect_checkpoints() needs it to extract the job_id.
    resolve_server_startup_dir
    local saved_server_log="$RESULTS_DIR/${model_name}_server_nohup.out"
    if [[ -n "$_server_startup_dir" && -f "$_server_startup_dir/nohup.out" ]]; then
        cp "$_server_startup_dir/nohup.out" "$saved_server_log" 2>/dev/null || true
    fi

    # Stop containers after training (pass or final failure)
    stop_all

    # Evaluate on UKA_1 by default. In startup-kit smoke mode, skip the legacy
    # Cosmos/UKA evaluation but still collect checkpoints for the dl0 smoke
    # evaluation script.
    if [[ "$train_status" == "pass" ]]; then
        if [[ "$SKIP_EVAL" == true ]]; then
            info "Skipping built-in evaluation for $model_name (--skip-eval); collecting checkpoints only"
            if collect_checkpoints "$model_name" >/dev/null; then
                checkpoint_status="pass"
            else
                checkpoint_status="fail"
            fi
        else
            if evaluate_model "$model_name" "$job_name"; then
                eval_status="pass"
                checkpoint_status="pass"
            else
                eval_status="fail"
                checkpoint_status="fail"
            fi
        fi
    else
        warn "Skipping evaluation for $model_name — training did not complete"
        eval_status="skipped"
    fi

    # Record result
    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))

    record_result "$model_name" "$job_name" "$train_status" "$eval_status" "$duration" "$checkpoint_status" "$LAST_JOB_ID"

    echo ""
    if [[ "$train_status" == "pass" && "$eval_status" == "pass" ]]; then
        ok "PASSED: $model_name (training + evaluation) in ${duration}s"
    elif [[ "$train_status" == "pass" && "$SKIP_EVAL" == true && "$checkpoint_status" == "pass" ]]; then
        ok "PASSED: $model_name (training + checkpoint collection; evaluation skipped) in ${duration}s"
    elif [[ "$train_status" == "pass" ]]; then
        warn "PARTIAL: $model_name (training passed, evaluation $eval_status, checkpoints $checkpoint_status) in ${duration}s"
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

    # Print individual results
    for result_file in "$RESULTS_DIR"/deploy_test_*.json; do
        [[ -f "$result_file" ]] || continue
        total=$((total + 1))

        local model train eval_stat checkpoint_stat duration
        model=$(jq -r '.model_name' "$result_file")
        train=$(jq -r '.training_status' "$result_file")
        eval_stat=$(jq -r '.evaluation_status' "$result_file")
        checkpoint_stat=$(jq -r '.checkpoint_status // "skipped"' "$result_file")
        duration=$(jq -r '.duration_seconds' "$result_file")

        if [[ "$train" == "pass" && ( "$eval_stat" == "pass" || ( "$eval_stat" == "skipped" && "$checkpoint_stat" == "pass" ) ) ]]; then
            passed=$((passed + 1))
            echo -e "  ${GREEN}PASS${NC}  $model (${duration}s)"
        else
            failed=$((failed + 1))
            echo -e "  ${RED}FAIL${NC}  $model (train=$train, eval=$eval_stat, checkpoints=$checkpoint_stat, ${duration}s)"
        fi
    done

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
    "version": "$VERSION",
    "docker_image": "$DOCKER_IMAGE",
    "git_sha": "$GIT_SHA",
    "project_file": "$PROJECT_FILE",
    "workspace_dir": "$WORKSPACE_DIR",
    "results_dir": "$RESULTS_DIR",
    "evaluation_skipped": $SKIP_EVAL
}
EOF

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

# Clean up any leftover containers and root-owned files from previous runs
stop_all

# Deploy startup kits to all sites (once — shared across all models)
step "Deploying startup kits to all sites"
deploy_kits

# Fix DNS on remote machines so they can reach the NVFlare server on Cosmos
fix_remote_dns

# Pre-pull Docker image on all remote machines (avoids slow pull during client startup)
pre_pull_images

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
