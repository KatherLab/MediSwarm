#!/usr/bin/env bash
# ============================================================================
# deploy_and_test.sh — Automated build, deploy, and test workflow for MediSwarm
#
# Usage:
#   ./deploy_and_test.sh <command> [args]
#
# Commands:
#   build                Build Docker image + startup kits
#   push                 Push Docker image to DockerHub
#   deploy               SCP startup kit zips to remote sites + unzip
#   start-server         Start NVFlare server locally
#   start-clients        Start NVFlare clients on remote sites
#   submit [job]         Open admin console and submit a job
#   status               Show running containers on all sites
#   logs <site>          Tail logs from a site (MHA, RSH, server)
#   stop                 Stop all containers on all sites
#   all [job]            Full pipeline: build -> push -> deploy -> start -> submit
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="$SCRIPT_DIR/deploy_sites.conf"

# ── Colors ──────────────────────────────────────────────────────────────────
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
step()  { echo -e "\n${BOLD}═══ $* ═══${NC}"; }

# ── Load Configuration ──────────────────────────────────────────────────────
if [[ ! -f "$CONF_FILE" ]]; then
    err "Configuration file not found: $CONF_FILE"
    echo "Create it with your site credentials. See deploy_sites.conf.example or the plan."
    exit 1
fi
# shellcheck source=deploy_sites.conf
source "$CONF_FILE"

# ── Derived Variables ───────────────────────────────────────────────────────
VERSION=$("$SCRIPT_DIR/scripts/build/getVersionNumber.sh")
DOCKER_IMAGE="jefftud/odelia:$VERSION"
GIT_SHORT_HASH=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD)

# The workspace directory name is derived from the project YAML "name:" field
# with the version placeholder replaced. For project_Challenge_test.yml this gives:
#   workspace/odelia_challenge_test_<VERSION>_model_test
PROJECT_NAME=$(grep "^name: " "$SCRIPT_DIR/$PROJECT_FILE" \
    | sed 's/^name: //' \
    | sed "s/__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__/$VERSION/")
WORKSPACE_DIR="$SCRIPT_DIR/workspace/$PROJECT_NAME"

# All sites to deploy to — configured in deploy_sites.conf via SITES=()
# Falls back to (MHA RSH) if deploy_sites.conf doesn't define SITES.
if [[ -z "${SITES+x}" || ${#SITES[@]} -eq 0 ]]; then
    SITES=(MHA RSH)
fi

# SSH options for sshpass
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# ── Helper Functions ────────────────────────────────────────────────────────

# Get a site variable by name, e.g.: site_var MHA HOST -> value of MHA_HOST
site_var() {
    local site=$1 var=$2
    local full_var="${site}_${var}"
    echo "${!full_var}"
}

# Run a command on a remote site via sshpass + SSH
remote_exec() {
    local site=$1; shift
    local host user pass
    host=$(site_var "$site" HOST)
    user=$(site_var "$site" USER)
    pass=$(site_var "$site" PASS)
    sshpass -p "$pass" ssh $SSH_OPTS "$user@$host" "$@"
}

# Copy a file to a remote site via sshpass + SCP
remote_copy() {
    local site=$1 src=$2 dst=$3
    local host user pass
    host=$(site_var "$site" HOST)
    user=$(site_var "$site" USER)
    pass=$(site_var "$site" PASS)
    sshpass -p "$pass" scp $SSH_OPTS "$src" "$user@$host:$dst"
}

# Find the latest prod_NN directory in the workspace
find_latest_prod() {
    if [[ ! -d "$WORKSPACE_DIR" ]]; then
        err "Workspace not found: $WORKSPACE_DIR"
        err "Run './deploy_and_test.sh build' first."
        exit 1
    fi
    ls -d "$WORKSPACE_DIR"/prod_* 2>/dev/null | sort -V | tail -n 1
}

# Check that required tools are installed
check_dependencies() {
    local missing=()
    if ! command -v sshpass &>/dev/null; then
        missing+=(sshpass)
    fi
    if [[ ${#missing[@]} -gt 0 ]]; then
        err "Missing required tools: ${missing[*]}"
        echo "Install with: sudo apt-get install ${missing[*]}"
        exit 1
    fi
}

# Check that expect is installed (only needed for submit)
check_expect() {
    if ! command -v expect &>/dev/null; then
        err "Missing required tool: expect (needed for admin console automation)"
        echo "Install with: sudo apt-get install expect"
        exit 1
    fi
}

# ── Commands ────────────────────────────────────────────────────────────────

cmd_build() {
    step "Building Docker image and startup kits"

    info "Project file: $PROJECT_FILE"
    info "Expected version: $VERSION"
    info "Expected image: $DOCKER_IMAGE"

    # buildDockerImageAndStartupKits.sh must be invoked from the repo root
    # with a RELATIVE project file path (it passes the path into a Docker
    # container where the absolute host path doesn't exist).
    (cd "$SCRIPT_DIR" && ./scripts/build/buildDockerImageAndStartupKits.sh -p "$PROJECT_FILE")

    # Verify
    local prod_dir
    prod_dir=$(find_latest_prod)
    info "Startup kits generated at: $prod_dir"

    for site in "${SITES[@]}"; do
        local site_name
        site_name=$(site_var "$site" SITE_NAME)
        local zip_file="$prod_dir/${site_name}_${VERSION}.zip"
        if [[ -f "$zip_file" ]]; then
            ok "Found startup kit: $(basename "$zip_file")"
        else
            err "Missing startup kit: $zip_file"
            exit 1
        fi
    done

    ok "Build complete"
}

cmd_push() {
    step "Pushing Docker image to DockerHub"

    info "Image: $DOCKER_IMAGE"
    docker push "$DOCKER_IMAGE"

    ok "Image pushed successfully"
}

cmd_deploy() {
    step "Deploying startup kits to remote sites"
    check_dependencies

    local prod_dir
    prod_dir=$(find_latest_prod)
    info "Using startup kits from: $prod_dir"

    for site in "${SITES[@]}"; do
        local site_name host user deploy_dir
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        user=$(site_var "$site" USER)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)

        local zip_file="$prod_dir/${site_name}_${VERSION}.zip"
        if [[ ! -f "$zip_file" ]]; then
            err "Startup kit not found: $zip_file"
            err "Run './deploy_and_test.sh build' first."
            exit 1
        fi

        echo ""
        info "Deploying to $site ($site_name @ $host)..."

        # Create deploy dir on remote if needed
        remote_exec "$site" "mkdir -p '$deploy_dir'"

        # Copy zip file
        info "  Copying $(basename "$zip_file")..."
        remote_copy "$site" "$zip_file" "$deploy_dir/"

        # Remove old kit directory and unzip new one
        info "  Unzipping on remote..."
        remote_exec "$site" "cd '$deploy_dir' && rm -rf '${site_name}' && unzip -qo '${site_name}_${VERSION}.zip'"

        ok "  Deployed $site_name to $host:$deploy_dir/$site_name/"
    done

    ok "All sites deployed"
}

cmd_start_server() {
    step "Starting NVFlare server (local)"

    local prod_dir
    prod_dir=$(find_latest_prod)
    local server_name="${SERVER_NAME:-dl3.tud.de}"
    local server_startup="$prod_dir/$server_name/startup"

    if [[ ! -d "$server_startup" ]]; then
        err "Server startup kit not found: $server_startup"
        exit 1
    fi

    info "Starting server from: $server_startup"
    cd "$server_startup"
    ./docker.sh --no_pull --start_server
    cd "$SCRIPT_DIR"

    info "Waiting 10s for server to initialize..."
    sleep 10

    # Verify
    if docker ps --format '{{.Names}}' | grep -qE "odelia_swarm|nvflare"; then
        ok "Server container is running"
    else
        warn "Server container not detected — it may still be starting"
    fi
}

cmd_start_clients() {
    step "Starting NVFlare clients on remote sites"
    check_dependencies

    for site in "${SITES[@]}"; do
        local site_name host deploy_dir datadir scratchdir gpu
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)
        datadir=$(site_var "$site" DATADIR)
        scratchdir=$(site_var "$site" SCRATCHDIR)
        gpu=$(site_var "$site" GPU)

        echo ""
        info "Starting client on $site ($site_name @ $host)..."

        remote_exec "$site" \
            "cd '$deploy_dir/$site_name/startup' && \
             export SITE_NAME='$site_name' && \
             export DATADIR='$datadir' && \
             export SCRATCHDIR='$scratchdir' && \
             ./docker.sh --data_dir '$datadir' --scratch_dir '$scratchdir' --GPU '$gpu' --start_client"

        ok "  Client started on $site_name"
    done

    ok "All clients started"
}

cmd_submit() {
    local job_name="${1:-$DEFAULT_JOB}"
    step "Submitting job: $job_name"
    check_expect

    local prod_dir
    prod_dir=$(find_latest_prod)
    local admin_startup="$prod_dir/$ADMIN_USER/startup"

    if [[ ! -d "$admin_startup" ]]; then
        err "Admin startup kit not found: $admin_startup"
        exit 1
    fi

    local job_path="MediSwarm/application/jobs/$job_name"
    info "Admin kit: $admin_startup"
    info "Job path: $job_path"
    info "Admin user: $ADMIN_USER"

    # Generate expect script on the fly
    local expect_script
    expect_script=$(mktemp /tmp/mediswarm_submit_XXXXXX.exp)
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
    info "Launching admin console and submitting job..."
    expect -f "$expect_script" || true
    cd "$SCRIPT_DIR"

    rm -f "$expect_script"
    ok "Job submitted: $job_name"
}

cmd_status() {
    step "Checking status on all sites"

    echo ""
    info "Local containers:"
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep -E "odelia|stamp|nvflare|NAMES" || echo "  (none)"

    check_dependencies

    for site in "${SITES[@]}"; do
        local host site_name
        host=$(site_var "$site" HOST)
        site_name=$(site_var "$site" SITE_NAME)

        echo ""
        info "$site ($site_name @ $host):"
        remote_exec "$site" \
            "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null | grep -E 'odelia|stamp|nvflare|NAMES' || echo '  (none)'" \
            2>/dev/null || warn "  Could not connect to $host"
    done
}

cmd_logs() {
    local target="${1:-}"
    if [[ -z "$target" ]]; then
        err "Usage: ./deploy_and_test.sh logs <site>"
        echo "  Sites: ${SITES[*]}, server"
        exit 1
    fi

    target=$(echo "$target" | tr '[:lower:]' '[:upper:]')

    if [[ "$target" == "SERVER" ]]; then
        local prod_dir
        prod_dir=$(find_latest_prod)
        local server_name="${SERVER_NAME:-dl3.tud.de}"
        local log_file="$prod_dir/$server_name/startup/nohup.out"
        if [[ -f "$log_file" ]]; then
            step "Server logs (last 50 lines)"
            tail -50 "$log_file"
        else
            err "Server log not found: $log_file"
        fi
        return
    fi

    check_dependencies

    # Check if it's a known site
    local found=false
    for site in "${SITES[@]}"; do
        if [[ "$target" == "$site" ]]; then
            found=true
            break
        fi
    done

    if [[ "$found" == false ]]; then
        err "Unknown site: $target"
        echo "  Available: ${SITES[*]} server"
        exit 1
    fi

    local host site_name deploy_dir
    host=$(site_var "$target" HOST)
    site_name=$(site_var "$target" SITE_NAME)
    deploy_dir=$(site_var "$target" DEPLOY_DIR)

    step "Logs from $target ($site_name @ $host) — last 50 lines"
    remote_exec "$target" \
        "tail -50 '$deploy_dir/$site_name/startup/nohup.out' 2>/dev/null || echo '(no logs found)'"
}

cmd_stop() {
    step "Stopping all containers"

    info "Stopping local containers..."
    # Kill all odelia containers locally
    local local_containers
    local_containers=$(docker ps --format '{{.Names}}' | grep -E "odelia_swarm|stamp|nvflare" || true)
    if [[ -n "$local_containers" ]]; then
        echo "$local_containers" | xargs docker kill 2>/dev/null || true
        ok "Stopped local containers"
    else
        info "No local odelia containers running"
    fi

    check_dependencies

    for site in "${SITES[@]}"; do
        local host site_name
        host=$(site_var "$site" HOST)
        site_name=$(site_var "$site" SITE_NAME)

        echo ""
        info "Stopping containers on $site ($host)..."
        remote_exec "$site" \
            "docker ps --format '{{.Names}}' | grep -E 'odelia_swarm|stamp|nvflare' | xargs -r docker kill 2>/dev/null || true" \
            2>/dev/null || warn "  Could not connect to $host"
        ok "  Stopped containers on $site"
    done

    ok "All containers stopped"
}

cmd_all() {
    local job_name="${1:-$DEFAULT_JOB}"
    step "Full deployment pipeline"
    info "Job: $job_name"
    echo ""

    cmd_build
    cmd_push
    cmd_deploy
    cmd_start_server
    cmd_start_clients

    info "Waiting 15s for clients to register with server..."
    sleep 15

    cmd_submit "$job_name"

    echo ""
    ok "Full pipeline complete!"
    info "Monitor progress at: http://172.24.4.65:8080/"
    info "Check status with: ./deploy_and_test.sh status"
    info "View logs with: ./deploy_and_test.sh logs <MHA|RSH|server>"
}

# ── Main ────────────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  build                Build Docker image + startup kits"
    echo "  push                 Push Docker image to DockerHub"
    echo "  deploy               SCP startup kit zips to remote sites + unzip"
    echo "  start-server         Start NVFlare server locally"
    echo "  start-clients        Start NVFlare clients on remote sites"
    echo "  submit [job]         Open admin console and submit a job (default: $DEFAULT_JOB)"
    echo "  status               Show running containers on all sites"
    echo "  logs <site>          Tail logs from a site (MHA, RSH, server)"
    echo "  stop                 Stop all containers on all sites"
    echo "  all [job]            Full pipeline: build -> push -> deploy -> start -> submit"
    echo ""
    echo "Examples:"
    echo "  $0 all challenge_1DivideAndConquer    # Full end-to-end"
    echo "  $0 deploy                              # Just redeploy kits"
    echo "  $0 submit challenge_3agaldran          # Submit a different job"
    echo "  $0 logs MHA                            # Check MHA logs"
    echo "  $0 stop                                # Kill everything"
    echo ""
    echo "Sites are configured in deploy_sites.conf via SITES=(SITE1 SITE2 ...)."
    echo "Server name is configured via SERVER_NAME=dl3.tud.de (default)."
}

COMMAND="${1:-}"
shift || true

case "$COMMAND" in
    build)          cmd_build ;;
    push)           cmd_push ;;
    deploy)         cmd_deploy ;;
    start-server)   cmd_start_server ;;
    start-clients)  cmd_start_clients ;;
    submit)         cmd_submit "${1:-}" ;;
    status)         cmd_status ;;
    logs)           cmd_logs "${1:-}" ;;
    stop)           cmd_stop ;;
    all)            cmd_all "${1:-}" ;;
    -h|--help|help) usage ;;
    "")             usage; exit 1 ;;
    *)              err "Unknown command: $COMMAND"; usage; exit 1 ;;
esac
