# shellcheck shell=bash
# ============================================================================
# deploy_common.sh — shared helpers for the deploy-test orchestrators (#276)
#
# Sourced by run_deploy_test.sh (ODELIA) and run_stamp_deploy_test.sh (STAMP).
# Contains only the byte-identical shared layer: colour/logging helpers and the
# SSH / workspace-path helpers. Pipeline-specific and structurally-divergent
# functions (deploy_kits, start_clients, stop_all, wait_for_completion, ...)
# stay in the respective scripts; parameterising those is a follow-up.
#
# Contract — the sourcing script must define before use:
#   SSH_OPTS       ssh/scp options            (used by remote_exec/remote_copy)
#   WORKSPACE_DIR  build workspace dir         (used by find_latest_prod)
#   DEPLOY_BASE    server/admin deploy dir     (used by resolve_server_startup_dir)
#   SERVER_NAME    server participant name     (optional; defaults to dl3.tud.de)
# and per-site variables <SITE>_HOST / _USER / _PASS (used by site_var lookups).
# This file defines no `set` options and runs no code beyond definitions.
# ============================================================================

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
