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

# ── Ensure the ADMIN/SERVER host itself resolves the server FQDN ──────────
# The admin startup kit's docker.sh runs with --net=host, so the admin container
# resolves $SERVER_NAME through *this* machine's /etc/hosts. The remote DNS fix
# deliberately skips this host, so without this check the admin silently fails to
# reach the server ("Cannot connect to host ...: Name or service not known"), the
# job is never submitted, and the run waits out its whole timeout on a job that
# never existed. Optional COSMOS_PASS is used for sudo when passwordless sudo is
# unavailable.
ensure_local_dns() {
    local server_fqdn="${SERVER_NAME:-dl3.tud.de}"
    local cosmos_ip
    cosmos_ip=$(tailscale ip -4 2>/dev/null) || {
        err "Cannot determine this host's Tailscale IP (is tailscale running?)"
        return 1
    }

    step "Ensuring $server_fqdn resolves to $cosmos_ip on this (admin/server) host"

    local current
    current=$(getent hosts "$server_fqdn" 2>/dev/null | awk '{print $1}' | head -1 || true)
    if [[ "$current" == "$cosmos_ip" ]]; then
        ok "  $server_fqdn -> $cosmos_ip"
        return 0
    fi

    warn "  $server_fqdn resolves to '${current:-nothing}' here (expected $cosmos_ip) — fixing /etc/hosts"

    local -a sudo_cmd
    if sudo -n true 2>/dev/null; then
        sudo_cmd=(sudo)
    elif [[ -n "${COSMOS_PASS:-}" ]]; then
        sudo_cmd=(sudo -S -p '')
    else
        err "  Cannot edit /etc/hosts (no sudo, no COSMOS_PASS). Add this line and re-run:"
        err "      $cosmos_ip $server_fqdn"
        return 1
    fi

    if [[ -n "$current" ]]; then
        printf '%s\n' "${COSMOS_PASS:-}" | "${sudo_cmd[@]}" \
            sed -i "s|^.*\\b${server_fqdn}\\b.*\$|${cosmos_ip} ${server_fqdn}|" /etc/hosts
    else
        printf '%s\n' "${COSMOS_PASS:-}" | "${sudo_cmd[@]}" \
            bash -c "echo '${cosmos_ip} ${server_fqdn}' >> /etc/hosts"
    fi

    current=$(getent hosts "$server_fqdn" 2>/dev/null | awk '{print $1}' | head -1 || true)
    if [[ "$current" != "$cosmos_ip" ]]; then
        err "  $server_fqdn still resolves to '${current:-nothing}' on this host."
        err "  The admin container (--net=host) will not reach the server. Aborting."
        return 1
    fi
    ok "  $server_fqdn -> $cosmos_ip"
}

# ── Verify an admin `submit_job` actually landed ──────────────────────────
# `expect` exits 0 even when the admin never reached the server, so its exit
# status must never be trusted. Prints the job id on stdout; diagnostics go to
# stderr. Returns non-zero if the job was not submitted.
#   usage: JOB_ID=$(assert_job_submitted "$submit_log" "$job_name") || exit 1
assert_job_submitted() {
    local submit_log="$1" job_name="$2"
    local server="${SERVER_NAME:-dl3.tud.de}"

    sed -i -E 's/\x1b\[[0-9;]*m//g' "$submit_log" 2>/dev/null || true

    if grep -qiE "ClientConnectorDNSError|FLCommunicationError|cannot connect to server" "$submit_log"; then
        err "Admin could not reach the server '$server' — job '$job_name' was NOT submitted."
        grep -iE "ClientConnectorDNSError|FLCommunicationError|cannot connect" "$submit_log" \
            | head -3 | sed 's/^/    /' >&2
        err "  The admin runs with --net=host: '$server' must resolve on THIS host."
        err "  Full admin session: $submit_log"
        return 1
    fi

    local job_id
    job_id=$(grep -oE "Submitted job: [0-9a-fA-F-]{36}" "$submit_log" | awk '{print $3}' | tail -1 || true)
    if [[ -z "$job_id" ]]; then
        err "Job submission failed for '$job_name' (no job id returned)"
        tail -15 "$submit_log" | sed 's/^/    /' >&2
        err "  Full admin session: $submit_log"
        return 1
    fi

    printf '%s\n' "$job_id"
}
