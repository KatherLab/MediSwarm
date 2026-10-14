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
# and per-site variables <SITE>_HOST / _USER (used by site_var lookups). The
# password comes from <SITE>_PASS, or the environment variable named by
# <SITE>_PASS_ENV, or the first line of the file <SITE>_PASS_FILE (see site_pass);
# with none of them set, ssh uses keys only.
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

# Optional per-site variables (INSTITUTION, DOCKER_OPTIONS, IMAGE, PASS_ENV, ...)
# may be absent from a conf. The '-' default keeps `set -u` callers alive: a bare
# ${!name} of an unset variable aborts the whole harness with "unbound variable".
site_var() {
    local site=$1 var=$2
    local full_var="${site}_${var}"
    printf '%s\n' "${!full_var-}"
}

# site_pass <site>: the site's SSH/sudo password, from <SITE>_PASS, else the
# environment variable named by <SITE>_PASS_ENV, else the first line of the file
# <SITE>_PASS_FILE. Empty when none is set, which means key-based SSH. Keeping the
# secret in the environment or a 0600 file lets a conf be shared without it.
site_pass() {
    local site=$1 pass env_name file
    pass=$(site_var "$site" PASS)
    if [[ -z "$pass" ]]; then
        env_name=$(site_var "$site" PASS_ENV)
        [[ -n "$env_name" ]] && pass="${!env_name-}"
    fi
    if [[ -z "$pass" ]]; then
        file=$(site_var "$site" PASS_FILE)
        if [[ -n "$file" && -r "$file" ]]; then
            IFS= read -r pass < "$file" || true
        fi
    fi
    printf '%s' "$pass"
}

# The password reaches sshpass through SSHPASS (sshpass -e), never through argv:
# `sshpass -p <secret>` is readable by every user of the host in `ps`.
remote_exec() {
    local site=$1; shift
    local host user pass
    host=$(site_var "$site" HOST)
    user=$(site_var "$site" USER)
    pass=$(site_pass "$site")

    if [[ -n "$pass" ]]; then
        SSHPASS="$pass" sshpass -e ssh $SSH_OPTS "$user@$host" "$@"
    else
        ssh -o BatchMode=yes $SSH_OPTS "$user@$host" "$@"
    fi
}

remote_copy() {
    local site=$1 src=$2 dst=$3
    local host user pass
    host=$(site_var "$site" HOST)
    user=$(site_var "$site" USER)
    pass=$(site_pass "$site")

    if [[ -n "$pass" ]]; then
        SSHPASS="$pass" sshpass -e scp $SSH_OPTS "$src" "$user@$host:$dst"
    else
        scp -o BatchMode=yes $SSH_OPTS "$src" "$user@$host:$dst"
    fi
}

# remote_fetch <site> <remote path> <local path>: copy a file back from a site.
remote_fetch() {
    local site=$1 src=$2 dst=$3
    local host user pass
    host=$(site_var "$site" HOST)
    user=$(site_var "$site" USER)
    pass=$(site_pass "$site")

    if [[ -n "$pass" ]]; then
        SSHPASS="$pass" sshpass -e scp $SSH_OPTS "$user@$host:$src" "$dst"
    else
        scp -o BatchMode=yes $SSH_OPTS "$user@$host:$src" "$dst"
    fi
}

# kit_script_with_image <startup_dir> <image>: write a copy of the kit's docker.sh
# whose built-in DOCKER_IMAGE line names <image>, next to the original (the script
# derives its kit directory from its own location), and print its path.
#
# Why: the server and admin templates pin the image their kit was built with and
# read neither image.conf nor an --image flag. To run a candidate image with the
# sites' existing kits (sites keep their 1.8.1 kits; only the image moves), the
# test must start the server and the admin on that image too, and the admin must
# submit the candidate's job code. Only the first plain assignment is replaced;
# client kits' later DOCKER_IMAGE="${CLI_IMAGE:-...}" lines contain '$' and stay.
kit_script_with_image() {
    local startup_dir=$1 image=$2
    local src="$startup_dir/docker.sh" out="$startup_dir/.docker_image_override.sh"
    [[ -f "$src" ]] || { err "kit_script_with_image: $src not found"; return 1; }
    [[ "$image" =~ ^[A-Za-z0-9._/:@-]+$ ]] || { err "kit_script_with_image: invalid image '$image'"; return 1; }
    awk -v img="$image" '
        !done && /^[[:space:]]*DOCKER_IMAGE=[^$"]*$/ {
            match($0, /^[[:space:]]*/)
            print substr($0, 1, RLENGTH) "DOCKER_IMAGE=" img
            done = 1
            next
        }
        { print }
        END { exit done ? 0 : 1 }
    ' "$src" > "$out" || { err "kit_script_with_image: no DOCKER_IMAGE= line in $src"; rm -f "$out"; return 1; }
    chmod +x "$out"
    printf '%s\n' "$out"
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
    local cosmos_pass
    cosmos_pass=$(site_pass COSMOS)  # COSMOS_PASS, COSMOS_PASS_ENV or COSMOS_PASS_FILE
    if sudo -n true 2>/dev/null; then
        sudo_cmd=(sudo)
    elif [[ -n "$cosmos_pass" ]]; then
        sudo_cmd=(sudo -S -p '')
    else
        err "  Cannot edit /etc/hosts (no sudo, no COSMOS_PASS[_ENV|_FILE]). Add this line and re-run:"
        err "      $cosmos_ip $server_fqdn"
        return 1
    fi

    if [[ -n "$current" ]]; then
        printf '%s\n' "$cosmos_pass" | "${sudo_cmd[@]}" \
            sed -i "s|^.*\\b${server_fqdn}\\b.*\$|${cosmos_ip} ${server_fqdn}|" /etc/hosts
    else
        printf '%s\n' "$cosmos_pass" | "${sudo_cmd[@]}" \
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
