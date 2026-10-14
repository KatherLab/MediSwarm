# shellcheck shell=bash
# ============================================================================
# container_scope.sh — which containers a deploy-test harness may remove
#
# The test hosts are shared. dl0, dl2 and dl3 also run another project's node
# containers (swarm-<hash>-dd2-node-A/B/C, all GPUs, --net host), and every
# ODELIA kit names its containers odelia_swarm_{client,server,admin}_..., the
# same scheme the production coordinator and the partner sites use. A cleanup
# that greps for 'odelia_swarm|nvflare|^swarm-' therefore tears down whatever
# else runs there (#472, and the swarm-* collateral found on 14 Oct 2026).
#
# Every built kit bakes its container names as
#     odelia_swarm_client_<SITE>_<suffix>
#     odelia_swarm_server_<svr_name>_<suffix>
#     odelia_swarm_admin_<suffix>
# where <suffix> is the git short hash of the commit the kits were built from.
# A harness removes ONLY names of that exact shape for its own kit and its own
# client sites, and refuses to clean up at all when the suffix cannot be read
# (fail closed: a leftover container is a nuisance, a removed foreign one is
# an outage).
#
# Usage (after sourcing):
#   sfx=$(kit_container_suffix "$prod_dir" odelia_swarm) || refuse
#   re=$(kit_container_filter odelia_swarm "$sfx" TEST_A_1 TEST_B_1)
#   docker ps -a --format '{{.Names}}' | grep -E "$re"
# This file defines functions only.
# ============================================================================

# Escape a literal for use inside an extended regular expression.
_ere_escape() {
    printf '%s' "$1" | sed -e 's/[][\.^$*+?(){}|\/]/\\&/g'
}

# kit_container_suffix <prod_dir> [prefix]
# Prints the container-name suffix baked into the kits under <prod_dir>; returns
# non-zero (and prints nothing) when no kit docker.sh carries one.
kit_container_suffix() {
    local prod_dir="$1" prefix="${2:-odelia_swarm}"
    [[ -n "$prod_dir" && -d "$prod_dir" ]] || return 1
    local docker_sh suffix
    while IFS= read -r docker_sh; do
        # The server kit's line is CONTAINER_NAME=<prefix>_server_${svr_name}_<suffix>.
        suffix=$(grep -hoE "^[[:space:]]*CONTAINER_NAME=${prefix}_[^[:space:]]*_[0-9a-f]{7,40}\$" "$docker_sh" 2>/dev/null \
            | head -1 | sed -E 's/.*_([0-9a-f]{7,40})$/\1/')
        if [[ -n "$suffix" ]]; then
            printf '%s\n' "$suffix"
            return 0
        fi
    done < <(find "$prod_dir" -mindepth 3 -maxdepth 3 -path '*/startup/docker.sh' 2>/dev/null | sort)
    return 1
}

# kit_container_filter <prefix> <suffix> [client_site...]
# Prints an anchored ERE matching exactly this kit's client containers for the
# given sites (any client when no site is given), its server and its admin.
kit_container_filter() {
    local prefix="$1" suffix="$2"
    shift 2
    if [[ ! "$suffix" =~ ^[0-9a-f]{7,40}$ ]]; then
        echo "kit_container_filter: refusing suffix '$suffix'" >&2
        return 1
    fi
    local clients site
    if [[ $# -gt 0 ]]; then
        clients=""
        for site in "$@"; do
            [[ -n "$site" ]] || continue
            clients+="${clients:+|}$(_ere_escape "$site")"
        done
        clients="(${clients})"
    else
        clients='[A-Za-z0-9_.-]+'
    fi
    local p
    p=$(_ere_escape "$prefix")
    printf '^%s_(client_%s|server_[A-Za-z0-9_.-]+|admin)_%s$\n' "$p" "$clients" "$suffix"
}

# kit_server_container_filter <prefix> <suffix>: this kit's server container only.
kit_server_container_filter() {
    local prefix="$1" suffix="$2"
    [[ "$suffix" =~ ^[0-9a-f]{7,40}$ ]] || return 1
    printf '^%s_server_[A-Za-z0-9_.-]+_%s$\n' "$(_ere_escape "$prefix")" "$suffix"
}
