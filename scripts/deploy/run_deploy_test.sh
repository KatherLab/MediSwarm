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
#
# Testing a candidate image with EXISTING kits (the way sites will run a release:
# they keep their 1.8.1 kits and only the image moves):
#   MEDISWARM_IMAGE_VERSION=<version the kits were built as>   # selects the kit workspace
#   ./scripts/deploy/run_deploy_test.sh --image jefftud/odelia:<candidate> --model MST --job ... --conf ...
# Clients get the image through the kit's own --image flag; the server and the admin
# run through a copy of their docker.sh with the image line rewritten (their templates
# have no image override), so the job code submitted is the candidate's as well. Each
# container's image id is checked against the candidate and recorded in the result.
# A conf may give one client another image with <SITE>_IMAGE (mixed-version test).
# --no-pull (or MEDISWARM_TEST_IMAGE_NO_PULL=1) uses images already on the hosts,
# e.g. a local *-local build on dl3; --cleanup-only removes this kit's containers.
#
# Container cleanup only ever touches this kit's containers for this conf's sites
# (container_scope.sh): the test hosts also run other projects' containers.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Shared colour/logging + SSH/workspace helpers (#276).
# shellcheck source=scripts/deploy/deploy_common.sh
source "$SCRIPT_DIR/deploy_common.sh"
# shellcheck source=scripts/deploy/container_scope.sh
source "$SCRIPT_DIR/container_scope.sh"

# ── Parse arguments ────────────────────────────────────────────────────────
RUN_ALL=false
SINGLE_MODEL=""
SINGLE_JOB=""
CONF_FILE=""
SKIP_BUILD=false
SKIP_EVAL=false
RESULTS_DIR_OVERRIDE=""
TIMEOUT_MINUTES=10080   # Per-model training timeout (7 days)
TEST_IMAGE="${MEDISWARM_TEST_IMAGE:-}"
NO_PULL="${MEDISWARM_TEST_IMAGE_NO_PULL:-}"
CLEANUP_ONLY=false

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
        --image)        TEST_IMAGE="$2"; shift ;;
        --no-pull)      NO_PULL=1 ;;
        --cleanup-only) CLEANUP_ONLY=true ;;
        -h|--help)
            echo "Usage: $0 [--all | --model NAME --job JOB_DIR] --conf CONF_FILE [--skip-build] [--skip-eval] [--results-dir DIR] [--timeout MINUTES]"
            echo "          [--image REF] [--no-pull] [--cleanup-only]"
            echo "  --image REF     run server, admin and clients on REF with the existing kits (env MEDISWARM_TEST_IMAGE)"
            echo "  --no-pull       use images already present on the hosts (env MEDISWARM_TEST_IMAGE_NO_PULL=1)"
            echo "  --cleanup-only  remove this kit's containers for this conf's sites, then exit"
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

# Shares a host with the self-hosted CI runner: same GPU, same Docker daemon,
# same 8002/8003. Queue behind any other GPU/Docker job rather than interleaving
# and failing an unrelated PR (#448, root cause behind #388). --cleanup-only
# touches nothing but this kit's own containers, so it does not queue.
if [[ "$CLEANUP_ONLY" != true ]]; then
    . "$SCRIPT_DIR/../ci/host_gpu_lock.sh"
    acquire_host_lock "ODELIA deploy test" || exit 1
fi

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
if [[ "$CLEANUP_ONLY" != true && "$RUN_ALL" == false && -z "$SINGLE_MODEL" ]]; then
    err "Must specify --all or --model NAME --job JOB_DIR"
    exit 1
fi

# ── Load configuration ─────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$CONF_FILE"

# VERSION names the kit workspace (and is the image tag of a normal build-and-test
# run). With --image the image under test is decoupled from it: the kits stay the
# ones built as VERSION, which is how a release candidate meets the sites' kits.
VERSION=$("$REPO_ROOT/scripts/build/getVersionNumber.sh")
KIT_VERSION="$VERSION"
# Recorded in the result JSON only. Must not kill the run when the harness is
# executed from an exported tree (no .git) -- it did exactly that once, one line
# after MEDISWARM_IMAGE_VERSION had been honoured, and the whole run died before
# stop_all with nothing but "fatal: not a git repository" in the log.
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
if [[ -n "$TEST_IMAGE" ]]; then
    if [[ ! "$TEST_IMAGE" =~ ^[A-Za-z0-9._/:@-]+$ ]]; then
        err "Invalid --image '$TEST_IMAGE'"
        exit 1
    fi
    DOCKER_IMAGE="$TEST_IMAGE"
else
    DOCKER_IMAGE="jefftud/odelia:$VERSION"
fi
MAX_RETRIES="${MAX_RETRIES:-2}"  # from the conf if it sets one; total attempts = 1 + MAX_RETRIES

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
# The post-training evaluation runs predict.py on THIS host against a held-out
# site's test split. The defaults are the historical Cosmos layout (UKA_1 on
# /mnt/sda1); a conf may override all three, which is how the CI runners point
# at the data they actually hold (dl0: RUMC_1 under /mnt/dlhd0/medswarmdata,
# dl2: MHA_1 under the challenge tree). Without an override the release-triggered
# run trained all six models and then failed every evaluation on a missing
# annotation.csv (v1.8.1, 13 Sep 2026).
EVAL_SITE_NAME="${EVAL_SITE_NAME:-UKA_1}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/mnt/sda1/ODELIA_Challenge_unilateral}"
EVAL_SCRATCH_DIR="${EVAL_SCRATCH_DIR:-/mnt/scratch/deploy_test_eval}"

# site_var / remote_exec / remote_copy / find_latest_prod /
# resolve_server_startup_dir are provided by deploy_common.sh (#276).

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

        # The sudo password travels on stdin, not inside the remote command line,
        # where any user of the remote host could read it with `ps`.
        if [[ -n "$current_ip" ]]; then
            info "  $host maps $server_fqdn → $current_ip (wrong, updating to $cosmos_ip)"
            # Replace the existing entry
            printf '%s\n' "$(site_pass "$site")" | remote_exec "$site" \
                "sudo -S -p '' sed -i 's|^.*\\b${server_fqdn}\\b.*\$|${cosmos_ip} ${server_fqdn}|' /etc/hosts" \
                2>/dev/null
        else
            info "  $host has no entry for $server_fqdn (adding)"
            printf '%s\n' "$(site_pass "$site")" | remote_exec "$site" \
                "sudo -S -p '' bash -c 'echo \"${cosmos_ip} ${server_fqdn}\" >> /etc/hosts'" \
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

# ── Container name filter, scoped to THIS test's kit (#472) ───────────────
# stop_all() used to kill every container matching 'odelia_swarm|nvflare' on this
# host AND on every client site. That pattern also matches PRODUCTION containers:
# the live ODELIA server here is odelia_swarm_server_flserver_<hash> and partner
# clients are odelia_swarm_client_<SITE>_<hash>. A deploy test (or a retry, or a
# concurrent run) would therefore tear down the running consortium swarm.
#
# Every built startup kit hardcodes its container names as
#   odelia_swarm_{client,server}_<name>_<git-short-hash>
# so the trailing hash identifies the kit this test drives. The filter matches only
# this kit's client containers for this conf's sites, its server and its admin
# (container_scope.sh). When the hash cannot be derived the harness refuses to
# remove anything: the old fallback ('odelia_swarm|nvflare|^swarm-') also matched
# the other project's swarm-*-node containers that run on dl0/dl2/dl3.
KIT_CONTAINER_SUFFIX=""
KIT_SUFFIX_RESOLVED=false

client_site_names() {
    local site
    for site in "${CLIENT_SITES[@]}"; do
        site_var "$site" SITE_NAME
    done
}

resolve_kit_container_suffix() {
    [[ "$KIT_SUFFIX_RESOLVED" == true ]] && return 0
    KIT_SUFFIX_RESOLVED=true

    local prod_dir
    prod_dir=$(find_latest_prod 2>/dev/null || true)
    KIT_CONTAINER_SUFFIX=$(kit_container_suffix "$prod_dir" odelia_swarm || true)

    if [[ -n "$KIT_CONTAINER_SUFFIX" ]]; then
        info "Container cleanup scoped to this kit: *_${KIT_CONTAINER_SUFFIX} (#472)"
    else
        err "Could not derive this kit's container suffix from ${prod_dir:-<no prod dir>}/*/startup/docker.sh;"
        err "  refusing to remove containers by a broad pattern (other projects run on these hosts)."
    fi
}

# Anchored extended regex matching only this test's containers; fails without a suffix.
container_filter() {
    resolve_kit_container_suffix
    [[ -n "$KIT_CONTAINER_SUFFIX" ]] || return 1
    local -a names
    mapfile -t names < <(client_site_names)
    kit_container_filter odelia_swarm "$KIT_CONTAINER_SUFFIX" "${names[@]}"
}

# This kit's server container only (the liveness check must not be satisfied by a
# client container that runs on the same host, as on the dl3-local test bed).
server_container_filter() {
    resolve_kit_container_suffix
    [[ -n "$KIT_CONTAINER_SUFFIX" ]] || return 1
    kit_server_container_filter odelia_swarm "$KIT_CONTAINER_SUFFIX"
}

client_container_name() {
    resolve_kit_container_suffix
    [[ -n "$KIT_CONTAINER_SUFFIX" ]] || return 1
    printf 'odelia_swarm_client_%s_%s\n' "$1" "$KIT_CONTAINER_SUFFIX"
}

# ── Stop all containers ───────────────────────────────────────────────────
stop_all() {
    local filter
    filter=$(container_filter) || {
        err "Not stopping containers: no kit-scoped filter (see above)."
        return 1
    }
    info "Stopping this kit's NVFlare containers (filter: $filter)..."

    # Stop local containers on Cosmos (server + admin)
    local local_containers
    local_containers=$(docker ps -a --format '{{.Names}}' | grep -E "$filter" || true)
    if [[ -n "$local_containers" ]]; then
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
            "docker ps -a --format '{{.Names}}' | grep -E '$filter' | xargs -r docker rm -f 2>/dev/null" \
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

clean_local_deploy_dir() {
    local dir_name=$1
    local target="$DEPLOY_BASE/$dir_name"

    [[ -e "$target" ]] || return 0
    rm -rf "$target" 2>/dev/null \
        || sudo rm -rf "$target" 2>/dev/null \
        || docker run --rm -v "$DEPLOY_BASE:/cleanup" alpine \
            rm -rf "/cleanup/$dir_name" 2>/dev/null \
        || {
            warn "Could not fully clean $target (root-owned files may remain)"
            return 1
        }
}

# ── Pre-pull Docker image on all remote machines ─────────────────────────
# Image a client runs: <SITE>_IMAGE from the conf (mixed-version tests), else the
# image under test.
site_image() {
    local image
    image=$(site_var "$1" IMAGE)
    printf '%s\n' "${image:-$DOCKER_IMAGE}"
}

pre_pull_images() {
    step "Pre-pulling Docker image on remote machines"
    info "Image: $DOCKER_IMAGE${NO_PULL:+ (no pull: must already be on every host)}"

    # Track unique host+image pairs to avoid pulling the same image twice
    local -A visited=()
    for site in "${CLIENT_SITES[@]}"; do
        local host image
        host=$(site_var "$site" HOST)
        image=$(site_image "$site")
        if [[ ! "$image" =~ ^[A-Za-z0-9._/:@-]+$ ]]; then
            err "Invalid image for $site: '$image'"
            exit 1
        fi
        local key="$host|$image"
        if [[ -n "${visited[$key]:-}" ]]; then
            continue
        fi
        visited[$key]=1

        if [[ -n "$NO_PULL" ]]; then
            remote_exec "$site" "docker image inspect '$image' >/dev/null 2>&1" || {
                err "Image $image is not on $host (--no-pull)"
                exit 1
            }
            ok "  $image present on $host"
            continue
        fi
        info "Pulling $image on $host ..."
        remote_exec "$site" "docker pull '$image'" || {
            err "Failed to pull image on $host"
            exit 1
        }
        ok "  Image pulled on $host"
    done

    # The server and admin run here with --no_pull. A normal run built the image on
    # this host; a candidate image may have to be fetched first.
    if [[ -n "$TEST_IMAGE" ]]; then
        if [[ -z "$NO_PULL" ]]; then
            docker pull "$DOCKER_IMAGE" >/dev/null || { err "Failed to pull $DOCKER_IMAGE on this host"; exit 1; }
        fi
        docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1 || { err "$DOCKER_IMAGE is not on this host"; exit 1; }
        ok "  $DOCKER_IMAGE present on this (server/admin) host"
    fi

    ok "Docker image available on all remote machines"
}

# Image id an image reference resolves to on a site's host (or on this host).
image_id_on() {
    local site=$1 image=$2
    if [[ -z "$site" ]]; then
        docker image inspect --format '{{.Id}}' "$image" 2>/dev/null
    else
        remote_exec "$site" "docker image inspect --format '{{.Id}}' '$image' 2>/dev/null"
    fi
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
        clean_local_deploy_dir "$server_name"
        cd "$DEPLOY_BASE" && unzip -qo "$(basename "$server_zip")"
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
        # Jobs staged into the deployed admin kit with prepare_odelia_job.sh
        # (local/mediswarm_jobs/, submitted by absolute --job path) must survive
        # the redeploy: the clean below rm -rf's the whole kit, and an absolute
        # --job then fails with "is not a valid folder" (13 Sep, guard test).
        local staged="$DEPLOY_BASE/$ADMIN_USER/local/mediswarm_jobs"
        local staged_keep="$DEPLOY_BASE/.staged_jobs.$$"
        if [[ -d "$staged" ]]; then
            mv "$staged" "$staged_keep"
        fi
        clean_local_deploy_dir "$ADMIN_USER"
        cd "$DEPLOY_BASE" && unzip -qo "$(basename "$admin_zip")"
        cd "$REPO_ROOT"
        if [[ -d "$staged_keep" ]]; then
            mkdir -p "$DEPLOY_BASE/$ADMIN_USER/local"
            mv "$staged_keep" "$staged"
            info "  Preserved staged jobs in the admin kit: $(ls "$staged" | tr '\n' ' ')"
        fi
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

    # nohup.out is appended across every server start in this kit directory. Remember
    # where this session begins, so the registration check cannot be satisfied by a
    # 'New client' line of an earlier session (#599: the harness once submitted a job
    # to an empty swarm and then waited out its whole timeout).
    SERVER_LOG_OFFSET=0
    if [[ -f "$server_startup/nohup.out" ]]; then
        SERVER_LOG_OFFSET=$(wc -l < "$server_startup/nohup.out" 2>/dev/null || echo 0)
    fi

    local script="./docker.sh"
    if [[ -n "$TEST_IMAGE" ]]; then
        script=$(kit_script_with_image "$server_startup" "$DOCKER_IMAGE") || return 1
        info "Server runs the image under test: $DOCKER_IMAGE (kit built as $KIT_VERSION)"
    fi

    info "Starting server from: $server_startup"
    cd "$server_startup"
    "$script" --no_pull --start_server
    cd "$REPO_ROOT"
    [[ "$script" == ./docker.sh ]] || rm -f "$script"

    info "Waiting 15s for server to initialize..."
    sleep 15

    local server_ctr filter
    filter=$(server_container_filter) || return 1
    server_ctr=$(docker ps --format '{{.Names}}' | grep -E "$filter" | head -1 || true)
    if [[ -z "$server_ctr" ]]; then
        warn "Server container not detected — it may still be starting"
        return 0
    fi
    ok "Server container is running: $server_ctr"
    SERVER_IMAGE_ID=$(docker inspect --format '{{.Image}}' "$server_ctr" 2>/dev/null || true)
    local want
    want=$(image_id_on "" "$DOCKER_IMAGE" || true)
    if [[ -n "$want" && "$SERVER_IMAGE_ID" != "$want" ]]; then
        err "Server container runs image $SERVER_IMAGE_ID, not $DOCKER_IMAGE ($want)"
        return 1
    fi
}

# Server log lines written since this server session started.
server_log_this_session() {
    local log="${_server_startup_dir}/nohup.out"
    [[ -f "$log" ]] || return 0
    tail -n +"$(( ${SERVER_LOG_OFFSET:-0} + 1 ))" "$log" 2>/dev/null || true
}

# ── Start clients (all remote) ───────────────────────────────────────────
CLIENT_IMAGE_IDS=()   # "<site name>=<image id>" of every started client, for the result JSON

start_clients() {
    local model_name="${1:-}"
    if [[ -n "$model_name" ]]; then
        info "Starting clients with MODEL_NAME=$model_name"
    fi
    CLIENT_IMAGE_IDS=()

    for site in "${CLIENT_SITES[@]}"; do
        local site_name host deploy_dir datadir scratchdir gpu institution docker_opts image container
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)
        datadir=$(site_var "$site" DATADIR)
        scratchdir=$(site_var "$site" SCRATCHDIR)
        gpu=$(site_var "$site" GPU)
        # The loader reads /data/$INSTITUTION/..., and INSTITUTION defaults to
        # SITE_NAME. Test clients are named TEST_A_1.. (not hospital names, see
        # the project YAML), so the data folder must be named explicitly.
        # Optional per site: <SITE>_INSTITUTION=<folder under DATADIR>.
        institution=$(site_var "$site" INSTITUTION)
        # Optional per site: <SITE>_DOCKER_OPTIONS, extra `docker run` options --
        # e.g. "--env ODELIA_RETURN_PER_CASE=1". Passed with the kit's own
        # --docker_options flag (which wins over image.conf, like the env var did).
        docker_opts=$(site_var "$site" DOCKER_OPTIONS)
        image=$(site_image "$site")
        container=$(client_container_name "$site_name") || return 1

        info "Starting client: $site_name @ $host"

        # Pin the image this test actually tests. Client kits ship startup/image.conf
        # pinning the ':current' release channel, and docker.sh's precedence is
        # --image > MEDISWARM_IMAGE > image.conf > built-in default. Without --image,
        # image.conf won and every client ran ':current' -- i.e. the last released
        # image, NOT the build under test, so the deploy test silently validated
        # something else. pre_pull_images() has already pulled the image on the
        # remote, so --no_pull stays correct.
        #
        # Every value is quoted with printf %q for the remote shell (#599: options
        # spliced into a hand-quoted string broke the client start on dl3 while the
        # harness still printed OK).
        local -a args=(--no_pull --image "$image" --data_dir "$datadir" --scratch_dir "$scratchdir" --GPU "$gpu")
        [[ -n "$institution" ]] && args+=(--institution "$institution")
        [[ -n "$model_name" ]] && args+=(--model_name "$model_name")
        [[ -n "$docker_opts" ]] && args+=(--docker_options "$docker_opts")
        args+=(--start_client)
        local cmd
        cmd="cd $(printf '%q' "$deploy_dir/$site_name/startup")"
        cmd+=" && export SITE_NAME=$(printf '%q' "$site_name") DATADIR=$(printf '%q' "$datadir")"
        cmd+=" SCRATCHDIR=$(printf '%q' "$scratchdir")"
        cmd+=" && ./docker.sh$(printf ' %q' "${args[@]}")"
        remote_exec "$site" "$cmd" || warn "  docker.sh exited non-zero for $site_name"

        # docker.sh does not pass docker run's failure on, so look at the result.
        local running got want
        running=$(remote_exec "$site" "docker ps --format '{{.Names}}' | grep -Fx $(printf '%q' "$container")" \
            2>/dev/null || true)
        if [[ "$running" != "$container" ]]; then
            err "  Client container $container is not running on $host"
            return 1
        fi
        got=$(remote_exec "$site" "docker inspect --format '{{.Image}}' $(printf '%q' "$container")" 2>/dev/null || true)
        want=$(image_id_on "$site" "$image" || true)
        if [[ -z "$got" || "$got" != "$want" ]]; then
            err "  $container runs image '${got:-?}', expected $image (${want:-not on $host})"
            return 1
        fi
        CLIENT_IMAGE_IDS+=("$site_name=$got")
        ok "  Client started: $site_name ($image)"
    done

    ok "All clients started"
}

# ── Verify model selection made it into each client container ─────────────
verify_client_model_env() {
    local model_name="$1"
    [[ -z "$model_name" ]] && return 0

    info "Verifying client containers use MODEL_NAME=$model_name"
    for site in "${CLIENT_SITES[@]}"; do
        local site_name host container
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        container=$(client_container_name "$site_name") || return 1

        remote_exec "$site" \
            "docker inspect $(printf '%q' "$container") --format '{{range .Config.Env}}{{println .}}{{end}}' \
                | grep -Fx $(printf '%q' "MODEL_NAME=$model_name") >/dev/null" || {
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
    info "Server log: $server_log (lines after $((${SERVER_LOG_OFFSET:-0})), i.e. this server session only)"

    # Registered = a 'New client <SITE>@' line in THIS session; earlier sessions in
    # the same nohup.out do not count (#599).
    _registered() {
        grep -q "New client ${1}@" <<< "$session" 2>/dev/null
    }

    local session
    while [[ $elapsed -lt $max_wait ]]; do
        session=$(server_log_this_session)
        local all_registered=true
        for site in "${expected_clients[@]}"; do
            local site_name
            site_name=$(site_var "$site" SITE_NAME)
            if ! _registered "$site_name"; then
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
            session=$(server_log_this_session)
            for site in "${expected_clients[@]}"; do
                local site_name
                site_name=$(site_var "$site" SITE_NAME)
                if _registered "$site_name"; then
                    registered=$((registered + 1))
                fi
            done
            info "  ${registered}/${#expected_clients[@]} clients registered (${elapsed}s elapsed)"
        fi
    done

    # Timeout: show which clients are missing
    warn "Timed out waiting for client registration. Status:"
    session=$(server_log_this_session)
    for site in "${expected_clients[@]}"; do
        local site_name
        site_name=$(site_var "$site" SITE_NAME)
        if _registered "$site_name"; then
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

    # A job name is resolved under the image's job tree. An ABSOLUTE path is
    # submitted as given, so a job staged with prepare_odelia_job.sh into the
    # admin kit (mounted at /fl_admin/local/) can be run through this harness
    # -- that is how the warm-start provenance guard (#545) gets exercised.
    local job_path
    if [[ "$job_name" == /* ]]; then
        job_path="$job_name"
    else
        job_path="MediSwarm/application/jobs/$job_name"
    fi
    info "Submitting job: $job_name (path: $job_path)"

    # With --image the admin runs the image under test too: a job given by name is
    # read from the admin container's /MediSwarm tree, and that must be the
    # candidate's job code, not the code of the image the kit was built with.
    local admin_script="./docker.sh"
    if [[ -n "$TEST_IMAGE" ]]; then
        admin_script=$(kit_script_with_image "$admin_startup" "$DOCKER_IMAGE") || return 1
    fi

    # Generate expect script
    local expect_script
    expect_script=$(mktemp /tmp/mediswarm_deploy_test_XXXXXX.exp)
    cat > "$expect_script" <<EXPECT_EOF
#!/usr/bin/env expect
set timeout 120
spawn $admin_script --no_pull
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

    # Capture the admin session: `expect` exits 0 even when the admin never
    # reached the server, so its exit status must not be trusted.
    # An absolute --job path contains slashes; keep the log name flat.
    local job_tag
    job_tag=$(basename "$job_name")
    local submit_log="$RESULTS_DIR/submit_${job_tag}.log"
    cd "$admin_startup"
    expect -f "$expect_script" > "$submit_log" 2>&1 || true
    cd "$REPO_ROOT"

    rm -f "$expect_script"
    [[ "$admin_script" == ./docker.sh ]] || rm -f "$admin_script"

    # Only a real job id proves the submission landed (deploy_common.sh). A failed
    # submission fails this attempt (the retry loop decides), not the whole run.
    local job_id
    job_id=$(assert_job_submitted "$submit_log" "$job_name") || return 1
    ok "Job submitted: $job_name (id: $job_id)"
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
                local site_name client_deploy_dir
                site_name=$(site_var "$site" SITE_NAME)
                client_deploy_dir=$(site_var "$site" DEPLOY_DIR)
                local client_nohup="$client_deploy_dir/$site_name/startup/nohup.out"
                local last_line
                last_line=$(remote_exec "$site" \
                    "tail -1 '$client_nohup' 2>/dev/null || echo '(no nohup.out yet)'" 2>/dev/null || echo "(SSH failed)")
                info "  Client $site_name last log: $last_line"
                # A worker that dies at launch leaves the client "healthy" and the
                # coordinator waiting for a site that never configures. Name it (#596).
                local worker_death
                worker_death=$(remote_exec "$site" \
                    "grep -hoE 'OMP: Error #[0-9]+.{0,60}|child worker process finished with RC [1-9][0-9]*' '$client_nohup' 2>/dev/null | tail -1" 2>/dev/null || true)
                [[ -n "$worker_death" ]] && warn "  Client $site_name WORKER DIED: $worker_death"
            done
        fi

        # Also check if the server container died
        if ! docker ps --format '{{.Names}}' | grep -qE "$(server_container_filter)"; then
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

# ── Warm-start mirror hygiene (#347, #545) ───────────────────────────────
# Every client mirrors its latest global to /scratch/mediswarm_latest_global.pt
# and, with warm_start_mode=auto, warm-starts from whatever it finds there. The
# deploy test reuses one SCRATCHDIR per site across models and across days, so
# model N inherits model N-1's weights. Before #545 that loaded silently and the
# run died at the first gather -- "None of the 187 incoming model parameter(s)
# matched the local model's 450" was MST aggregating into a 1DivideAndConquer
# mirror left by a failed run three days earlier (2026-09-11). Since #545 the
# provenance guard panics instead. Either way the model is lost and --all can
# never get past its first model. A deploy test must start every model from
# scratch, so drop the mirror and its sidecar on every client before the first
# attempt. Retries within one model keep it: same architecture, and resuming
# from the last good round is exactly what warm-continue is for.
clear_stale_mirrors() {
    local mirror="mediswarm_latest_global.pt"
    # A test that exercises warm-start itself plants a mirror on purpose -- e.g.
    # the provenance guard's refuse path plants a wrong-architecture one and
    # expects WARM_START_MODEL_MISMATCH. Let it opt out of the wipe explicitly.
    if [[ -n "${DEPLOY_TEST_KEEP_MIRROR:-}" ]]; then
        warn "DEPLOY_TEST_KEEP_MIRROR set: leaving every client's $mirror in place"
        return 0
    fi
    for site in "${CLIENT_SITES[@]}"; do
        local site_name host scratchdir
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        scratchdir=$(site_var "$site" SCRATCHDIR)
        [[ -n "$scratchdir" ]] || continue
        info "Clearing warm-start mirror on $site_name @ $host: $scratchdir/$mirror"
        remote_exec "$site" "rm -f '$scratchdir/$mirror' '$scratchdir/$mirror.provenance.json'" 2>/dev/null \
            || warn "  Could not clear $scratchdir/$mirror on $host -- this model may warm-start from a stale one"
    done
}

# ── Save server-side job outputs before stop_all deletes the server dir ───
# The only server file the harness keeps is nohup.out; stop_all() then rm -rf's
# $DEPLOY_BASE/<server>/ and every <job_id>/ run dir with it. That is where the
# opt-in per-case predictions (#557: cross_site_val/per_case_predictions/<site>.csv)
# and the per-site metrics (cross_site_val/cross_val_results.json) land, so a
# per-case deploy test would "pass" and leave nothing to inspect. Copy each
# run's cross_site_val/ into RESULTS_DIR first. Runs that wrote none are skipped.
#
# Use the directory the server actually ran from, not $DEPLOY_BASE: every
# attempt's stop_all() deletes the deployed server kit BEFORE start_server(),
# which then falls back to the kit under workspace/.../prod_00. In practice the
# server always runs there, and its run dirs pile up across models and days.
# Copy only the run this model produced (LAST_JOB_ID) when we know it.
save_server_artifacts() {
    local model_name="$1"
    resolve_server_startup_dir
    [[ -n "$_server_startup_dir" ]] || return 0
    local server_root
    server_root=$(dirname "$_server_startup_dir")
    [[ -d "$server_root" ]] || return 0

    # collect_checkpoints() has not run yet, so find this run's job id the same
    # way it does: the last "Server runner finished." line in the server log.
    local want_job="$LAST_JOB_ID"
    if [[ -z "$want_job" && -f "$_server_startup_dir/nohup.out" ]]; then
        want_job=$(grep 'Server runner finished\.' "$_server_startup_dir/nohup.out" \
            | tail -1 | grep -oP 'run=\K[0-9a-f-]+' || true)
    fi

    local dest_root="$RESULTS_DIR/${model_name}_server_artifacts"
    local run_dir job_id src
    for run_dir in "$server_root"/*/; do
        [[ -d "$run_dir" ]] || continue
        job_id=$(basename "$run_dir")
        [[ -z "$want_job" || "$job_id" == "$want_job" ]] || continue
        src="$run_dir/cross_site_val"
        [[ -d "$src" ]] || continue
        mkdir -p "$dest_root/$job_id"
        if cp -r "$src" "$dest_root/$job_id/" 2>/dev/null; then
            info "Saved server artifacts: $dest_root/$job_id/cross_site_val"
            find "$dest_root/$job_id/cross_site_val" -type f 2>/dev/null | while read -r f; do
                info "  $(basename "$f") ($(wc -l < "$f" 2>/dev/null || echo '?') lines)"
            done
        else
            warn "Could not copy $src (root-owned files?)"
        fi
    done

    # A job that COMPLETED has no run dir any more: NVFlare packs the server
    # workspace into its job store -- inside the container, as the zip
    # /tmp/nvflare/jobs-storage/<job_id>/workspace -- and deletes the run dir.
    # Only aborted runs leave a run dir behind, which is why the copy above
    # worked on 12 Sep (aborted) and found nothing on 13 Sep (20 rounds, clean).
    # The server role runs on this host, so read the store with docker cp.
    if [[ -n "$want_job" && ! -d "$dest_root/$want_job/cross_site_val" ]]; then
        local server_ctr filter
        # Only this kit's server: an unscoped '^odelia_swarm_server_' would read the job
        # store of whatever ODELIA server runs on this host (the coordinator on agh1).
        filter=$(server_container_filter) || return 0
        server_ctr=$(docker ps --format '{{.Names}}' | grep -E "$filter" | head -1 || true)
        if [[ -z "$server_ctr" ]]; then
            warn "No cross_site_val for job $want_job and no running server container to read the job store from"
            return 0
        fi
        local tmp tries=0
        tmp=$(mktemp -d)
        # The store is written a few seconds after "Server runner finished."
        while [[ $tries -lt 6 ]]; do
            docker cp "$server_ctr:/tmp/nvflare/jobs-storage/$want_job/workspace" "$tmp/workspace.zip" 2>/dev/null && break
            tries=$((tries + 1)); sleep 10
        done
        if [[ -f "$tmp/workspace.zip" ]] \
            && python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$tmp/workspace.zip" "$tmp/ws" 2>/dev/null \
            && [[ -d "$tmp/ws/cross_site_val" ]]; then
            mkdir -p "$dest_root/$want_job"
            cp -r "$tmp/ws/cross_site_val" "$dest_root/$want_job/"
            info "Saved server artifacts from the job store: $dest_root/$want_job/cross_site_val"
            find "$dest_root/$want_job/cross_site_val" -type f 2>/dev/null | while read -r f; do
                info "  $(basename "$f") ($(wc -l < "$f" 2>/dev/null || echo '?') lines)"
            done
        else
            warn "No cross_site_val for job $want_job: no run dir, and the job store in $server_ctr had no workspace zip with one"
        fi
        rm -rf "$tmp"
    fi
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
        local site_name host deploy_dir
        site_name=$(site_var "$site" SITE_NAME)
        host=$(site_var "$site" HOST)
        deploy_dir=$(site_var "$site" DEPLOY_DIR)

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
        else
            remote_files=$(remote_exec "$site" "$search_cmd" 2>/dev/null || true)
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
            else
                remote_fetch "$site" "$remote_file" "$local_app_dir/$basename_file" || {
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

    if [[ ! -f "$EVAL_DATA_DIR/$EVAL_SITE_NAME/metadata_unilateral/annotation.csv" ]]; then
        err "Evaluation data for $EVAL_SITE_NAME not found under $EVAL_DATA_DIR on this host"
        err "  (set EVAL_SITE_NAME / EVAL_DATA_DIR / EVAL_SCRATCH_DIR in the conf, or use --skip-eval)"
        return 1
    fi
    mkdir -p "$EVAL_SCRATCH_DIR" 2>/dev/null || true

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

    # Which image was really exercised: the reference, its local id and digests,
    # the id the server container ran and the id each client container ran.
    local image_id repo_digests clients_json
    image_id=$(docker image inspect --format '{{.Id}}' "$DOCKER_IMAGE" 2>/dev/null || true)
    repo_digests=$(docker image inspect --format '{{json .RepoDigests}}' "$DOCKER_IMAGE" 2>/dev/null || echo '[]')
    clients_json=$(printf '%s\n' "${CLIENT_IMAGE_IDS[@]:-}" | jq -R 'select(length > 0) | split("=") | {(.[0]): .[1]}' \
        | jq -s 'add // {}')

    jq -n \
        --arg model_name "$model_name" \
        --arg job_name "$job_name" \
        --arg job_id "$job_id" \
        --arg training_status "$train_status" \
        --arg evaluation_status "$eval_status" \
        --arg checkpoint_status "$checkpoint_status" \
        --argjson duration_seconds "$duration_seconds" \
        --arg timestamp "$timestamp" \
        --arg docker_image "$DOCKER_IMAGE" \
        --arg docker_image_id "$image_id" \
        --argjson docker_image_repo_digests "${repo_digests:-[]}" \
        --arg test_image "$TEST_IMAGE" \
        --arg kit_version "$KIT_VERSION" \
        --arg version "$VERSION" \
        --arg git_sha "$GIT_SHA" \
        --arg project_file "$PROJECT_FILE" \
        --arg workspace_dir "$WORKSPACE_DIR" \
        --arg server_image_id "${SERVER_IMAGE_ID:-}" \
        --argjson client_image_ids "$clients_json" \
        --argjson client_sites "$(printf '%s\n' "${CLIENT_SITES[@]}" | jq -R . | jq -s .)" \
        '{model_name: $model_name, job_name: $job_name, job_id: $job_id,
          training_status: $training_status, evaluation_status: $evaluation_status,
          checkpoint_status: $checkpoint_status, duration_seconds: $duration_seconds,
          timestamp: $timestamp, docker_image: $docker_image, docker_image_id: $docker_image_id,
          docker_image_repo_digests: $docker_image_repo_digests,
          test_image: (if $test_image == "" then null else $test_image end),
          kit_version: $kit_version, version: $version, git_sha: $git_sha,
          project_file: $project_file, workspace_dir: $workspace_dir,
          server_image_id: $server_image_id, client_image_ids: $client_image_ids,
          client_sites: $client_sites}' > "$result_file"

    info "Result recorded: $result_file"
}

# ── Run one training attempt ──────────────────────────────────────────────
# Runs inside `if`, where `set -e` is suspended, so every step's failure must be
# returned explicitly. Before #599 a failed MODEL_NAME check or registration wait
# fell through to submit_job and the run waited out its timeout on an empty swarm.
_run_training_attempt() {
    local job_name="$1"
    local model_name="$2"

    # 1. Stop any existing containers
    stop_all || return 1

    # 2. Start server on Cosmos
    start_server || return 1

    # 3. Start clients on remote machines
    start_clients "$model_name" || return 1

    # 4. Verify the selected model was propagated to every client container
    verify_client_model_env "$model_name" || return 1

    # 5. Wait for ALL clients to register with the server
    wait_for_client_registration || return 1

    # 6. Submit job via admin on Cosmos
    submit_job "$job_name" || return 1

    # 7. Wait for training completion
    if wait_for_completion "$model_name"; then
        return 0
    else
        return 1
    fi
}

# ── Run one model through the full cycle ──────────────────────────────────
# MAX_RETRIES is set near the top from the conf (default 2); total attempts =
# 1 + MAX_RETRIES. It used to be reassigned here, after the conf was sourced, so a
# conf's MAX_RETRIES=1 was silently ignored (#599).

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

    # Start this model from scratch on every client (see clear_stale_mirrors).
    clear_stale_mirrors

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
    # Same reason: the per-case CSVs and per-site metrics live in the server's
    # run dir, which stop_all is about to delete.
    save_server_artifacts "$model_name"

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
    "kit_version": "$KIT_VERSION",
    "docker_image": "$DOCKER_IMAGE",
    "test_image": "$TEST_IMAGE",
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

# Resolve the kit's container suffix once, here: the filters are later built in
# $(...) subshells, which would otherwise each look it up again.
resolve_kit_container_suffix
if [[ -z "$KIT_CONTAINER_SUFFIX" ]]; then
    err "Cannot scope container cleanup to this kit; refusing to run (see above)."
    exit 1
fi
if [[ -n "$TEST_IMAGE" ]]; then
    info "Image under test: $DOCKER_IMAGE; kits: $WORKSPACE_DIR (built as $KIT_VERSION)"
fi

if [[ "$CLEANUP_ONLY" == true ]]; then
    stop_all
    exit $?
fi

# Clean up any leftover containers and root-owned files from previous runs
stop_all

# Deploy startup kits to all sites (once — shared across all models)
step "Deploying startup kits to all sites"
deploy_kits

# Fix DNS on this (admin/server) host first — the admin container uses --net=host
ensure_local_dns

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
