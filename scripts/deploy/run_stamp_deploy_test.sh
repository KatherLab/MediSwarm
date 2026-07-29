#!/usr/bin/env bash
# ============================================================================
# run_stamp_deploy_test.sh — Orchestrate 2-node STAMP deploy test
#
# Runs STAMP_classification through one or more full federated training cycles
# across 2 physical machines connected via Tailscale VPN, with Cosmos as the
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
#  10. Record pass/fail result for each model
#
# Unlike the ODELIA deploy test (run_deploy_test.sh), this script:
#   - Can run one or more STAMP models against STAMP_classification
#   - Exports STAMP_* environment variables on remote hosts before docker.sh
#   - Uses stamp_swarm container name prefix (not odelia_swarm)
#   - Optional post-training evaluation via --evaluate (#270): collects the
#     global checkpoints and runs stamp_predict.py to record AUROC/accuracy.
#     Needs the eval site's data staged on this machine (see --eval-data-dir).
#   - Has no retry loop (simpler for initial testing)
#
# Usage:
#   ./scripts/deploy/run_stamp_deploy_test.sh --conf deploy_sites_2node_stamp_test.conf
#   ./scripts/deploy/run_stamp_deploy_test.sh --conf deploy_sites_2node_stamp_test.conf --timeout 60
#   ./scripts/deploy/run_stamp_deploy_test.sh --conf deploy_sites_2node_stamp_test.conf --models vit,mlp
#
# The Docker image and startup kits must be built BEFORE running this script:
#   ./scripts/build/buildDockerImageAndStartupKits.sh \
#       -p application/provision/project_deploy_test_2site.yml \
#       -d docker_config/Dockerfile_STAMP --num-rounds 2
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Shared colour/logging + SSH/workspace helpers (#276).
# shellcheck source=scripts/deploy/deploy_common.sh
source "$SCRIPT_DIR/deploy_common.sh"

# Shares a host with the self-hosted CI runner: same GPU, same Docker daemon,
# same 8002/8003. Queue behind any other GPU/Docker job rather than interleaving
# and failing an unrelated PR (#448, root cause behind #388).
. "$SCRIPT_DIR/../ci/host_gpu_lock.sh"
acquire_host_lock "STAMP/DECADE 2-node deploy test" || exit 1

# ── Parse arguments ────────────────────────────────────────────────────────
CONF_FILE=""
TIMEOUT_MINUTES=120   # 2 hours for a quick 2-round test
MODELS_ARG=""
# Models supported by the STAMP version shipped in the image (2.4.0), i.e.
# stamp.modeling.registry.ModelName == {vit, mlp, trans_mil, linear}.
# `barspoon` was only added in STAMP 2.5.0 (models/barspoon.py) and raises
# "'barspoon' is not a valid ModelName" here — see BARSPOON_MIN_STAMP below.
ALL_STAMP_MODELS=(vit mlp trans_mil linear)
BARSPOON_MIN_STAMP="2.5.0"
TEST_MODELS=()

# Populated by run_single_model(); drive the script's exit status.
PASSED_MODELS=()
FAILED_MODELS=()
TRAINING_ONLY_MODELS=()  # passed training, but a requested evaluation did not verify (#476)
EVALUATE=false          # run the post-training evaluation step (#270)
EVAL_SITE_ARG=""        # site whose data to evaluate on (default: first client site)
EVAL_DATA_DIR_ARG=""    # host path to eval data on this (server) machine
TASK="classification"   # STAMP task: classification | survival | regression (#271)
COMPARE_LOCAL=false     # also train a local-only baseline and compare (#275)
STRICT=false            # treat a requested-but-skipped/failed evaluation as a failure (#476)

while [[ $# -gt 0 ]]; do
    case $1 in
        --conf)          CONF_FILE="$2"; shift ;;
        --timeout)       TIMEOUT_MINUTES="$2"; shift ;;
        --models)        MODELS_ARG="$2"; shift ;;
        --task)          TASK="$2"; shift ;;
        --evaluate)      EVALUATE=true ;;
        --compare-local) COMPARE_LOCAL=true ;;
        --strict)        STRICT=true ;;
        --eval-site)     EVAL_SITE_ARG="$2"; shift ;;
        --eval-data-dir) EVAL_DATA_DIR_ARG="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 --conf CONF_FILE [--timeout MINUTES] [--models MODEL1,MODEL2,...]"
            echo "          [--task classification|survival|regression]"
            echo "          [--evaluate [--eval-site SITE] [--eval-data-dir DIR]]"
            echo ""
            echo "  --conf           Path to site configuration file (required)"
            echo "  --timeout        Per-model training timeout in minutes (default: 120)"
            echo "  --models         Comma-separated STAMP models to test"
            echo "                   (default: vit,mlp,trans_mil,linear; barspoon needs STAMP >= 2.5.0)"
            echo "  --task           STAMP task type (default: classification). The site"
            echo "                   data must be generated for this task, e.g."
            echo "                   create_synthetic_stamp_dataset.py <dir> --task survival (#271)."
            echo "  --evaluate       After training, collect global checkpoints and run"
            echo "                   stamp_predict.py to record task metrics (#270)."
            echo "  --compare-local  With --evaluate: also train a local-only baseline on the"
            echo "                   eval site and compare federated vs local (#275)."
            echo "  --strict         With --evaluate: fail (non-zero exit) if evaluation was"
            echo "                   requested but skipped or errored, instead of reporting a"
            echo "                   plain PASS. Otherwise such a model is 'PASSED (training only)' (#476)."
            echo "  --eval-site      Site whose data (on this machine) to evaluate on"
            echo "                   (default: first client site; needs its data locally)"
            echo "  --eval-data-dir  Host dir mounted as /data for evaluation"
            echo "                   (default: the eval site's DATADIR from the conf)"
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

# ── Model selection ────────────────────────────────────────────────────────
is_supported_stamp_model() {
    local candidate="$1"
    local model
    for model in "${ALL_STAMP_MODELS[@]}"; do
        if [[ "$model" == "$candidate" ]]; then
            return 0
        fi
    done
    return 1
}

resolve_test_models() {
    if [[ -z "$MODELS_ARG" ]]; then
        TEST_MODELS=("${ALL_STAMP_MODELS[@]}")
        return
    fi

    local raw_models=()
    IFS=',' read -r -a raw_models <<< "$MODELS_ARG"
    TEST_MODELS=()

    local model
    for model in "${raw_models[@]}"; do
        model="${model//[[:space:]]/}"
        [[ -z "$model" ]] && continue

        if [[ "$model" == "barspoon" ]]; then
            err "STAMP model 'barspoon' requires STAMP >= $BARSPOON_MIN_STAMP."
            err "  The image ships STAMP 2.4.0, whose ModelName enum has no 'barspoon';"
            err "  training would abort with \"'barspoon' is not a valid ModelName\"."
            err "  Supported models: ${ALL_STAMP_MODELS[*]}"
            exit 1
        fi

        if ! is_supported_stamp_model "$model"; then
            err "Unsupported STAMP model: $model"
            err "Supported models: ${ALL_STAMP_MODELS[*]}"
            exit 1
        fi

        TEST_MODELS+=("$model")
    done

    if [[ ${#TEST_MODELS[@]} -eq 0 ]]; then
        err "No valid STAMP models specified via --models"
        exit 1
    fi
}

resolve_test_models

# ── Validate task (#271) ───────────────────────────────────────────────────
case "$TASK" in
    classification|survival|regression) ;;
    *) err "Invalid --task: $TASK (expected classification|survival|regression)"; exit 1 ;;
esac

# ── Load configuration ─────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$CONF_FILE"

VERSION=$("$REPO_ROOT/scripts/build/getVersionNumber.sh")
# Derive the image name from the provision YAML's docker_image line (same source
# the build script uses), so non-odelia projects (e.g. DECADE = jefftud/decade)
# work without editing this script. Falls back to the historical default.
DOCKER_IMAGE=$(grep -m1 "docker_image:" "$REPO_ROOT/$PROJECT_FILE" 2>/dev/null \
    | sed 's/.*docker_image:[[:space:]]*//' \
    | sed "s/__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__/$VERSION/")
DOCKER_IMAGE="${DOCKER_IMAGE:-jefftud/odelia:$VERSION}"

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

# Set by submit_job() once the admin returns a real job id.
SUBMITTED_JOB_ID=""

# ── Evaluation configuration (#270) ────────────────────────────────────────
# Evaluation runs on THIS (server) machine and needs the eval site's data
# locally (mounted as /data). CLI flags override; conf may set EVAL_SITE /
# EVAL_DATA_DIR. Default eval site = first client site.
EVAL_SITE="${EVAL_SITE_ARG:-${EVAL_SITE:-${CLIENT_SITES[0]:-}}}"
EVAL_SCRATCH_DIR="${EVAL_SCRATCH_DIR:-$RESULTS_DIR/eval_scratch}"

# ── STAMP environment variables ──────────────────────────────────────────
# These are exported on each remote host BEFORE calling docker.sh.
# The STAMP master template's env var forwarding loop (for _var in STAMP_*)
# picks them up and passes them into the Docker container.

setup_stamp_env() {
    local site_name="$1"
    local model_name="$2"
    # Inside the container, --data_dir is mounted at /data/ (read-only).
    # STAMP env vars must reference the container-internal path, not the host path.
    cat <<EOF
export STAMP_CLINI_TABLE="/data/${site_name}/clini_table.csv"
export STAMP_FEATURE_DIR="/data/${site_name}/features"
export STAMP_PATIENT_LABEL="PATIENT"
export STAMP_TASK="${TASK}"
export STAMP_MODEL_NAME="${model_name}"
export STAMP_DIM_INPUT="1024"
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
    # Task-specific label columns — must match create_synthetic_stamp_dataset.py (#271).
    case "$TASK" in
        classification)
            echo 'export STAMP_GROUND_TRUTH_LABEL="Diagnosis"'
            echo 'export STAMP_NUM_CLASSES="3"'
            ;;
        survival)
            echo 'export STAMP_TIME_LABEL="Time"'
            echo 'export STAMP_STATUS_LABEL="Event"'
            echo 'export STAMP_NUM_CLASSES="1"'
            ;;
        regression)
            echo 'export STAMP_GROUND_TRUTH_LABEL="Target"'
            echo 'export STAMP_NUM_CLASSES="1"'
            ;;
    esac
}

# site_var / remote_exec / remote_copy / find_latest_prod /
# resolve_server_startup_dir are provided by deploy_common.sh (#276).

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

# ── Container name filter, scoped to THIS test's kit (#472) ───────────────
# stop_all() used to kill every container matching "stamp_swarm|nvflare" on every
# site, so a retry (or a concurrently running deploy test) tore down containers
# belonging to a run that was still training — clients died ~1s after round 0 was
# dispatched, with no error in any log.
#
# Every built startup kit hardcodes its own container names as
#   stamp_swarm_{client,server}_<name>_<git-short-hash>
# so the trailing hash identifies the kit this test drives. Scope the filter to it.
# If it cannot be derived we fall back to the old broad pattern (and say so) —
# a silently no-op cleanup would leave containers behind and break later runs.
KIT_CONTAINER_SUFFIX=""
KIT_SUFFIX_RESOLVED=false

resolve_kit_container_suffix() {
    [[ "$KIT_SUFFIX_RESOLVED" == true ]] && return 0
    KIT_SUFFIX_RESOLVED=true

    local prod_dir docker_sh
    prod_dir=$(find_latest_prod 2>/dev/null || true)
    [[ -n "$prod_dir" && -d "$prod_dir" ]] || return 0

    docker_sh=$(find "$prod_dir" -mindepth 3 -maxdepth 3 -path '*/startup/docker.sh' 2>/dev/null | head -1)
    [[ -n "$docker_sh" ]] || return 0

    KIT_CONTAINER_SUFFIX=$(grep -hoE '^CONTAINER_NAME=stamp_swarm_.*_[0-9a-f]{7,40}$' "$docker_sh" 2>/dev/null \
        | head -1 | sed -E 's/.*_([0-9a-f]{7,40})$/\1/')

    if [[ -n "$KIT_CONTAINER_SUFFIX" ]]; then
        info "Container cleanup scoped to this kit: *_${KIT_CONTAINER_SUFFIX} (#472)"
    else
        warn "Could not derive this kit's container suffix — cleanup falls back to the"
        warn "  broad 'stamp_swarm|nvflare' filter, which can disturb a concurrent run (#472)."
    fi
}

# Extended-regex matching only this test's containers (or the broad legacy filter).
container_filter() {
    resolve_kit_container_suffix
    if [[ -n "$KIT_CONTAINER_SUFFIX" ]]; then
        printf 'stamp_swarm_.*_%s$' "$KIT_CONTAINER_SUFFIX"
    else
        printf 'stamp_swarm|nvflare'
    fi
}

# ── Stop all containers ───────────────────────────────────────────────────
stop_all() {
    local filter
    filter=$(container_filter)
    info "Stopping STAMP/NVFlare containers (filter: $filter)..."

    # Stop local containers on Cosmos (server + admin)
    local local_containers
    local_containers=$(docker ps --format '{{.Names}}' | grep -E "$filter" || true)
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
            "docker ps --format '{{.Names}}' | grep -E '$filter' | xargs -r docker kill 2>/dev/null; \
             docker ps -a --format '{{.Names}}' | grep -E '$filter' | xargs -r docker rm -f 2>/dev/null" \
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

    if docker ps --format '{{.Names}}' | grep -qE "$(container_filter)"; then
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
    local model_name="$1"
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
        stamp_exports=$(setup_stamp_env "$site_name" "$model_name")

        # Pin the client to the image this test actually built and pre-pulled.
        # Client kits ship an image.conf pinning the release channel (:current), and
        # docker.sh's precedence is --image > MEDISWARM_IMAGE > image.conf > built-in
        # default. Without --image, image.conf won and every client ran :current --
        # i.e. the deploy test pre-pulled the new image and then trained with the old
        # one, so a "passing" deploy test said nothing about the build under test.
        # (Server/admin kits do not source image.conf, so they were already correct.)
        remote_exec "$site" \
            "$stamp_exports && \
             cd '$deploy_dir/$site_name/startup' && \
             ./docker.sh --no_pull --image '$DOCKER_IMAGE' --data_dir '$datadir' --scratch_dir '$scratchdir' --GPU '$gpu' --start_client"

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

    # Capture the admin session: `expect` exits 0 even when the admin failed to
    # reach the server, so its exit status alone must never be trusted.
    local submit_log="$RESULTS_DIR/submit_${job_name}.log"
    cd "$admin_startup"
    local rc=0
    expect -f "$expect_script" > "$submit_log" 2>&1 || rc=$?
    cd "$REPO_ROOT"
    rm -f "$expect_script"

    # `expect` exits 0 even when the admin never reached the server (rc=$rc is
    # not trustworthy); only a real job id proves the submission landed.
    SUBMITTED_JOB_ID=$(assert_job_submitted "$submit_log" "$job_name") || exit 1
    ok "Job submitted: $job_name (id: $SUBMITTED_JOB_ID)"
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
        if ! docker ps --format '{{.Names}}' | grep -qE "$(container_filter)"; then
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

# ── Collect global checkpoints from client machines (#270) ─────────────────
collect_checkpoints() {
    local model_name="$1"

    step "Collecting global checkpoints for $model_name" >&2

    # Find the job ID of the run that finished, from the server log.
    resolve_server_startup_dir
    local server_log="${_server_startup_dir}/nohup.out"
    local job_id=""
    if [[ -f "$server_log" ]]; then
        job_id=$(grep 'Server runner finished\.' "$server_log" | tail -1 \
            | grep -oP 'run=\K[0-9a-f-]+' || true)
    fi
    if [[ -n "$job_id" ]]; then
        info "  Job ID from server log: $job_id" >&2
    else
        warn "  Could not determine job ID — falling back to glob" >&2
    fi

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
        local remote_base="$deploy_dir/$site_name"

        local search_cmd
        if [[ -n "$job_id" ]]; then
            search_cmd="ls -1 '$remote_base/$job_id/app_${site_name}/FL_global_model.pt' '$remote_base/$job_id/app_${site_name}/best_FL_global_model.pt' 2>/dev/null || true"
        else
            search_cmd="find '$remote_base' -maxdepth 3 \( -name 'FL_global_model.pt' -o -name 'best_FL_global_model.pt' \) 2>/dev/null || true"
        fi

        local remote_files
        remote_files=$(sshpass -p "$pass" ssh $SSH_OPTS "$user@$host" "$search_cmd" 2>/dev/null || true)
        if [[ -z "$remote_files" ]]; then
            info "  No checkpoints found on $site_name" >&2
            continue
        fi

        local local_app_dir="$staging_dir/app_${site_name}"
        mkdir -p "$local_app_dir"
        while IFS= read -r remote_file; do
            [[ -z "$remote_file" ]] && continue
            local basename_file
            basename_file=$(basename "$remote_file")
            if sshpass -p "$pass" scp $SSH_OPTS "$user@$host:$remote_file" "$local_app_dir/$basename_file" 2>/dev/null; then
                ok "  Collected $basename_file from $site_name" >&2
                found_any=true
            else
                warn "  Failed to SCP $remote_file from $site_name" >&2
            fi
        done <<< "$remote_files"
    done

    if [[ "$found_any" == true ]]; then
        echo "$staging_dir"
        return 0
    fi
    err "No checkpoints collected for $model_name" >&2
    return 1
}

# ── Evaluate the global model with stamp_predict.py (#270) ─────────────────
# Runs on THIS machine; needs the eval site's data staged locally. Echoes the
# eval status (pass|fail|skipped). Non-fatal: a skip/fail never fails the run.
evaluate_model() {
    local model_name="$1"

    step "Evaluating $model_name" >&2

    if [[ -z "$EVAL_SITE" ]]; then
        warn "  No eval site resolved — skipping evaluation" >&2
        echo "skipped"; return 0
    fi

    local eval_site_name eval_data_dir
    eval_site_name=$(site_var "$EVAL_SITE" SITE_NAME 2>/dev/null || echo "$EVAL_SITE")
    eval_data_dir="${EVAL_DATA_DIR_ARG:-$(site_var "$EVAL_SITE" DATADIR 2>/dev/null || true)}"

    if [[ -z "$eval_data_dir" || ! -d "$eval_data_dir" ]]; then
        warn "  Eval data not available on this machine ('$eval_data_dir') — skipping evaluation" >&2
        warn "  (evaluation runs here; stage $eval_site_name's data locally or pass --eval-data-dir)" >&2
        echo "skipped"; return 0
    fi

    local checkpoint_dir
    if ! checkpoint_dir=$(collect_checkpoints "$model_name"); then
        err "  Cannot evaluate $model_name — no checkpoints collected" >&2
        echo "fail"; return 0
    fi

    mkdir -p "$EVAL_SCRATCH_DIR"
    local eval_output_dir="$RESULTS_DIR/${model_name}_evaluation"
    mkdir -p "$eval_output_dir"

    # STAMP_* env for the eval site (same layout: /data/<site>/...).
    local envfile="$eval_output_dir/stamp_env.list"
    setup_stamp_env "$eval_site_name" "$model_name" | sed 's/^export //; s/"//g' > "$envfile"

    # Local-only baseline (#275): train on the eval site's data and evaluate that
    # checkpoint on the SAME split as the federated model. Best-effort — a failure
    # here just omits the comparison, it never fails the deploy test.
    local baseline_args=()
    if [[ "$COMPARE_LOCAL" == true ]]; then
        local baseline_ckpt="$EVAL_SCRATCH_DIR/local_baseline_${model_name}.ckpt"
        rm -f "$baseline_ckpt"
        info "  Training local-only baseline for $model_name on $eval_site_name data" >&2
        docker run --rm \
            ${EVAL_GPU:+--gpus="$EVAL_GPU"} \
            --net=host --ipc=host \
            -v "$eval_data_dir:/data/:ro" \
            -v "$EVAL_SCRATCH_DIR:/scratch/" \
            --env-file "$envfile" \
            --env SITE_NAME="$eval_site_name" \
            --env SCRATCH_DIR=/scratch \
            --env TRAINING_MODE=local_training \
            --env MEDISWARM_VERSION="$VERSION" \
            -w /MediSwarm/application/jobs/STAMP_classification/app/custom \
            "$DOCKER_IMAGE" \
            python3 main.py \
            > "$eval_output_dir/local_training_stdout.log" 2>&1 \
            || warn "  Local baseline training failed (see local_training_stdout.log) — comparison omitted" >&2
        # Local training writes best_model.ckpt under /scratch/runs/<site>/STAMP_*/.
        local produced
        produced=$(find "$EVAL_SCRATCH_DIR/runs/$eval_site_name" -name best_model.ckpt 2>/dev/null | sort | tail -1)
        if [[ -n "$produced" && -f "$produced" ]]; then
            cp "$produced" "$baseline_ckpt"
            baseline_args=(--baseline "/scratch/local_baseline_${model_name}.ckpt")
            ok "  Local baseline checkpoint ready for comparison" >&2
        else
            warn "  No local baseline checkpoint produced — comparison omitted" >&2
        fi
    fi

    info "  Running stamp_predict.py in $DOCKER_IMAGE on $eval_site_name data (CPU unless EVAL_GPU set)" >&2
    local rc=0
    docker run --rm \
        ${EVAL_GPU:+--gpus="$EVAL_GPU"} \
        --net=host --ipc=host \
        -v "$eval_data_dir:/data/:ro" \
        -v "$EVAL_SCRATCH_DIR:/scratch/" \
        -v "$checkpoint_dir:/workspace/:ro" \
        -v "$eval_output_dir:/output/" \
        --env-file "$envfile" \
        --env SITE_NAME="$eval_site_name" \
        --env SCRATCH_DIR=/scratch \
        --env MEDISWARM_VERSION="$VERSION" \
        "$DOCKER_IMAGE" \
        python3 /MediSwarm/application/jobs/STAMP_classification/app/custom/stamp_predict.py \
            --workspace /workspace --output-dir /output "${baseline_args[@]}" \
        > "$eval_output_dir/stamp_predict_stdout.log" 2>&1 || rc=$?

    EVAL_RESULTS_FILE="$eval_output_dir/stamp_eval_results.json"
    if [[ $rc -eq 0 && -f "$EVAL_RESULTS_FILE" ]]; then
        ok "  Evaluation completed for $model_name" >&2
        echo "pass"; return 0
    fi
    err "  Evaluation failed for $model_name (exit $rc); see $eval_output_dir/stamp_predict_stdout.log" >&2
    echo "fail"; return 0
}

# ── Record result ─────────────────────────────────────────────────────────
record_result() {
    local model_name="$1"
    local job_name="$2"
    local train_status="$3"       # pass or fail
    local duration_seconds="$4"
    local eval_status="${5:-skipped}"   # pass, fail, or skipped

    local result_file="$RESULTS_DIR/deploy_test_stamp_${model_name}.json"
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Embed the eval metrics JSON if present, else null.
    local eval_metrics="null"
    local eval_results_file="$RESULTS_DIR/${model_name}_evaluation/stamp_eval_results.json"
    if [[ "$eval_status" != "skipped" && -f "$eval_results_file" ]]; then
        eval_metrics=$(cat "$eval_results_file")
    fi

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
    "client_sites": $(printf '%s\n' "${CLIENT_SITES[@]}" | jq -R . | jq -s .),
    "evaluation_metrics": $eval_metrics
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
    start_clients "$model_name"

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

    # 7. Evaluate the trained global model (#270) — before stop_all, so the
    #    client checkpoints are still collectable from the remote workspaces.
    local eval_status="skipped"
    if [[ "$EVALUATE" == true && "$train_status" == "pass" ]]; then
        eval_status=$(evaluate_model "$model_name")
    fi

    # 8. Stop all containers
    stop_all

    # 9. Record result
    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))

    record_result "$model_name" "$job_name" "$train_status" "$duration" "$eval_status"

    echo ""
    if [[ "$train_status" != "pass" ]]; then
        err "FAILED: $model_name in ${duration}s"
        FAILED_MODELS+=("$model_name")
    elif [[ "$EVALUATE" == true && "$eval_status" != "pass" ]]; then
        # Evaluation was explicitly requested but did not produce a result. Never
        # report this as a plain PASS — a skipped evaluation previously looked
        # identical to a verified one (#476).
        if [[ "$STRICT" == true ]]; then
            err "FAILED: $model_name in ${duration}s — evaluation requested but ${eval_status} (--strict)"
            FAILED_MODELS+=("$model_name")
        else
            warn "PASSED (training only): $model_name in ${duration}s — evaluation ${eval_status}, NOT verified"
            warn "  Re-run with --strict to treat this as a failure."
            TRAINING_ONLY_MODELS+=("$model_name")
            PASSED_MODELS+=("$model_name")
        fi
    else
        ok "PASSED: $model_name in ${duration}s (evaluation: $eval_status)"
        PASSED_MODELS+=("$model_name")
    fi

    # Always return 0 so the caller keeps testing the remaining models; the
    # script's exit status is decided from FAILED_MODELS at the end.
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

# Fix DNS: this (admin/server) host first — the admin container uses --net=host
ensure_local_dns

# Fix DNS on remote machines
fix_remote_dns

# Pre-pull Docker image on all remote machines
pre_pull_images

# Run the STAMP deploy test
step "Selected STAMP models: ${TEST_MODELS[*]}"
for model_name in "${TEST_MODELS[@]}"; do
    run_single_model "$JOB_NAME" "$model_name"
done

# ── Summary + exit status ────────────────────────────────────────────────
# Previously the script always exited 0, so a failing model (e.g. barspoon,
# which the shipped STAMP cannot instantiate) still reported success to CI.
step "STAMP deploy test summary"
for m in "${PASSED_MODELS[@]:-}"; do
    [[ -n "$m" ]] || continue
    # Mark models whose requested evaluation never verified, so a training-only
    # result is never mistaken for a fully verified pass (#476).
    if [[ " ${TRAINING_ONLY_MODELS[*]:-} " == *" $m "* ]]; then
        warn "  PASS  $m  (training only — evaluation NOT verified)"
    else
        ok "  PASS  $m"
    fi
done
for m in "${FAILED_MODELS[@]:-}"; do [[ -n "$m" ]] && err "  FAIL  $m"; done
info "  ${#PASSED_MODELS[@]} passed, ${#FAILED_MODELS[@]} failed (results: $RESULTS_DIR)"

if [[ ${#TRAINING_ONLY_MODELS[@]} -gt 0 ]]; then
    warn "  Evaluation was requested but not verified for: ${TRAINING_ONLY_MODELS[*]}"
    warn "  These passed TRAINING only. Re-run with --strict to fail on this."
fi

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    err "Deploy test FAILED for: ${FAILED_MODELS[*]}"
    exit 1
fi
ok "All ${#PASSED_MODELS[@]} model(s) passed"
