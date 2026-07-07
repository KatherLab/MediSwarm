#!/usr/bin/env bash
# End-to-end Duke IID validation on DL servers:
#   dl3 = local NVFlare server/admin
#   dl0 = node_A client + prediction evaluation host
#   dl2 = node_B client

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

CONF_FILE="$REPO_ROOT/deploy_sites_duke_iid_2client.local.conf"
PROJECT_FILE="$REPO_ROOT/application/provision/project_duke_iid_2client_dl.yml"
JOB_NAME="challenge_1DivideAndConquer"
MODEL_NAME="1DivideAndConquer"
RUN_ID="duke_iid_2client_validation_$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$REPO_ROOT/workspace/duke_iid_2client_validation/$RUN_ID"
TIMEOUT_MINUTES=10080
SMOKE_ROUNDS=2
FULL_ROUNDS=20
CONTINUE_ROUNDS=2
DROP_ROUNDS=3
ABORT_ROUNDS=3
RESUME_ROUNDS=1
PHASES_CSV="all"
SKIP_BUILD=false
SKIP_PUSH=false
SKIP_EVAL=false
USE_DOCKER_CACHE=false
PROJECT_FILE_FROM_CLI=false

ALL_MODELS=(
    "ODELIA_ternary_classification:MST"
    "challenge_1DivideAndConquer:1DivideAndConquer"
    "challenge_2BCN_AIM:2BCN_AIM"
    "challenge_3agaldran:3agaldran"
    "challenge_4abmil:4LME_ABMIL"
    "challenge_5pimed:5Pimed"
)

usage() {
    cat <<'EOF'
Usage:
  scripts/deploy/run_duke_iid_2client_validation.sh \
    --conf deploy_sites_duke_iid_2client.local.conf \
    [options]

Run from dl3, which acts as the NVFlare server/admin host. The wrapper builds
startup kits for dl3.tud.de + node_A/node_B, deploys clients to dl0/dl2, patches
admin-submitted job copies for the desired client counts, and stores artifacts
under workspace/duke_iid_2client_validation/<run_id>/.

Options:
  --conf FILE              Local deploy config (default: deploy_sites_duke_iid_2client.local.conf)
  --project FILE           Provision project (default: application/provision/project_duke_iid_2client_dl.yml)
  --job JOB                Robustness job directory (default: challenge_1DivideAndConquer)
  --model MODEL            Robustness model name (default: 1DivideAndConquer)
  --results-dir DIR        Output directory
  --timeout MIN            Per-phase timeout in minutes (default: 10080)
  --smoke-rounds N         Rounds for all-model smoke jobs (default: 2)
  --full-rounds N          Rounds for the full robustness model run (default: 20)
  --continue-rounds N      Rounds for fresh/continue phases (default: 2)
  --drop-rounds N          Rounds for one-client-drop phase (default: 3)
  --abort-rounds N         Fresh rounds before abort-recovery crash (default: 3)
  --resume-rounds N        Strict continue rounds after crash (default: 1)
  --phases CSV             all or comma list:
                           all_model_smoke,one_dc_full,negative_continue,
                           fresh_then_continue,single_client_drop,abort_recovery
  --skip-build             Reuse existing Docker image/startup kits
  --skip-push              Do not docker push the image
  --skip-eval              Skip dl0 prediction evaluation
  --use-docker-cache       Pass --use-docker-cache to the build script
  -h, --help               Show this help

Environment:
  DL_SWARM_PASS            Recommended password env var used by the example config.
  EVAL_DEVICE              Passed to the dl0 evaluator through predict.py defaults.
  SSH_CONNECT_TIMEOUT      Seconds to wait for SSH connection setup (default: 10).
  START_CLIENT_ATTEMPTS    Attempts per client startup before failing (default: 12).
  START_CLIENT_RETRY_SLEEP Seconds between client startup attempts (default: 30).
  VERSION_OVERRIDE         Reuse an already-built image/startup-kit version.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conf)             CONF_FILE="$2"; shift ;;
        --project)          PROJECT_FILE="$2"; PROJECT_FILE_FROM_CLI=true; shift ;;
        --job)              JOB_NAME="$2"; shift ;;
        --model)            MODEL_NAME="$2"; shift ;;
        --results-dir)      RESULTS_DIR="$2"; shift ;;
        --timeout)          TIMEOUT_MINUTES="$2"; shift ;;
        --smoke-rounds)     SMOKE_ROUNDS="$2"; shift ;;
        --full-rounds)      FULL_ROUNDS="$2"; shift ;;
        --continue-rounds)  CONTINUE_ROUNDS="$2"; shift ;;
        --drop-rounds)      DROP_ROUNDS="$2"; shift ;;
        --abort-rounds)     ABORT_ROUNDS="$2"; shift ;;
        --resume-rounds)    RESUME_ROUNDS="$2"; shift ;;
        --phases)           PHASES_CSV="$2"; shift ;;
        --skip-build)       SKIP_BUILD=true ;;
        --skip-push)        SKIP_PUSH=true ;;
        --skip-eval)        SKIP_EVAL=true ;;
        --use-docker-cache) USE_DOCKER_CACHE=true ;;
        -h|--help)          usage; exit 0 ;;
        *)                  err "Unknown argument: $1"; usage; exit 2 ;;
    esac
    shift
done

resolve_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$REPO_ROOT" "$path"
    fi
}

CONF_FILE="$(resolve_path "$CONF_FILE")"
PROJECT_FILE="$(resolve_path "$PROJECT_FILE")"
RESULTS_DIR="$(resolve_path "$RESULTS_DIR")"
CLI_PROJECT_FILE="$PROJECT_FILE"

if [[ ! -f "$CONF_FILE" ]]; then
    err "Config file not found: $CONF_FILE"
    err "Copy deploy_sites_duke_iid_2client.local.conf.example and set DL_SWARM_PASS."
    exit 2
fi
if [[ ! -f "$PROJECT_FILE" ]]; then
    err "Project file not found: $PROJECT_FILE"
    exit 2
fi
if [[ "$PROJECT_FILE" != "$REPO_ROOT/"* ]]; then
    err "--project must point to a file under the repository root: $PROJECT_FILE"
    exit 2
fi

# shellcheck source=/dev/null
source "$CONF_FILE"

if [[ "$PROJECT_FILE_FROM_CLI" == true ]]; then
    PROJECT_FILE="$CLI_PROJECT_FILE"
fi
PROJECT_FILE="$(resolve_path "$PROJECT_FILE")"
if [[ ! -f "$PROJECT_FILE" ]]; then
    err "Project file not found after loading config: $PROJECT_FILE"
    exit 2
fi
if [[ "$PROJECT_FILE" != "$REPO_ROOT/"* ]]; then
    err "PROJECT_FILE must point to a file under the repository root: $PROJECT_FILE"
    exit 2
fi

PROJECT_FILE_FOR_BUILD="${PROJECT_FILE#$REPO_ROOT/}"
VERSION="${VERSION_OVERRIDE:-$("$REPO_ROOT/scripts/build/getVersionNumber.sh")}"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
DOCKER_IMAGE="$(grep 'docker_image:' "$PROJECT_FILE" \
    | sed 's/.*docker_image:[[:space:]]*//' \
    | sed "s#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#$VERSION#")"
PROJECT_NAME="$(grep '^name: ' "$PROJECT_FILE" \
    | sed 's/^name: //' \
    | sed "s#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#$VERSION#")"
WORKSPACE_DIR="$REPO_ROOT/workspace/$PROJECT_NAME"
# Local (dl3) deploy dir for the server/admin kits. Default under $HOME so it works
# for whichever user runs this (not just 'swarm'); override with COSMOS_DEPLOY_DIR.
DEPLOY_BASE="${COSMOS_DEPLOY_DIR:-$HOME/deploy_test_duke_iid_2client}"
SERVER_NAME="${SERVER_NAME:-dl3.tud.de}"
ADMIN_USER="${ADMIN_USER:-jiefu.zhu@tu-dresden.de}"
CLIENT_COUNT="${#CLIENT_SITES[@]}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
SSH_SERVER_ALIVE_INTERVAL="${SSH_SERVER_ALIVE_INTERVAL:-15}"
SSH_SERVER_ALIVE_COUNT_MAX="${SSH_SERVER_ALIVE_COUNT_MAX:-2}"
START_CLIENT_ATTEMPTS="${START_CLIENT_ATTEMPTS:-12}"
START_CLIENT_RETRY_SLEEP="${START_CLIENT_RETRY_SLEEP:-30}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o ServerAliveInterval=$SSH_SERVER_ALIVE_INTERVAL -o ServerAliveCountMax=$SSH_SERVER_ALIVE_COUNT_MAX"
NVFLARE_CONTAINER_RE='odelia_swarm|nvflare|^swarm-'

mkdir -p "$RESULTS_DIR"

declare -A PASS_CACHE=()
declare -A MIRROR_MD5=()

site_var() {
    local site="$1" var="$2" name
    name="${site}_${var}"
    printf '%s\n' "${!name-}"
}

ensure_site_pass() {
    local site="$1" pass env_name host user
    if [[ -n "${PASS_CACHE[$site]+set}" ]]; then
        return 0
    fi

    pass="$(site_var "$site" PASS)"
    env_name="$(site_var "$site" PASS_ENV)"
    if [[ -z "$pass" && -n "$env_name" && -n "${!env_name-}" ]]; then
        pass="${!env_name}"
    fi

    if [[ -z "$pass" ]]; then
        host="$(site_var "$site" HOST)"
        user="$(site_var "$site" USER)"
        read -r -s -p "Password for $user@$host ($site, hidden; leave empty for SSH key/no sudo): " pass
        echo >&2
    fi

    PASS_CACHE[$site]="$pass"
}

site_scratch() {
    local site="$1" root
    root="$(site_var "$site" SCRATCHDIR)"
    printf '%s/%s\n' "${root%/}" "$RUN_ID"
}

remote_exec() {
    local site="$1"; shift
    local host user pass
    host="$(site_var "$site" HOST)"
    user="$(site_var "$site" USER)"
    ensure_site_pass "$site"
    pass="${PASS_CACHE[$site]}"

    if [[ -n "$pass" ]]; then
        SSHPASS="$pass" sshpass -e ssh $SSH_OPTS "$user@$host" "$@"
    else
        ssh $SSH_OPTS "$user@$host" "$@"
    fi
}

remote_sudo_exec() {
    local site="$1"; shift
    local cmd="$*" host user pass remote_cmd
    host="$(site_var "$site" HOST)"
    user="$(site_var "$site" USER)"
    ensure_site_pass "$site"
    pass="${PASS_CACHE[$site]}"
    remote_cmd="sudo -S bash -lc $(printf '%q' "$cmd")"

    if [[ -n "$pass" ]]; then
        printf '%s\n' "$pass" | SSHPASS="$pass" sshpass -e ssh $SSH_OPTS "$user@$host" "$remote_cmd"
    else
        ssh $SSH_OPTS "$user@$host" "sudo -n bash -lc $(printf '%q' "$cmd")"
    fi
}

local_sudo_exec() {
    local cmd="$*"
    if sudo -n true >/dev/null 2>&1; then
        sudo bash -lc "$cmd"
    elif [[ -n "${DL_SWARM_PASS:-}" ]]; then
        printf '%s\n' "$DL_SWARM_PASS" | sudo -S bash -lc "$cmd"
    else
        err "sudo is required locally to update /etc/hosts; set DL_SWARM_PASS or configure passwordless sudo"
        return 1
    fi
}

remote_copy() {
    local site="$1" src="$2" dst="$3"
    local host user pass
    host="$(site_var "$site" HOST)"
    user="$(site_var "$site" USER)"
    ensure_site_pass "$site"
    pass="${PASS_CACHE[$site]}"

    if [[ -n "$pass" ]]; then
        SSHPASS="$pass" sshpass -e scp $SSH_OPTS "$src" "$user@$host:$dst"
    else
        scp $SSH_OPTS "$src" "$user@$host:$dst"
    fi
}

remote_download() {
    local site="$1" src="$2" dst="$3"
    local host user pass
    host="$(site_var "$site" HOST)"
    user="$(site_var "$site" USER)"
    ensure_site_pass "$site"
    pass="${PASS_CACHE[$site]}"

    if [[ -n "$pass" ]]; then
        SSHPASS="$pass" sshpass -e scp $SSH_OPTS "$user@$host:$src" "$dst"
    else
        scp $SSH_OPTS "$user@$host:$src" "$dst"
    fi
}

phase_enabled() {
    local phase="$1"
    [[ "$PHASES_CSV" == "all" ]] && return 0
    [[ ",$PHASES_CSV," == *",$phase,"* ]]
}

safe_phase_name() {
    local value="$1"
    value="${value//[^A-Za-z0-9_.-]/_}"
    printf '%s\n' "$value"
}

find_latest_prod() {
    if [[ ! -d "$WORKSPACE_DIR" ]]; then
        err "Workspace not found: $WORKSPACE_DIR"
        err "Build startup kits first or run without --skip-build."
        return 1
    fi
    ls -d "$WORKSPACE_DIR"/prod_* 2>/dev/null | sort -V | tail -n 1
}

clean_local_deploy_dir() {
    local dir_name="$1" target
    target="$DEPLOY_BASE/$dir_name"
    [[ -e "$target" ]] || return 0
    rm -rf "$target" 2>/dev/null \
        || sudo rm -rf "$target" 2>/dev/null \
        || docker run --rm -v "$DEPLOY_BASE:/cleanup" alpine rm -rf "/cleanup/$dir_name" 2>/dev/null \
        || warn "Could not fully clean $target"
}

stop_all() {
    docker ps -a --format '{{.Names}}' \
        | grep -E "$NVFLARE_CONTAINER_RE" \
        | xargs -r docker rm -f >/dev/null 2>&1 || true

    local -A visited=()
    local site host
    for site in "${CLIENT_SITES[@]}"; do
        host="$(site_var "$site" HOST)"
        [[ -n "${visited[$host]+set}" ]] && continue
        visited[$host]=1
        remote_exec "$site" \
            "docker ps -a --format '{{.Names}}' | grep -E 'odelia_swarm|nvflare|^swarm-' | xargs -r docker rm -f >/dev/null 2>&1" \
            >/dev/null 2>&1 || true
    done
}

server_client_ip() {
    if [[ -n "${SERVER_HOST_IP:-}" ]]; then
        printf '%s\n' "$SERVER_HOST_IP"
        return 0
    fi
    if [[ -n "${COSMOS_HOST_IP:-}" ]]; then
        printf '%s\n' "$COSMOS_HOST_IP"
        return 0
    fi
    tailscale ip -4 2>/dev/null && return 0

    local first_host
    first_host="$(site_var "${CLIENT_SITES[0]}" HOST)"
    ip route get "$first_host" 2>/dev/null \
        | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}'
}

preflight() {
    step "Preflight"
    if [[ "$CLIENT_COUNT" -ne 2 ]]; then
        err "This validation expects exactly two clients; CLIENT_SITES has $CLIENT_COUNT."
        return 1
    fi
    for cmd in docker expect jq sshpass unzip; do
        command -v "$cmd" >/dev/null 2>&1 || { err "Missing required command: $cmd"; return 1; }
    done
    docker info >/dev/null
    ok "Local Docker is available"

    if [[ "$SKIP_BUILD" != true ]]; then
        if ! git -C "$REPO_ROOT" diff --quiet || ! git -C "$REPO_ROOT" diff --staged --quiet; then
            err "Tracked local changes exist. The build script archives HEAD and refuses dirty tracked state."
            err "Commit/stash tracked edits, or rerun with --skip-build after building startup kits separately."
            return 1
        fi
    fi

    local server_ip
    server_ip="$(server_client_ip)"
    if [[ -z "$server_ip" ]]; then
        err "Cannot determine the dl3 client-reachable IP. Set SERVER_HOST_IP in $CONF_FILE."
        return 1
    fi
    ok "$SERVER_NAME client-reachable IP: $server_ip"

    local site site_name datadir scratch
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        datadir="$(site_var "$site" DATADIR)"
        scratch="$(site_scratch "$site")"
        info "Checking $site_name data/GPU/Docker"
        remote_exec "$site" "
            command -v docker >/dev/null &&
            command -v nvidia-smi >/dev/null &&
            nvidia-smi >/dev/null &&
            test -d '$datadir/$site_name/data_unilateral' &&
            test -d '$datadir/$site_name/metadata_unilateral' &&
            test -n \"\$(find '$datadir/$site_name/data_unilateral' -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)\"
        " || return 1
        remote_exec "$site" "rm -rf '$scratch' && mkdir -p '$scratch' && chmod 777 '$scratch'" \
            || remote_sudo_exec "$site" "rm -rf '$scratch' && mkdir -p '$scratch' && chmod 777 '$scratch'" \
            || return 1
        ok "$site_name preflight passed"
    done

    remote_exec "DL0" "
        test -d '${DL0_DUKE_DATA:-/mnt/dlhd0/DUKE_iid}/test/data_unilateral' &&
        test -d '${DL0_DUKE_DATA:-/mnt/dlhd0/DUKE_iid}/test/metadata_unilateral' &&
        test -d '${DL0_ODELIA_DATA:-/mnt/dlhd0/medswarmdata}'
    " || return 1
    ok "dl0 evaluation data paths are present"
}

build_and_push() {
    if [[ "$SKIP_BUILD" == true ]]; then
        warn "Skipping build; reusing existing startup kits/image for $DOCKER_IMAGE"
        return 0
    fi

    step "Build Docker image and startup kits"
    local build_args=("-p" "$PROJECT_FILE_FOR_BUILD")
    if [[ "$USE_DOCKER_CACHE" == true ]]; then
        build_args+=("--use-docker-cache")
    fi
    (cd "$REPO_ROOT" && bash scripts/build/buildDockerImageAndStartupKits.sh "${build_args[@]}") || return 1

    if [[ "$SKIP_PUSH" == true ]]; then
        warn "Skipping docker push for $DOCKER_IMAGE"
    else
        step "Push Docker image"
        docker push "$DOCKER_IMAGE" || return 1
        ok "Pushed $DOCKER_IMAGE"
    fi
}

deploy_kits() {
    local prod_dir
    prod_dir="$(find_latest_prod)" || return 1
    step "Deploy startup kits"
    info "Source: $prod_dir"

    local site site_name deploy_dir zip_file
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        deploy_dir="$(site_var "$site" DEPLOY_DIR)"
        zip_file="$prod_dir/${site_name}_${VERSION}.zip"
        [[ -f "$zip_file" ]] || zip_file="$(ls "$prod_dir"/${site_name}*.zip 2>/dev/null | head -1 || true)"
        [[ -n "$zip_file" && -f "$zip_file" ]] || { err "Missing startup kit for $site_name"; return 1; }

        remote_exec "$site" "mkdir -p '$deploy_dir'" || return 1
        remote_copy "$site" "$zip_file" "$deploy_dir/" || return 1
        remote_exec "$site" "cd '$deploy_dir' && rm -rf '$site_name' && unzip -qo '$(basename "$zip_file")'" || return 1
        ok "Deployed $site_name"
    done

    local server_zip admin_zip
    server_zip="$prod_dir/${SERVER_NAME}_${VERSION}.zip"
    [[ -f "$server_zip" ]] || server_zip="$(ls "$prod_dir"/${SERVER_NAME}*.zip 2>/dev/null | head -1 || true)"
    [[ -n "$server_zip" && -f "$server_zip" ]] || { err "Missing server startup kit for $SERVER_NAME"; return 1; }

    admin_zip="$prod_dir/${ADMIN_USER}_${VERSION}.zip"
    [[ -f "$admin_zip" ]] || admin_zip="$(ls "$prod_dir"/${ADMIN_USER}*.zip 2>/dev/null | head -1 || true)"
    [[ -n "$admin_zip" && -f "$admin_zip" ]] || { err "Missing admin startup kit for $ADMIN_USER"; return 1; }

    mkdir -p "$DEPLOY_BASE"
    cp "$server_zip" "$admin_zip" "$DEPLOY_BASE/"
    clean_local_deploy_dir "$SERVER_NAME"
    clean_local_deploy_dir "$ADMIN_USER"
    (cd "$DEPLOY_BASE" && unzip -qo "$(basename "$server_zip")" && unzip -qo "$(basename "$admin_zip")") || return 1
    ok "Deployed server/admin kits locally"
}

fix_remote_dns() {
    local server_ip current site
    server_ip="$(server_client_ip)"
    [[ -n "$server_ip" ]] || { err "Cannot determine server IP"; return 1; }

    step "Ensure server/admin host and clients resolve $SERVER_NAME to $server_ip"
    current="$(grep -E "\\b${SERVER_NAME}\\b" /etc/hosts 2>/dev/null | awk '{print $1}' | head -1 || true)"
    if [[ "$current" == "$server_ip" ]]; then
        ok "local host already maps $SERVER_NAME to $server_ip"
    elif [[ -n "$current" ]]; then
        local_sudo_exec "sed -i 's|^.*\\b${SERVER_NAME}\\b.*\$|${server_ip} ${SERVER_NAME}|' /etc/hosts" || return 1
        ok "local host maps $SERVER_NAME to $server_ip"
    else
        local_sudo_exec "echo '${server_ip} ${SERVER_NAME}' >> /etc/hosts" || return 1
        ok "local host maps $SERVER_NAME to $server_ip"
    fi

    for site in "${CLIENT_SITES[@]}"; do
        current="$(remote_exec "$site" "grep -E '\\b${SERVER_NAME}\\b' /etc/hosts 2>/dev/null | awk '{print \$1}' | head -1" 2>/dev/null || true)"
        if [[ "$current" == "$server_ip" ]]; then
            ok "$site already maps $SERVER_NAME to $server_ip"
            continue
        fi
        if [[ -n "$current" ]]; then
            remote_sudo_exec "$site" "sed -i 's|^.*\\b${SERVER_NAME}\\b.*\$|${server_ip} ${SERVER_NAME}|' /etc/hosts" || return 1
        else
            remote_sudo_exec "$site" "echo '${server_ip} ${SERVER_NAME}' >> /etc/hosts" || return 1
        fi
        ok "$site maps $SERVER_NAME to $server_ip"
    done
}

pre_pull_images() {
    step "Pull $DOCKER_IMAGE on clients"
    # Clients pull from the registry, so the image must be there. With --skip-build
    # (reusing a locally-built image) the push is skipped, which otherwise leaves the
    # clients failing with "manifest unknown". If the tag isn't on the registry but
    # exists locally, push it now so the pull below can succeed.
    if ! docker manifest inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
            warn "$DOCKER_IMAGE is not on the registry (e.g. --skip-push); pushing it so clients can pull"
            docker push "$DOCKER_IMAGE" || { err "Failed to push $DOCKER_IMAGE — run 'docker login' or drop --skip-push"; return 1; }
        else
            err "$DOCKER_IMAGE is neither on the registry nor built locally; build it or drop --skip-build"
            return 1
        fi
    fi
    local site gpu
    for site in "${CLIENT_SITES[@]}"; do
        gpu="$(site_var "$site" GPU)"
        remote_exec "$site" "docker pull '$DOCKER_IMAGE' && docker run --rm --gpus='$gpu' '$DOCKER_IMAGE' nvidia-smi -L >/dev/null" || return 1
        ok "$site image/GPU check passed"
    done
}

server_startup_dir() {
    printf '%s/%s/startup\n' "$DEPLOY_BASE" "$SERVER_NAME"
}

admin_startup_dir() {
    printf '%s/%s/startup\n' "$DEPLOY_BASE" "$ADMIN_USER"
}

server_log() {
    printf '%s/nohup.out\n' "$(server_startup_dir)"
}

start_server() {
    local startup
    startup="$(server_startup_dir)"
    [[ -d "$startup" ]] || { err "Missing server startup dir: $startup"; return 1; }
    (cd "$startup" && ./docker.sh --no_pull --start_server) || return 1
    sleep 15
}

client_model_running() {
    local site="$1" model_name="$2" site_name
    site_name="$(site_var "$site" SITE_NAME)"
    remote_exec "$site" "
        container=\$(docker ps --filter 'name=odelia_swarm_client_${site_name}_' --format '{{.Names}}' | head -1)
        test -n \"\$container\"
        docker inspect \"\$container\" --format '{{range .Config.Env}}{{println .}}{{end}}' |
          grep -Fx 'MODEL_NAME=$model_name' >/dev/null
    "
}

start_clients() {
    local model_name="$1" site site_name deploy_dir datadir scratch gpu attempt
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        deploy_dir="$(site_var "$site" DEPLOY_DIR)"
        datadir="$(site_var "$site" DATADIR)"
        scratch="$(site_scratch "$site")"
        gpu="$(site_var "$site" GPU)"
        for ((attempt = 1; attempt <= START_CLIENT_ATTEMPTS; attempt++)); do
            if remote_exec "$site" \
                "cd '$deploy_dir/$site_name/startup' && ./docker.sh --no_pull --data_dir '$datadir' --scratch_dir '$scratch' --GPU '$gpu' --model_name '$model_name' --start_client"; then
                ok "Started $site_name with MODEL_NAME=$model_name"
                break
            fi

            if client_model_running "$site" "$model_name" >/dev/null 2>&1; then
                ok "Started $site_name with MODEL_NAME=$model_name"
                break
            fi

            if [[ "$attempt" -eq "$START_CLIENT_ATTEMPTS" ]]; then
                err "Failed to start $site_name with MODEL_NAME=$model_name after $START_CLIENT_ATTEMPTS attempts"
                return 1
            fi

            warn "Starting $site_name failed (attempt $attempt/$START_CLIENT_ATTEMPTS); retrying in ${START_CLIENT_RETRY_SLEEP}s"
            sleep "$START_CLIENT_RETRY_SLEEP"
        done
    done
}

verify_client_model_env() {
    local model_name="$1" site
    for site in "${CLIENT_SITES[@]}"; do
        client_model_running "$site" "$model_name" || return 1
    done
}

wait_for_registration() {
    local max_wait=600 elapsed=0 log_file all_registered site site_name
    log_file="$(server_log)"
    while [[ $elapsed -lt $max_wait ]]; do
        all_registered=true
        for site in "${CLIENT_SITES[@]}"; do
            site_name="$(site_var "$site" SITE_NAME)"
            grep -q "New client ${site_name}@" "$log_file" 2>/dev/null || all_registered=false
        done
        if [[ "$all_registered" == true ]]; then
            ok "All clients registered"
            sleep 5
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    err "Timed out waiting for clients to register"
    return 1
}

prepare_admin_job() {
    local job="$1" warm_start="$2" rounds="$3" min_clients="$4" min_responses="$5" configure_min_clients="$6" broadcast_last_result="${7:-true}"
    local admin_startup args=()
    admin_startup="$(admin_startup_dir)"
    [[ -d "$admin_startup" ]] || { err "Missing admin startup dir: $admin_startup"; return 1; }

    args=(
        --job "$job"
        --warm-start "$warm_start"
        --num-rounds "$rounds"
        --min-clients "$min_clients"
        --min-responses "$min_responses"
        --broadcast-last-result "$broadcast_last_result"
    )
    if [[ -n "$configure_min_clients" ]]; then
        args+=(--configure-min-clients "$configure_min_clients")
    fi
    (cd "$admin_startup" && ./prepare_odelia_job.sh "${args[@]}") || return 1
}

submit_job() {
    local job="$1" warm_start="$2"
    local job_path="/fl_admin/local/mediswarm_jobs/${job}_${warm_start}"
    local admin_startup expect_script
    admin_startup="$(admin_startup_dir)"
    expect_script="$(mktemp /tmp/mediswarm_duke_2client_XXXXXX.exp)"
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
    (cd "$admin_startup" && expect -f "$expect_script") || {
        rm -f "$expect_script"
        return 1
    }
    rm -f "$expect_script"
}

log_start_line() {
    local log_file
    log_file="$(server_log)"
    if [[ -f "$log_file" ]]; then
        wc -l < "$log_file"
    else
        echo 0
    fi
}

phase_lines() {
    local start_line="$1"
    tail -n +"$((start_line + 1))" "$(server_log)" 2>/dev/null || true
}

save_phase_logs() {
    local phase="$1" phase_dir site site_name deploy_dir
    phase_dir="$RESULTS_DIR/$phase"
    mkdir -p "$phase_dir"
    cp "$(server_log)" "$phase_dir/server_nohup.out" 2>/dev/null || true
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        deploy_dir="$(site_var "$site" DEPLOY_DIR)"
        remote_download "$site" "$deploy_dir/$site_name/startup/nohup.out" "$phase_dir/${site_name}_nohup.out" \
            >/dev/null 2>&1 || true
    done
}

wait_for_success() {
    local phase="$1" start_line="$2" max_attempts=$((TIMEOUT_MINUTES * 2)) attempt=0 lines
    while [[ $attempt -lt $max_attempts ]]; do
        lines="$(phase_lines "$start_line")"
        if echo "$lines" | grep -q "WARM_START_REQUIRED_MISSING"; then
            err "$phase failed with WARM_START_REQUIRED_MISSING"
            return 1
        fi
        if echo "$lines" | grep -q "FATAL_SYSTEM_ERROR\\|ABORT_RUN"; then
            err "$phase hit fatal server error"
            echo "$lines" | grep -i "FATAL\\|ABORT\\|ERROR\\|EXCEPTION" | tail -20 >&2 || true
            return 1
        fi
        if echo "$lines" | grep -q "Server runner finished\\."; then
            ok "$phase completed"
            return 0
        fi
        sleep 30
        attempt=$((attempt + 1))
    done
    err "Timed out waiting for $phase"
    return 1
}

wait_for_negative_continue() {
    local phase="$1" start_line="$2" max_attempts=$((TIMEOUT_MINUTES * 2)) attempt=0 lines
    while [[ $attempt -lt $max_attempts ]]; do
        lines="$(phase_lines "$start_line")"
        if echo "$lines" | grep -q "WARM_START_REQUIRED_MISSING"; then
            if echo "$lines" | grep -Ei "WARM_START_REQUIRED_MISSING.*(prun|tolerat)|(prun|tolerat).*WARM_START_REQUIRED_MISSING" >/dev/null; then
                err "Warm-start missing error was pruned/tolerated"
                return 1
            fi
            if echo "$lines" | grep -q "Server runner finished\\."; then
                ok "Negative continue aborted with WARM_START_REQUIRED_MISSING"
                return 0
            fi
        fi
        if echo "$lines" | grep -q "Server runner finished\\."; then
            save_phase_logs "$phase"
            if echo "$lines" | grep -Ei "WARM_START_REQUIRED_MISSING.*(prun|tolerat)|(prun|tolerat).*WARM_START_REQUIRED_MISSING" >/dev/null; then
                err "Warm-start missing error was pruned/tolerated"
                return 1
            fi
            local all_clients_logged_error=true site site_name log_file
            for site in "${CLIENT_SITES[@]}"; do
                site_name="$(site_var "$site" SITE_NAME)"
                log_file="$RESULTS_DIR/$phase/${site_name}_nohup.out"
                grep -q "WARM_START_REQUIRED_MISSING" "$log_file" || all_clients_logged_error=false
            done
            if [[ "$all_clients_logged_error" == true ]]; then
                ok "Negative continue aborted after clients reported WARM_START_REQUIRED_MISSING"
                return 0
            fi
            err "Negative continue finished without WARM_START_REQUIRED_MISSING in all client logs"
            return 1
        fi
        sleep 30
        attempt=$((attempt + 1))
    done
    err "Timed out waiting for negative continue failure"
    return 1
}

clear_mirrors() {
    local site scratch
    for site in "${CLIENT_SITES[@]}"; do
        scratch="$(site_scratch "$site")"
        remote_exec "$site" "rm -f '$scratch/mediswarm_latest_global.pt'" >/dev/null 2>&1 || true
    done
}

assert_latest_globals() {
    local site site_name scratch
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        scratch="$(site_scratch "$site")"
        remote_exec "$site" "test -s '$scratch/mediswarm_latest_global.pt'" || {
            err "Missing latest global checkpoint for $site_name"
            return 1
        }
    done
}

mirror_md5() {
    local site="$1" scratch
    scratch="$(site_scratch "$site")"
    remote_exec "$site" "md5sum '$scratch/mediswarm_latest_global.pt' 2>/dev/null | awk '{print \$1}'" 2>/dev/null || true
}

record_mirror_hashes() {
    local phase="$1" site site_name h
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        h="$(mirror_md5 "$site")"
        MIRROR_MD5["$phase:$site"]="$h"
        info "$site_name mirror md5 [$phase]: ${h:-<none>}"
    done
}

assert_mirror_unchanged_since() {
    local ref_phase="$1" site site_name cur ref
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        ref="${MIRROR_MD5["$ref_phase:$site"]:-}"
        cur="$(mirror_md5 "$site")"
        if [[ -z "$ref" || -z "$cur" || "$cur" != "$ref" ]]; then
            err "$site_name mirror continuity failed for $ref_phase (ref=${ref:-<none>}, cur=${cur:-<none>})"
            return 1
        fi
    done
}

mirror_hashes_json() {
    local phase="$1" site site_name h obj='{}'
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        h="${MIRROR_MD5["$phase:$site"]:-}"
        obj="$(echo "$obj" | jq --arg k "$site_name" --arg v "$h" '. + {($k):$v}')"
    done
    echo "$obj"
}

wait_for_mirror() {
    local max_attempts=$((TIMEOUT_MINUTES * 4)) attempt=0 site scratch all
    while [[ $attempt -lt $max_attempts ]]; do
        all=true
        for site in "${CLIENT_SITES[@]}"; do
            scratch="$(site_scratch "$site")"
            remote_exec "$site" "test -s '$scratch/mediswarm_latest_global.pt'" >/dev/null 2>&1 || all=false
        done
        if [[ "$all" == true ]]; then
            ok "All clients mirrored a global"
            return 0
        fi
        sleep 15
        attempt=$((attempt + 1))
    done
    err "Timed out waiting for client mirrors"
    return 1
}

assert_log_contains() {
    local phase="$1" pattern="$2" site site_name log_file
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        log_file="$RESULTS_DIR/$phase/${site_name}_nohup.out"
        grep -q "$pattern" "$log_file" || {
            err "$phase log for $site_name does not contain: $pattern"
            return 1
        }
    done
}

assert_drop_tolerance_logged() {
    local phase="$1" phase_dir
    phase_dir="$RESULTS_DIR/$phase"
    if grep -E "FaultTolerant:.*(prun|tolerat)|configure_min_clients=.*continuing|did not configure.*continuing" \
        "$phase_dir/server_nohup.out" "$phase_dir"/*_nohup.out >/dev/null 2>&1; then
        ok "Drop-tolerance log evidence found"
        return 0
    fi
    err "No fault-tolerant prune/tolerance log evidence found for $phase"
    return 1
}

kill_client_site() {
    local site="$1" site_name
    site_name="$(site_var "$site" SITE_NAME)"
    remote_exec "$site" "
        container=\$(docker ps --filter 'name=odelia_swarm_client_${site_name}_' --format '{{.Names}}' | head -1)
        test -n \"\$container\"
        docker rm -f \"\$container\"
    "
}

collect_latest_globals_to() {
    local phase="$1" target_dir="$2"
    local site site_name scratch app_dir unique_hashes
    rm -rf "$target_dir"
    mkdir -p "$target_dir"

    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        scratch="$(site_scratch "$site")"
        app_dir="$target_dir/app_${site_name}"
        mkdir -p "$app_dir"
        remote_download "$site" "$scratch/mediswarm_latest_global.pt" "$app_dir/FL_global_model.pt" || return 1
    done

    md5sum "$target_dir"/app_*/FL_global_model.pt | tee "$RESULTS_DIR/$phase/checkpoint_md5s.txt" >/dev/null
    unique_hashes="$(awk '{print $1}' "$RESULTS_DIR/$phase/checkpoint_md5s.txt" | sort -u | wc -l | tr -d ' ')"
    if [[ "$unique_hashes" != "1" ]]; then
        err "$phase final globals are not byte-identical"
        return 1
    fi
    ok "$phase final globals are byte-identical"
}

evaluate_checkpoint_set() {
    local phase="$1" checkpoint_dir="$2" model="$3"
    local input_root output_dir
    if [[ "$SKIP_EVAL" == true ]]; then
        echo ""
        return 0
    fi

    input_root="$RESULTS_DIR/$phase/eval_input"
    output_dir="$RESULTS_DIR/$phase/eval"
    rm -rf "$input_root" "$output_dir"
    mkdir -p "$input_root"
    cp -a "$checkpoint_dir" "$input_root/${model}_checkpoints"

    "$REPO_ROOT/scripts/deploy/run_startupkit_smoke_eval_dl0.sh" \
        --conf "$CONF_FILE" \
        --checkpoint-root "$input_root" \
        --output-dir "$output_dir" \
        --image "$DOCKER_IMAGE" \
        --models "$model" \
        --expected-checkpoints "$CLIENT_COUNT" || return 1
    echo "$output_dir"
}

latest_job_id_from_phase() {
    local phase="$1" log_file="$RESULTS_DIR/$phase/server_nohup.out"
    grep 'Server runner finished\.' "$log_file" 2>/dev/null \
        | tail -1 \
        | grep -oP 'run=\K[0-9a-f-]+' \
        || true
}

record_phase() {
    local phase="$1" status="$2" checkpoint_dir="${3:-}" eval_dir="${4:-}" extra="${5:-}" job_id
    [[ -n "$extra" ]] || extra='{}'
    job_id="$(latest_job_id_from_phase "$phase")"
    jq -n \
        --arg phase "$phase" \
        --arg status "$status" \
        --arg job_id "$job_id" \
        --arg checkpoint_dir "$checkpoint_dir" \
        --arg eval_dir "$eval_dir" \
        --arg docker_image "$DOCKER_IMAGE" \
        --arg git_sha "$GIT_SHA" \
        --argjson extra "$extra" \
        '{phase:$phase,status:$status,job_id:$job_id,checkpoint_dir:$checkpoint_dir,evaluation_dir:$eval_dir,docker_image:$docker_image,git_sha:$git_sha} + $extra' \
        > "$RESULTS_DIR/${phase}_result.json"
}

run_job_phase() {
    local phase="$1" job="$2" model="$3" warm_start="$4" rounds="$5"
    local min_clients="$6" min_responses="$7" configure_min_clients="$8" expectation="$9" continuity_ref="${10:-}"
    local start_line

    step "Phase: $phase"
    stop_all
    start_server || return 1
    start_clients "$model" || return 1
    verify_client_model_env "$model" || return 1
    wait_for_registration || return 1

    if [[ -n "$continuity_ref" ]]; then
        assert_mirror_unchanged_since "$continuity_ref" || return 1
    fi

    prepare_admin_job "$job" "$warm_start" "$rounds" "$min_clients" "$min_responses" "$configure_min_clients" || return 1
    start_line="$(log_start_line)"
    submit_job "$job" "$warm_start" || return 1

    if [[ "$expectation" == "negative" ]]; then
        wait_for_negative_continue "$phase" "$start_line" || return 1
        save_phase_logs "$phase"
        stop_all
        return 0
    fi

    wait_for_success "$phase" "$start_line" || return 1
    save_phase_logs "$phase"
    stop_all
}

run_all_model_smoke() {
    local phase="all_model_smoke" training_dir="$RESULTS_DIR/$phase/training" eval_dir="" overall="pass"
    local entry job model model_phase checkpoint_dir model_jsonl="$RESULTS_DIR/$phase/models.jsonl"
    mkdir -p "$training_dir" "$RESULTS_DIR/$phase"
    : > "$model_jsonl"

    for entry in "${ALL_MODELS[@]}"; do
        job="${entry%%:*}"
        model="${entry#*:}"
        model_phase="${phase}_$(safe_phase_name "$model")"
        checkpoint_dir="$training_dir/${model}_checkpoints"
        clear_mirrors
        if run_job_phase "$model_phase" "$job" "$model" "fresh" "$SMOKE_ROUNDS" "$CLIENT_COUNT" "$CLIENT_COUNT" "$CLIENT_COUNT" "success"; then
            if assert_latest_globals && collect_latest_globals_to "$model_phase" "$checkpoint_dir"; then
                jq -n --arg model "$model" --arg job "$job" --arg status "pass" --arg checkpoint_dir "$checkpoint_dir" \
                    '{model:$model,job:$job,status:$status,checkpoint_dir:$checkpoint_dir}' >> "$model_jsonl"
            else
                overall="fail"
                jq -n --arg model "$model" --arg job "$job" --arg status "fail" \
                    '{model:$model,job:$job,status:$status}' >> "$model_jsonl"
            fi
        else
            overall="fail"
            save_phase_logs "$model_phase"
            stop_all
            jq -n --arg model "$model" --arg job "$job" --arg status "fail" \
                '{model:$model,job:$job,status:$status}' >> "$model_jsonl"
        fi
    done

    if [[ "$overall" == "pass" && "$SKIP_EVAL" != true ]]; then
        eval_dir="$RESULTS_DIR/$phase/eval"
        "$REPO_ROOT/scripts/deploy/run_startupkit_smoke_eval_dl0.sh" \
            --conf "$CONF_FILE" \
            --checkpoint-root "$training_dir" \
            --output-dir "$eval_dir" \
            --image "$DOCKER_IMAGE" \
            --expected-checkpoints "$CLIENT_COUNT" || overall="fail"
    fi

    record_phase "$phase" "$overall" "$training_dir" "$eval_dir" \
        "$(jq -s '{models:.}' "$model_jsonl")"
    [[ "$overall" == "pass" ]]
}

run_one_dc_full() {
    local phase="one_dc_full" checkpoint_dir="$RESULTS_DIR/$phase/checkpoints" eval_dir=""
    clear_mirrors
    if ! run_job_phase "$phase" "$JOB_NAME" "$MODEL_NAME" "fresh" "$FULL_ROUNDS" "$CLIENT_COUNT" "$CLIENT_COUNT" "$CLIENT_COUNT" "success"; then
        save_phase_logs "$phase"
        stop_all
        record_phase "$phase" "fail"
        return 1
    fi
    assert_latest_globals || { record_phase "$phase" "fail"; return 1; }
    record_mirror_hashes "$phase"
    collect_latest_globals_to "$phase" "$checkpoint_dir" || { record_phase "$phase" "fail"; return 1; }
    eval_dir="$(evaluate_checkpoint_set "$phase" "$checkpoint_dir" "$MODEL_NAME")" || { record_phase "$phase" "fail" "$checkpoint_dir"; return 1; }
    record_phase "$phase" "pass" "$checkpoint_dir" "$eval_dir" \
        "$(jq -n --argjson m "$(mirror_hashes_json "$phase")" '{mirror_md5:$m}')"
}

run_negative_continue() {
    local phase="negative_continue"
    clear_mirrors
    if run_job_phase "$phase" "$JOB_NAME" "$MODEL_NAME" "continue" "$CONTINUE_ROUNDS" "$CLIENT_COUNT" "$CLIENT_COUNT" "$CLIENT_COUNT" "negative"; then
        record_phase "$phase" "pass"
        return 0
    fi
    save_phase_logs "$phase"
    stop_all
    record_phase "$phase" "fail"
    return 1
}

run_fresh_then_continue() {
    local phase="fresh_then_continue" fresh_phase="${phase}_fresh" continue_phase="${phase}_continue"
    clear_mirrors
    if ! run_job_phase "$fresh_phase" "$JOB_NAME" "$MODEL_NAME" "fresh" "$CONTINUE_ROUNDS" "$CLIENT_COUNT" "$CLIENT_COUNT" "$CLIENT_COUNT" "success"; then
        record_phase "$phase" "fail"
        return 1
    fi
    assert_latest_globals || { record_phase "$phase" "fail"; return 1; }
    record_mirror_hashes "$fresh_phase"

    if ! run_job_phase "$continue_phase" "$JOB_NAME" "$MODEL_NAME" "continue" "$CONTINUE_ROUNDS" "$CLIENT_COUNT" "$CLIENT_COUNT" "$CLIENT_COUNT" "success" "$fresh_phase"; then
        record_phase "$phase" "fail"
        return 1
    fi
    assert_latest_globals || { record_phase "$phase" "fail"; return 1; }
    assert_log_contains "$continue_phase" "WarmStart: will warm-start from checkpoint /scratch/mediswarm_latest_global.pt (mode=require)" \
        || { record_phase "$phase" "fail"; return 1; }
    record_phase "$phase" "pass" "" "" \
        "$(jq -n --argjson m "$(mirror_hashes_json "$fresh_phase")" '{continuity_verified:true,fresh_mirror_md5:$m}')"
}

run_single_client_drop() {
    local phase="single_client_drop" start_line
    clear_mirrors
    step "Phase: $phase"
    stop_all
    start_server || return 1
    start_clients "$MODEL_NAME" || return 1
    verify_client_model_env "$MODEL_NAME" || return 1
    wait_for_registration || return 1
    prepare_admin_job "$JOB_NAME" "fresh" "$DROP_ROUNDS" "1" "1" "$CLIENT_COUNT" "false" || return 1
    start_line="$(log_start_line)"
    submit_job "$JOB_NAME" "fresh" || return 1
    wait_for_mirror || return 1
    record_mirror_hashes "$phase"
    warn "Killing node_B client container to exercise one-client-drop tolerance"
    kill_client_site "DL2" || return 1
    wait_for_success "$phase" "$start_line" || return 1
    save_phase_logs "$phase"
    stop_all
    assert_drop_tolerance_logged "$phase" || { record_phase "$phase" "fail"; return 1; }
    record_phase "$phase" "pass" "" "" \
        "$(jq -n --argjson m "$(mirror_hashes_json "$phase")" '{pre_drop_mirror_md5:$m,dropped_site:"node_B"}')"
}

run_abort_recovery() {
    local phase="abort_recovery" start_line
    clear_mirrors
    step "Phase: $phase"
    stop_all
    start_server || return 1
    start_clients "$MODEL_NAME" || return 1
    verify_client_model_env "$MODEL_NAME" || return 1
    wait_for_registration || return 1
    prepare_admin_job "$JOB_NAME" "fresh" "$ABORT_ROUNDS" "$CLIENT_COUNT" "$CLIENT_COUNT" "$CLIENT_COUNT" || return 1
    start_line="$(log_start_line)"
    submit_job "$JOB_NAME" "fresh" || return 1
    wait_for_mirror || return 1
    record_mirror_hashes "$phase"
    save_phase_logs "$phase"
    warn "Aborting all NVFlare containers after first mirrored global"
    stop_all
    assert_latest_globals || { record_phase "$phase" "fail"; return 1; }
    assert_mirror_unchanged_since "$phase" || { record_phase "$phase" "fail"; return 1; }

    start_server || return 1
    start_clients "$MODEL_NAME" || return 1
    verify_client_model_env "$MODEL_NAME" || return 1
    wait_for_registration || return 1
    assert_mirror_unchanged_since "$phase" || { record_phase "$phase" "fail"; return 1; }
    prepare_admin_job "$JOB_NAME" "continue" "$RESUME_ROUNDS" "$CLIENT_COUNT" "$CLIENT_COUNT" "$CLIENT_COUNT" || return 1
    start_line="$(log_start_line)"
    submit_job "$JOB_NAME" "continue" || return 1
    wait_for_success "$phase" "$start_line" || return 1
    save_phase_logs "$phase"
    stop_all
    assert_latest_globals || { record_phase "$phase" "fail"; return 1; }
    assert_log_contains "$phase" "WarmStart: will warm-start from checkpoint /scratch/mediswarm_latest_global.pt (mode=require)" \
        || { record_phase "$phase" "fail"; return 1; }
    record_phase "$phase" "pass" "" "" \
        "$(jq -n --argjson m "$(mirror_hashes_json "$phase")" '{resumed_from_crash:true,pre_abort_mirror_md5:$m}')"
}

write_summary() {
    local result_files=()
    while IFS= read -r f; do
        result_files+=("$f")
    done < <(find "$RESULTS_DIR" -maxdepth 1 -name '*_result.json' -type f | sort)

    if [[ "${#result_files[@]}" -eq 0 ]]; then
        warn "No phase result files found; not writing summary.json"
        return 0
    fi

    jq -s \
        --arg run_id "$RUN_ID" \
        --arg docker_image "$DOCKER_IMAGE" \
        --arg git_sha "$GIT_SHA" \
        --arg project_file "$PROJECT_FILE" \
        --arg conf_file "$CONF_FILE" \
        --arg results_dir "$RESULTS_DIR" \
        --arg phases "$PHASES_CSV" \
        '{run_id:$run_id,docker_image:$docker_image,git_sha:$git_sha,project_file:$project_file,conf_file:$conf_file,results_dir:$results_dir,requested_phases:$phases,phases:.}' \
        "${result_files[@]}" \
        > "$RESULTS_DIR/summary.json"

    {
        echo "# Duke IID 2-Client DL Validation"
        echo ""
        echo "- Run ID: \`$RUN_ID\`"
        echo "- Docker image: \`$DOCKER_IMAGE\`"
        echo "- Git SHA: \`$GIT_SHA\`"
        echo "- Project: \`$PROJECT_FILE\`"
        echo "- Config: \`$CONF_FILE\`"
        echo "- Requested phases: \`$PHASES_CSV\`"
        echo "- Summary: \`$RESULTS_DIR/summary.json\`"
        echo ""
        echo "| Phase | Status |"
        echo "|-------|--------|"
        jq -r '.phases[] | [.phase, .status] | @tsv' "$RESULTS_DIR/summary.json" \
            | while IFS=$'\t' read -r phase status; do
                echo "| $phase | $status |"
              done
    } > "$RESULTS_DIR/RUN.md"
    ok "Summary written to $RESULTS_DIR/RUN.md"
}

run_selected_phase() {
    local phase="$1"
    phase_enabled "$phase" || return 0
    case "$phase" in
        all_model_smoke)       run_all_model_smoke ;;
        one_dc_full)           run_one_dc_full ;;
        negative_continue)     run_negative_continue ;;
        fresh_then_continue)   run_fresh_then_continue ;;
        single_client_drop)    run_single_client_drop ;;
        abort_recovery)        run_abort_recovery ;;
        *)                     err "Unknown phase: $phase"; return 1 ;;
    esac
}

step "Duke IID 2-client DL validation"
info "Run ID: $RUN_ID"
info "Docker image: $DOCKER_IMAGE"
info "Results: $RESULTS_DIR"
info "Clients: ${CLIENT_SITES[*]}"

trap 'stop_all >/dev/null 2>&1 || true' EXIT

preflight
build_and_push
deploy_kits
fix_remote_dns
pre_pull_images

overall=0
for phase in all_model_smoke one_dc_full negative_continue fresh_then_continue single_client_drop abort_recovery; do
    if phase_enabled "$phase"; then
        if ! run_selected_phase "$phase"; then
            overall=1
            if [[ ! -f "$RESULTS_DIR/${phase}_result.json" ]]; then
                save_phase_logs "$phase"
                record_phase "$phase" "fail"
            fi
            stop_all
            warn "Phase failed: $phase"
        fi
    fi
done

write_summary
stop_all
trap - EXIT

if [[ "$overall" -eq 0 ]]; then
    ok "Duke IID 2-client validation completed"
    exit 0
fi

err "Duke IID 2-client validation completed with failures. See $RESULTS_DIR/RUN.md"
exit 1
