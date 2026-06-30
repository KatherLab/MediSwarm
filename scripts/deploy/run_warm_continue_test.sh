#!/usr/bin/env bash
# Orchestrate the RSH/MHA warm-continue full-eval test.

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

usage() {
    cat <<'EOF'
Usage:
  scripts/deploy/run_warm_continue_test.sh \
    --conf deploy_sites_rsh_mha.local.conf \
    --project application/provision/project_warm_continue_rsh_mha.yml \
    [--job challenge_5pimed] [--model 5Pimed] [--results-dir DIR]

Options:
  --skip-build      Reuse existing Docker image/startup kits.
  --skip-push       Do not push the built Docker image before pulling on clients.
  --timeout MIN     Per-phase wait timeout in minutes (default: 240).

Environment:
  EVAL_DEVICE       Prediction device for Cosmos eval: cpu, auto, cuda, cuda:0 (default: cpu).
  EVAL_GPU          Docker --gpus value when EVAL_DEVICE is not cpu (default: device=0).
EOF
}

CONF_FILE=""
PROJECT_FILE="$REPO_ROOT/application/provision/project_warm_continue_rsh_mha.yml"
JOB_NAME="challenge_5pimed"
MODEL_NAME="5Pimed"
RESULTS_DIR=""
SKIP_BUILD=false
SKIP_PUSH=false
TIMEOUT_MINUTES=240
RUN_ID="warm_continue_$(date -u +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conf)        CONF_FILE="$2"; shift ;;
        --project)     PROJECT_FILE="$2"; shift ;;
        --job)         JOB_NAME="$2"; shift ;;
        --model)       MODEL_NAME="$2"; shift ;;
        --results-dir) RESULTS_DIR="$2"; shift ;;
        --skip-build)  SKIP_BUILD=true ;;
        --skip-push)   SKIP_PUSH=true ;;
        --timeout)     TIMEOUT_MINUTES="$2"; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             err "Unknown argument: $1"; usage; exit 2 ;;
    esac
    shift
done

if [[ -z "$CONF_FILE" ]]; then
    err "Missing --conf"
    usage
    exit 2
fi

resolve_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$REPO_ROOT/$path"
    fi
}

CONF_FILE="$(resolve_path "$CONF_FILE")"
PROJECT_FILE="$(resolve_path "$PROJECT_FILE")"
if [[ "$PROJECT_FILE" != "$REPO_ROOT/"* ]]; then
    err "--project must point to a file under the repository root: $PROJECT_FILE"
    exit 1
fi
PROJECT_FILE_FOR_BUILD="${PROJECT_FILE#$REPO_ROOT/}"

if [[ ! -f "$CONF_FILE" ]]; then
    err "Config file not found: $CONF_FILE"
    exit 1
fi
if [[ ! -f "$PROJECT_FILE" ]]; then
    err "Project file not found: $PROJECT_FILE"
    exit 1
fi

# shellcheck source=/dev/null
source "$CONF_FILE"

VERSION="$("$REPO_ROOT/scripts/build/getVersionNumber.sh")"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
DOCKER_IMAGE="$(grep 'docker_image:' "$PROJECT_FILE" \
    | sed 's/.*docker_image:[[:space:]]*//' \
    | sed "s#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#$VERSION#")"
PROJECT_NAME="$(grep '^name: ' "$PROJECT_FILE" \
    | sed 's/^name: //' \
    | sed "s#__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__#$VERSION#")"
WORKSPACE_DIR="$REPO_ROOT/workspace/$PROJECT_NAME"
DEPLOY_BASE="${COSMOS_DEPLOY_DIR:-/home/jeff/deploy_test}"
SERVER_NAME="${SERVER_NAME:-dl3.tud.de}"
ADMIN_USER="${ADMIN_USER:-jiefu.zhu@tu-dresden.de}"
EVAL_SITE_NAME="${EVAL_SITE_NAME:-UKA_1}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/mnt/sda1/ODELIA_Challenge_unilateral}"
EVAL_SCRATCH_DIR="${EVAL_SCRATCH_DIR:-/mnt/scratch/deploy_test_eval}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EVAL_GPU="${EVAL_GPU:-device=0}"
CLIENT_COUNT="${#CLIENT_SITES[@]}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
NVFLARE_CONTAINER_RE='odelia_swarm|nvflare|^swarm-'

if [[ -z "$RESULTS_DIR" ]]; then
    RESULTS_DIR="$REPO_ROOT/workspace/warm_continue_results/$RUN_ID"
elif [[ "$RESULTS_DIR" != /* ]]; then
    RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR"
fi
mkdir -p "$RESULTS_DIR"

declare -A PASS_CACHE=()
declare -A MIRROR_MD5=()   # "phase:site" -> md5 of that client's /scratch/mediswarm_latest_global.pt

site_var() {
    local site="$1"
    local var="$2"
    local name="${site}_${var}"
    echo "${!name-}"
}

ensure_site_pass() {
    local site="$1"
    if [[ -n "${PASS_CACHE[$site]+set}" ]]; then
        return
    fi

    local pass env_name host user
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
    echo "${root%/}/$RUN_ID"
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
    local cmd="$*"
    local host user pass remote_cmd
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

find_latest_prod() {
    if [[ ! -d "$WORKSPACE_DIR" ]]; then
        err "Workspace not found: $WORKSPACE_DIR"
        exit 1
    fi
    ls -d "$WORKSPACE_DIR"/prod_* 2>/dev/null | sort -V | tail -n 1
}

clean_local_deploy_dir() {
    local dir_name="$1"
    local target="$DEPLOY_BASE/$dir_name"
    [[ -e "$target" ]] || return 0
    rm -rf "$target" 2>/dev/null \
        || sudo rm -rf "$target" 2>/dev/null \
        || docker run --rm -v "$DEPLOY_BASE:/cleanup" alpine rm -rf "/cleanup/$dir_name" 2>/dev/null \
        || warn "Could not fully clean $target"
}

stop_all() {
    info "Stopping local and remote NVFlare containers"
    docker ps -a --format '{{.Names}}' | grep -E "$NVFLARE_CONTAINER_RE" | xargs -r docker rm -f 2>/dev/null || true

    local -A visited=()
    for site in "${CLIENT_SITES[@]}"; do
        local host
        host="$(site_var "$site" HOST)"
        [[ -n "${visited[$host]+set}" ]] && continue
        visited[$host]=1
        remote_exec "$site" \
            "docker ps -a --format '{{.Names}}' | grep -E 'odelia_swarm|nvflare|^swarm-' | xargs -r docker rm -f 2>/dev/null" \
            >/dev/null 2>&1 || warn "Could not stop containers on $host"
    done
}

build_and_push() {
    if [[ "$SKIP_BUILD" == true ]]; then
        info "Skipping build; reusing $DOCKER_IMAGE"
        return
    fi

    step "Building Docker image and startup kits"
    bash "$REPO_ROOT/scripts/build/buildDockerImageAndStartupKits.sh" \
        -p "$PROJECT_FILE_FOR_BUILD" \
        --use-docker-cache

    if [[ "$SKIP_PUSH" == true ]]; then
        warn "Skipping docker push for $DOCKER_IMAGE"
    else
        docker push "$DOCKER_IMAGE"
    fi
}

deploy_kits() {
    local prod_dir
    prod_dir="$(find_latest_prod)"
    step "Deploying startup kits from $prod_dir"

    for site in "${CLIENT_SITES[@]}"; do
        local site_name deploy_dir zip_file
        site_name="$(site_var "$site" SITE_NAME)"
        deploy_dir="$(site_var "$site" DEPLOY_DIR)"
        zip_file="$prod_dir/${site_name}_${VERSION}.zip"
        [[ -f "$zip_file" ]] || zip_file="$(ls "$prod_dir"/${site_name}*.zip 2>/dev/null | head -1 || true)"
        [[ -n "$zip_file" && -f "$zip_file" ]] || { err "Missing startup kit for $site_name"; exit 1; }

        remote_exec "$site" "mkdir -p '$deploy_dir'"
        remote_copy "$site" "$zip_file" "$deploy_dir/"
        remote_exec "$site" "cd '$deploy_dir' && rm -rf '$site_name' && unzip -qo '$(basename "$zip_file")'"
        ok "Deployed $site_name"
    done

    local server_zip admin_zip
    server_zip="$prod_dir/${SERVER_NAME}_${VERSION}.zip"
    [[ -f "$server_zip" ]] || server_zip="$(ls "$prod_dir"/${SERVER_NAME}*.zip 2>/dev/null | head -1 || true)"
    [[ -n "$server_zip" && -f "$server_zip" ]] || { err "Missing server startup kit for $SERVER_NAME"; exit 1; }

    admin_zip="$prod_dir/${ADMIN_USER}_${VERSION}.zip"
    [[ -f "$admin_zip" ]] || admin_zip="$(ls "$prod_dir"/${ADMIN_USER}*.zip 2>/dev/null | head -1 || true)"
    [[ -n "$admin_zip" && -f "$admin_zip" ]] || { err "Missing admin startup kit for $ADMIN_USER"; exit 1; }

    mkdir -p "$DEPLOY_BASE"
    cp "$server_zip" "$admin_zip" "$DEPLOY_BASE/"
    clean_local_deploy_dir "$SERVER_NAME"
    clean_local_deploy_dir "$ADMIN_USER"
    (cd "$DEPLOY_BASE" && unzip -qo "$(basename "$server_zip")" && unzip -qo "$(basename "$admin_zip")")
    ok "Deployed server/admin kits on Cosmos"
}

fix_remote_dns() {
    local cosmos_ip="${COSMOS_HOST_IP:-}"
    if [[ -z "$cosmos_ip" ]]; then
        local first_host
        first_host="$(site_var "${CLIENT_SITES[0]}" HOST)"
        cosmos_ip="$(ip route get "$first_host" 2>/dev/null \
            | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}')"
    fi
    if [[ -z "$cosmos_ip" ]]; then
        cosmos_ip="$(tailscale ip -4 2>/dev/null)" || {
            err "Cannot determine Cosmos client-reachable IP"
            exit 1
        }
    fi

    step "Ensuring $SERVER_NAME resolves to Cosmos ($cosmos_ip) on clients"
    for site in "${CLIENT_SITES[@]}"; do
        local current
        current="$(remote_exec "$site" "grep -E '\\b${SERVER_NAME}\\b' /etc/hosts 2>/dev/null | awk '{print \$1}' | head -1" 2>/dev/null || true)"
        if [[ "$current" == "$cosmos_ip" ]]; then
            ok "$site already maps $SERVER_NAME to $cosmos_ip"
            continue
        fi

        if [[ -n "$current" ]]; then
            remote_sudo_exec "$site" "sed -i 's|^.*\\b${SERVER_NAME}\\b.*\$|${cosmos_ip} ${SERVER_NAME}|' /etc/hosts"
        else
            remote_sudo_exec "$site" "echo '${cosmos_ip} ${SERVER_NAME}' >> /etc/hosts"
        fi
        ok "$site maps $SERVER_NAME to $cosmos_ip"
    done
}

preflight_clients() {
    step "Client preflight"
    for site in "${CLIENT_SITES[@]}"; do
        local site_name datadir scratch
        site_name="$(site_var "$site" SITE_NAME)"
        datadir="$(site_var "$site" DATADIR)"
        scratch="$(site_scratch "$site")"
        info "Checking $site_name"
        remote_exec "$site" "command -v docker >/dev/null && command -v nvidia-smi >/dev/null && test -d '$datadir/$site_name/data_unilateral' && test -d '$datadir/$site_name/metadata_unilateral'"
        remote_exec "$site" "rm -rf '$scratch' && mkdir -p '$scratch' && chmod 777 '$scratch'" \
            || remote_sudo_exec "$site" "rm -rf '$scratch' && mkdir -p '$scratch' && chmod 777 '$scratch'"
        ok "$site_name preflight passed"
    done
}

pre_pull_images() {
    step "Pulling $DOCKER_IMAGE on clients"
    for site in "${CLIENT_SITES[@]}"; do
        local gpu
        gpu="$(site_var "$site" GPU)"
        remote_exec "$site" "docker pull '$DOCKER_IMAGE' && docker run --rm --gpus='$gpu' '$DOCKER_IMAGE' nvidia-smi -L >/dev/null"
        ok "$site image/GPU check passed"
    done
}

server_startup_dir() {
    echo "$DEPLOY_BASE/$SERVER_NAME/startup"
}

admin_startup_dir() {
    echo "$DEPLOY_BASE/$ADMIN_USER/startup"
}

server_log() {
    echo "$(server_startup_dir)/nohup.out"
}

start_server() {
    local startup
    startup="$(server_startup_dir)"
    [[ -d "$startup" ]] || { err "Missing server startup dir: $startup"; exit 1; }
    (cd "$startup" && ./docker.sh --no_pull --start_server)
    sleep 15
}

start_clients() {
    for site in "${CLIENT_SITES[@]}"; do
        local site_name deploy_dir datadir scratch gpu
        site_name="$(site_var "$site" SITE_NAME)"
        deploy_dir="$(site_var "$site" DEPLOY_DIR)"
        datadir="$(site_var "$site" DATADIR)"
        scratch="$(site_scratch "$site")"
        gpu="$(site_var "$site" GPU)"
        remote_exec "$site" \
            "cd '$deploy_dir/$site_name/startup' && ./docker.sh --no_pull --data_dir '$datadir' --scratch_dir '$scratch' --GPU '$gpu' --model_name '$MODEL_NAME' --start_client"
        ok "Started $site_name"
    done
}

wait_for_registration() {
    local max_wait=600 elapsed=0 log_file
    log_file="$(server_log)"
    while [[ $elapsed -lt $max_wait ]]; do
        local all_registered=true
        for site in "${CLIENT_SITES[@]}"; do
            local site_name
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
    local warm_start="$1" rounds="$2" admin_startup
    admin_startup="$(admin_startup_dir)"
    [[ -d "$admin_startup" ]] || { err "Missing admin startup dir: $admin_startup"; exit 1; }
    (cd "$admin_startup" && ./prepare_odelia_job.sh \
        --job "$JOB_NAME" \
        --warm-start "$warm_start" \
        --num-rounds "$rounds" \
        --min-clients "${MIN_CLIENTS:-$CLIENT_COUNT}" \
        --configure-min-clients "${CONFIGURE_MIN_CLIENTS:-$CLIENT_COUNT}" \
        --min-responses "${MIN_RESPONSES:-$CLIENT_COUNT}")
}

submit_job() {
    # NB: keep these on separate lines -- `local a="$1" b="...$a..."` expands $a before
    # the assignment lands, which under `set -u` aborts when no outer `warm_start` is in
    # scope (e.g. when called from run_abort_recovery_phase rather than run_phase).
    local warm_start="$1"
    local job_path="/fl_admin/local/mediswarm_jobs/${JOB_NAME}_${warm_start}"
    local admin_startup expect_script
    admin_startup="$(admin_startup_dir)"
    expect_script="$(mktemp /tmp/mediswarm_warm_continue_XXXXXX.exp)"
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
    (cd "$admin_startup" && expect -f "$expect_script")
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

wait_for_negative_continue() {
    local phase="$1" start_line="$2" max_attempts=$((TIMEOUT_MINUTES * 2)) attempt=0
    while [[ $attempt -lt $max_attempts ]]; do
        local lines
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

            local all_clients_logged_error=true
            for site in "${CLIENT_SITES[@]}"; do
                local site_name log_file
                site_name="$(site_var "$site" SITE_NAME)"
                log_file="$RESULTS_DIR/$phase/${site_name}_nohup.out"
                if ! grep -q "WARM_START_REQUIRED_MISSING" "$log_file"; then
                    all_clients_logged_error=false
                    warn "$site_name log did not contain WARM_START_REQUIRED_MISSING"
                fi
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

wait_for_success() {
    local phase="$1" start_line="$2" max_attempts=$((TIMEOUT_MINUTES * 2)) attempt=0
    while [[ $attempt -lt $max_attempts ]]; do
        local lines
        lines="$(phase_lines "$start_line")"
        if echo "$lines" | grep -q "WARM_START_REQUIRED_MISSING"; then
            err "$phase failed with WARM_START_REQUIRED_MISSING"
            return 1
        fi
        if echo "$lines" | grep -q "FATAL_SYSTEM_ERROR\\|ABORT_RUN"; then
            err "$phase hit fatal server error"
            echo "$lines" | grep -i "FATAL\\|ABORT\\|ERROR\\|EXCEPTION" | tail -20 >&2
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

save_phase_logs() {
    local phase="$1"
    local phase_dir="$RESULTS_DIR/$phase"
    mkdir -p "$phase_dir"
    cp "$(server_log)" "$phase_dir/server_nohup.out" 2>/dev/null || true
    for site in "${CLIENT_SITES[@]}"; do
        local site_name deploy_dir
        site_name="$(site_var "$site" SITE_NAME)"
        deploy_dir="$(site_var "$site" DEPLOY_DIR)"
        remote_download "$site" "$deploy_dir/$site_name/startup/nohup.out" "$phase_dir/${site_name}_nohup.out" \
            >/dev/null 2>&1 || true
    done
}

assert_latest_globals() {
    for site in "${CLIENT_SITES[@]}"; do
        local site_name scratch
        site_name="$(site_var "$site" SITE_NAME)"
        scratch="$(site_scratch "$site")"
        remote_exec "$site" "test -s '$scratch/mediswarm_latest_global.pt'" || {
            err "Missing latest global checkpoint for $site_name"
            return 1
        }
        ok "$site_name has latest global checkpoint"
    done
}

assert_log_contains() {
    local phase="$1" pattern="$2"
    for site in "${CLIENT_SITES[@]}"; do
        local site_name log_file
        site_name="$(site_var "$site" SITE_NAME)"
        log_file="$RESULTS_DIR/$phase/${site_name}_nohup.out"
        grep -q "$pattern" "$log_file" || {
            err "$phase log for $site_name does not contain: $pattern"
            return 1
        }
    done
}

mirror_md5() {
    # md5 of a client's mirrored global, or empty if absent
    local site="$1" scratch
    scratch="$(site_scratch "$site")"
    remote_exec "$site" "md5sum '$scratch/mediswarm_latest_global.pt' 2>/dev/null | awk '{print \$1}'" 2>/dev/null || true
}

clear_mirrors() {
    # delete every client's mirror so the next run's resume point is unambiguously the
    # global produced by THAT run (not a leftover mirror from a previous phase)
    local site site_name scratch
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        scratch="$(site_scratch "$site")"
        remote_exec "$site" "rm -f '$scratch/mediswarm_latest_global.pt'" >/dev/null 2>&1 || true
        info "$site_name mirror cleared"
    done
}

record_mirror_hashes() {
    # snapshot every client's current mirror md5 under MIRROR_MD5["$phase:$site"]
    local phase="$1" site site_name h
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        h="$(mirror_md5 "$site")"
        MIRROR_MD5["$phase:$site"]="$h"
        info "$site_name mirror md5 [$phase]: ${h:-<none>}"
    done
}

assert_mirror_unchanged_since() {
    # each client's CURRENT mirror md5 must equal the one snapshotted for <ref_phase>;
    # proves the mirror survived a client restart / crash and is byte-identical to what
    # the reference run produced -- i.e. a continue loads exactly the prior run's global.
    local ref_phase="$1" site site_name cur ref
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        ref="${MIRROR_MD5["$ref_phase:$site"]:-}"
        cur="$(mirror_md5 "$site")"
        if [[ -z "$ref" || -z "$cur" ]]; then
            err "$site_name continuity check missing a hash (ref[$ref_phase]=${ref:-<none>}, cur=${cur:-<none>})"
            return 1
        fi
        if [[ "$cur" != "$ref" ]]; then
            err "$site_name mirror changed since $ref_phase (continuity broken): $ref -> $cur"
            return 1
        fi
        ok "$site_name mirror unchanged since $ref_phase ($cur) -- continuity holds"
    done
}

mirror_hashes_json() {
    # {"<site_name>": "<md5>", ...} from MIRROR_MD5 for a given phase
    local phase="$1" site site_name h obj='{}'
    for site in "${CLIENT_SITES[@]}"; do
        site_name="$(site_var "$site" SITE_NAME)"
        h="${MIRROR_MD5["$phase:$site"]:-}"
        obj="$(echo "$obj" | jq --arg k "$site_name" --arg v "$h" '. + {($k):$v}')"
    done
    echo "$obj"
}

wait_for_mirror() {
    # poll until EVERY client has a non-empty mirror (a round's global has been saved+mirrored)
    local max_attempts=$((TIMEOUT_MINUTES * 4)) attempt=0 site scratch all
    while [[ $attempt -lt $max_attempts ]]; do
        all=true
        for site in "${CLIENT_SITES[@]}"; do
            scratch="$(site_scratch "$site")"
            remote_exec "$site" "test -s '$scratch/mediswarm_latest_global.pt'" >/dev/null 2>&1 || all=false
        done
        if [[ "$all" == true ]]; then
            ok "All clients mirrored a global (a round completed) -- safe to abort mid-run"
            return 0
        fi
        sleep 15
        attempt=$((attempt + 1))
    done
    err "Timed out waiting for a mirrored global before the abort"
    return 1
}

collect_latest_globals() {
    local phase="$1"
    local checkpoint_dir="$RESULTS_DIR/$phase/checkpoints"
    rm -rf "$checkpoint_dir"
    mkdir -p "$checkpoint_dir"
    for site in "${CLIENT_SITES[@]}"; do
        local site_name scratch app_dir
        site_name="$(site_var "$site" SITE_NAME)"
        scratch="$(site_scratch "$site")"
        app_dir="$checkpoint_dir/app_${site_name}"
        mkdir -p "$app_dir"
        remote_download "$site" "$scratch/mediswarm_latest_global.pt" "$app_dir/FL_global_model.pt"
    done
    echo "$checkpoint_dir"
}

evaluate_phase() {
    local phase="$1"
    local checkpoint_dir="$2"
    local output_dir="$RESULTS_DIR/$phase/evaluation"
    local gpu_args=()
    mkdir -p "$EVAL_SCRATCH_DIR" "$output_dir"

    if [[ "${EVAL_DEVICE,,}" != "cpu" && "${EVAL_GPU,,}" != "none" ]]; then
        gpu_args=(--gpus="$EVAL_GPU")
    fi

    docker run --rm \
        "${gpu_args[@]}" \
        --net=host \
        --ipc=host \
        -v "$EVAL_DATA_DIR:/data/:ro" \
        -v "$EVAL_SCRATCH_DIR:/scratch/" \
        -v "$checkpoint_dir:/workspace/:ro" \
        -v "$output_dir:/output/" \
        --env SITE_NAME="$EVAL_SITE_NAME" \
        --env DATA_DIR=/data \
        --env SCRATCH_DIR=/scratch \
        --env MODEL_NAME="$MODEL_NAME" \
        --env TORCH_HOME=/torch_home \
        --env CONFIG=unilateral \
        --env PREDICT_DEVICE="$EVAL_DEVICE" \
        --env MEDISWARM_ALLOW_CPU_MODEL=1 \
        "$DOCKER_IMAGE" \
        python3 /MediSwarm/scripts/evaluation/predict.py \
            --workspace /workspace \
            --model-name "$MODEL_NAME" \
            --output-dir /output \
            --split test \
            --device "$EVAL_DEVICE" \
        2>&1 | tee "$output_dir/predict_stdout.log" >&2
    echo "$output_dir"
}

latest_job_id_from_phase() {
    local phase="$1"
    local log_file="$RESULTS_DIR/$phase/server_nohup.out"
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

run_phase() {
    local phase="$1" warm_start="$2" rounds="$3" expectation="$4"
    local start_line checkpoint_dir="" eval_dir=""

    step "Phase: $phase"
    stop_all
    start_server
    start_clients
    wait_for_registration

    # Continuity (P1): before a continue submits, the mirror it is about to load must be
    # byte-identical to what the fresh run produced and have survived the client restart.
    if [[ "$phase" == "continue" ]]; then
        assert_mirror_unchanged_since "fresh"
    fi

    prepare_admin_job "$warm_start" "$rounds"
    start_line="$(log_start_line)"
    submit_job "$warm_start"

    if [[ "$expectation" == "negative" ]]; then
        wait_for_negative_continue "$phase" "$start_line"
        save_phase_logs "$phase"
        record_phase "$phase" "pass"
        stop_all
        return
    fi

    wait_for_success "$phase" "$start_line"
    save_phase_logs "$phase"
    assert_latest_globals
    record_mirror_hashes "$phase"

    if [[ "$phase" == "continue" ]]; then
        assert_log_contains "$phase" "WarmStart: will warm-start from checkpoint /scratch/mediswarm_latest_global.pt (mode=require)"
    elif [[ "$phase" == "fresh_probe" ]]; then
        assert_log_contains "$phase" "WarmStart: warm_start_mode=fresh; initializing fresh and ignoring any local checkpoint"
    fi

    stop_all

    if [[ "$expectation" == "eval" ]]; then
        checkpoint_dir="$(collect_latest_globals "$phase")"
        eval_dir="$(evaluate_phase "$phase" "$checkpoint_dir")"
    fi

    local extra='{}'
    if [[ "$phase" == "continue" ]]; then
        extra="$(jq -n --argjson m "$(mirror_hashes_json "fresh")" \
            '{continuity_verified:true, reference_phase:"fresh", fresh_mirror_md5:$m}')"
    fi
    record_phase "$phase" "pass" "$checkpoint_dir" "$eval_dir" "$extra"
}

run_abort_recovery_phase() {
    # The real #347 scenario: a run CRASHES mid-flight; a strict (require) continue must
    # resume from the last mirrored global instead of restarting. (The other phases only
    # chain two cleanly-completed runs.)
    local phase="abort_recovery" start_line
    step "Phase: $phase (crash mid-run, then resume from the last mirror)"

    # 1. Start a fresh run with enough rounds that we can abort it mid-way.
    stop_all
    start_server
    start_clients
    wait_for_registration
    clear_mirrors   # so the resume point is THIS run's global, not a leftover from a prior phase
    prepare_admin_job "fresh" 3
    start_line="$(log_start_line)"
    submit_job "fresh"

    # 2. As soon as every client has mirrored a round's global, snapshot it and simulate a crash.
    wait_for_mirror
    record_mirror_hashes "$phase"
    save_phase_logs "$phase"
    warn "Aborting the run mid-flight (docker rm -f all NVFlare containers) to simulate a crash"
    stop_all

    # 3. The mirror must survive the crash on the host /scratch mount.
    assert_latest_globals
    assert_mirror_unchanged_since "$phase"

    # 4. Resume: a strict continue must load the surviving mirror and finish cleanly.
    start_server
    start_clients
    wait_for_registration
    assert_mirror_unchanged_since "$phase"   # mirror still intact after the client restart
    prepare_admin_job "continue" 1
    start_line="$(log_start_line)"
    submit_job "continue"
    wait_for_success "$phase" "$start_line"
    save_phase_logs "$phase"
    assert_latest_globals
    assert_log_contains "$phase" "WarmStart: will warm-start from checkpoint /scratch/mediswarm_latest_global.pt (mode=require)"
    stop_all

    record_phase "$phase" "pass" "" "" \
        "$(jq -n --argjson m "$(mirror_hashes_json "$phase")" \
            '{resumed_from_crash:true, pre_abort_mirror_md5:$m}')"
    ok "Abort-recovery: a crashed run resumed from the last mirrored global"
}

write_summary() {
    jq -s \
        --arg run_id "$RUN_ID" \
        --arg docker_image "$DOCKER_IMAGE" \
        --arg git_sha "$GIT_SHA" \
        --arg project_file "$PROJECT_FILE" \
        --arg results_dir "$RESULTS_DIR" \
        '{run_id:$run_id,docker_image:$docker_image,git_sha:$git_sha,project_file:$project_file,results_dir:$results_dir,phases:.}' \
        "$RESULTS_DIR"/*_result.json \
        > "$RESULTS_DIR/summary.json"
    ok "Summary written to $RESULTS_DIR/summary.json"
}

for cmd in docker expect jq sshpass unzip; do
    command -v "$cmd" >/dev/null 2>&1 || { err "Missing required command: $cmd"; exit 1; }
done

if [[ "$CLIENT_COUNT" -lt 2 ]]; then
    err "This wrapper expects at least two clients; got $CLIENT_COUNT"
    exit 1
fi
if [[ ! -d "$EVAL_DATA_DIR/$EVAL_SITE_NAME" ]]; then
    err "Missing eval site directory: $EVAL_DATA_DIR/$EVAL_SITE_NAME"
    exit 1
fi

step "Warm-continue RSH/MHA test"
info "Run ID: $RUN_ID"
info "Docker image: $DOCKER_IMAGE"
info "Results: $RESULTS_DIR"

build_and_push
preflight_clients
deploy_kits
fix_remote_dns
pre_pull_images

run_phase "negative_continue" "continue" 2 "negative"
run_phase "fresh" "fresh" 2 "eval"
run_phase "continue" "continue" 2 "eval"
run_phase "fresh_probe" "fresh" 1 "no_eval"
run_abort_recovery_phase

write_summary
stop_all
ok "Warm-continue full-eval test completed"
