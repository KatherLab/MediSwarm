#!/usr/bin/env bash
# ============================================================================
# launch_testbed.sh — start one deploy test on the validation test bed (dl3)
#
# Tracked replacement for the untracked launch_mvp.sh in dl3's checkout. That
# script pinned the September test image, hard-coded the model and the timeout,
# and a sibling script held the test bed's password in plain text. This one takes
# everything as flags, refuses credentials that are readable by other users,
# records what it launched next to the results, and starts run_deploy_test.sh in
# the background (or attached with --foreground).
#
# Credentials: the conf names them, never this script. Per site (and for COSMOS):
#   <SITE>_PASS_ENV=DL_SWARM_PASS          an environment variable that holds it, or
#   <SITE>_PASS_FILE=/home/swarm/.mediswarm_testbed_pass   a file (mode 0600), or
#   <SITE>_PASS=...                        only in a conf that is itself mode 0600.
# See scripts/deploy/deploy_sites_4node_local.dl3.conf.example.
#
# Usage:
#   scripts/deploy/launch_testbed.sh --conf deploy_sites_4node_local.dl3.conf \
#       --kit-version 1.8.1-dev.260913.f7c66bb [--image jefftud/odelia:<candidate>] \
#       [--model MST] [--job ODELIA_ternary_classification | /fl_admin/local/mediswarm_jobs/<job>] \
#       [--timeout 150] [--eval] [--tag rc_mst4c] [--keep-mirror] [--no-pull] [--foreground]
#
#   --kit-version  version the test kits were built as (selects workspace/<project>_<version>);
#                  the sites' kits are 1.8.1-template kits, e.g. 1.8.1-dev.260913.f7c66bb on dl3
#   --image        image under test; server, admin and clients run it with those kits
#                  (default: the image the kits were built for)
#   --job          a job name from the image's job tree, or an absolute path of a job staged in
#                  the admin kit (build it with scripts/admin/build_consortium_job.py)
#   --eval         run the post-training evaluation (needs EVAL_* in the conf); default: skip
#   --keep-mirror  leave the clients' warm-start mirrors in place (warm-start tests)
#
# Prints "<pid> <results dir>"; follow <results dir>/harness.log. Progress of the swarm:
#   python3 scripts/admin/watch_run.py <server kit>/startup/nohup.out --follow
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONF=""
KIT_VERSION="${MEDISWARM_IMAGE_VERSION:-}"
IMAGE="${MEDISWARM_TEST_IMAGE:-}"
MODEL="MST"
JOB="ODELIA_ternary_classification"
TIMEOUT=150
EVAL=false
TAG="testbed"
KEEP_MIRROR=false
NO_PULL=false
FOREGROUND=false

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | sed '$d; s/^# \{0,1\}//'
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conf)         CONF="$2"; shift ;;
        --kit-version)  KIT_VERSION="$2"; shift ;;
        --image)        IMAGE="$2"; shift ;;
        --model)        MODEL="$2"; shift ;;
        --job)          JOB="$2"; shift ;;
        --timeout)      TIMEOUT="$2"; shift ;;
        --eval)         EVAL=true ;;
        --tag)          TAG="$2"; shift ;;
        --keep-mirror)  KEEP_MIRROR=true ;;
        --no-pull)      NO_PULL=true ;;
        --foreground)   FOREGROUND=true ;;
        -h|--help)      usage; exit 0 ;;
        *)              usage >&2; die "unknown argument: $1" ;;
    esac
    shift
done

[[ -n "$CONF" ]] || die "--conf is required"
[[ -f "$CONF" ]] || { [[ -f "$REPO_ROOT/$CONF" ]] && CONF="$REPO_ROOT/$CONF"; } || die "conf not found: $CONF"
CONF="$(cd "$(dirname "$CONF")" && pwd)/$(basename "$CONF")"
[[ -n "$KIT_VERSION" ]] || die "--kit-version is required (the version the test kits were built as)"
[[ "$KIT_VERSION" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid --kit-version '$KIT_VERSION'"
[[ -z "$IMAGE" || "$IMAGE" =~ ^[A-Za-z0-9._/:@-]+$ ]] || die "invalid --image '$IMAGE'"
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || die "--timeout must be minutes"
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid --tag '$TAG'"
[[ "$MODEL" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid --model '$MODEL'"

mode_of() { stat -c '%a' "$1"; }
private_file() {
    # True when neither group nor others have any permission on the file.
    local mode
    mode=$(mode_of "$1") || return 1
    (( (8#$mode & 8#077) == 0 ))
}

# Read the conf's credential settings in a subshell (the harness sources it too).
# Only variables the conf itself sets or changes count: an exported DL_SWARM_PASS
# in this environment is the recommended setup, not a password in the conf.
# shellcheck disable=SC1090
mapfile -t CRED_LINES < <(
    set +u
    declare -A before=()
    for name in $(compgen -v | grep -E '_PASS(_ENV|_FILE)?$' || true); do
        before[$name]="${!name}"
    done
    source "$CONF" >/dev/null 2>&1
    for name in $(compgen -v | grep -E '_PASS(_ENV|_FILE)?$' || true); do
        if [[ "${before[$name]+set}" == set && "${before[$name]}" == "${!name}" ]]; then
            continue
        fi
        printf '%s=%s\n' "$name" "${!name}"
    done
    printf 'PROJECT_FILE=%s\n' "${PROJECT_FILE:-}"
)

PROJECT_FILE=""
plaintext=()
for line in "${CRED_LINES[@]}"; do
    name="${line%%=*}"
    value="${line#*=}"
    case "$name" in
        PROJECT_FILE)
            PROJECT_FILE="$value" ;;
        *_PASS_ENV)
            [[ -n "$value" ]] || continue
            [[ -n "${!value-}" ]] || die "$name names \$$value, which is not set in this environment"
            ;;
        *_PASS_FILE)
            [[ -n "$value" ]] || continue
            [[ -r "$value" ]] || die "$name: cannot read $value"
            private_file "$value" || die "$name: $value is readable by other users; chmod 600 it"
            ;;
        *_PASS)
            [[ -n "$value" ]] && plaintext+=("$name")
            ;;
    esac
done
if [[ ${#plaintext[@]} -gt 0 ]] && ! private_file "$CONF"; then
    die "$CONF holds passwords (${plaintext[*]}) and is readable by other users: chmod 600 it, or use *_PASS_ENV / *_PASS_FILE"
fi

[[ -n "$PROJECT_FILE" ]] || die "the conf sets no PROJECT_FILE"
[[ -f "$REPO_ROOT/$PROJECT_FILE" ]] || die "project file not found: $REPO_ROOT/$PROJECT_FILE"
PROJECT_NAME=$(grep "^name: " "$REPO_ROOT/$PROJECT_FILE" | sed 's/^name: //' \
    | sed "s/__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__/$KIT_VERSION/")
KIT_WORKSPACE="$REPO_ROOT/workspace/$PROJECT_NAME"
[[ -d "$KIT_WORKSPACE" ]] || die "no kits for $KIT_VERSION: $KIT_WORKSPACE (build them with scripts/build/_buildStartupKits.sh)"

HARNESS_SHA=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
HARNESS_DIRTY=false
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no 2>/dev/null || true)" ]]; then
    HARNESS_DIRTY=true
    echo "[WARN] the harness checkout has uncommitted changes; they are part of this test" >&2
fi

RES_REL="workspace/deploy_test_results/${TAG}_$(date -u +%y%m%d_%H%M%S)"
RES="$REPO_ROOT/$RES_REL"
mkdir -p "$(dirname "$RES")"
mkdir "$RES" || die "results dir exists: $RES"

args=(--model "$MODEL" --job "$JOB" --conf "$CONF" --skip-build --timeout "$TIMEOUT" --results-dir "$RES")
[[ "$EVAL" == true ]] || args+=(--skip-eval)
[[ -n "$IMAGE" ]] && args+=(--image "$IMAGE")
[[ "$NO_PULL" == true ]] && args+=(--no-pull)

jq -n \
    --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg host "$(hostname)" \
    --arg harness_commit "$HARNESS_SHA" \
    --argjson harness_dirty "$HARNESS_DIRTY" \
    --arg kit_version "$KIT_VERSION" \
    --arg kit_workspace "$KIT_WORKSPACE" \
    --arg image "${IMAGE:-jefftud/odelia:$KIT_VERSION}" \
    --arg model "$MODEL" \
    --arg job "$JOB" \
    --arg conf "$CONF" \
    --argjson timeout_min "$TIMEOUT" \
    --argjson eval "$EVAL" \
    --argjson keep_mirror "$KEEP_MIRROR" \
    --arg command "scripts/deploy/run_deploy_test.sh ${args[*]}" \
    '$ARGS.named' > "$RES/launch.json"

export MEDISWARM_IMAGE_VERSION="$KIT_VERSION"
if [[ "$KEEP_MIRROR" == true ]]; then
    export DEPLOY_TEST_KEEP_MIRROR=1
fi

cd "$REPO_ROOT"
if [[ "$FOREGROUND" == true ]]; then
    set +e
    ./scripts/deploy/run_deploy_test.sh "${args[@]}" 2>&1 | tee "$RES/harness.log"
    status=${PIPESTATUS[0]}
    echo "results: $RES" >&2
    exit "$status"
fi

nohup ./scripts/deploy/run_deploy_test.sh "${args[@]}" > "$RES/harness.log" 2>&1 &
echo "$! $RES"
