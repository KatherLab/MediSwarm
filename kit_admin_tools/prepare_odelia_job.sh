#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./prepare_odelia_job.sh --job JOB_NAME --warm-start fresh|continue [--num-rounds N] [--min-clients N] [--configure-min-clients N] [--min-responses N] [--strict-clients CLIENT_1,CLIENT_2,...] [--broadcast-last-result true|false] [--fold N]

Client policy:
  With no --min-* override, the current exact eight-site ODELIA production roster
  is required automatically. Use --strict-clients to supply another exact roster.
  Supplying any --min-* option opts into an explicit non-default/test policy.

Examples:
  ./prepare_odelia_job.sh --job ODELIA_ternary_classification --warm-start fresh
  ./prepare_odelia_job.sh --job ODELIA_ternary_classification --warm-start continue
  ./prepare_odelia_job.sh --job challenge_1DivideAndConquer --warm-start continue --strict-clients CAM_1,MHA_1,RSH_1,RUMC_1,UKA_1,UMCU_1,USZ_1,VHIO_1
EOF
}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_STRICT_CLIENTS="CAM_1,VHIO_1,USZ_1,RUMC_1,MHA_1,RSH_1,UMCU_1,UKA_1"
JOB_NAME=""
WARM_START=""
FOLD=""
OUTPUT_DIR="$DIR/../local/mediswarm_jobs"
NUM_ROUNDS=""
MIN_CLIENTS=""
CONFIGURE_MIN_CLIENTS=""
MIN_RESPONSES=""
STRICT_CLIENTS="$DEFAULT_STRICT_CLIENTS"
STRICT_CLIENTS_SET=true
STRICT_CLIENTS_EXPLICIT=false
CUSTOM_CLIENT_COUNTS_SET=false
BROADCAST_LAST_RESULT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --job)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --job" >&2
        exit 2
      fi
      JOB_NAME="${2:-}"
      shift 2
      ;;
    --warm-start)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --warm-start" >&2
        exit 2
      fi
      WARM_START="${2:-}"
      shift 2
      ;;
    --output-dir)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --output-dir" >&2
        exit 2
      fi
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --fold)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --fold" >&2
        exit 2
      fi
      FOLD="${2:-}"
      shift 2
      ;;
    --num-rounds)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --num-rounds" >&2
        exit 2
      fi
      NUM_ROUNDS="${2:-}"
      shift 2
      ;;
    --min-clients)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --min-clients" >&2
        exit 2
      fi
      MIN_CLIENTS="${2:-}"
      CUSTOM_CLIENT_COUNTS_SET=true
      shift 2
      ;;
    --configure-min-clients)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --configure-min-clients" >&2
        exit 2
      fi
      CONFIGURE_MIN_CLIENTS="${2:-}"
      CUSTOM_CLIENT_COUNTS_SET=true
      shift 2
      ;;
    --min-responses)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --min-responses" >&2
        exit 2
      fi
      MIN_RESPONSES="${2:-}"
      CUSTOM_CLIENT_COUNTS_SET=true
      shift 2
      ;;
    --strict-clients)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --strict-clients" >&2
        exit 2
      fi
      STRICT_CLIENTS="${2:-}"
      STRICT_CLIENTS_SET=true
      STRICT_CLIENTS_EXPLICIT=true
      shift 2
      ;;
    --broadcast-last-result)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --broadcast-last-result" >&2
        exit 2
      fi
      BROADCAST_LAST_RESULT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Bare production preparation is strict by default. Existing validation tools
# that deliberately provide custom quorum values keep their explicit policy.
if [ "$STRICT_CLIENTS_EXPLICIT" != true ] && [ "$CUSTOM_CLIENT_COUNTS_SET" = true ]; then
  STRICT_CLIENTS=""
  STRICT_CLIENTS_SET=false
fi

if [ -z "$JOB_NAME" ] || [ -z "$WARM_START" ]; then
  usage >&2
  exit 2
fi

if [[ ! "$JOB_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid job name: $JOB_NAME" >&2
  echo "Use the job directory name only, for example ODELIA_ternary_classification." >&2
  exit 2
fi

case "$WARM_START" in
  fresh)
    CONFIG_MODE="fresh"
    ;;
  continue)
    CONFIG_MODE="require"
    ;;
  *)
    echo "Invalid --warm-start value: $WARM_START" >&2
    echo "Expected fresh or continue." >&2
    exit 2
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to copy the job from the MediSwarm image." >&2
  exit 1
fi

DOCKER_SH="$DIR/docker.sh"
if [ ! -f "$DOCKER_SH" ]; then
  echo "Missing startup docker.sh next to this helper: $DOCKER_SH" >&2
  exit 1
fi

# Keep the helper and patcher version together. Provisioned admin kits place
# the patcher next to this script; a repository checkout uses the source copy.
PATCHER_HOST="$DIR/patch_warm_start_job.py"
if [ ! -f "$PATCHER_HOST" ]; then
  PATCHER_HOST="$DIR/../scripts/admin/patch_warm_start_job.py"
fi
if [ ! -f "$PATCHER_HOST" ]; then
  echo "Missing job patcher next to the helper or in scripts/admin: $PATCHER_HOST" >&2
  exit 1
fi
PATCHER_HOST="$(cd "$(dirname "$PATCHER_HOST")" && pwd -P)/$(basename "$PATCHER_HOST")"

DOCKER_IMAGE="$(awk -F= '/^[[:space:]]*DOCKER_IMAGE=/{print $2; exit}' "$DOCKER_SH" | tr -d '[:space:]' | tr -d '"')"
if [ -z "$DOCKER_IMAGE" ]; then
  echo "Could not read DOCKER_IMAGE from $DOCKER_SH" >&2
  exit 1
fi

if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "Docker image is not available locally: $DOCKER_IMAGE" >&2
  echo "Pull or build the image, then rerun this helper." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_ROOT="$(cd "$OUTPUT_DIR" && pwd -P)"
DEST_NAME="${JOB_NAME}_${WARM_START}"
if [ -n "$FOLD" ]; then
  DEST_NAME="${DEST_NAME}_fold${FOLD}"
fi
DEST_HOST="$OUTPUT_ROOT/$DEST_NAME"
TEMP_NAME=".${DEST_NAME}.prepare.$$"
TEMP_HOST="$OUTPUT_ROOT/$TEMP_NAME"
if [ -e "$TEMP_HOST" ]; then
  echo "Refusing to overwrite unexpected staging path: $TEMP_HOST" >&2
  exit 1
fi
JOB_SRC="/MediSwarm/application/jobs/$JOB_NAME"
PATCH_ARGS=(--job-dir "/job_out/$TEMP_NAME" --mode "$CONFIG_MODE")
if [ -n "$FOLD" ]; then
  PATCH_ARGS+=(--fold "$FOLD")
fi
if [ -n "$NUM_ROUNDS" ]; then
  PATCH_ARGS+=(--num-rounds "$NUM_ROUNDS")
fi
if [ -n "$MIN_CLIENTS" ]; then
  PATCH_ARGS+=(--min-clients "$MIN_CLIENTS")
fi
if [ -n "$CONFIGURE_MIN_CLIENTS" ]; then
  PATCH_ARGS+=(--configure-min-clients "$CONFIGURE_MIN_CLIENTS")
fi
if [ -n "$MIN_RESPONSES" ]; then
  PATCH_ARGS+=(--min-responses "$MIN_RESPONSES")
fi
if [ "$STRICT_CLIENTS_SET" = true ]; then
  PATCH_ARGS+=(--strict-clients "$STRICT_CLIENTS")
fi
if [ -n "$BROADCAST_LAST_RESULT" ]; then
  PATCH_ARGS+=(--broadcast-last-result "$BROADCAST_LAST_RESULT")
fi

printf -v PATCH_ARGS_QUOTED ' %q' "${PATCH_ARGS[@]}"

cleanup_staging() {
  if [ -e "$TEMP_HOST" ]; then
    rm -rf -- "$TEMP_HOST"
  fi
}
trap cleanup_staging EXIT

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$OUTPUT_ROOT":/job_out \
  -v "$PATCHER_HOST":/mediswarm_tools/patch_warm_start_job.py:ro \
  "$DOCKER_IMAGE" \
  bash -lc "set -euo pipefail; test -d '$JOB_SRC'; cp -R '$JOB_SRC' '/job_out/$TEMP_NAME'; python3 /mediswarm_tools/patch_warm_start_job.py$PATCH_ARGS_QUOTED"

# Preserve any previously prepared job until the new copy has passed every
# patch/validation step. If the final rename fails, restore the old copy.
BACKUP_HOST=""
if [ -e "$DEST_HOST" ]; then
  BACKUP_HOST="${DEST_HOST}.previous.$$"
  if [ -e "$BACKUP_HOST" ]; then
    echo "Refusing to overwrite unexpected backup path: $BACKUP_HOST" >&2
    exit 1
  fi
  mv -- "$DEST_HOST" "$BACKUP_HOST"
fi
if mv -- "$TEMP_HOST" "$DEST_HOST"; then
  trap - EXIT
  if [ -n "$BACKUP_HOST" ]; then
    rm -rf -- "$BACKUP_HOST"
  fi
else
  if [ -n "$BACKUP_HOST" ] && [ -e "$BACKUP_HOST" ]; then
    mv -- "$BACKUP_HOST" "$DEST_HOST"
  fi
  exit 1
fi

echo
echo "Prepared job: $DEST_HOST"
echo "Client config warm_start_mode: $CONFIG_MODE"
if [ -n "$NUM_ROUNDS" ]; then
  echo "Server config num_rounds: $NUM_ROUNDS"
fi
if [ -n "$MIN_CLIENTS" ]; then
  echo "Server config min_clients: $MIN_CLIENTS"
fi
if [ -n "$CONFIGURE_MIN_CLIENTS" ]; then
  echo "Server config configure_min_clients: $CONFIGURE_MIN_CLIENTS"
fi
if [ -n "$MIN_RESPONSES" ]; then
  echo "Client config min_responses_required: $MIN_RESPONSES"
fi
if [ "$STRICT_CLIENTS_SET" = true ]; then
  if [ "$STRICT_CLIENTS_EXPLICIT" = true ]; then
    echo "Strict required clients: $STRICT_CLIENTS"
  else
    echo "Strict required clients (default ODELIA production profile): $STRICT_CLIENTS"
  fi
fi
if [ -n "$BROADCAST_LAST_RESULT" ]; then
  echo "Client config broadcast_last_result: $BROADCAST_LAST_RESULT"
fi
echo
echo "Submit from the admin console:"
echo "  submit_job /fl_admin/local/mediswarm_jobs/$DEST_NAME"
