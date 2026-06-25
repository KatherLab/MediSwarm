#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./prepare_odelia_job.sh --job JOB_NAME --warm-start fresh|continue

Examples:
  ./prepare_odelia_job.sh --job ODELIA_ternary_classification --warm-start fresh
  ./prepare_odelia_job.sh --job ODELIA_ternary_classification --warm-start continue
EOF
}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
JOB_NAME=""
WARM_START=""
OUTPUT_DIR="$DIR/../local/mediswarm_jobs"

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
DEST_HOST="$OUTPUT_ROOT/$DEST_NAME"
JOB_SRC="/MediSwarm/application/jobs/$JOB_NAME"

rm -rf "$DEST_HOST"

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$OUTPUT_ROOT":/job_out \
  "$DOCKER_IMAGE" \
  bash -lc "set -euo pipefail; test -d '$JOB_SRC'; cp -R '$JOB_SRC' '/job_out/$DEST_NAME'; python3 /MediSwarm/scripts/admin/patch_warm_start_job.py --job-dir '/job_out/$DEST_NAME' --mode '$CONFIG_MODE'"

echo
echo "Prepared job: $DEST_HOST"
echo "Client config warm_start_mode: $CONFIG_MODE"
echo
echo "Submit from the admin console:"
echo "  submit_job /fl_admin/local/mediswarm_jobs/$DEST_NAME"
