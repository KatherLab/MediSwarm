#!/usr/bin/env bash
#
# One-command external validation for a predict-only node (#412).
#
# A center can validate a released ODELIA global model on its OWN data without
# joining the swarm: no VPN, no aggregation, no network. This wrapper just chains the
# two kit commands a site would otherwise run by hand --
#
#   1. docker.sh --preflight_check       (the same data-integrity/GPU guard that caught
#                                          CAM's corrupt inputs -- reject a bad dataset
#                                          BEFORE predicting)
#   2. docker.sh --external_validation   (run predict.py on the local test/ext split)
#
# Metrics and per-sample predictions are written under your scratch dir and stay on the
# node. Only the aggregate metrics in prediction_results.json need to be shared back.
#
# Usage:
#   run_external_validation.sh \
#       --data_dir    /path/to/your/data \
#       --scratch_dir /path/to/scratch \
#       --model_name  <released model name> \
#       [--checkpoint /scratch/FL_global_model.pt]   # container path; default shown
#       [--split test] [--GPU device=0] [--kit_dir /path/to/kit/startup]
#
# Put the delivered global model where --checkpoint points (default: drop
# FL_global_model.pt into your scratch dir, which is mounted at /scratch).

set -euo pipefail

DATA_DIR="" SCRATCH_DIR="" MODEL_NAME="" CHECKPOINT="" SPLIT="test"
GPU="device=0" KIT_DIR=""
SKIP_PREFLIGHT=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir)       DATA_DIR="$2"; shift ;;
        --scratch_dir)    SCRATCH_DIR="$2"; shift ;;
        --model_name)     MODEL_NAME="$2"; shift ;;
        --checkpoint)     CHECKPOINT="$2"; shift ;;
        --split)          SPLIT="$2"; shift ;;
        --GPU)            GPU="$2"; shift ;;
        --kit_dir)        KIT_DIR="$2"; shift ;;
        --skip_preflight) SKIP_PREFLIGHT="1" ;;
        -h|--help)        grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
    shift
done

# Locate docker.sh: --kit_dir wins, else next to this script (a site can copy this
# wrapper into its kit's startup/ dir), else the current directory.
SELF_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
for cand in "$KIT_DIR/docker.sh" "$SELF_DIR/docker.sh" "./docker.sh"; do
    [ -n "$cand" ] && [ -f "$cand" ] && { DOCKER_SH="$cand"; break; }
done
: "${DOCKER_SH:?could not find docker.sh -- pass --kit_dir <kit>/startup}"

for req in DATA_DIR SCRATCH_DIR MODEL_NAME; do
    [ -n "${!req}" ] || { echo "--${req,,} is required" >&2; exit 1; }
done

echo "== External validation =="
echo "   docker.sh:   $DOCKER_SH"
echo "   model:       $MODEL_NAME    split: $SPLIT"
echo "   data (ro):   $DATA_DIR"
echo "   scratch:     $SCRATCH_DIR"

if [ -z "$SKIP_PREFLIGHT" ]; then
    echo "-- Step 1/2: preflight (data-integrity + GPU) --"
    "$DOCKER_SH" --data_dir "$DATA_DIR" --scratch_dir "$SCRATCH_DIR" --GPU "$GPU" \
        --model_name "$MODEL_NAME" --preflight_check
fi

echo "-- Step 2/2: predict-only external validation --"
EV_ARGS=(--data_dir "$DATA_DIR" --scratch_dir "$SCRATCH_DIR" --GPU "$GPU"
         --model_name "$MODEL_NAME" --split "$SPLIT" --external_validation)
[ -n "$CHECKPOINT" ] && EV_ARGS+=(--checkpoint "$CHECKPOINT")
"$DOCKER_SH" "${EV_ARGS[@]}"

echo
echo "== Done. Results are LOCAL, under: $SCRATCH_DIR =="
echo "   prediction_results.json   <- aggregate metrics (the only file to share back)"
echo "   predictions_*.csv         <- per-sample predictions (keep on the node)"
