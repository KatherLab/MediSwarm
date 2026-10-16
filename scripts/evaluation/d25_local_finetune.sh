#!/usr/bin/env bash
# D2.5 regional fine-tuning for a cohort of one site (the Greek cohort, MHA_1).
#
# A swarm job needs at least two sites, so a single-site cohort fine-tunes the August pan-European
# model by local training and compares it with the unchanged model on its own test split, with the
# same comparison code as the Dutch swarm run (compare_regional_finetune.py). Runs inside the MediSwarm
# image (1.9.3 or later); the site starts it with
#
#   ./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 \
#       --run_script /MediSwarm/scripts/evaluation/d25_local_finetune.sh
#
# Needs the start model in the scratch directory (default /scratch/odelia_87c5bbee_global.pt). Steps:
# predict with the start model, fine-tune it locally, predict with the fine-tuned model, compare.
# Everything stays under /scratch/d25_local/<UTC time>/; only report.json (AUROCs, bootstrap intervals,
# support counts; no case identifiers) is meant to be sent to TUD.
#
# Settings (environment, all optional):
#   D25_START_MODEL   container path of the start model   (default /scratch/odelia_87c5bbee_global.pt)
#   D25_START_SHA256  its sha256                           (default: the August model, job 87c5bbee)
#   D25_EPOCHS        local fine-tuning epochs             (default 10: what MHA_1 trains in 5 swarm rounds)
#   D25_LABEL         cohort name in the report            (default "Greek cohort")
set -euo pipefail

START_MODEL="${D25_START_MODEL:-/scratch/odelia_87c5bbee_global.pt}"
START_SHA256="${D25_START_SHA256:-ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d}"
EPOCHS="${D25_EPOCHS:-10}"
LABEL="${D25_LABEL:-Greek cohort}"
MODEL=1DivideAndConquer
MAIN=/MediSwarm/application/jobs/challenge_1DivideAndConquer/app/custom/main.py
EVAL=/MediSwarm/scripts/evaluation
SITE="${INSTITUTION:-${SITE_NAME:?SITE_NAME is not set: start this script with docker.sh --run_script}}"
OUT="/scratch/d25_local/$(date -u +%Y%m%d_%H%M%S)"

step() { echo; echo "=== D2.5 local fine-tuning, $SITE: $* ($(date -u +%H:%M:%S) UTC)"; }

step "checking the start model"
[ -f "$START_MODEL" ] || { echo "[FAIL] $START_MODEL not found. Copy the start model into your scratch directory first."; exit 1; }
actual="$(sha256sum "$START_MODEL" | cut -d' ' -f1)"
[ "$actual" = "$START_SHA256" ] || { echo "[FAIL] $START_MODEL has sha256 $actual, expected $START_SHA256 (another or an incomplete file)."; exit 1; }
mkdir -p "$OUT/baseline/$SITE" "$OUT/finetuned/$SITE"
echo "[OK] start model verified; results go to $OUT"

step "1/4 predicting the test split with the start model"
python3 "$EVAL/predict.py" --checkpoint "$START_MODEL" --model-name "$MODEL" --split test \
    --output-dir "$OUT/baseline/$SITE"

step "2/4 fine-tuning the start model for $EPOCHS epochs (local training)"
touch "$OUT/.started"
TRAINING_MODE=local_training MODEL_NAME="$MODEL" NUM_EPOCHS="$EPOCHS" \
    ODELIA_LOCAL_INIT_CHECKPOINT="$START_MODEL" ODELIA_LOCAL_INIT_SHA256="$START_SHA256" \
    python3 "$MAIN"
# The run directory of this training is the one main.py wrote local_init.json into after $OUT/.started.
RUN_DIR="$(find /scratch/runs -mindepth 3 -maxdepth 3 -name local_init.json -newer "$OUT/.started" -printf '%h\n' | sort | tail -1)"
[ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/last_global_model.ckpt" ] || {
    echo "[FAIL] no run directory with local_init.json and last_global_model.ckpt under /scratch/runs from this training"; exit 1; }
echo "[OK] fine-tuned model: $RUN_DIR/last_global_model.ckpt"

step "3/4 predicting the test split with the fine-tuned model"
python3 "$EVAL/predict.py" --checkpoint "$RUN_DIR/last_global_model.ckpt" --model-name "$MODEL" --split test \
    --output-dir "$OUT/finetuned/$SITE"

step "4/4 comparing the two on the same cases"
python3 "$EVAL/compare_regional_finetune.py" --baseline "$OUT/baseline" --finetuned "$OUT/finetuned" \
    --sites "$SITE" --label "$LABEL" --json-out "$OUT/report.json"

echo
echo "[OK] done. Please send $OUT/report.json to TUD (aggregate numbers only; the per-case files stay here)."
echo "     On the host it is under your scratch directory: d25_local/$(basename "$OUT")/report.json"
