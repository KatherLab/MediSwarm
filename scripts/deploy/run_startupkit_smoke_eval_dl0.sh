#!/usr/bin/env bash
# run_startupkit_smoke_eval_dl0.sh — Validate collected startup-kit smoke
# checkpoints and run post-training prediction smoke tests on dl0.
#
# Expected input:
#   <checkpoint-root>/<MODEL>_checkpoints/app_<SITE>/FL_global_model.pt
#
# Typical use:
#   scripts/deploy/run_startupkit_smoke_eval_dl0.sh \
#       --conf deploy_sites_duke_iid.conf \
#       --checkpoint-root workspace/deploy_test_results/startupkit_smoke_<ts>/training \
#       --output-dir workspace/deploy_test_results/startupkit_smoke_<ts>/eval

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

CONF_FILE="$REPO_ROOT/deploy_sites_duke_iid.conf"
CHECKPOINT_ROOT="$REPO_ROOT/workspace/deploy_test_results"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$REPO_ROOT/workspace/deploy_test_results/startupkit_smoke_${RUN_ID}/eval"
DOCKER_IMAGE=""
REMOTE_STAGE=""
MODELS_CSV=""
ODELIA_SITES_CSV="CAM,MHA,RSH,RUMC,UKA,UMCU"
EXPECTED_CHECKPOINTS=3

usage() {
    cat <<EOF
Usage: $0 --conf FILE --checkpoint-root DIR [options]

Options:
  --conf FILE              Deploy-site configuration file (default: deploy_sites_duke_iid.conf)
  --checkpoint-root DIR    Directory containing <MODEL>_checkpoints dirs
  --output-dir DIR         Local output directory for reports/predictions
  --image IMAGE            Docker image to run on dl0 (default: jefftud/odelia:<current version>)
  --remote-stage DIR       dl0 staging dir (default: <DL0_SCRATCHDIR>/startupkit_smoke_eval/<run_id>)
  --models CSV             Models to evaluate (default: all six deploy-test models)
  --odelia-sites CSV       ODELIA institutions (default: CAM,MHA,RSH,RUMC,UKA,UMCU)
  --expected-checkpoints N Expected number of collected client checkpoints (default: 3)
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conf)            CONF_FILE="$2"; shift ;;
        --checkpoint-root) CHECKPOINT_ROOT="$2"; shift ;;
        --output-dir)      OUTPUT_DIR="$2"; shift ;;
        --image)           DOCKER_IMAGE="$2"; shift ;;
        --remote-stage)    REMOTE_STAGE="$2"; shift ;;
        --models)          MODELS_CSV="$2"; shift ;;
        --odelia-sites)    ODELIA_SITES_CSV="$2"; shift ;;
        --expected-checkpoints) EXPECTED_CHECKPOINTS="$2"; shift ;;
        -h|--help)         usage; exit 0 ;;
        *)                 err "Unknown argument: $1"; usage; exit 1 ;;
    esac
    shift
done

resolve_path() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        printf '%s\n' "$p"
    else
        printf '%s/%s\n' "$REPO_ROOT" "$p"
    fi
}

CONF_FILE="$(resolve_path "$CONF_FILE")"
CHECKPOINT_ROOT="$(resolve_path "$CHECKPOINT_ROOT")"
OUTPUT_DIR="$(resolve_path "$OUTPUT_DIR")"

if [[ ! -f "$CONF_FILE" ]]; then
    err "Configuration file not found: $CONF_FILE"
    exit 1
fi

# shellcheck source=/dev/null
source "$CONF_FILE"

VERSION="$("$REPO_ROOT/scripts/build/getVersionNumber.sh")"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
DOCKER_IMAGE="${DOCKER_IMAGE:-jefftud/odelia:$VERSION}"
DL0_SCRATCHDIR="${DL0_SCRATCHDIR:-/mnt/dlhd0/deploy_test_duke_iid}"
REMOTE_STAGE="${REMOTE_STAGE:-$DL0_SCRATCHDIR/startupkit_smoke_eval/$RUN_ID}"

IFS=',' read -r -a MODELS <<< "${MODELS_CSV:-MST,1DivideAndConquer,2BCN_AIM,3agaldran,4LME_ABMIL,5Pimed}"
IFS=',' read -r -a ODELIA_SITES <<< "$ODELIA_SITES_CSV"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
DL0_HOST="${DL0_HOST:?DL0_HOST must be set in $CONF_FILE}"
DL0_USER="${DL0_USER:?DL0_USER must be set in $CONF_FILE}"
DL0_PASS="${DL0_PASS:-}"
DL0_PASS_ENV="${DL0_PASS_ENV:-}"
if [[ -z "$DL0_PASS" && -n "$DL0_PASS_ENV" && -n "${!DL0_PASS_ENV-}" ]]; then
    DL0_PASS="${!DL0_PASS_ENV}"
fi
DL0_DUKE_DATA="${DL0_DUKE_DATA:-/mnt/dlhd0/DUKE_iid}"
DL0_ODELIA_DATA="${DL0_ODELIA_DATA:-/mnt/dlhd0/medswarmdata}"
DL0_GPU="${DL0_GPU:-device=0}"

mkdir -p "$OUTPUT_DIR"
: > "$OUTPUT_DIR/model_results.jsonl"

remote_exec() {
    if [[ -n "$DL0_PASS" ]]; then
        sshpass -p "$DL0_PASS" ssh $SSH_OPTS "$DL0_USER@$DL0_HOST" "$@"
    else
        ssh $SSH_OPTS "$DL0_USER@$DL0_HOST" "$@"
    fi
}

remote_copy_to() {
    local src="$1" dst="$2"
    if [[ -n "$DL0_PASS" ]]; then
        sshpass -p "$DL0_PASS" scp $SSH_OPTS "$src" "$DL0_USER@$DL0_HOST:$dst"
    else
        scp $SSH_OPTS "$src" "$DL0_USER@$DL0_HOST:$dst"
    fi
}

remote_copy_from_dir() {
    local remote_dir="$1" local_dir="$2"
    mkdir -p "$local_dir"
    if [[ -n "$DL0_PASS" ]]; then
        sshpass -p "$DL0_PASS" scp -r $SSH_OPTS "$DL0_USER@$DL0_HOST:$remote_dir/." "$local_dir/"
    else
        scp -r $SSH_OPTS "$DL0_USER@$DL0_HOST:$remote_dir/." "$local_dir/"
    fi
}

run_predict_on_dl0() {
    local model="$1"
    local dataset_name="$2"
    local site_name="$3"
    local data_dir="$4"
    local remote_ckpt_dir="$5"
    local model_remote_out="$REMOTE_STAGE/outputs/$model/$dataset_name"
    local model_local_out="$OUTPUT_DIR/$model/$dataset_name"
    local stdout_log="$model_local_out/predict_stdout.log"

    mkdir -p "$model_local_out"
    remote_exec "rm -rf '$model_remote_out' && mkdir -p '$model_remote_out'"

    info "Evaluating $model on $dataset_name (SITE_NAME=$site_name, DATA_DIR=$data_dir)"
    if remote_exec "
        docker run --rm \
            --gpus='$DL0_GPU' \
            --ulimit nofile=65536:65536 \
            --net=host --ipc=host \
            -v '$data_dir:/data:ro' \
            -v '$remote_ckpt_dir:/workspace:ro' \
            -v '$model_remote_out:/output' \
            --env SITE_NAME='$site_name' \
            --env DATA_DIR=/data \
            --env SCRATCH_DIR=/output \
            --env MODEL_NAME='$model' \
            --env TORCH_HOME=/torch_home \
            --env CONFIG=unilateral \
            --env TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_system \
            '$DOCKER_IMAGE' \
            python3 /MediSwarm/scripts/evaluation/predict.py \
                --checkpoint /workspace/FL_global_model.pt \
                --model-name '$model' \
                --output-dir /output \
                --split test
    " > "$stdout_log" 2>&1; then
        remote_copy_from_dir "$model_remote_out" "$model_local_out"
        if [[ -s "$model_local_out/prediction_results.json" ]] && compgen -G "$model_local_out/predictions_*.csv" >/dev/null; then
            ok "Prediction smoke passed: $model / $dataset_name"
            return 0
        fi
        err "Prediction outputs missing for $model / $dataset_name"
        return 1
    fi

    remote_copy_from_dir "$model_remote_out" "$model_local_out" 2>/dev/null || true
    err "Prediction command failed for $model / $dataset_name (see $stdout_log)"
    return 1
}

step "Preflight on dl0"
remote_exec "command -v docker >/dev/null && docker image inspect '$DOCKER_IMAGE' >/dev/null && nvidia-smi >/dev/null && test -d '$DL0_DUKE_DATA' && test -d '$DL0_ODELIA_DATA'"
ok "dl0 has Docker image, GPU, Duke data, and ODELIA data"

overall_status="pass"

for model in "${MODELS[@]}"; do
    step "Smoke evaluation: $model"
    model_status="pass"
    checkpoint_status="pass"
    ckpts=()
    eval_results_jsonl="$OUTPUT_DIR/${model}_eval_results.jsonl"
    : > "$eval_results_jsonl"

    checkpoint_dir="$CHECKPOINT_ROOT/${model}_checkpoints"
    model_out="$OUTPUT_DIR/$model"
    mkdir -p "$model_out"

    if [[ ! -d "$checkpoint_dir" ]]; then
        err "Checkpoint directory missing: $checkpoint_dir"
        checkpoint_status="fail"
        model_status="fail"
    else
        mapfile -t ckpts < <(find "$checkpoint_dir" -path '*/app_*/FL_global_model.pt' -type f | sort)
        if [[ "${#ckpts[@]}" -ne "$EXPECTED_CHECKPOINTS" ]]; then
            err "Expected exactly $EXPECTED_CHECKPOINTS FL_global_model.pt files for $model, found ${#ckpts[@]}"
            checkpoint_status="fail"
            model_status="fail"
        else
            sha224sum "${ckpts[@]}" | tee "$model_out/checkpoint_hash.txt" >/dev/null
            unique_hashes="$(awk '{print $1}' "$model_out/checkpoint_hash.txt" | sort -u | wc -l | tr -d ' ')"
            if [[ "$unique_hashes" != "1" ]]; then
                err "Final global checkpoints are not byte-identical for $model"
                checkpoint_status="fail"
                model_status="fail"
            else
                ok "All $EXPECTED_CHECKPOINTS final global checkpoints are byte-identical for $model"
            fi
        fi
    fi

    if [[ "$checkpoint_status" == "pass" ]]; then
        remote_ckpt_dir="$REMOTE_STAGE/checkpoints/$model"
        remote_exec "rm -rf '$remote_ckpt_dir' && mkdir -p '$remote_ckpt_dir'"
        remote_copy_to "${ckpts[0]}" "$remote_ckpt_dir/FL_global_model.pt"

        if run_predict_on_dl0 "$model" "duke_test" "test" "$DL0_DUKE_DATA" "$remote_ckpt_dir"; then
            jq -n --arg dataset "duke_test" --arg status "pass" '{dataset:$dataset,status:$status}' >> "$eval_results_jsonl"
        else
            jq -n --arg dataset "duke_test" --arg status "fail" '{dataset:$dataset,status:$status}' >> "$eval_results_jsonl"
            model_status="fail"
        fi

        for inst in "${ODELIA_SITES[@]}"; do
            dataset="odelia_$inst"
            if run_predict_on_dl0 "$model" "$dataset" "$inst" "$DL0_ODELIA_DATA" "$remote_ckpt_dir"; then
                jq -n --arg dataset "$dataset" --arg status "pass" '{dataset:$dataset,status:$status}' >> "$eval_results_jsonl"
            else
                jq -n --arg dataset "$dataset" --arg status "fail" '{dataset:$dataset,status:$status}' >> "$eval_results_jsonl"
                model_status="fail"
            fi
        done
    fi

    evals_json="$(jq -s '.' "$eval_results_jsonl")"
    hash_json="[]"
    if [[ -f "$model_out/checkpoint_hash.txt" ]]; then
        hash_json="$(jq -R 'split("  ") | {hash:.[0], path:.[1]}' "$model_out/checkpoint_hash.txt" | jq -s '.')"
    fi

    checkpoint_count="${#ckpts[@]}"
    jq -n \
        --arg model "$model" \
        --arg status "$model_status" \
        --arg checkpoint_status "$checkpoint_status" \
        --argjson checkpoint_count "$checkpoint_count" \
        --argjson hash "$hash_json" \
        --argjson evals "$evals_json" \
        '{model:$model,status:$status,checkpoint_status:$checkpoint_status,checkpoint_count:$checkpoint_count,hash:$hash,evals:$evals}' \
        >> "$OUTPUT_DIR/model_results.jsonl"

    if [[ "$model_status" != "pass" ]]; then
        overall_status="fail"
    fi
done

jq -s \
    --arg run_id "$RUN_ID" \
    --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    --arg git_sha "$GIT_SHA" \
    --arg docker_image "$DOCKER_IMAGE" \
    --arg conf_file "$CONF_FILE" \
    --arg checkpoint_root "$CHECKPOINT_ROOT" \
    --argjson expected_checkpoints "$EXPECTED_CHECKPOINTS" \
    --arg output_dir "$OUTPUT_DIR" \
    --arg remote_stage "$REMOTE_STAGE" \
    --arg status "$overall_status" \
    '{
        run_id:$run_id,
        timestamp:$timestamp,
        status:$status,
        git_sha:$git_sha,
        docker_image:$docker_image,
        conf_file:$conf_file,
        checkpoint_root:$checkpoint_root,
        expected_checkpoints:$expected_checkpoints,
        output_dir:$output_dir,
        remote_stage:$remote_stage,
        models:.
    }' "$OUTPUT_DIR/model_results.jsonl" > "$OUTPUT_DIR/summary.json"

{
    echo "# Duke IID Startup Kit Smoke Evaluation"
    echo ""
    echo "- Status: \`$overall_status\`"
    echo "- Git SHA: \`$GIT_SHA\`"
    echo "- Docker image: \`$DOCKER_IMAGE\`"
    echo "- Checkpoint root: \`$CHECKPOINT_ROOT\`"
    echo "- dl0 remote stage: \`$REMOTE_STAGE\`"
    echo ""
    echo "| Model | Status | Checkpoints | Eval datasets |"
    echo "|-------|--------|-------------|---------------|"
    jq -r '.models[] | [.model, .status, (.checkpoint_status + " (" + (.checkpoint_count|tostring) + ")"), (([.evals[]? | select(.status=="pass")] | length | tostring) + "/" + ((.evals|length)|tostring))] | @tsv' "$OUTPUT_DIR/summary.json" \
        | while IFS=$'\t' read -r model status checkpoints evals; do
            echo "| $model | $status | $checkpoints | $evals |"
          done
    echo ""
    echo "Prediction outputs are under per-model dataset directories in this folder."
} > "$OUTPUT_DIR/REPORT.md"

if [[ "$overall_status" == "pass" ]]; then
    ok "Startup-kit smoke evaluation passed. Report: $OUTPUT_DIR/REPORT.md"
    exit 0
fi

err "Startup-kit smoke evaluation failed. Report: $OUTPUT_DIR/REPORT.md"
exit 1
