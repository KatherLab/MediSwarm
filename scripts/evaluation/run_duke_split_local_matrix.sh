#!/usr/bin/env bash
# ============================================================================
# run_duke_split_local_matrix.sh
#
# Serial local-training matrix for the custom DUKE split on dd-dl0.
#
# Runs 6 models x 4 partitions:
#   node_A, node_B, node_C, node_all
#
# Workflow:
#   1. Ensure node_all exists in /mnt/dlhd0/DUKE_swarm_new
#   2. Verify remote prerequisites and dataset structure
#   3. Smoke test each model on node_A (preflight_check + local_training, 1 epoch)
#   4. Run the full matrix with held-out test evaluation on `test`
#   5. Write per-cell artifacts, summary CSV/JSON, and a markdown report
#
# Outputs:
#   - Local summary root: duke_results/duke_split_local_matrix_260417/
#   - Report: docs/DUKE_SPLIT_LOCAL_TRAINING_REPORT.md
#
# Notes:
#   - Uses direct `docker run`, not startup-kit docker.sh local_training mode,
#     because we must inject INSTITUTION=node_A/node_B/node_C/node_all.
#   - Reuses the already deployed image on dd-dl0:
#       jefftud/odelia:1.4.2-dev.260417.50e783b
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCAL_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$LOCAL_PYTHON" ]]; then
    LOCAL_PYTHON="python3"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[DUKE-MATRIX]${NC} $*"; }
ok()    { echo -e "${GREEN}[DUKE-MATRIX]${NC} $*"; }
warn()  { echo -e "${YELLOW}[DUKE-MATRIX]${NC} $*"; }
err()   { echo -e "${RED}[DUKE-MATRIX]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}═══ $* ═══${NC}"; }

REMOTE_HOST="${REMOTE_HOST:-dd-dl0}"
REMOTE_USER="${REMOTE_USER:-swarm}"
REMOTE_PASS="${REMOTE_PASS:?REMOTE_PASS must be set (export REMOTE_PASS=<password>)}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

SITE_NAME="${SITE_NAME:-RUMC_1}"
VERSION="${VERSION:-1.4.2-dev.260417.50e783b}"
IMAGE="${IMAGE:-jefftud/odelia:${VERSION}}"
GPU_SPEC="${GPU_SPEC:-device=0}"

REMOTE_DATASET_ROOT="${REMOTE_DATASET_ROOT:-/mnt/dlhd0/DUKE_swarm_new}"
REMOTE_NODE_ALL_METADATA_SOURCE="${REMOTE_NODE_ALL_METADATA_SOURCE:-/mnt/dlhd0/DUKE_swarm/node_all/metadata_unilateral}"
REMOTE_DUKE_DATA_ROOT="${REMOTE_DUKE_DATA_ROOT:-/mnt/dlhd0/DUKE/data_unilateral}"
REMOTE_SCRATCH_ROOT="${REMOTE_SCRATCH_ROOT:-/mnt/dlhd0/duke_local_matrix_260417}"
REMOTE_STARTUP_DIR="${REMOTE_STARTUP_DIR:-/home/swarm/deploy_test/RUMC_1/startup}"

LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-$PROJECT_ROOT/duke_results/duke_split_local_matrix_260417}"
REPORT_FILE="${REPORT_FILE:-$PROJECT_ROOT/docs/DUKE_SPLIT_LOCAL_TRAINING_REPORT.md}"
DATASET_STATS_JSON="$LOCAL_OUTPUT_ROOT/dataset_stats.json"
SUMMARY_JSON="$LOCAL_OUTPUT_ROOT/summary.json"
SUMMARY_CSV="$LOCAL_OUTPUT_ROOT/summary.csv"

DRY_RUN=0
VERIFY_ONLY=0
SMOKE_ONLY=0
MATRIX_ONLY=0
PILOT_ONLY=0
FORCE=0

MODEL_SPECS=(
    "ODELIA_ternary_classification:MST"
    "challenge_1DivideAndConquer:1DivideAndConquer"
    "challenge_2BCN_AIM:2BCN_AIM"
    "challenge_3agaldran:3agaldran"
    "challenge_4abmil:4LME_ABMIL"
    "challenge_5pimed:5Pimed"
)
NODES=(node_A node_C node_B node_all)

usage() {
    cat <<EOF
Usage:
  $0 [options]

Options:
  --dry-run              Print actions without executing them
  --verify-only          Prepare node_all, verify dataset/prereqs, and render report skeleton
  --smoke-only           Run only the smoke stage on node_A
  --matrix-only          Skip smoke and run the full matrix only
  --pilot-only           Run only node_A + MST end-to-end
  --force                Re-run cells even if result.json already exists
  --output-dir DIR       Override local output root
  --report-file FILE     Override markdown report path
  -h, --help             Show help

Environment overrides:
  REMOTE_HOST, REMOTE_USER, REMOTE_PASS
  REMOTE_DATASET_ROOT, REMOTE_SCRATCH_ROOT
  REMOTE_STARTUP_DIR, VERSION, IMAGE, GPU_SPEC
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        --smoke-only) SMOKE_ONLY=1; shift ;;
        --matrix-only) MATRIX_ONLY=1; shift ;;
        --pilot-only) PILOT_ONLY=1; shift ;;
        --force) FORCE=1; shift ;;
        --output-dir) LOCAL_OUTPUT_ROOT="$2"; shift 2 ;;
        --report-file) REPORT_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) err "Unknown option: $1"; usage; exit 1 ;;
    esac
done

mkdir -p "$LOCAL_OUTPUT_ROOT" \
         "$LOCAL_OUTPUT_ROOT/cells" \
         "$LOCAL_OUTPUT_ROOT/artifacts" \
         "$LOCAL_OUTPUT_ROOT/logs"
mkdir -p "$(dirname "$REPORT_FILE")"

remote_bash() {
    local script="$1"
    shift
    if [[ "$DRY_RUN" -eq 1 ]]; then
        info "[dry-run] remote bash on ${REMOTE_USER}@${REMOTE_HOST} with args: $*"
        printf '%s\n' "$script"
        return 0
    fi
    sshpass -p "$REMOTE_PASS" ssh $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" \
        bash -s -- "$@" <<<"$script"
}

copy_from_remote() {
    local remote_path="$1"
    local local_path="$2"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        info "[dry-run] scp ${REMOTE_USER}@${REMOTE_HOST}:${remote_path} -> ${local_path}"
        return 0
    fi
    mkdir -p "$(dirname "$local_path")"
    sshpass -p "$REMOTE_PASS" scp $SSH_OPTS -r \
        "${REMOTE_USER}@${REMOTE_HOST}:${remote_path}" "$local_path"
}

copy_to_remote() {
    local local_path="$1"
    local remote_path="$2"
    local remote_dir
    remote_dir="$(dirname "$remote_path")"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        info "[dry-run] scp ${local_path} -> ${REMOTE_USER}@${REMOTE_HOST}:${remote_path}"
        return 0
    fi
    sshpass -p "$REMOTE_PASS" ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
        "mkdir -p '$remote_dir' && cat > '$remote_path'" < "$local_path"
}

verify_remote_prereqs() {
    step "Verifying remote prerequisites on ${REMOTE_HOST}"
    local script
    script=$(cat <<'EOF'
set -euo pipefail
startup_dir="$1"
image="$2"
dataset_root="$3"
if [[ ! -x "$startup_dir/docker.sh" ]]; then
    echo "Missing startup docker.sh at $startup_dir" >&2
    exit 1
fi
docker image inspect "$image" >/dev/null
if [[ ! -d "$dataset_root" ]]; then
    echo "Missing dataset root: $dataset_root" >&2
    exit 1
fi
echo "startup=$startup_dir"
echo "image=$image"
echo "dataset_root=$dataset_root"
EOF
)
    remote_bash "$script" "$REMOTE_STARTUP_DIR" "$IMAGE" "$REMOTE_DATASET_ROOT" >/dev/null
    ok "Remote startup kit, image, and dataset root are present"
}

prepare_remote_node_all() {
    step "Preparing node_all under ${REMOTE_DATASET_ROOT}"
    local script
    script=$(cat <<'EOF'
set -euo pipefail
dataset_root="$1"
metadata_src="$2"
duke_data_root="$3"
node_all_root="$dataset_root/node_all"
mkdir -p "$node_all_root/metadata_unilateral"
cp "$metadata_src/annotation.csv" "$node_all_root/metadata_unilateral/annotation.csv"
cp "$metadata_src/split.csv" "$node_all_root/metadata_unilateral/split.csv"
ln -sfn "$duke_data_root" "$node_all_root/data_unilateral"
ls -ld "$node_all_root" "$node_all_root/data_unilateral" "$node_all_root/metadata_unilateral"
EOF
)
    remote_bash "$script" \
        "$REMOTE_DATASET_ROOT" \
        "$REMOTE_NODE_ALL_METADATA_SOURCE" \
        "$REMOTE_DUKE_DATA_ROOT" >/dev/null
    ok "node_all prepared at ${REMOTE_DATASET_ROOT}/node_all"
}

collect_dataset_stats() {
    step "Collecting dataset statistics"
    local script
    script=$(cat <<'EOF'
set -euo pipefail
dataset_root="$1"
python3 - <<'PY' "$dataset_root"
import json
import sys
from pathlib import Path

import pandas as pd

dataset_root = Path(sys.argv[1])
institutions = ["node_A", "node_B", "node_C", "node_all", "test"]
output = {
    "dataset_root": str(dataset_root),
    "nodes": {},
}
for institution in institutions:
    metadata_dir = dataset_root / institution / "metadata_unilateral"
    annotation_csv = metadata_dir / "annotation.csv"
    split_csv = metadata_dir / "split.csv"
    data_dir = dataset_root / institution / "data_unilateral"
    if not annotation_csv.exists() or not split_csv.exists():
        raise FileNotFoundError(f"Missing metadata for {institution} under {metadata_dir}")

    ann = pd.read_csv(annotation_csv)
    spl = pd.read_csv(split_csv)
    merged = spl.merge(ann, on="UID", how="left")

    node_stats = {
        "patients": int(ann["PatientID"].nunique()),
        "uids": int(len(ann)),
        "data_unilateral_exists": bool(data_dir.exists()),
        "splits": {},
    }

    for split_name in sorted(merged["Split"].unique().tolist()):
        subset = merged[merged["Split"] == split_name]
        lesion_counts = {
            str(int(k)): int(v)
            for k, v in subset["Lesion"].value_counts().sort_index().to_dict().items()
        }
        node_stats["splits"][split_name] = {
            "uids": int(len(subset)),
            "lesions": lesion_counts,
        }
    output["nodes"][institution] = node_stats

print(json.dumps(output, indent=2))
PY
EOF
)
    if [[ "$DRY_RUN" -eq 1 ]]; then
        info "[dry-run] skipping dataset stats collection"
        return 0
    fi
    remote_bash "$script" "$REMOTE_DATASET_ROOT" > "$DATASET_STATS_JSON"
    ok "Dataset stats written to $DATASET_STATS_JSON"
}

render_outputs() {
    step "Refreshing summary outputs and markdown report"
    "$LOCAL_PYTHON" - <<'PY' "$LOCAL_OUTPUT_ROOT" "$DATASET_STATS_JSON" "$SUMMARY_JSON" "$SUMMARY_CSV" "$REPORT_FILE"
import csv
import json
from pathlib import Path
from collections import defaultdict

output_root = Path(__import__("sys").argv[1])
dataset_stats_path = Path(__import__("sys").argv[2])
summary_json_path = Path(__import__("sys").argv[3])
summary_csv_path = Path(__import__("sys").argv[4])
report_path = Path(__import__("sys").argv[5])

dataset_stats = {}
if dataset_stats_path.exists():
    dataset_stats = json.loads(dataset_stats_path.read_text())

results = []
for result_path in sorted((output_root / "cells" / "full").glob("*/*/result.json")):
    results.append(json.loads(result_path.read_text()))

summary_json_path.parent.mkdir(parents=True, exist_ok=True)
summary_json_path.write_text(json.dumps(results, indent=2))

fieldnames = [
    "node",
    "job_dir",
    "model_name",
    "status",
    "run_dir_remote",
    "best_checkpoint_remote",
    "training_duration_seconds",
    "binary_auroc",
    "binary_accuracy",
    "binary_f1",
    "num_samples",
]
with summary_csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        metrics = row.get("metrics", {})
        writer.writerow({
            "node": row.get("node"),
            "job_dir": row.get("job_dir"),
            "model_name": row.get("model_name"),
            "status": row.get("status"),
            "run_dir_remote": row.get("run_dir_remote"),
            "best_checkpoint_remote": row.get("best_checkpoint_remote"),
            "training_duration_seconds": row.get("training_duration_seconds"),
            "binary_auroc": metrics.get("binary_auroc"),
            "binary_accuracy": metrics.get("binary_accuracy"),
            "binary_f1": metrics.get("binary_f1"),
            "num_samples": metrics.get("num_samples"),
        })

best_by_node = {}
best_by_model = {}
successful = [row for row in results if row.get("status") == "success"]
failed = [row for row in results if row.get("status") and row.get("status") != "success"]

for row in successful:
    score = row["metrics"]["binary_auroc"]
    node = row["node"]
    model = row["model_name"]
    if node not in best_by_node or score > best_by_node[node]["metrics"]["binary_auroc"]:
        best_by_node[node] = row
    if model not in best_by_model or score > best_by_model[model]["metrics"]["binary_auroc"]:
        best_by_model[model] = row

overall = sorted(
    successful,
    key=lambda row: row["metrics"]["binary_auroc"],
    reverse=True,
)

missing = []
expected_nodes = ["node_A", "node_B", "node_C", "node_all"]
expected_models = ["MST", "1DivideAndConquer", "2BCN_AIM", "3agaldran", "4LME_ABMIL", "5Pimed"]
present = {(row.get("node"), row.get("model_name")) for row in results}
for node in expected_nodes:
    for model in expected_models:
        if (node, model) not in present:
            missing.append((node, model))

lines = []
lines.append("# DUKE Split Local Training Report")
lines.append("")
lines.append("> Auto-generated by `scripts/evaluation/run_duke_split_local_matrix.sh`.")
lines.append("")
if results and not missing and not failed:
    lines.append(f"**Status:** Completed `{len(results)}` of `24` full training/evaluation cells with no recorded failures.")
else:
    lines.append(
        f"**Status:** Recorded `{len(successful)}` successful full cells, "
        f"`{len(failed)}` failed cells, and `{len(missing)}` missing cells."
    )
lines.append("")
lines.append("## Dataset Split")
lines.append("")
lines.append("| Node | Patients | Total UIDs | Train UIDs | Val/Test UIDs | Lesion Distribution |")
lines.append("|------|----------|------------|------------|---------------|---------------------|")
for name in ["node_A", "node_B", "node_C", "node_all", "test"]:
    node = dataset_stats.get("nodes", {}).get(name, {})
    splits = node.get("splits", {})
    if name == "test":
        test_split = splits.get("test", {})
        lines.append(
            f"| {name} | {node.get('patients', '-')} | {node.get('uids', '-')} | - | "
            f"{test_split.get('uids', '-')} | {test_split.get('lesions', {})} |"
        )
    else:
        train_split = splits.get("train", {})
        val_split = splits.get("val", {})
        lines.append(
            f"| {name} | {node.get('patients', '-')} | {node.get('uids', '-')} | "
            f"{train_split.get('uids', '-')} | {val_split.get('uids', '-')} | "
            f"train={train_split.get('lesions', {})}, val={val_split.get('lesions', {})} |"
        )
lines.append("")

lines.append("## Results")
lines.append("")
lines.append("| Node | Job Dir | Model | Status | Run Dir | Best Checkpoint | Train Sec | Test AUROC | Test Acc | Test F1 |")
lines.append("|------|---------|-------|--------|---------|-----------------|-----------|------------|----------|---------|")
for row in results:
    metrics = row.get("metrics", {})
    lines.append(
        f"| {row.get('node')} | {row.get('job_dir')} | {row.get('model_name')} | {row.get('status')} | "
        f"`{row.get('run_dir_remote', '-')}` | `{row.get('best_checkpoint_remote', '-')}` | "
        f"{row.get('training_duration_seconds', '-')} | "
        f"{metrics.get('binary_auroc', '-')} | {metrics.get('binary_accuracy', '-')} | {metrics.get('binary_f1', '-')} |"
    )
lines.append("")

lines.append("## Best Model Per Node")
lines.append("")
if best_by_node:
    lines.append("| Node | Model | AUROC | Accuracy | F1 |")
    lines.append("|------|-------|-------|----------|----|")
    for node in expected_nodes:
        row = best_by_node.get(node)
        if row:
            m = row["metrics"]
            lines.append(f"| {node} | {row['model_name']} | {m['binary_auroc']:.4f} | {m['binary_accuracy']:.4f} | {m['binary_f1']:.4f} |")
lines.append("")

lines.append("## Best Node Per Model")
lines.append("")
if best_by_model:
    lines.append("| Model | Node | AUROC | Accuracy | F1 |")
    lines.append("|-------|------|-------|----------|----|")
    for model in expected_models:
        row = best_by_model.get(model)
        if row:
            m = row["metrics"]
            lines.append(f"| {model} | {row['node']} | {m['binary_auroc']:.4f} | {m['binary_accuracy']:.4f} | {m['binary_f1']:.4f} |")
lines.append("")

lines.append("## Overall Ranking")
lines.append("")
if overall:
    lines.append("| Rank | Node | Model | AUROC | Accuracy | F1 |")
    lines.append("|------|------|-------|-------|----------|----|")
    for idx, row in enumerate(overall, start=1):
        m = row["metrics"]
        lines.append(f"| {idx} | {row['node']} | {row['model_name']} | {m['binary_auroc']:.4f} | {m['binary_accuracy']:.4f} | {m['binary_f1']:.4f} |")
lines.append("")

if missing:
    lines.append("## Missing Cells")
    lines.append("")
    for node, model in missing:
        lines.append(f"- `{node}` / `{model}`")
    lines.append("")

report_path.write_text("\n".join(lines) + "\n")
print(f"Summary JSON: {summary_json_path}")
print(f"Summary CSV: {summary_csv_path}")
print(f"Report: {report_path}")
PY
    ok "Summary artifacts refreshed"
}

run_remote_training_mode() {
    local mode="$1"
    local job_dir="$2"
    local model_name="$3"
    local institution="$4"
    local cell_remote_dir="$5"
    local epochs="$6"

    local script
    script=$(cat <<'EOF'
set -euo pipefail
image="$1"
gpu_spec="$2"
dataset_root="$3"
duke_data_root="$4"
cell_dir="$5"
mode="$6"
job_dir="$7"
model_name="$8"
institution="$9"
version="${10}"
num_epochs="${11}"
site_name="${12}"

mkdir -p "$cell_dir"
log_file="$cell_dir/${mode}.log"
set -o pipefail
docker run --rm \
  --gpus="$gpu_spec" \
  -u "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd \
  -v /etc/group:/etc/group \
  -v "$dataset_root:/data:ro" \
  -v "$duke_data_root:$duke_data_root:ro" \
  -v "$cell_dir:/scratch" \
  --ipc=host \
  -w /scratch \
  --env SITE_NAME="$site_name" \
  --env INSTITUTION="$institution" \
  --env DATA_DIR=/data \
  --env SCRATCH_DIR=/scratch \
  --env CONFIG=unilateral \
  --env TORCH_HOME=/scratch/torch_home \
  --env GPU_DEVICE="$gpu_spec" \
  --env MODEL_NAME="$model_name" \
  --env NUM_EPOCHS="$num_epochs" \
  --env MEDISWARM_VERSION="$version" \
  --env TRAINING_MODE="$mode" \
  "$image" \
  /bin/bash -lc "/MediSwarm/application/jobs/${job_dir}/app/custom/main.py" \
  2>&1 | tee "$log_file"
EOF
)

    remote_bash "$script" \
        "$IMAGE" \
        "$GPU_SPEC" \
        "$REMOTE_DATASET_ROOT" \
        "$REMOTE_DUKE_DATA_ROOT" \
        "$cell_remote_dir" \
        "$mode" \
        "$job_dir" \
        "$model_name" \
        "$institution" \
        "$VERSION" \
        "$epochs" \
        "$SITE_NAME"
}

resolve_remote_run_dir() {
    local cell_remote_dir="$1"
    local institution="$2"
    local script
    script=$(cat <<'EOF'
set -euo pipefail
cell_dir="$1"
institution="$2"
run_root="$cell_dir/runs/$institution"
if [[ ! -d "$run_root" ]]; then
    exit 0
fi
find "$run_root" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1
EOF
)
    remote_bash "$script" "$cell_remote_dir" "$institution"
}

resolve_remote_best_checkpoint() {
    local run_remote_dir="$1"
    local script
    script=$(cat <<'EOF'
set -euo pipefail
run_dir="$1"
best_json="$run_dir/lightning_logs/version_0/best_checkpoint.json"
if [[ -f "$best_json" ]]; then
    best_name=$(python3 - <<'PY' "$best_json"
import json
import sys
with open(sys.argv[1]) as handle:
    print(json.load(handle)["best_model_epoch"])
PY
)
    if [[ -f "$run_dir/$best_name" ]]; then
        echo "$run_dir/$best_name"
        exit 0
    fi
fi

epoch_ckpt=$(find "$run_dir" -maxdepth 1 -type f -name 'epoch=*.ckpt' | sort -V | tail -n 1 || true)
if [[ -n "${epoch_ckpt:-}" ]]; then
    echo "$epoch_ckpt"
    exit 0
fi

if [[ -f "$run_dir/last_global_model.ckpt" ]]; then
    echo "$run_dir/last_global_model.ckpt"
    exit 0
fi

if [[ -f "$run_dir/last.ckpt" ]]; then
    echo "$run_dir/last.ckpt"
    exit 0
fi
EOF
)
    remote_bash "$script" "$run_remote_dir"
}

verify_model_logged() {
    local log_file="$1"
    local model_name="$2"
    if grep -q "Using model: ${model_name}" "$log_file"; then
        return 0
    fi
    grep -q "Model name: .*${model_name}" "$log_file"
}

run_remote_prediction() {
    local model_name="$1"
    local checkpoint_remote_path="$2"
    local cell_remote_dir="$3"
    local script
    script=$(cat <<'EOF'
set -euo pipefail
image="$1"
gpu_spec="$2"
dataset_root="$3"
duke_data_root="$4"
cell_dir="$5"
model_name="$6"
checkpoint_path="$7"
version="$8"
site_name="$9"

mkdir -p "$cell_dir/predictions"
log_file="$cell_dir/predict.log"
container_checkpoint_path="$checkpoint_path"
case "$checkpoint_path" in
  "$cell_dir"/*)
    container_checkpoint_path="/scratch/${checkpoint_path#"$cell_dir"/}"
    ;;
esac
set -o pipefail
docker run --rm \
  --gpus="$gpu_spec" \
  -u "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd \
  -v /etc/group:/etc/group \
  -v "$dataset_root:/data:ro" \
  -v "$duke_data_root:$duke_data_root:ro" \
  -v "$cell_dir:/scratch" \
  --ipc=host \
  -w /scratch \
  --env SITE_NAME="$site_name" \
  --env INSTITUTION=test \
  --env DATA_DIR=/data \
  --env SCRATCH_DIR=/scratch \
  --env CONFIG=unilateral \
  --env MODEL_NAME="$model_name" \
  --env MEDISWARM_VERSION="$version" \
  --env MEDISWARM_PROJECT_ROOT=/MediSwarm \
  "$image" \
  /bin/bash -lc "python /scratch/predict.py --checkpoint '$container_checkpoint_path' --checkpoint-type lightning --output-dir /scratch/predictions --split test --num-classes 3 --model-name '$model_name'" \
  2>&1 | tee "$log_file"
EOF
)
    remote_bash "$script" \
        "$IMAGE" \
        "$GPU_SPEC" \
        "$REMOTE_DATASET_ROOT" \
        "$REMOTE_DUKE_DATA_ROOT" \
        "$cell_remote_dir" \
        "$model_name" \
        "$checkpoint_remote_path" \
        "$VERSION" \
        "$SITE_NAME"
}

write_result_json() {
    local result_json="$1"
    local node="$2"
    local job_dir="$3"
    local model_name="$4"
    local run_remote_dir="$5"
    local checkpoint_remote_path="$6"
    local training_duration_seconds="$7"
    local metrics_json="$8"
    local train_log_local="$9"
    local predict_log_local="${10}"
    local predictions_csv_local="${11}"

    "$LOCAL_PYTHON" - <<'PY' "$result_json" "$node" "$job_dir" "$model_name" "$run_remote_dir" "$checkpoint_remote_path" "$training_duration_seconds" "$metrics_json" "$train_log_local" "$predict_log_local" "$predictions_csv_local"
import json
import sys
from pathlib import Path

result_json = Path(sys.argv[1])
metrics_json = Path(sys.argv[8])
result_json.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "node": sys.argv[2],
    "job_dir": sys.argv[3],
    "model_name": sys.argv[4],
    "status": "success",
    "run_dir_remote": sys.argv[5],
    "best_checkpoint_remote": sys.argv[6],
    "training_duration_seconds": int(sys.argv[7]),
    "train_log_local": sys.argv[9],
    "predict_log_local": sys.argv[10],
    "predictions_csv_local": sys.argv[11],
    "metrics": json.loads(metrics_json.read_text()),
}
result_json.write_text(json.dumps(payload, indent=2))
PY
}

run_smoke_cell() {
    local job_dir="$1"
    local model_name="$2"

    local smoke_root_remote="$REMOTE_SCRATCH_ROOT/smoke/${SITE_NAME}/${model_name}"
    local smoke_root_local="$LOCAL_OUTPUT_ROOT/cells/smoke/${SITE_NAME}/${model_name}"
    mkdir -p "$smoke_root_local"

    step "Smoke: ${model_name} on node_A"

    for mode in preflight_check local_training; do
        local epochs=1
        local remote_dir="${smoke_root_remote}/${mode}"
        local local_dir="${smoke_root_local}/${mode}"
        local local_log="${local_dir}/${mode}.log"
        mkdir -p "$local_dir"

        run_remote_training_mode "$mode" "$job_dir" "$model_name" "node_A" "$remote_dir" "$epochs"
        copy_from_remote "${remote_dir}/${mode}.log" "$local_log"

        if ! verify_model_logged "$local_log" "$model_name"; then
            err "Smoke ${mode} did not log the expected model name for ${model_name}"
            exit 1
        fi

        local run_remote_dir
        run_remote_dir="$(resolve_remote_run_dir "$remote_dir" "node_A" || true)"
        if [[ -z "$run_remote_dir" ]]; then
            err "Smoke ${mode} did not create a run directory for ${model_name}"
            exit 1
        fi
        printf '%s\n' "$run_remote_dir" > "${local_dir}/run_remote_dir.txt"
        ok "Smoke ${mode} passed for ${model_name}"
    done
}

run_full_cell() {
    local node="$1"
    local job_dir="$2"
    local model_name="$3"

    local local_cell_dir="$LOCAL_OUTPUT_ROOT/cells/full/${node}/${model_name}"
    local local_result_json="$local_cell_dir/result.json"
    if [[ -f "$local_result_json" && "$FORCE" -eq 0 ]]; then
        info "Skipping ${node}/${model_name} — existing result.json found"
        return 0
    fi

    mkdir -p "$local_cell_dir"
    local remote_cell_dir="$REMOTE_SCRATCH_ROOT/full/${node}/${model_name}"
    local train_log_local="$local_cell_dir/train.log"
    local predict_log_local="$local_cell_dir/predict.log"
    local predictions_local_dir="$local_cell_dir/predictions"
    local metrics_json="$local_cell_dir/metrics.json"

    step "Full cell: ${node} / ${model_name}"

    local train_start train_end training_duration_seconds
    train_start="$(date +%s)"
    run_remote_training_mode "local_training" "$job_dir" "$model_name" "$node" "$remote_cell_dir" "100"
    train_end="$(date +%s)"
    training_duration_seconds="$((train_end - train_start))"

    copy_from_remote "${remote_cell_dir}/local_training.log" "$train_log_local"

    if ! verify_model_logged "$train_log_local" "$model_name"; then
        err "Training log did not report the expected model for ${node}/${model_name}"
        exit 1
    fi

    local run_remote_dir
    run_remote_dir="$(resolve_remote_run_dir "$remote_cell_dir" "$node" || true)"
    if [[ -z "$run_remote_dir" ]]; then
        err "No run directory found for ${node}/${model_name}"
        exit 1
    fi

    local checkpoint_remote_path
    checkpoint_remote_path="$(resolve_remote_best_checkpoint "$run_remote_dir" || true)"
    if [[ -z "$checkpoint_remote_path" ]]; then
        err "No usable checkpoint found for ${node}/${model_name}"
        exit 1
    fi

    copy_to_remote "$PROJECT_ROOT/scripts/evaluation/predict.py" "${remote_cell_dir}/predict.py"
    run_remote_prediction "$model_name" "$checkpoint_remote_path" "$remote_cell_dir"
    copy_from_remote "${remote_cell_dir}/predict.log" "$predict_log_local"
    rm -rf "$predictions_local_dir"
    copy_from_remote "${remote_cell_dir}/predictions" "$predictions_local_dir"

    local predictions_csv_local
    predictions_csv_local="$(find "$predictions_local_dir" -maxdepth 1 -type f -name 'predictions_*.csv' ! -name 'predictions_ensemble.csv' | sort | head -n 1)"
    if [[ -z "${predictions_csv_local:-}" ]]; then
        err "No predictions CSV found for ${node}/${model_name}"
        exit 1
    fi

    "$LOCAL_PYTHON" "$PROJECT_ROOT/scripts/evaluation/evaluate_duke_binary.py" \
        --predictions-csv "$predictions_csv_local" \
        --output-json "$metrics_json"

    write_result_json \
        "$local_result_json" \
        "$node" \
        "$job_dir" \
        "$model_name" \
        "$run_remote_dir" \
        "$checkpoint_remote_path" \
        "$training_duration_seconds" \
        "$metrics_json" \
        "$train_log_local" \
        "$predict_log_local" \
        "$predictions_csv_local"

    ok "Completed ${node}/${model_name}"
    render_outputs
}

run_smoke_stage() {
    step "Running smoke stage on node_A"
    for spec in "${MODEL_SPECS[@]}"; do
        IFS=":" read -r job_dir model_name <<<"$spec"
        run_smoke_cell "$job_dir" "$model_name"
    done
}

run_full_matrix() {
    step "Running full local-training matrix"
    for node in "${NODES[@]}"; do
        for spec in "${MODEL_SPECS[@]}"; do
            IFS=":" read -r job_dir model_name <<<"$spec"
            run_full_cell "$node" "$job_dir" "$model_name"
        done
    done
}

run_pilot() {
    step "Running pilot cell node_A / MST"
    run_full_cell "node_A" "ODELIA_ternary_classification" "MST"
}

step "Configuration"
info "Remote host:        ${REMOTE_USER}@${REMOTE_HOST}"
info "Image:              ${IMAGE}"
info "Dataset root:       ${REMOTE_DATASET_ROOT}"
info "Remote scratch:     ${REMOTE_SCRATCH_ROOT}"
info "Local output root:  ${LOCAL_OUTPUT_ROOT}"
info "Report file:        ${REPORT_FILE}"
info "Dry run:            ${DRY_RUN}"
info "Force:              ${FORCE}"

verify_remote_prereqs
prepare_remote_node_all
collect_dataset_stats
render_outputs

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    ok "Verification only complete"
    exit 0
fi

if [[ "$PILOT_ONLY" -eq 1 ]]; then
    run_pilot
    ok "Pilot run complete"
    exit 0
fi

if [[ "$MATRIX_ONLY" -eq 0 && "$SMOKE_ONLY" -eq 0 ]]; then
    run_smoke_stage
fi

if [[ "$SMOKE_ONLY" -eq 1 ]]; then
    run_smoke_stage
    ok "Smoke stage complete"
    exit 0
fi

run_full_matrix
ok "DUKE split local-training matrix complete"
