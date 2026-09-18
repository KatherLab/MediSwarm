#!/usr/bin/env python3
"""Evaluate 20260513 challenge swarm/local artifacts on ODELIA external sites.

This script is intentionally narrow: it supports the artifact package summarized
by ``generate_challenge_swarm_local_report.py`` and writes external-validation
tables that the report generator can embed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from generate_challenge_swarm_local_report import (
    DEFAULT_EXTRACTED,
    DEFAULT_WORK,
    EXPECTED_SITES,
    FINAL_MODEL_ORDER,
    MODEL_TO_FINAL_JOB,
    ROOT,
)


DL0_HOST = "swarm@dd-dl0"
DL0_STAGE = Path("/mnt/dlhd0/challenge_swarm_local_external_eval_20260513")
DL0_DATA = Path("/mnt/dlhd0/medswarmdata")
DOCKER_IMAGE = "jefftud/odelia:1.4.4-dev.260518.c2bd423"
EVAL_SITES = tuple(site.removesuffix("_1") for site in EXPECTED_SITES)


@dataclass
class EvalTarget:
    target_id: str
    model: str
    artifact: str
    checkpoint_type: str
    checkpoint_path: str
    source_site: str
    checkpoint_label: str
    internal_val_auroc: float | None
    internal_val_note: str


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def remote(command: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return run(["ssh", "-o", "ConnectTimeout=20", DL0_HOST, command], check=check, capture=capture)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def as_float(value: str | None) -> float | None:
    if value in (None, "", "None", "NA"):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return None if math.isnan(result) else result


def weighted_mean(rows: list[dict], metric: str, weight_key: str = "samples") -> float | None:
    total = 0.0
    weight = 0
    for row in rows:
        value = row.get(metric)
        if value in (None, "", "None", "NA"):
            continue
        value = float(value)
        if math.isnan(value):
            continue
        w = int(row.get(weight_key) or 0)
        total += value * w
        weight += w
    return total / weight if weight else None


def load_validation_rows(work_dir: Path) -> list[dict]:
    path = work_dir / "validation_metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation summary: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_inventory(work_dir: Path) -> dict:
    path = work_dir / "run_inventory.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run inventory: {path}")
    return json.loads(path.read_text())


def retained_epoch_ckpt(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("epoch=*.ckpt"))
    if ckpts:
        return ckpts[0]
    last = run_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No retained epoch or last checkpoint in {run_dir}")


def weighted_internal_last_auroc(rows: list[dict], *, model: str, source: str) -> float | None:
    subset = [
        row for row in rows
        if row["model"] == model and row["role"] == "final" and row["source"] == source
    ]
    weighted_rows = []
    for row in subset:
        value = as_float(row.get("last_auroc"))
        samples = int(row.get("samples_per_epoch") or 0)
        if value is not None and samples:
            weighted_rows.append({"last_auroc": value, "samples": samples})
    return weighted_mean(weighted_rows, "last_auroc")


def select_targets(work_dir: Path) -> list[EvalTarget]:
    inventory = load_inventory(work_dir)
    rows = load_validation_rows(work_dir)
    targets: list[EvalTarget] = []

    for model in FINAL_MODEL_ORDER:
        job_id = MODEL_TO_FINAL_JOB[model]
        job = inventory["jobs"][job_id]

        global_site = "CAM_1"
        global_model = Path(job["sites"][global_site].get("global_model") or "")
        if not global_model.exists():
            for site in EXPECTED_SITES:
                candidate = Path(job["sites"][site].get("global_model") or "")
                if candidate.exists():
                    global_site = site
                    global_model = candidate
                    break
        if not global_model.exists():
            raise FileNotFoundError(f"Missing final global checkpoint for {model}")

        swarm_internal = weighted_internal_last_auroc(rows, model=model, source="aggregated")
        targets.append(
            EvalTarget(
                target_id=safe_id(f"{model}_swarm_global_final"),
                model=model,
                artifact="swarm_global_final",
                checkpoint_type="state_dict",
                checkpoint_path=str(global_model),
                source_site=global_site,
                checkpoint_label="FL_global_model.pt",
                internal_val_auroc=swarm_internal,
                internal_val_note="weighted mean of aggregated-stream last validation AUROC across sites",
            )
        )

        site_rows = [
            row for row in rows
            if row["model"] == model and row["role"] == "final" and row["source"] == "site"
        ]
        best_site_row = max(site_rows, key=lambda row: as_float(row.get("best_auroc")) or -1.0)
        best_site = best_site_row["site"]
        run_dirs = [Path(path) for path in job["sites"][best_site].get("run_dirs", [])]
        run_dirs = [path for path in run_dirs if path.exists()]
        if not run_dirs:
            raise FileNotFoundError(f"No run dir for best site-local {model} target at {best_site}")
        local_ckpt = retained_epoch_ckpt(run_dirs[0])
        targets.append(
            EvalTarget(
                target_id=safe_id(f"{model}_{best_site}_site_local_retained"),
                model=model,
                artifact="site_local_retained",
                checkpoint_type="lightning",
                checkpoint_path=str(local_ckpt),
                source_site=best_site,
                checkpoint_label=local_ckpt.name,
                internal_val_auroc=as_float(best_site_row.get("best_auroc")),
                internal_val_note=f"best site-stream validation AUROC for {best_site}",
            )
        )

    return targets


def write_manifest(targets: list[EvalTarget], work_dir: Path) -> None:
    out = {
        "dl0_host": DL0_HOST,
        "dl0_stage": str(DL0_STAGE),
        "dl0_data": str(DL0_DATA),
        "docker_image": DOCKER_IMAGE,
        "eval_sites": list(EVAL_SITES),
        "targets": [asdict(target) for target in targets],
        "comparison_definition": (
            "swarm_global_final = representative FL_global_model.pt from the final "
            "swarm run; site_local_retained = retained non-last epoch checkpoint "
            "from the site with the best internal site-stream validation AUROC."
        ),
    }
    (work_dir / "external_eval_manifest.json").write_text(json.dumps(out, indent=2))


def stage_artifacts(targets: list[EvalTarget]) -> None:
    remote(f"mkdir -p {DL0_STAGE}/checkpoints {DL0_STAGE}/tools {DL0_STAGE}/odelia_eval")
    run(["rsync", "-aL", str(ROOT / "scripts" / "evaluation" / "predict.py"), f"{DL0_HOST}:{DL0_STAGE}/tools/predict.py"])
    for target in targets:
        local = Path(target.checkpoint_path)
        remote_path = f"{DL0_STAGE}/checkpoints/{target.target_id}{local.suffix}"
        exists = remote(f"test -s {remote_path} && echo yes || echo no", capture=True).stdout.strip()
        if exists == "yes":
            continue
        run(["rsync", "-aL", "--progress", str(local), f"{DL0_HOST}:{remote_path}"])


def eval_targets(targets: list[EvalTarget], *, force: bool) -> None:
    for target in targets:
        local = Path(target.checkpoint_path)
        staged_ckpt = f"/workspace/{target.target_id}{local.suffix}"
        for eval_site in EVAL_SITES:
            out_dir = f"{DL0_STAGE}/odelia_eval/{target.model}/{target.target_id}/{eval_site}"
            done = remote(
                f"test -s {out_dir}/prediction_results.json && grep -q '\"metrics\"' {out_dir}/prediction_results.json && echo yes || echo no",
                capture=True,
            ).stdout.strip()
            if done == "yes" and not force:
                print(f"Skipping existing eval: {target.target_id} on {eval_site}")
                continue
            command = f"""
set -euo pipefail
mkdir -p {out_dir}
docker run --rm \\
  --gpus=device=0 \\
  --ulimit nofile=65536:65536 \\
  --net=host --ipc=host \\
  -v {DL0_DATA}:/data:ro \\
  -v {DL0_STAGE}/checkpoints:/workspace:ro \\
  -v {DL0_STAGE}/tools/predict.py:/MediSwarm/scripts/evaluation/predict.py:ro \\
  -v {out_dir}:/output \\
  --env SITE_NAME={eval_site} \\
  --env DATA_DIR=/data \\
  --env SCRATCH_DIR=/output \\
  --env MODEL_NAME={target.model} \\
  --env TORCH_HOME=/torch_home \\
  --env CONFIG=unilateral \\
  --env TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_system \\
  {DOCKER_IMAGE} \\
  python3 /MediSwarm/scripts/evaluation/predict.py \\
    --checkpoint {staged_ckpt} \\
    --checkpoint-type {target.checkpoint_type} \\
    --model-name {target.model} \\
    --output-dir /output \\
    --split test > {out_dir}/predict_stdout.log 2>&1
"""
            remote(command)


def pull_results(work_dir: Path) -> None:
    out = work_dir / "external_eval"
    out.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", f"{DL0_HOST}:{DL0_STAGE}/odelia_eval/", str(out) + "/"])


def summarize_results(targets: list[EvalTarget], work_dir: Path) -> None:
    target_by_id = {target.target_id: target for target in targets}
    rows: list[dict] = []
    raw_root = work_dir / "external_eval"
    for result_path in sorted(raw_root.glob("*/*/*/prediction_results.json")):
        model = result_path.parents[2].name
        target_id = result_path.parents[1].name
        eval_site = result_path.parent.name
        target = target_by_id[target_id]
        results = json.loads(result_path.read_text())
        if not results:
            continue
        metrics = results[0]["metrics"]
        per_auc = metrics.get("auroc_per_class", {})
        per_recall = metrics.get("per_class_recall", {})
        per_f1 = metrics.get("per_class_f1", {})
        rows.append(
            {
                "model": model,
                "artifact": target.artifact,
                "target_id": target_id,
                "source_site": target.source_site,
                "checkpoint_label": target.checkpoint_label,
                "checkpoint_type": target.checkpoint_type,
                "eval_site": eval_site,
                "samples": metrics.get("num_samples"),
                "accuracy": metrics.get("accuracy"),
                "macro_auroc": metrics.get("auroc_macro"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "class0_auroc": per_auc.get("0"),
                "class1_auroc": per_auc.get("1"),
                "class2_auroc": per_auc.get("2"),
                "class2_recall": per_recall.get("2"),
                "class2_f1": per_f1.get("2"),
                "internal_val_auroc": target.internal_val_auroc,
                "internal_val_note": target.internal_val_note,
            }
        )

    by_site_path = work_dir / "external_metrics_by_eval_site.csv"
    fieldnames = [
        "model", "artifact", "target_id", "source_site", "checkpoint_label", "checkpoint_type",
        "eval_site", "samples", "accuracy", "macro_auroc", "macro_f1", "weighted_f1",
        "class0_auroc", "class1_auroc", "class2_auroc", "class2_recall", "class2_f1",
        "internal_val_auroc", "internal_val_note",
    ]
    with by_site_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["target_id"]].append(row)
    aggregate_rows: list[dict] = []
    for target in targets:
        group = grouped.get(target.target_id, [])
        if not group:
            continue
        aggregate_rows.append(
            {
                "model": target.model,
                "artifact": target.artifact,
                "target_id": target.target_id,
                "source_site": target.source_site,
                "checkpoint_label": target.checkpoint_label,
                "checkpoint_type": target.checkpoint_type,
                "external_sites": len(group),
                "samples": sum(int(row["samples"] or 0) for row in group),
                "external_accuracy": weighted_mean(group, "accuracy"),
                "external_macro_auroc": weighted_mean(group, "macro_auroc"),
                "external_macro_f1": weighted_mean(group, "macro_f1"),
                "external_class2_auroc": weighted_mean(group, "class2_auroc"),
                "external_class2_recall": weighted_mean(group, "class2_recall"),
                "external_class2_f1": weighted_mean(group, "class2_f1"),
                "internal_val_auroc": target.internal_val_auroc,
                "internal_val_note": target.internal_val_note,
            }
        )

    aggregate_path = work_dir / "external_comparison_metrics.csv"
    agg_fields = [
        "model", "artifact", "target_id", "source_site", "checkpoint_label", "checkpoint_type",
        "external_sites", "samples", "external_accuracy", "external_macro_auroc",
        "external_macro_f1", "external_class2_auroc", "external_class2_recall",
        "external_class2_f1", "internal_val_auroc", "internal_val_note",
    ]
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--force", action="store_true", help="Re-run completed dl0 predictions")
    parser.add_argument("--skip-eval", action="store_true", help="Only summarize already-pulled/present outputs")
    args = parser.parse_args()

    # The root path is validated here so callers get an early error if the
    # report package has not been extracted, even though the inventory carries
    # the resolved paths used for actual target selection.
    if not args.root.exists():
        raise FileNotFoundError(args.root)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    targets = select_targets(args.work_dir)
    write_manifest(targets, args.work_dir)
    if not args.skip_eval:
        stage_artifacts(targets)
        eval_targets(targets, force=args.force)
        pull_results(args.work_dir)
    summarize_results(targets, args.work_dir)
    print(f"Wrote {(args.work_dir / 'external_comparison_metrics.csv').relative_to(ROOT)}")
    print(f"Wrote {(args.work_dir / 'external_metrics_by_eval_site.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
