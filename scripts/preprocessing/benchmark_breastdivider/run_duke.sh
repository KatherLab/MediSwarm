#!/usr/bin/env bash
# BreastDivider benchmark, Duke leg: bilateral NIfTI (from dl0) -> BD seg -> BD crop -> old crop -> MIPs -> eval roots
set -uo pipefail
B=/home/swarm/bd_bench; R=/mnt/swarm_alpha/bd_bench/duke; PY=$B/venv/bin/python; PKG=$B/bdpp
cd $PKG; ts(){ date -u +%H:%M:%SZ; }
echo "$(ts) cases: $(ls $R/data | wc -l)"
rm -rf $R/BD_Input_seq $R/BD_Output
echo "$(ts) 2. BreastDivider segmentation"; $PY BD_run_seg.py $R --bd_gpu=device=0 --single-site --force > $B/duke_seg.log 2>&1; echo "  error lines: $(grep -cE "\[ERROR\]" $B/duke_seg.log)  segmentations: $(ls $R/BD_Output 2>/dev/null | wc -l)"
echo "$(ts) 3. BD crops"; rm -rf $R/data_unilateral_BD; $PY BD_main.py $R --mode=normal --num_workers=8 --single-site --force --subs 1 --include_pre_post --log_level=PROGRESS --diagnostic_plots > $B/duke_main.log 2>&1; echo "  BD cases: $(ls $R/data_unilateral_BD 2>/dev/null | wc -l)  fallback mentions: $(grep -ciE fallback $B/duke_main.log)"
echo "$(ts) 4. reference crops"; $PY $B/scripts/old_unilateral.py $R --out data_unilateral_old --workers 8 > $B/duke_old.log 2>&1; tail -1 $B/duke_old.log
echo "$(ts) 5. MIPs"; $PY compute_mips.py $R data_unilateral_old data_unilateral_BD --single-site --force > $B/duke_mips.log 2>&1
echo "$(ts) 6. eval roots"; for v in old BD; do d=/mnt/swarm_alpha/bd_bench/eval_duke_$v/DUKE_1; mkdir -p $d/metadata_unilateral; $PY - $R/data_unilateral_$v $d/metadata_unilateral <<PY
import csv, sys, os
crops, meta = sys.argv[1], sys.argv[2]
have = set(os.listdir(crops))
rows = [r for r in csv.DictReader(open("/mnt/swarm_alpha/bd_bench/duke/metadata_src/annotation.csv", encoding="utf-8-sig")) if r["UID"] in have]
with open(os.path.join(meta, "annotation.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["UID", "PatientID", "Lesion"]); w.writeheader()
    for r in rows: w.writerow({"UID": r["UID"], "PatientID": r["UID"].rsplit("_", 1)[0], "Lesion": r["Lesion"]})
with open(os.path.join(meta, "split.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["UID", "Fold", "Split"]); [w.writerow([r["UID"], 0, "test"]) for r in rows]
print(meta, len(rows), "uids")
PY
done
echo "$(ts) DONE"
