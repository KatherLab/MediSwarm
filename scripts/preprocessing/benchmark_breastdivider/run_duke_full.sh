#!/usr/bin/env bash
# Phase 2: MST trained from scratch on Duke fold 0, old crops vs BreastDivider crops, then a train x test matrix
set -uo pipefail
B=/home/swarm/bd_bench; R=/mnt/swarm_alpha/bd_bench/duke_full; PY=$B/venv/bin/python; PKG=$B/bdpp; ROOT=/mnt/swarm_alpha/bd_bench
cd $PKG; ts(){ date -u +%H:%M:%SZ; }
echo "$(ts) cases: $(ls $R/data | wc -l)"
rm -rf $R/BD_Input_seq $R/BD_Output
echo "$(ts) 1. BreastDivider segmentation"; $PY BD_run_seg.py $R --bd_gpu=device=0 --single-site --force > $B/dukefull_seg.log 2>&1; echo "  segmentations: $(ls $R/BD_Output 2>/dev/null | wc -l)"
echo "$(ts) 2. BD crops"; rm -rf $R/data_unilateral_BD; nice -n 10 $PY BD_main.py $R --mode=normal --num_workers=14 --single-site --force --subs 1 --include_pre_post --log_level=PROGRESS > $B/dukefull_main.log 2>&1; echo "  BD sides: $(ls $R/data_unilateral_BD 2>/dev/null | wc -l)"
echo "$(ts) 3. reference crops"; nice -n 10 $PY $B/scripts/old_unilateral.py $R --out data_unilateral_old --workers 14 > $B/dukefull_old.log 2>&1; tail -1 $B/dukefull_old.log
echo "$(ts) 4. train roots (paired uids)"; $PY - "$R" "$ROOT" <<'PY'
import csv, os, sys, collections
R, ROOT = sys.argv[1], sys.argv[2]
def ok(v, u): return os.path.exists(f"{R}/data_unilateral_{v}/{u}/Sub_1.nii.gz")
have = {u for u in os.listdir(f"{R}/data_unilateral_old") if ok("old", u) and ok("BD", u)}
an = [r for r in csv.DictReader(open(f"{R}/metadata_src/annotation.csv", encoding="utf-8-sig")) if r["UID"] in have]
sp = [r for r in csv.DictReader(open(f"{R}/metadata_src/split.csv", encoding="utf-8-sig")) if r["UID"] in have]
for v in ("old", "BD"):
    d = f"{ROOT}/train_duke_{v}/DUKE_1/metadata_unilateral"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/annotation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["UID", "PatientID", "Age", "Lesion"]); w.writeheader(); w.writerows(an)
    with open(f"{d}/split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["UID", "Fold", "Split"]); w.writeheader(); w.writerows(sp)
print("paired uids:", len(have), "fold 0:", dict(collections.Counter(r["Split"] for r in sp if r["Fold"] == "0")))
PY
echo "$(ts) 5. train MST, 20 epochs"; for v in old BD; do echo "-- train $v $(ts)"; $B/scripts/train_duke.sh $v MST 20; done
echo "$(ts) 6. evaluate, train x test"; for tv in old BD; do for ev in old BD; do echo "-- trained on $tv, tested on $ev"; $B/scripts/eval_duke_trained.sh $tv $ev MST; done; done
echo "$(ts) DONE"
