#!/usr/bin/env bash
set -uo pipefail
B=/home/swarm/bd_bench; R=/mnt/swarm_alpha/bd_bench/challenge; PY=$B/venv/bin/python; PKG=$B/bdpp
cd $PKG; ts(){ date -u +%H:%M:%SZ; }
echo "$(ts) 1. layout (hard links)"; $PY $B/scripts/make_layout.py /mnt/swarm_alpha/bd_bench/challenge_extracted $R --ids $B/scripts/umcu_challenge_ids.txt
rm -rf $R/BD_Input_seq $R/BD_Output
echo "$(ts) 2. BreastDivider segmentation"; $PY BD_run_seg.py $R --bd_gpu=device=0 --single-site --force > $B/bd_seg.log 2>&1; grep -cE "\[ERROR\]" $B/bd_seg.log | sed "s/^/  error lines: /"; echo "  segmentations: $(ls $R/BD_Output 2>/dev/null | wc -l)"
echo "$(ts) 3. BD crops"; rm -rf $R/data_unilateral_BD; $PY BD_main.py $R --mode=normal --num_workers=8 --single-site --force --subs 1 --include_pre_post --log_level=PROGRESS --diagnostic_plots > $B/bd_main.log 2>&1; echo "  BD cases: $(ls $R/data_unilateral_BD 2>/dev/null | wc -l)"; grep -ciE "fallback" $B/bd_main.log | sed "s/^/  fallback mentions: /"
echo "$(ts) 4. MIPs"; $PY compute_mips.py $R data_unilateral_old data_unilateral_BD --single-site --force > $B/bd_mips.log 2>&1; ls -d $R/*mips* 2>/dev/null
echo "$(ts) DONE"
