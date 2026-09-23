#!/usr/bin/env bash
# Folds 1 and 2 of the from-scratch comparison (fold 0 already done): train MST on old and BD crops, evaluate train x test, compare
set -uo pipefail
B=/home/swarm/bd_bench; PY=$B/venv/bin/python; O=/mnt/swarm_alpha/bd_bench/eval_out; ts(){ date -u +%H:%M:%SZ; }
for F in 1 2; do
  for v in old BD; do echo "-- $(ts) train $v fold $F"; $B/scripts/train_duke.sh $v MST 20 $F; done
  for tv in old BD; do for ev in old BD; do echo "-- $(ts) trained on $tv, tested on $ev, fold $F"; $B/scripts/eval_duke_trained.sh $tv $ev MST $F 2>&1 | grep -E "^exit|missing" | cut -c1-60; done; done
  f(){ ls $O/duke_train$1_test$2/MST_fold$F/pred/predictions_epoch*_single.csv | head -1; }
  echo "######## fold $F, best-epoch checkpoints"; $PY $B/scripts/compare_variants.py --label old=$(f old old) --label BD=$(f BD BD) --label old_on_BD=$(f old BD) --label BD_on_old=$(f BD old)
done
echo "$(ts) DONE"
