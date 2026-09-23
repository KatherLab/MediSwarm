#!/usr/bin/env bash
# eval_duke_trained.sh <train variant> <eval variant> <model>   evaluate a locally trained model on the Duke fold-0 test split
set -uo pipefail
TV=$1; EV=$2; M=$3; F=${4:-0}; SITE=DUKE_1; ROOT=/mnt/swarm_alpha/bd_bench
DATA=$ROOT/duke_full/data_unilateral_$EV; META=$ROOT/train_duke_$EV/$SITE/metadata_unilateral
TR=$ROOT/train_out/duke_$TV/${M}_fold$F; RUN=$(ls -d $TR/runs/$SITE/*/ | tail -1)
BEST=$(ls $RUN/epoch=*.ckpt 2>/dev/null | tail -1 | xargs -r basename)
OUT=$ROOT/eval_out/duke_train${TV}_test${EV}/${M}_fold$F; mkdir -p $OUT/pred $OUT/cache
for CK in "$BEST" last_global_model.ckpt; do
  [ -n "$CK" ] && [ -f "$RUN/$CK" ] || { echo "missing checkpoint $CK"; continue; }
  docker run --rm --gpus device=0 --ipc=host -u $(id -u):$(id -g) -v /etc/passwd:/etc/passwd:ro \
    -v $DATA:/data/$SITE/data_unilateral:ro -v $META:/data/$SITE/metadata_unilateral:ro -v $RUN:/ckpt:ro -v $OUT:/scratch \
    -e HOME=/scratch -e DATA_DIR=/data -e SITE_NAME=$SITE -e SCRATCH_DIR=/scratch -e FOLD=$F -e ODELIA_PREPROCESS_CACHE_DIR=/scratch/cache \
    jefftud/odelia:current python3 /MediSwarm/scripts/evaluation/predict.py --checkpoint "/ckpt/$CK" --checkpoint-type lightning --model-name $M --split test --batch-size 4 --output-dir /scratch/pred > $OUT/predict_$(basename "$CK" .ckpt).log 2>&1
  echo "exit $? ckpt=$CK ; $(grep -iE "AUC-ROC:|Samples:" $OUT/predict_$(basename "$CK" .ckpt).log | tail -2 | tr '\n' ' ')"
done
ls $OUT/pred
