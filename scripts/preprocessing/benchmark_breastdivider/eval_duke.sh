#!/usr/bin/env bash
# eval_duke.sh <variant: old|BD> <model>   (Duke leg: crops under duke/data_unilateral_<v>, metadata under eval_duke_<v>/DUKE_1)
set -uo pipefail
V=$1; M=$2; SITE=DUKE_1; ROOT=/mnt/swarm_alpha/bd_bench
DATA=$ROOT/duke/data_unilateral_$V; META=$ROOT/eval_duke_$V/$SITE/metadata_unilateral
OUT=$ROOT/eval_out/duke_$V/$M; mkdir -p $OUT/pred $OUT/cache
docker run --rm --gpus device=0 --ipc=host \
  -v $DATA:/data/$SITE/data_unilateral:ro -v $META:/data/$SITE/metadata_unilateral:ro \
  -v $ROOT/eval:/ckpt:ro -v $OUT:/scratch \
  -e DATA_DIR=/data -e SITE_NAME=$SITE -e SCRATCH_DIR=/scratch -e ODELIA_PREPROCESS_CACHE_DIR=/scratch/cache \
  jefftud/odelia:current python3 /MediSwarm/scripts/evaluation/predict.py --checkpoint /ckpt/${M}_swarm_global_final.pt --model-name $M --split test --batch-size 4 --output-dir /scratch/pred > $OUT/predict.log 2>&1
echo "exit $? ; $(ls $OUT/pred 2>/dev/null | tr "\n" " ")"; grep -iE "AUC-ROC:|Samples:|Traceback|Error" $OUT/predict.log | tail -4 | cut -c1-140
