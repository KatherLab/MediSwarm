#!/usr/bin/env bash
# eval_variant.sh <variant: old|BD|existing> <model: MST|1DivideAndConquer> [site=UMCU_1] [root=/mnt/swarm_alpha/bd_bench]
# The dataset loader reads /data/<SITE>/{data_unilateral,metadata_unilateral} inside the container,
# so the crop folder of the variant is bind-mounted straight to that path.
set -uo pipefail
V=$1; M=$2; SITE=${3:-UMCU_1}; ROOT=${4:-/mnt/swarm_alpha/bd_bench}
case $V in old|BD) DATA=$ROOT/challenge/data_unilateral_$V;; existing) DATA=$ROOT/eval_existing/$SITE/data_unilateral;; *) echo "variant?"; exit 2;; esac
META=$ROOT/eval_$V/$SITE/metadata_unilateral; [ -d $META ] || META=$ROOT/eval_old/$SITE/metadata_unilateral
OUT=$ROOT/eval_out/$V/$M; mkdir -p $OUT/pred $OUT/cache
docker run --rm --gpus device=0 --ipc=host \
  -v $DATA:/data/$SITE/data_unilateral:ro -v $META:/data/$SITE/metadata_unilateral:ro \
  -v $ROOT/eval:/ckpt:ro -v $OUT:/scratch \
  -e DATA_DIR=/data -e SITE_NAME=$SITE -e SCRATCH_DIR=/scratch -e ODELIA_PREPROCESS_CACHE_DIR=/scratch/cache \
  jefftud/odelia:current python3 /MediSwarm/scripts/evaluation/predict.py --checkpoint /ckpt/${M}_swarm_global_final.pt --model-name $M --split test --batch-size 4 --output-dir /scratch/pred > $OUT/predict.log 2>&1
echo "exit $? ; outputs: $(ls $OUT/pred 2>/dev/null | tr "\n" " ")"; grep -iE "auc|accuracy|samples|Loaded|error|Traceback" $OUT/predict.log | tail -8 | cut -c1-160
