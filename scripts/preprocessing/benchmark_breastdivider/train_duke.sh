#!/usr/bin/env bash
# train_duke.sh <variant: old|BD> <model> <epochs>   local (non-swarm) MediSwarm training on Duke fold 0
set -uo pipefail
V=$1; M=$2; E=$3; F=${4:-0}; SITE=DUKE_1; ROOT=/mnt/swarm_alpha/bd_bench
DATA=$ROOT/duke_full/data_unilateral_$V; META=$ROOT/train_duke_$V/$SITE/metadata_unilateral
OUT=$ROOT/train_out/duke_$V/${M}_fold$F; mkdir -p $OUT/home $OUT/torch_home $OUT/hf_home
docker run --rm --gpus device=0 --ipc=host --shm-size=16g -u $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v $DATA:/data/$SITE/data_unilateral:ro -v $META:/data/$SITE/metadata_unilateral:ro -v $OUT:/scratch \
  -e HOME=/scratch/home -e SITE_NAME=$SITE -e DATA_DIR=/data -e SCRATCH_DIR=/scratch \
  -e TORCH_HOME=/scratch/torch_home -e HF_HOME=/scratch/hf_home \
  -e GPU_DEVICE=0 -e MODEL_NAME=$M -e NUM_EPOCHS=$E -e FOLD=$F -e CONFIG=unilateral -e TRAINING_MODE=local_training \
  -e ODELIA_NUM_WORKERS=8 -e MEDISWARM_VERSION=bd_bench -e TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_system \
  jefftud/odelia:current /bin/bash -c "/MediSwarm/application/jobs/ODELIA_ternary_classification/app/custom/main.py" > $OUT/train.log 2>&1
echo "train exit $?"; grep -E "Best model checkpoint|Last model saved|Traceback|Error" $OUT/train.log | tail -4 | cut -c1-160
