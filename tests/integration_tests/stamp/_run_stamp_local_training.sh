#!/usr/bin/env bash
# STAMP local training: full local training on synthetic data (3 epochs).
# Must run inside the STAMP Docker image.
set -e

echo "=== STAMP Local Training ==="

cd /MediSwarm

# Set environment for STAMP training
export TRAINING_MODE="local_training"
export SITE_NAME="${SITE_NAME:-client_A}"
export DATA_DIR="${DATA_DIR:-/data}"
export SCRATCH_DIR="${SCRATCH_DIR:-/scratch}"
export NUM_EPOCHS=3
export TORCH_HOME=/torch_home

# STAMP-specific env vars
export STAMP_CLINI_TABLE="${DATA_DIR}/${SITE_NAME}/clini_table.csv"
export STAMP_FEATURE_DIR="${DATA_DIR}/${SITE_NAME}/features"
export STAMP_GROUND_TRUTH_LABEL="Diagnosis"
export STAMP_PATIENT_LABEL="PATIENT"
export STAMP_TASK="classification"
export STAMP_MODEL_NAME="vit"
export STAMP_DIM_INPUT="1024"
export STAMP_NUM_CLASSES="3"
export STAMP_BAG_SIZE="64"
export STAMP_BATCH_SIZE="8"
export STAMP_MAX_EPOCHS="3"
export STAMP_PATIENCE="3"
export STAMP_NUM_WORKERS="0"
export STAMP_SEED="42"

echo "STAMP_CLINI_TABLE=$STAMP_CLINI_TABLE"
echo "STAMP_FEATURE_DIR=$STAMP_FEATURE_DIR"
echo "SITE_NAME=$SITE_NAME"

# Run STAMP training in local mode
python3 application/jobs/STAMP_classification/app/custom/main.py

echo "=== STAMP Local Training PASSED ==="
