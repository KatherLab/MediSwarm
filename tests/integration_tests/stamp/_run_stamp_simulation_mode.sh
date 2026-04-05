#!/usr/bin/env bash
# STAMP NVFlare simulation mode: 2 clients, 2 rounds.
# Must run inside the STAMP Docker image.
set -e

echo "=== STAMP Simulation Mode ==="

cd /MediSwarm

export TMPDIR=$(mktemp -d)
export DATA_DIR="${DATA_DIR:-/data}"
export SCRATCH_DIR="${SCRATCH_DIR:-/scratch}"

# Copy the job definition so we can modify num_rounds
cp -RL application/jobs/STAMP_classification ${TMPDIR}/STAMP_classification

# Override to 2 rounds for testing
sed -i 's/num_rounds = .*/num_rounds = 2/' ${TMPDIR}/STAMP_classification/app/config/config_fed_server.conf

# Set environment for STAMP training
export TRAINING_MODE="swarm"
export SITE_NAME="client_A"
export TORCH_HOME=/torch_home
# Must match num_rounds above so OneCycleLR scheduler has enough total_steps
export STAMP_NUM_ROUNDS="2"

# STAMP-specific env vars — both clients read the same data keyed by SITE_NAME
# In simulation mode, SITE_NAME is overridden per-client by NVFlare, but
# DATA_DIR layout needs both client_A/ and client_B/ subdirectories.
# For simulation we use client_A's data for all clients via symlink or
# by setting the env vars to point at a shared location.
export STAMP_CLINI_TABLE="${DATA_DIR}/client_A/clini_table.csv"
export STAMP_FEATURE_DIR="${DATA_DIR}/client_A/features"
export STAMP_GROUND_TRUTH_LABEL="Diagnosis"
export STAMP_PATIENT_LABEL="PATIENT"
export STAMP_TASK="classification"
export STAMP_MODEL_NAME="vit"
export STAMP_DIM_INPUT="1024"
export STAMP_NUM_CLASSES="3"
export STAMP_BAG_SIZE="64"
export STAMP_BATCH_SIZE="8"
export STAMP_MAX_EPOCHS="2"
export STAMP_PATIENCE="2"
export STAMP_NUM_WORKERS="0"
export STAMP_SEED="42"

# Override weighted epoch computation to keep test fast.
# With 15 patients: epochs = 2 * (15 / 15) = 2, clamped to [1, 4].
export STAMP_EPOCHS_PER_ROUND="2"
export STAMP_EPOCHS_REFERENCE_DATASET_SIZE="15"
export STAMP_EPOCHS_MAX_CAP="4"

echo "Running NVFlare simulator: 2 clients, 2 rounds"
nvflare simulator \
    -w /tmp/stamp_simulation \
    -n 2 \
    -t 2 \
    ${TMPDIR}/STAMP_classification \
    -c client_A,client_B

rm -rf ${TMPDIR}

echo "=== STAMP Simulation Mode PASSED ==="
