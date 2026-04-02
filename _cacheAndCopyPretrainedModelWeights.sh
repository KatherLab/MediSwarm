#!/usr/bin/env bash

set -e

# prepare pre-trained model weights for being included in Docker image

SOURCE_DIR=$1
TARGET_DIR=$2
MODEL_WEIGHTS_FILE=$SOURCE_DIR'/docker_config/torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth'
MODEL_LICENSE_FILE=$SOURCE_DIR'/docker_config/torch_home_cache/hub/facebookresearch_dinov2_main/LICENSE'

cache_files () {
    if [[ ! -f $MODEL_WEIGHTS_FILE || ! -f $MODEL_LICENSE_FILE ]]; then
        echo "Pre-trained model not available. Attempting download"
        HUBDIR=$(dirname $(dirname $MODEL_LICENSE_FILE))
        mkdir -p $(dirname $MODEL_WEIGHTS_FILE)
        wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -O $MODEL_WEIGHTS_FILE
        wget https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip -O /tmp/dinov2.zip
        unzip /tmp/dinov2.zip -d $HUBDIR
        mv $HUBDIR/dinov2-main $HUBDIR/$(basename $(dirname $MODEL_LICENSE_FILE))
        touch $HUBDIR/trusted_list
    fi
}

verify_files () {
    if echo 2e405cee1bad14912278296d4f42e993 $MODEL_WEIGHTS_FILE | md5sum --check - && echo 153d2db1c329326a2d9f881317ea942e $MODEL_LICENSE_FILE | md5sum --check -; then
        echo "File contents verified successfully."
    else
        echo "Unexpected file contents."
        exit 1
    fi
}

copy_files() {
    cp -r $SOURCE_DIR/docker_config/torch_home_cache $TARGET_DIR/torch_home_cache
    chmod a+rX $TARGET_DIR/torch_home_cache -R

    # Copy challenge model weights to a SEPARATE directory outside the job folders.
    # This is critical: NVFlare packages the entire job folder when submitting a job,
    # so .pth files inside job dirs would be transferred over the network to every client.
    # Instead we store them at /MediSwarm/pretrained_weights/ in the Docker image, and
    # the model code falls back to that path at runtime.
    WEIGHTS_DIR="$TARGET_DIR/MediSwarm/pretrained_weights"
    mkdir -p "$WEIGHTS_DIR"
    echo "Copy pretrained model weights to $WEIGHTS_DIR/ ..."

    # challenge_1DivideAndConquer: checkpoint_final.pth
    echo "1DivideAndConquer: caching checkpoint_final.pth"
    if [[ -f "$SOURCE_DIR/application/jobs/challenge_1DivideAndConquer/app/custom/models/checkpoint_final.pth" ]]; then
        cp "$SOURCE_DIR/application/jobs/challenge_1DivideAndConquer/app/custom/models/checkpoint_final.pth" \
           "$WEIGHTS_DIR/"
    else
        echo "Downloading 1DivideAndConquer checkpoint from Google Drive..."
        GDOWN_CMD=$(command -v gdown || echo "")
        if [[ -z "$GDOWN_CMD" && -x "$SOURCE_DIR/.venv/bin/gdown" ]]; then
            GDOWN_CMD="$SOURCE_DIR/.venv/bin/gdown"
        fi
        if [[ -z "$GDOWN_CMD" ]]; then
            echo "gdown not found, installing into temporary venv..."
            TMPVENV=$(mktemp -d)/gdown_venv
            python3 -m venv "$TMPVENV"
            "$TMPVENV/bin/pip" install --quiet gdown
            GDOWN_CMD="$TMPVENV/bin/gdown"
        fi
        "$GDOWN_CMD" 1bVmZHvI7H1H9YTIMy11zwU2p95W4Y_W6 -O "$WEIGHTS_DIR/checkpoint_final.pth"
    fi

    # challenge_3agaldran: mvit_v2_s-ae3be167.pth (PyTorch pretrained weights)
    echo "3agaldran: caching mvit_v2_s-ae3be167.pth"
    if [[ -f "$SOURCE_DIR/application/jobs/challenge_3agaldran/app/custom/models/mvit_v2_s-ae3be167.pth" ]]; then
        cp "$SOURCE_DIR/application/jobs/challenge_3agaldran/app/custom/models/mvit_v2_s-ae3be167.pth" \
           "$WEIGHTS_DIR/"
    else
        echo "Downloading 3agaldran checkpoint..."
        wget https://download.pytorch.org/models/mvit_v2_s-ae3be167.pth -O "$WEIGHTS_DIR/mvit_v2_s-ae3be167.pth"
    fi

    chmod a+rX "$WEIGHTS_DIR" -R
}

cache_files
verify_files
copy_files
