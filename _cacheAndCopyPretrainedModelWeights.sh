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
    # Copy challenge model weights
    echo "Copy pretrained model weights for challenge models... "
    CHALLENGE_MODEL_DIR="$TARGET_DIR/MediSwarm/application/jobs/ODELIA_ternary_classification/app/custom/models/challenge"
    
    echo "1DivideAndConquer: from $SOURCE_DIR/.../checkpoint_final.pth to $CHALLENGE_MODEL_DIR/1DivideAndConquer/"
    mkdir -p "$CHALLENGE_MODEL_DIR/1DivideAndConquer"
    cp "$SOURCE_DIR/application/jobs/ODELIA_ternary_classification/app/custom/models/challenge/1DivideAndConquer/checkpoint_final.pth" \
       "$CHALLENGE_MODEL_DIR/1DivideAndConquer/"

    echo "3agaldran: from $SOURCE_DIR/.../mvit_v2_s-ae3be167.pth to $CHALLENGE_MODEL_DIR/3agaldran/"
    mkdir -p "$CHALLENGE_MODEL_DIR/3agaldran"
    cp "$SOURCE_DIR/application/jobs/ODELIA_ternary_classification/app/custom/models/challenge/3agaldran/mvit_v2_s-ae3be167.pth" \
       "$CHALLENGE_MODEL_DIR/3agaldran/"
}

cache_files
verify_files
copy_files
