#!/usr/bin/env bash

set -e

# prepare pre-trained model weights for being included in Docker image

SOURCE_DIR=$1
TARGET_DIR=$2

MODEL_WEIGHTS_FILE_DINO=$SOURCE_DIR'/docker_config/torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth'
MODEL_WEIGHTS_FILE_DINO_URL=https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
MODEL_WEIGHTS_FILE_DINO_SHA=cf1f2360da4adbffe57342f0fa067fe759d9223a

MODEL_LICENSE_FILE_DINO=$SOURCE_DIR'/docker_config/torch_home_cache/hub/facebookresearch_dinov2_main/LICENSE'
MODEL_LICENSE_FILE_DINO_URL=https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip
MODEL_LICENSE_FILE_DINO_SHA=83fe23afe70f538ae3ea0969cf8b9d0701a976b1

MODEL_WEIGHTS_FILE_MVIT=$SOURCE_DIR'/application/jobs/challenge_3agaldran/app/custom/models/mvit_v2_s-ae3be167.pth'
MODEL_WEIGHTS_FILE_MVIT_URL=https://download.pytorch.org/models/mvit_v2_s-ae3be167.pth
MODEL_WEIGHTS_FILE_MVIT_SHA=94826d379879465b184689212bd62e62d50f40df

# b6d0badeb218ec2eb0b07300a53b8b855810019b  checkpoint_final.pth


_cache_file_wget () {
    url=$1
    filename=$2

    if [[ ! -f $filename ]]; then
        echo "File" $filename "not available, attempting download from" $url
        mkdir -p $(dirname $filename)
        wget $url -O $filename
    fi
}

cache_files () {
    _cache_file_wget $MODEL_WEIGHTS_FILE_DINO_URL $MODEL_WEIGHTS_FILE_DINO

    if [[ ! -f $MODEL_LICENSE_FILE_DINO ]]; then
        echo "Pre-trained model license not available. Attempting download."
        HUBDIR=$(dirname $(dirname $MODEL_LICENSE_FILE_DINO))
        _cache_file_wget $MODEL_LICENSE_FILE_DINO_URL $SOURCE_DIR/tmp/dinov2.zip
        unzip $SOURCE_DIR/tmp/dinov2.zip -d $HUBDIR
        mv $HUBDIR/dinov2-main $HUBDIR/$(basename $(dirname $MODEL_LICENSE_FILE_DINO))
        rm -f $SOURCE_DIR/tmp/dinov2.zip
        touch $HUBDIR/trusted_list
    fi

    _cache_file_wget $MODEL_WEIGHTS_FILE_MVIT_URL $MODEL_WEIGHTS_FILE_MVIT
}

_verify_hash() {
    hash_value=$1
    filename=$2

    echo $hash_value $filename

    if echo $hash_value"  "$filename | shasum --check -; then
        echo "Hash" $1 "for" $2 "verified successfully."
    else
        echo "Unexpected file hash."
        exit 1
    fi
}

verify_files () {
    _verify_hash $MODEL_WEIGHTS_FILE_DINO_SHA $MODEL_WEIGHTS_FILE_DINO
    _verify_hash $MODEL_LICENSE_FILE_DINO_SHA $MODEL_LICENSE_FILE_DINO
    _verify_hash $MODEL_WEIGHTS_FILE_MVIT_SHA $MODEL_WEIGHTS_FILE_MVIT
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
        # Verify gdown actually works (not just a stale shim with missing module)
        if [[ -n "$GDOWN_CMD" ]] && ! "$GDOWN_CMD" --version &>/dev/null; then
            echo "Found gdown at $GDOWN_CMD but it is broken, ignoring..."
            GDOWN_CMD=""
        fi
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

    cp "$MODEL_WEIGHTS_FILE_MVIT" "$WEIGHTS_DIR/"

    chmod a+rX "$WEIGHTS_DIR" -R
}

cache_files
verify_files
copy_files
