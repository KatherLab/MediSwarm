#!/usr/bin/env bash

set -e

# prepare pre-trained model weights for being included in Docker image

if [[ -n "$MEDISWARM_BUILD_CACHE_DIR" ]]; then
    CACHE_DIR=$MEDISWARM_BUILD_CACHE_DIR
else
    CACHE_DIR=$(mktemp -d)
fi
TARGET_DIR=$1

MODEL_WEIGHTS_FILE_DINO=$CACHE_DIR'/torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth'
MODEL_WEIGHTS_FILE_DINO_URL=https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
MODEL_WEIGHTS_FILE_DINO_HASH=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9

MODEL_LICENSE_FILE_DINO=$CACHE_DIR'/torch_home_cache/hub/facebookresearch_dinov2_main/LICENSE'
MODEL_LICENSE_FILE_DINO_URL=https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip
MODEL_LICENSE_FILE_DINO_HASH=600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7

MODEL_WEIGHTS_FILE_MVIT=$CACHE_DIR'/application/jobs/challenge_3agaldran/app/custom/models/mvit_v2_s-ae3be167.pth'
MODEL_WEIGHTS_FILE_MVIT_URL=https://download.pytorch.org/models/mvit_v2_s-ae3be167.pth
MODEL_WEIGHTS_FILE_MVIT_HASH=ae3be16733081f6d1cd40e4ab980ca23d6df6dc6486d15ada05a5e8ab8c9b975

MODEL_WEIGHTS_FILE_ODAC=$CACHE_DIR'/application/jobs/challenge_1DivideAndConquer/app/custom/models/checkpoint_final.pth'
MODEL_WEIGHTS_FILE_ODAC_HASH=ed686907205fb0cb752dc987851eb9d0191034599c5d204c7ec1ad9ff91dd758

# SAM-Med2D ViT-B (MST_SAMMed2D backbone). Optional: the release is distributed via Google
# Drive / Baidu (https://github.com/OpenGVLab/SAM-Med2D), so it is not fetched automatically.
# Drop the file into the build cache and it is shipped at /MediSwarm/pretrained_weights/.
MODEL_WEIGHTS_FILE_SAMMED2D=$CACHE_DIR'/pretrained_weights/sam-med2d_b.pth'

MODEL_WEIGHTS_FILE_RESNETTEN_HASH=afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605
MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_CONTENTS=deeeaa1e2a342b63e28b61e9d993b63e79c4b437
MODEL_WEIGHTS_FILE_RESNETTEN=$CACHE_DIR'/hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet10/blobs/'$MODEL_WEIGHTS_FILE_RESNETTEN_HASH
MODEL_WEIGHTS_FILE_RESNETTEN_SYMLINK=$CACHE_DIR'/hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet10/snapshots/'$MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_CONTENTS'/resnet_10_23dataset.pth'
MODEL_WEIGHTS_FILE_RESNETTEN_URL='https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/resolve/'$MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_CONTENTS'/resnet_10_23dataset.pth'
MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_FILE=$CACHE_DIR'/hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet10/refs/main'
# e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_HASH=61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4
MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_CONTENTS=758fe285bc8ab565eb4f9f965810f1d1a3f79491
MODEL_WEIGHTS_FILE_RESNETEIGHTEEN=$CACHE_DIR'/hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/blobs/'$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_HASH                 # this is different from ResNet10
MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_SYMLINK=$CACHE_DIR'/hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/snapshots/'$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_CONTENTS'/resnet_18_23dataset.pth'
MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_URL=https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/resolve/9d7c4c66c77bff89b79631369426612f09d0fe9b/resnet_18_23dataset.pth   # this is different from ResNet10
MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_FILE=$CACHE_DIR'/hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/refs/main'

_cache_file_wget () {
    url=$1
    filename=$2

    if [[ ! -f "$filename" ]]; then
        echo "File" $filename "not available, attempting download from" $url
        mkdir -p $(dirname "$filename")
        wget "$url" -O "$filename"
    fi
}

_cache_odac_model () {
    if [[ ! -f "$MODEL_WEIGHTS_FILE_ODAC" ]]; then
        echo "Downloading 1DivideAndConquer checkpoint from Google Drive..."
        mkdir -p $(dirname "$MODEL_WEIGHTS_FILE_ODAC")
        GDOWN_CMD=$(command -v gdown || echo "")
        # Verify gdown actually works (not just a stale shim with missing module)
        if [[ -n "$GDOWN_CMD" ]] && ! "$GDOWN_CMD" --version &>/dev/null; then
            echo "Found gdown at $GDOWN_CMD but it is broken, ignoring..."
            GDOWN_CMD=""
        fi
        if [[ -z "$GDOWN_CMD" && -x "$CACHE_DIR/.venv/bin/gdown" ]]; then
            GDOWN_CMD="$CACHE_DIR/.venv/bin/gdown"
        fi
        if [[ -z "$GDOWN_CMD" ]]; then
            echo "gdown not found, installing into temporary venv..."
            TMPVENV=$(mktemp -d)/gdown_venv
            python3 -m venv "$TMPVENV"
            "$TMPVENV/bin/pip" install --quiet gdown
            GDOWN_CMD="$TMPVENV/bin/gdown"
        fi
        "$GDOWN_CMD" 1bVmZHvI7H1H9YTIMy11zwU2p95W4Y_W6 -O "$MODEL_WEIGHTS_FILE_ODAC"
        if [[ -n $TMPVENV ]]; then
            echo "deleting " $TMPENV
            rm -rf "$TMPVENV"
        fi
    fi
}

_cache_file_resnet () {
    MODEL_WEIGHTS_FILE_URL=$1
    MODEL_WEIGHTS_FILE=$2
    MODEL_WEIGHTS_FILE_SYMLINK=$3
    MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_FILE=$4
    MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_CONTENTS=$5

    _cache_file_wget "$MODEL_WEIGHTS_FILE_URL" "$MODEL_WEIGHTS_FILE"
    if [[ ! -f "$MODEL_WEIGHTS_FILE_SYMLINK" ]]; then
        mkdir -p $(dirname $MODEL_WEIGHTS_FILE_RESNETTEN_SYMLINK)
        cd $(dirname $MODEL_WEIGHTS_FILE_RESNETTEN_SYMLINK)
        ln -s -f ../../blobs/$(basename $MODEL_WEIGHTS_FILE_RESNETTEN) $(basename $MODEL_WEIGHTS_FILE_RESNETTEN_SYMLINK)
    fi
    if [[ ! -f "$MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_FILE" ]]; then
        mkdir -p $(dirname $MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_FILE)
        echo $MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_CONTENTS > $MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_FILE
    fi

}

cache_files () {
    _cache_file_wget "$MODEL_WEIGHTS_FILE_DINO_URL" "$MODEL_WEIGHTS_FILE_DINO"

    if [[ ! -f "$MODEL_LICENSE_FILE_DINO" ]]; then
        echo "Pre-trained model license not available. Attempting download."
        HUBDIR=$(dirname $(dirname "$MODEL_LICENSE_FILE_DINO"))
        _cache_file_wget "$MODEL_LICENSE_FILE_DINO_URL" "$CACHE_DIR/tmp/dinov2.zip"
        mkdir -p $(dirname "$HUBDIR")
        unzip "$CACHE_DIR/tmp/dinov2.zip" -d "$HUBDIR"
        mv "$HUBDIR/dinov2-main" "$HUBDIR/""$(basename $(dirname "$MODEL_LICENSE_FILE_DINO"))"
        rm -f "$CACHE_DIR/tmp/dinov2.zip"
        touch "$HUBDIR/trusted_list"
    fi

    _cache_odac_model

    _cache_file_wget "$MODEL_WEIGHTS_FILE_MVIT_URL" "$MODEL_WEIGHTS_FILE_MVIT"

    _cache_file_resnet "$MODEL_WEIGHTS_FILE_RESNETTEN_URL" "$MODEL_WEIGHTS_FILE_RESNETTEN" "$MODEL_WEIGHTS_FILE_RESNETTEN_SYMLINK" "$MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_FILE" "$MODEL_WEIGHTS_FILE_RESNETTEN_MAIN_CONTENTS"
    _cache_file_resnet "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_URL" "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN" "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_SYMLINK" "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_FILE" "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_CONEIGHTEENTS"
}


_verify_hash() {
    hash_value=$1
    filename=$2

    echo $hash_value $filename

    if echo $hash_value"  "$filename | sha256sum --check -; then
        echo "Hash" $1 "for" $2 "verified successfully."
    else
        echo "Unexpected file hash."
        exit 1
    fi
}

verify_files () {
    _verify_hash $MODEL_WEIGHTS_FILE_DINO_HASH $MODEL_WEIGHTS_FILE_DINO
    _verify_hash $MODEL_LICENSE_FILE_DINO_HASH $MODEL_LICENSE_FILE_DINO
    _verify_hash $MODEL_WEIGHTS_FILE_MVIT_HASH $MODEL_WEIGHTS_FILE_MVIT
    _verify_hash $MODEL_WEIGHTS_FILE_ODAC_HASH $MODEL_WEIGHTS_FILE_ODAC
    _verify_hash $MODEL_WEIGHTS_FILE_RESNETTEN_HASH $MODEL_WEIGHTS_FILE_RESNETTEN
    _verify_hash $MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_HASH $MODEL_WEIGHTS_FILE_RESNETEIGHTEEN
}


copy_files() {
    # Copy MST pre-trained weights.
    #
    # #408: the torch.hub cache holds the checkpoint (~85 MB, required) *and* a full
    # clone of the dinov2 repo. The clone cannot simply be dropped -- models/mst.py
    # calls torch.hub.load('facebookresearch/dinov2', ...), which imports hubconf.py
    # and the dinov2 package from it. But its docs/ (1.8 MB of PNGs), notebooks/
    # (1.3 MB of .ipynb) and .github/ are documentation and media that are never
    # imported, and they end up in a Docker image layer. Prune only those.
    cp -r "$CACHE_DIR/torch_home_cache" "$TARGET_DIR/torch_home_cache"
    _dinov2_dir="$TARGET_DIR/torch_home_cache/hub/facebookresearch_dinov2_main"
    if [[ -d "$_dinov2_dir" ]]; then
        rm -rf "$_dinov2_dir/docs" "$_dinov2_dir/notebooks" "$_dinov2_dir/.github"
        # Fail loudly rather than shipping an image whose backbone cannot load: a
        # missing hub artifact only surfaces at training time, not in CI.
        for _required in \
            "$TARGET_DIR/torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth" \
            "$_dinov2_dir/hubconf.py" \
            "$_dinov2_dir/dinov2" ; do
            if [[ ! -e "$_required" ]]; then
                echo "ERROR: required torch.hub artifact missing after prune: $_required" >&2
                exit 1
            fi
        done
    fi
    chmod a+rX "$TARGET_DIR/torch_home_cache" -R

    # Copy ResNet18 pre-trained weights
    cp -r "$CACHE_DIR/hf_home_cache" "$TARGET_DIR/hf_home_cache"
    chmod a+rX "$TARGET_DIR/hf_home_cache" -R

    # Copy challenge model weights to a SEPARATE directory outside the job folders.
    # This is critical: NVFlare packages the entire job folder when submitting a job,
    # so .pth files inside job dirs would be transferred over the network to every client.
    # Instead we store them at /MediSwarm/pretrained_weights/ in the Docker image, and
    # the model code falls back to that path at runtime.
    WEIGHTS_DIR="$TARGET_DIR/MediSwarm/pretrained_weights"
    mkdir -p "$WEIGHTS_DIR"
    echo "Copy pretrained model weights to $WEIGHTS_DIR/ ..."

    # challenge_1DivideAndConquer: checkpoint_final.pth
    echo "1DivideAndConquer: copying checkpoint_final.pth"
    cp "$CACHE_DIR/application/jobs/challenge_1DivideAndConquer/app/custom/models/checkpoint_final.pth"  "$WEIGHTS_DIR/"

    # challenge_3agaldran: mvit_v2_s-ae3be167.pth (PyTorch pretrained weights)
    echo "3agaldran: copying mvit_v2_s-ae3be167.pth"
    cp "$MODEL_WEIGHTS_FILE_MVIT"  "$WEIGHTS_DIR/"

    # MST_SAMMed2D: sam-med2d_b.pth, only if the operator placed it in the cache
    if [[ -f "$MODEL_WEIGHTS_FILE_SAMMED2D" ]]; then
        echo "SAM-Med2D: copying sam-med2d_b.pth"
        cp "$MODEL_WEIGHTS_FILE_SAMMED2D" "$WEIGHTS_DIR/"
    else
        echo "SAM-Med2D: $MODEL_WEIGHTS_FILE_SAMMED2D not in the build cache — MST_SAMMed2D will start from random weights in this image"
    fi

    chmod a+rX "$WEIGHTS_DIR" -R
}


cache_files
verify_files
copy_files

if [[ -z "$MEDISWARM_BUILD_CACHE_DIR" ]]; then
    rm -rf "$CACHE_DIR"
fi
