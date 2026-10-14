#!/usr/bin/env bash
#
# Put the pre-trained model weights the ODELIA image ships into a Docker build context.
#
#   _cacheAndCopyPretrainedModelWeights.sh <build-context-dir>
#
# Fills, next to (not inside) <build-context-dir>/MediSwarm:
#   torch_home_cache/    -> /torch_home                     DINOv2 (MST and its variants)
#   hf_home_cache/       -> /huggingface_home               MedicalNet ResNet18
#   pretrained_weights/  -> /MediSwarm/pretrained_weights   1DivideAndConquer, 3agaldran, MST_SAMMed2D
# Dockerfile_ODELIA copies each into its own layer below the per-commit code, so a code-only
# rebuild that uses the docker cache does not ship the ~3.5 GB of weights again.
#
# MEDISWARM_BUILD_CACHE_DIR (absolute path): persistent cache. Downloads land there and are reused;
# without it a temporary cache is filled and deleted. Every cached file is checked against its
# SHA-256 BEFORE anything is copied, so a truncated download or a stale file never reaches an image.
#
# SAM-Med2D (pretrained_weights/sam-med2d_b.pth in the cache) is published only on Google Drive and
# Baidu (https://github.com/OpenGVLab/SAM-Med2D), so it cannot be fetched here: an operator places it
# in the cache. It is hash-checked whenever present. It is REQUIRED (the build stops) when
#   MEDISWARM_REQUIRE_SAMMED2D=1, or
#   MEDISWARM_REQUIRE_SAMMED2D is unset and MEDISWARM_IMAGE_VERSION names a release, i.e. anything but
#   the dev form <version>-dev.<date>.<sha> that getVersionNumber.sh computes and every CI workflow
#   exports for its whole run (#428). CI runners hold no SAM-Med2D copy.
# MEDISWARM_REQUIRE_SAMMED2D=0 makes it optional again; an image without it gives MST_SAMMed2D no
# pretrained encoder, which is why a release must never be built that way by accident.

# --------------------------------------------------------------------------- what we ship

MODEL_WEIGHTS_FILE_DINO_URL=https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
MODEL_WEIGHTS_FILE_DINO_HASH=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9

MODEL_LICENSE_FILE_DINO_URL=https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip
MODEL_LICENSE_FILE_DINO_HASH=600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7

MODEL_WEIGHTS_FILE_MVIT_URL=https://download.pytorch.org/models/mvit_v2_s-ae3be167.pth
MODEL_WEIGHTS_FILE_MVIT_HASH=ae3be16733081f6d1cd40e4ab980ca23d6df6dc6486d15ada05a5e8ab8c9b975

MODEL_WEIGHTS_FILE_ODAC_GDRIVE_ID=1bVmZHvI7H1H9YTIMy11zwU2p95W4Y_W6
MODEL_WEIGHTS_FILE_ODAC_HASH=ed686907205fb0cb752dc987851eb9d0191034599c5d204c7ec1ad9ff91dd758

# SAM-Med2D ViT-B, the MST_SAMMed2D backbone (2,561,829,701 bytes; the copy CAM validated).
MODEL_WEIGHTS_FILE_SAMMED2D_HASH=921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18

MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_URL=https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/resolve/9d7c4c66c77bff89b79631369426612f09d0fe9b/resnet_18_23dataset.pth
MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_HASH=61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4
MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_CONTENTS=758fe285bc8ab565eb4f9f965810f1d1a3f79491

# Cache layout; separate from the constants above so the unit tests can point it at a fake cache.
set_cache_paths () {
    CACHE_DIR=$1
    MODEL_WEIGHTS_FILE_DINO=$CACHE_DIR'/torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth'
    MODEL_LICENSE_FILE_DINO=$CACHE_DIR'/torch_home_cache/hub/facebookresearch_dinov2_main/LICENSE'
    MODEL_WEIGHTS_FILE_MVIT=$CACHE_DIR'/application/jobs/challenge_3agaldran/app/custom/models/mvit_v2_s-ae3be167.pth'
    MODEL_WEIGHTS_FILE_ODAC=$CACHE_DIR'/application/jobs/challenge_1DivideAndConquer/app/custom/models/checkpoint_final.pth'
    MODEL_WEIGHTS_FILE_SAMMED2D=$CACHE_DIR'/pretrained_weights/sam-med2d_b.pth'
    MODEL_WEIGHTS_FILE_RESNETEIGHTEEN=$CACHE_DIR'/hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/blobs/'$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_HASH
    MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_SYMLINK=$CACHE_DIR'/hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/snapshots/'$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_CONTENTS'/resnet_18_23dataset.pth'
    MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_FILE=$CACHE_DIR'/hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/refs/main'
}

# --------------------------------------------------------------------------- download

_cache_file_wget () {
    url=$1
    filename=$2

    if [[ ! -f "$filename" ]]; then
        echo "File $filename not available, attempting download from $url"
        mkdir -p "$(dirname "$filename")"
        # Download next to the target and rename only on success: an interrupted download must not
        # leave a partial file in a persistent cache, where it would fail the hash check every build.
        wget "$url" -O "$filename.part"
        mv "$filename.part" "$filename"
    fi
}

_cache_odac_model () {
    if [[ ! -f "$MODEL_WEIGHTS_FILE_ODAC" ]]; then
        echo "Downloading 1DivideAndConquer checkpoint from Google Drive..."
        mkdir -p "$(dirname "$MODEL_WEIGHTS_FILE_ODAC")"
        GDOWN_CMD=$(command -v gdown || echo "")
        # Verify gdown actually works (not just a stale shim with missing module)
        if [[ -n "$GDOWN_CMD" ]] && ! "$GDOWN_CMD" --version &>/dev/null; then
            echo "Found gdown at $GDOWN_CMD but it is broken, ignoring..."
            GDOWN_CMD=""
        fi
        if [[ -z "$GDOWN_CMD" && -x "$CACHE_DIR/.venv/bin/gdown" ]]; then
            GDOWN_CMD="$CACHE_DIR/.venv/bin/gdown"
        fi
        TMPVENV_ROOT=""
        if [[ -z "$GDOWN_CMD" ]]; then
            echo "gdown not found, installing into temporary venv..."
            TMPVENV_ROOT=$(mktemp -d)
            python3 -m venv "$TMPVENV_ROOT/gdown_venv"
            "$TMPVENV_ROOT/gdown_venv/bin/pip" install --quiet gdown
            GDOWN_CMD="$TMPVENV_ROOT/gdown_venv/bin/gdown"
        fi
        "$GDOWN_CMD" "$MODEL_WEIGHTS_FILE_ODAC_GDRIVE_ID" -O "$MODEL_WEIGHTS_FILE_ODAC.part"
        mv "$MODEL_WEIGHTS_FILE_ODAC.part" "$MODEL_WEIGHTS_FILE_ODAC"
        if [[ -n "$TMPVENV_ROOT" ]]; then
            echo "deleting $TMPVENV_ROOT"
            rm -rf "$TMPVENV_ROOT"
        fi
    fi
}

write_resnet18_ref () {
    # refs/main must hold the bare commit hash. huggingface_hub reads it verbatim, so the trailing
    # newline that `echo` used to add made the snapshot path '<hash>\n/...': the baked cache never
    # resolved offline and ResNet18 failed at sites without internet. Rewritten on every build so a
    # cache filled by the old script is repaired as well.
    mkdir -p "$(dirname "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_FILE")"
    printf '%s' "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_CONTENTS" > "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_FILE"
}

cache_files () {
    _cache_file_wget "$MODEL_WEIGHTS_FILE_DINO_URL" "$MODEL_WEIGHTS_FILE_DINO"

    if [[ ! -f "$MODEL_LICENSE_FILE_DINO" ]]; then
        echo "Pre-trained model license not available. Attempting download."
        HUBDIR=$(dirname "$(dirname "$MODEL_LICENSE_FILE_DINO")")
        _cache_file_wget "$MODEL_LICENSE_FILE_DINO_URL" "$CACHE_DIR/tmp/dinov2.zip"
        mkdir -p "$(dirname "$HUBDIR")"
        unzip "$CACHE_DIR/tmp/dinov2.zip" -d "$HUBDIR"
        mv "$HUBDIR/dinov2-main" "$HUBDIR/$(basename "$(dirname "$MODEL_LICENSE_FILE_DINO")")"
        rm -f "$CACHE_DIR/tmp/dinov2.zip"
        touch "$HUBDIR/trusted_list"
    fi

    _cache_odac_model

    _cache_file_wget "$MODEL_WEIGHTS_FILE_MVIT_URL" "$MODEL_WEIGHTS_FILE_MVIT"

    _cache_file_wget "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_URL" "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN"
    if [[ ! -f "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_SYMLINK" ]]; then
        mkdir -p "$(dirname "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_SYMLINK")"
        # A relative target, as huggingface_hub lays out its cache; no cd needed for that.
        ln -s -f "../../blobs/$(basename "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN")" "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_SYMLINK"
    fi
    write_resnet18_ref
}

# --------------------------------------------------------------------------- verify

_verify_hash () {
    hash_value=$1
    filename=$2

    if [[ ! -f "$filename" ]]; then
        echo "ERROR: $filename is missing from the build cache." >&2
        exit 1
    fi
    if printf '%s  %s\n' "$hash_value" "$filename" | sha256sum --check --quiet - ; then
        echo "Hash $hash_value for $filename verified successfully."
    else
        echo "ERROR: unexpected SHA-256 for $filename" >&2
        echo "       expected $hash_value" >&2
        echo "       got      $(sha256sum "$filename" | cut -d' ' -f1)" >&2
        echo "       Replace or delete the file (a deleted download is fetched again) and rebuild." >&2
        exit 1
    fi
}

# True when this build must ship SAM-Med2D (see the header for the rule).
sammed2d_required () {
    case "${MEDISWARM_REQUIRE_SAMMED2D:-}" in
        1) return 0 ;;
        0) return 1 ;;
        "") ;;
        *)
            echo "ERROR: MEDISWARM_REQUIRE_SAMMED2D must be 0 or 1, not '$MEDISWARM_REQUIRE_SAMMED2D'." >&2
            exit 1
            ;;
    esac
    [[ -n "${MEDISWARM_IMAGE_VERSION:-}" && "$MEDISWARM_IMAGE_VERSION" != *-dev.* ]]
}

# Sets SAMMED2D_AVAILABLE=1|0. Runs first so a release build stops before any download.
check_sammed2d () {
    local required=0
    if sammed2d_required; then   # evaluated up front so a malformed switch is reported either way
        required=1
    fi
    if [[ -f "$MODEL_WEIGHTS_FILE_SAMMED2D" ]]; then
        _verify_hash "$MODEL_WEIGHTS_FILE_SAMMED2D_HASH" "$MODEL_WEIGHTS_FILE_SAMMED2D"
        SAMMED2D_AVAILABLE=1
    elif [[ "$required" == 1 ]]; then
        {
            echo "ERROR: SAM-Med2D weights missing: $MODEL_WEIGHTS_FILE_SAMMED2D"
            echo "       This is a release build (MEDISWARM_IMAGE_VERSION='${MEDISWARM_IMAGE_VERSION:-}'"
            echo "       MEDISWARM_REQUIRE_SAMMED2D='${MEDISWARM_REQUIRE_SAMMED2D:-}'), and without the file"
            echo "       MST_SAMMed2D would get no pretrained encoder at any site."
            echo "       Point MEDISWARM_BUILD_CACHE_DIR at the durable build cache (docs/RELEASE_RUNBOOK.md, section 2)"
            echo "       or place sam-med2d_b.pth (sha256 $MODEL_WEIGHTS_FILE_SAMMED2D_HASH) there, e.g. from"
            echo "       an image that has it:  docker create --name sam jefftud/odelia:1.8.1-cam.261009 &&"
            echo "       docker cp sam:/MediSwarm/pretrained_weights/sam-med2d_b.pth <cache>/pretrained_weights/ && docker rm sam"
            echo "       A deliberate build without it: MEDISWARM_REQUIRE_SAMMED2D=0."
        } >&2
        exit 1
    else
        echo "WARNING: $MODEL_WEIGHTS_FILE_SAMMED2D is not in the build cache. This dev/CI image gives" >&2
        echo "         MST_SAMMed2D no pretrained encoder; never publish it as a release." >&2
        SAMMED2D_AVAILABLE=0
    fi
}

verify_files () {
    _verify_hash "$MODEL_WEIGHTS_FILE_DINO_HASH" "$MODEL_WEIGHTS_FILE_DINO"
    _verify_hash "$MODEL_LICENSE_FILE_DINO_HASH" "$MODEL_LICENSE_FILE_DINO"
    _verify_hash "$MODEL_WEIGHTS_FILE_MVIT_HASH" "$MODEL_WEIGHTS_FILE_MVIT"
    _verify_hash "$MODEL_WEIGHTS_FILE_ODAC_HASH" "$MODEL_WEIGHTS_FILE_ODAC"
    _verify_hash "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_HASH" "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN"
    if [[ "$(cat "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_FILE")" != "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_CONTENTS" \
          || "$(wc -c < "$MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_FILE")" -ne ${#MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_CONTENTS} ]]; then
        echo "ERROR: $MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_MAIN_FILE must contain exactly the commit hash, no newline." >&2
        exit 1
    fi
}

# --------------------------------------------------------------------------- copy

copy_files () {
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

    # Copy ResNet18 pre-trained weights. World-writable as before: huggingface_hub creates lock
    # files in the cache at load time and the containers run as the site's user. Setting the mode
    # here, in the build context, replaces a 'RUN chmod -R' in the Dockerfile that stored the
    # whole tree a second time in its own layer.
    cp -r "$CACHE_DIR/hf_home_cache" "$TARGET_DIR/hf_home_cache"
    chmod a+rwX "$TARGET_DIR/hf_home_cache" -R

    # Challenge model weights go to a SEPARATE directory outside the job folders.
    # This is critical: NVFlare packages the entire job folder when submitting a job,
    # so .pth files inside job dirs would be transferred over the network to every client.
    # Instead they are stored at /MediSwarm/pretrained_weights/ in the Docker image, and
    # the model code falls back to that path at runtime. The context directory sits next to
    # MediSwarm/ (not inside it) so the image gets it as its own layer.
    WEIGHTS_DIR="$TARGET_DIR/pretrained_weights"
    mkdir -p "$WEIGHTS_DIR"
    echo "Copy pretrained model weights to $WEIGHTS_DIR/ ..."

    # challenge_1DivideAndConquer: checkpoint_final.pth
    echo "1DivideAndConquer: copying checkpoint_final.pth"
    cp -p "$MODEL_WEIGHTS_FILE_ODAC" "$WEIGHTS_DIR/"

    # challenge_3agaldran: mvit_v2_s-ae3be167.pth (PyTorch pretrained weights)
    echo "3agaldran: copying mvit_v2_s-ae3be167.pth"
    cp -p "$MODEL_WEIGHTS_FILE_MVIT" "$WEIGHTS_DIR/"

    # MST_SAMMed2D: sam-med2d_b.pth, verified (and, for a release, required) in check_sammed2d
    if [[ "${SAMMED2D_AVAILABLE:-0}" == 1 ]]; then
        echo "SAM-Med2D: copying sam-med2d_b.pth"
        cp -p "$MODEL_WEIGHTS_FILE_SAMMED2D" "$WEIGHTS_DIR/"
    fi

    chmod a+rX "$WEIGHTS_DIR" -R
}

# --------------------------------------------------------------------------- main

main () {
    if [[ $# -ne 1 || -z "$1" ]]; then
        echo "Usage: _cacheAndCopyPretrainedModelWeights.sh <build-context-dir>" >&2
        exit 1
    fi
    TARGET_DIR=$1

    TEMP_CACHE=""
    if [[ -n "${MEDISWARM_BUILD_CACHE_DIR:-}" ]]; then
        # The build script calls this from inside the temporary build context, so a relative path
        # would silently resolve there and the cache would be thrown away with the context.
        if [[ "$MEDISWARM_BUILD_CACHE_DIR" != /* ]]; then
            echo "ERROR: MEDISWARM_BUILD_CACHE_DIR must be an absolute path, not '$MEDISWARM_BUILD_CACHE_DIR'." >&2
            exit 1
        fi
        mkdir -p "$MEDISWARM_BUILD_CACHE_DIR"
        set_cache_paths "$MEDISWARM_BUILD_CACHE_DIR"
    else
        TEMP_CACHE=$(mktemp -d)
        set_cache_paths "$TEMP_CACHE"
    fi

    check_sammed2d
    cache_files
    verify_files
    copy_files

    if [[ -n "$TEMP_CACHE" ]]; then
        rm -rf "$TEMP_CACHE"
    fi
}

# Run only when executed; the unit tests source this file and call the steps one by one.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -e
    main "$@"
fi
