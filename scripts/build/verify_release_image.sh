#!/usr/bin/env bash
#
# Verify a built ODELIA image before it is pushed, re-tagged to :current or handed to sites.
#
#   scripts/build/verify_release_image.sh <image> [--reference <image>] [--trivy]
#
#   1. the shipped weights and their SHA-256 (SAM-Med2D, 1DivideAndConquer, MViT, DINOv2, ResNet18)
#   2. image config: ENTRYPOINT/CMD, KMP_DUPLICATE_LIB_OK, ODELIA_IMAGE_VERSION, OCI labels
#   3. Python packages: NVFlare's wire stack and the training stack equal to --reference (default
#      jefftud/odelia:1.8.1; compared only when present locally, nothing is pulled), the packages
#      1.9.0 removed are absent
#   4. every module of the job code imports; MST and MST_SAMMed2D build and run on CPU, and the
#      MST_SAMMed2D encoder equals the shipped SAM-Med2D checkpoint
#   5. the baked MedicalNet ResNet18 cache resolves offline, as a non-root user
#   6. the entrypoint: a simulated 1.8.1 client kit holding a stale daemon_pid.fl starts in a fresh
#      container, and a plain command sees the same environment as with the entrypoint bypassed
#   7. --trivy: the CVE gate of daily-image-verify.yml (HIGH/CRITICAL with a fix, this repo's
#      .trivyignore) on a `docker save` of the image; the image tar needs ~20 GB under $TMPDIR
# Ends with the image size and its layer table. Exit 1 if any check failed.
#
# Test containers run with --network none (the Trivy container needs network for its DB).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRIVY_IMAGE=aquasec/trivy:0.70.0   # the version the daily workflow's trivy-action runs

IMAGE=""
REFERENCE=jefftud/odelia:1.8.1
RUN_TRIVY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --reference) REFERENCE=$2; shift ;;
        --trivy)     RUN_TRIVY=1 ;;
        -*) echo "Unknown option $1" >&2; exit 2 ;;
        *)  IMAGE=$1 ;;
    esac
    shift
done
if [[ -z "$IMAGE" ]]; then
    sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Image $IMAGE is not present locally." >&2; exit 2; }

FAILED=0
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; FAILED=1; }
section() { echo; echo "== $*"; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/verify_release_image.XXXXXX")
EP_CONTAINER="verify_release_entrypoint_$$"
cleanup() {
    docker rm -f "$EP_CONTAINER" >/dev/null 2>&1 || true
    rm -rf "$WORK"
}
trap cleanup EXIT

USER_OPTS=(-u "$(id -u):$(id -g)" -e HOME=/tmp)
INSIDE=(-v "$SCRIPT_DIR/_verify_image_inside.py:/verify/inside.py:ro")

# ---------------------------------------------------------------- 1. weights
section "1. weights"
declare -A WEIGHTS=(
    [/MediSwarm/pretrained_weights/sam-med2d_b.pth]=921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18
    [/MediSwarm/pretrained_weights/checkpoint_final.pth]=ed686907205fb0cb752dc987851eb9d0191034599c5d204c7ec1ad9ff91dd758
    [/MediSwarm/pretrained_weights/mvit_v2_s-ae3be167.pth]=ae3be16733081f6d1cd40e4ab980ca23d6df6dc6486d15ada05a5e8ab8c9b975
    [/torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth]=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
    [/huggingface_home/hub/models--TencentMedicalNet--MedicalNet-Resnet18/blobs/61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4]=61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4
)
sums=$(docker run --rm --network none --entrypoint sha256sum "$IMAGE" "${!WEIGHTS[@]}" 2>&1)
for path in "${!WEIGHTS[@]}"; do
    got=$(awk -v p="$path" '$2 == p {print $1}' <<< "$sums")
    if [[ "$got" == "${WEIGHTS[$path]}" ]]; then
        pass "$path ${got:0:12}"
    else
        fail "$path: expected ${WEIGHTS[$path]:0:12}, got ${got:-missing}"
    fi
done
ref_bytes=$(docker run --rm --network none --entrypoint stat "$IMAGE" -c %s \
    /huggingface_home/hub/models--TencentMedicalNet--MedicalNet-Resnet18/refs/main 2>/dev/null)
if [[ "$ref_bytes" == 40 ]]; then pass "HF refs/main holds the bare 40-character hash"; else fail "HF refs/main is ${ref_bytes:-missing} bytes, not 40"; fi

# ---------------------------------------------------------------- 2. config
section "2. image config"
ep=$(docker image inspect "$IMAGE" --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}')
if [[ "$ep" == '["/usr/local/bin/mediswarm_entrypoint.sh"] ["/bin/bash"]' ]]; then pass "ENTRYPOINT/CMD $ep"; else fail "ENTRYPOINT/CMD $ep"; fi
envs=$(docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}')
grep -qx 'KMP_DUPLICATE_LIB_OK=TRUE' <<< "$envs" && pass "KMP_DUPLICATE_LIB_OK=TRUE (#596)" || fail "KMP_DUPLICATE_LIB_OK missing (#596)"
version=$(sed -n 's/^ODELIA_IMAGE_VERSION=//p' <<< "$envs")
if [[ -n "$version" && "$version" != unknown ]]; then pass "ODELIA_IMAGE_VERSION=$version"; else fail "ODELIA_IMAGE_VERSION not set"; fi
for label in org.opencontainers.image.version org.opencontainers.image.revision mediswarm.nvflare.revision; do
    value=$(docker image inspect "$IMAGE" --format "{{index .Config.Labels \"$label\"}}")
    if [[ -n "$value" && "$value" != unknown && "$value" != "<no value>" ]]; then pass "label $label=$value"; else fail "label $label missing"; fi
done

# ---------------------------------------------------------------- 3. packages
section "3. Python packages"
docker run --rm --network none "${INSIDE[@]}" --entrypoint python3 "$IMAGE" /verify/inside.py packages > "$WORK/image.json"
reference_json=""
if docker image inspect "$REFERENCE" >/dev/null 2>&1; then
    docker run --rm --network none "${INSIDE[@]}" --entrypoint python3 "$REFERENCE" /verify/inside.py packages > "$WORK/reference.json"
    reference_json="$WORK/reference.json"
else
    echo "[SKIP] reference $REFERENCE not present locally: frozen versions are not compared"
fi
python3 - "$WORK/image.json" "$reference_json" "$REFERENCE" <<'EOF' || FAILED=1
import json, re, sys
image = json.load(open(sys.argv[1]))
reference = json.load(open(sys.argv[2])) if sys.argv[2] else None
frozen = ["nvflare_mediswarm", "grpcio", "protobuf", "msgpack", "aiohttp", "websockets", "cryptography", "torch",
          "torchvision", "numpy", "monai", "pytorch_lightning", "torchmetrics", "timm", "SimpleITK", "torchio",
          "pandas", "scikit-learn", "huggingface_hub", "x-transformers"]
removed = ["wandb", "datasets", "GitPython", "pyarrow", "pyOpenSSL"]
bad = False
def base(v):  # the fork's version carries its build date (2.7.2.devYYMMDD)
    return re.sub(r"\.dev\d+$", "", v) if v else v
for name in frozen:
    if reference is None:
        print(f"[INFO] {name} {image[name]}")
    elif base(image[name]) == base(reference[name]):
        print(f"[PASS] {name} {image[name]} (as {sys.argv[3]})")
    else:
        print(f"[FAIL] {name} {image[name]}, {sys.argv[3]} has {reference[name]}")
        bad = True
for name in removed:
    if image[name] is None:
        print(f"[PASS] {name} not installed")
    else:
        print(f"[FAIL] {name} {image[name]} is installed")
        bad = True
for name in ["PyJWT", "Werkzeug", "urllib3", "setuptools", "pillow"]:
    print(f"[INFO] {name} {image[name]}")
sys.exit(1 if bad else 0)
EOF

# ---------------------------------------------------------------- 4. job code and models
section "4. job code imports, models on CPU"
# The environment a kit's docker.sh gives the training code (main.py reads TRAINING_MODE on import;
# preflight_check needs no NVFlare connection).
JOB_ENV=(-e TORCH_HOME=/torch_home -e HF_HOME=/huggingface_home -e SITE_NAME=VERIFY_1 -e SCRATCH_DIR=/tmp
         -e DATA_DIR=/tmp -e TRAINING_MODE=preflight_check -e MEDISWARM_ALLOW_CPU_MODEL=1)
docker run --rm --network none "${USER_OPTS[@]}" "${JOB_ENV[@]}" "${INSIDE[@]}" --entrypoint python3 "$IMAGE" \
    /verify/inside.py imports 2>&1 | sed 's/^PASS /[PASS] /; s/^FAIL /[FAIL] /' | grep -E '^\[(PASS|FAIL)\]' > "$WORK/imports.txt"
grep '^\[FAIL\]' "$WORK/imports.txt" && FAILED=1
echo "[INFO] $(grep -c '^\[PASS\]' "$WORK/imports.txt") modules imported"
docker run --rm --network none "${USER_OPTS[@]}" "${JOB_ENV[@]}" "${INSIDE[@]}" --entrypoint python3 "$IMAGE" \
    /verify/inside.py models > "$WORK/models.txt" 2>&1
grep -E 'SAM-Med2D backbone weights|RANDOM weights' "$WORK/models.txt" | sed 's/^/[INFO] /'
sed -n 's/^PASS /[PASS] /p; s/^FAIL /[FAIL] /p' "$WORK/models.txt"
grep -q '^FAIL ' "$WORK/models.txt" && FAILED=1
grep -q '^PASS MST_SAMMed2D encoder' "$WORK/models.txt" || { echo "[FAIL] model check did not complete:"; tail -20 "$WORK/models.txt"; FAILED=1; }

# ---------------------------------------------------------------- 5. offline HF cache
section "5. offline Hugging Face cache"
docker run --rm --network none "${USER_OPTS[@]}" -e HF_HOME=/huggingface_home "${INSIDE[@]}" --entrypoint python3 "$IMAGE" \
    /verify/inside.py hf-offline 2>/dev/null | sed 's/^PASS /[PASS] /; s/^FAIL /[FAIL] /' | tee "$WORK/hf.txt"
grep -q '^\[PASS\]' "$WORK/hf.txt" || FAILED=1

# ---------------------------------------------------------------- 6. entrypoint
section "6. entrypoint"
mkdir -p "$WORK/kit"
docker run --rm --network none "${USER_OPTS[@]}" -v "$WORK/kit:/out" --entrypoint python3 "$IMAGE" -c '
import os, yaml, nvflare.lighter
t = yaml.safe_load(open(os.path.join(os.path.dirname(nvflare.lighter.__file__), "templates", "master_template.yml")))
sub = t["sub_start_sh"]
for key, value in {"type": "client", "app_name": "client_train", "cln_uid": "uid=VERIFY_1",
                   "org_name": "verify", "config_folder": "config"}.items():
    sub = sub.replace("{~~%s~~}" % key, value)
os.makedirs("/out/startup")
for name, text in [("sub_start.sh", sub), ("start.sh", t["start_cln_sh"]), ("fed_client.json", "{}\n")]:
    with open("/out/startup/" + name, "w") as fh:
        fh.write(text)
    os.chmod("/out/startup/" + name, 0o755)
for lock in ("daemon_pid.fl", "pid.fl"):
    with open("/out/" + lock, "w") as fh:
        fh.write("1\n")   # PID 1 always exists in the new container: the lock looks alive
'
# The exact container command of a 1.8.1 client kit (no lock clearing of its own, #630 came later).
docker run -d -t --name "$EP_CONTAINER" --network none "${USER_OPTS[@]}" -v "$WORK/kit:/startupkit/" \
    -w /startupkit/startup/ --env TRAINING_MODE=swarm "$IMAGE" \
    /bin/bash -c "nohup ./start.sh >> nohup.out 2>&1 && /bin/bash" >/dev/null
for _ in $(seq 1 30); do
    grep -qE 'new pid|There seems to be one instance' "$WORK/kit/startup/nohup.out" 2>/dev/null && break
    sleep 1
done
if grep -q 'new pid' "$WORK/kit/startup/nohup.out" 2>/dev/null && \
   ! grep -q 'There seems to be one instance' "$WORK/kit/startup/nohup.out"; then
    pass "1.8.1 client kit with a stale daemon_pid.fl starts ($(docker logs "$EP_CONTAINER" 2>&1 | grep -c 'removed stale') stale locks removed)"
else
    fail "1.8.1 client kit did not start: $(tail -3 "$WORK/kit/startup/nohup.out" 2>/dev/null | tr '\n' ' ')"
fi
docker rm -f "$EP_CONTAINER" >/dev/null 2>&1
probe='import json, os; e = dict(os.environ); e.pop("HOSTNAME", None); print(json.dumps(e, sort_keys=True))'
with_ep=$(docker run --rm --network none "$IMAGE" python3 -c "$probe")
without_ep=$(docker run --rm --network none --entrypoint python3 "$IMAGE" -c "$probe")
if [[ -n "$with_ep" && "$with_ep" == "$without_ep" ]]; then pass "a plain command sees the same environment as without the entrypoint"; else fail "environment differs through the entrypoint"; fi

# ---------------------------------------------------------------- 7. CVE gate
if [[ -n "$RUN_TRIVY" ]]; then
    section "7. Trivy (HIGH/CRITICAL with a fix, .trivyignore)"
    # A saved image instead of the Docker socket: the scanner gets no control over the daemon.
    docker save "$IMAGE" -o "$WORK/image.tar"
    TRIVY_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/mediswarm-trivy"
    mkdir -p "$TRIVY_CACHE"
    if docker run --rm -u "$(id -u):$(id -g)" -e HOME=/cache -v "$TRIVY_CACHE:/cache" \
        -v "$WORK/image.tar:/image.tar:ro" -v "$REPO_ROOT/.trivyignore:/trivyignore:ro" "$TRIVY_IMAGE" \
        image --input /image.tar --cache-dir /cache/trivy --ignorefile /trivyignore --severity CRITICAL,HIGH \
        --ignore-unfixed --scanners vuln --skip-version-check --exit-code 1 --format table > "$WORK/trivy.txt" 2>&1; then
        pass "Trivy gate: no unaccepted HIGH/CRITICAL finding with a fix"
    else
        fail "Trivy gate:"; grep -E 'Total:|│' "$WORK/trivy.txt" | head -40
    fi
    rm -f "$WORK/image.tar"
fi

# ---------------------------------------------------------------- size and layers
section "image size and layers"
docker image inspect "$IMAGE" --format 'size (as stored here): {{.Size}} bytes'
docker history "$IMAGE" --format '{{.Size}}	{{.CreatedBy}}' | cut -c1-110 | head -24

echo
if [[ "$FAILED" == 0 ]]; then echo "ALL CHECKS PASSED for $IMAGE"; else echo "CHECKS FAILED for $IMAGE"; fi
exit "$FAILED"
