"""The image build ships only verified weights, and a release cannot lose SAM-Med2D silently.

scripts/build/_cacheAndCopyPretrainedModelWeights.sh fills the Docker build context. Before 1.9.0
the SAM-Med2D backbone was optional and never hash-checked: a release built on a host without the
file shipped an image in which MST_SAMMed2D had no pretrained encoder, with one log line as the only
trace. Now the file is verified whenever present and required for release builds, while CI builds
(which export MEDISWARM_IMAGE_VERSION=<v>-dev.<date>.<sha> and hold no copy) stay optional.

The tests source the script and drive its steps against a small fake cache whose expected hashes
are set to the fakes' real digests; nothing is downloaded.
"""

import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build" / "_cacheAndCopyPretrainedModelWeights.sh"
SAM_SHA256 = "921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18"
RESNET_COMMIT = "758fe285bc8ab565eb4f9f965810f1d1a3f79491"

# script variable holding the path (relative to the cache) -> its hash variable
WEIGHTS = {
    "MODEL_WEIGHTS_FILE_DINO": ("torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth",
                                "MODEL_WEIGHTS_FILE_DINO_HASH"),
    "MODEL_LICENSE_FILE_DINO": ("torch_home_cache/hub/facebookresearch_dinov2_main/LICENSE",
                                "MODEL_LICENSE_FILE_DINO_HASH"),
    "MODEL_WEIGHTS_FILE_MVIT": ("application/jobs/challenge_3agaldran/app/custom/models/mvit_v2_s-ae3be167.pth",
                                "MODEL_WEIGHTS_FILE_MVIT_HASH"),
    "MODEL_WEIGHTS_FILE_ODAC": ("application/jobs/challenge_1DivideAndConquer/app/custom/models/checkpoint_final.pth",
                                "MODEL_WEIGHTS_FILE_ODAC_HASH"),
    "MODEL_WEIGHTS_FILE_RESNETEIGHTEEN": (None, "MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_HASH"),  # path holds the hash
    "MODEL_WEIGHTS_FILE_SAMMED2D": ("pretrained_weights/sam-med2d_b.pth", "MODEL_WEIGHTS_FILE_SAMMED2D_HASH"),
    "MODEL_WEIGHTS_FILE_MEDICALNET": ("pretrained_weights/resnet_34_23dataset.pth", "MODEL_WEIGHTS_FILE_MEDICALNET_HASH"),
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def cache(tmp_path):
    """A complete fake cache; returns (cache_dir, {hash variable: digest})."""
    root = tmp_path / "cache"
    digests = {}
    for var, (rel, hash_var) in WEIGHTS.items():
        data = f"fake {var}\n".encode()
        if rel is None:
            # The blob is stored under its own digest, as huggingface_hub does.
            rel = f"hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/blobs/{_sha(data)}"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digests[hash_var] = _sha(data)
    hub = root / "torch_home_cache" / "hub" / "facebookresearch_dinov2_main"
    (hub / "dinov2").mkdir(parents=True)
    (hub / "hubconf.py").write_text("dependencies = ['torch']\n")
    (hub / "docs").mkdir()
    snapshot = root / "hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/snapshots" / RESNET_COMMIT
    snapshot.mkdir(parents=True)
    os.symlink(f"../../blobs/{digests['MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_HASH']}",
               snapshot / "resnet_18_23dataset.pth")
    return root, digests


def run_steps(steps, cache_dir, target, digests, env_extra=None, keep_real=()):
    """Source the script, point it at the fake cache, override the expected hashes, run `steps`.

    keep_real: hash variables that keep the script's real value (to test the constants).
    """
    overrides = "".join(f"{var}={digest}; " for var, digest in digests.items() if var not in keep_real)
    # set_cache_paths derives the ResNet18 blob path from its hash, so the override comes first.
    snippet = (f'source "$1"; {overrides}set_cache_paths "$2"; TARGET_DIR="$3"; '
               "set -e; " + steps)
    env = {k: v for k, v in os.environ.items()
           if k not in ("MEDISWARM_IMAGE_VERSION", "MEDISWARM_REQUIRE_SAMMED2D", "MEDISWARM_BUILD_CACHE_DIR")}
    env.update(env_extra or {})
    target.mkdir(exist_ok=True)
    return subprocess.run(["bash", "-c", snippet, "weights-under-test", str(SCRIPT), str(cache_dir), str(target)],
                          env=env, capture_output=True, text=True, timeout=120)


ALL_STEPS = "check_sammed2d; check_medicalnet; write_resnet18_ref; verify_files; copy_files"


def test_release_build_puts_verified_weights_in_their_own_context_dirs(cache, tmp_path):
    cache_dir, digests = cache
    target = tmp_path / "context"
    result = run_steps(ALL_STEPS, cache_dir, target, digests, {"MEDISWARM_IMAGE_VERSION": "1.9.0"})
    assert result.returncode == 0, result.stderr

    weights = target / "pretrained_weights"
    assert sorted(p.name for p in weights.iterdir()) == \
        ["checkpoint_final.pth", "mvit_v2_s-ae3be167.pth", "resnet_34_23dataset.pth", "sam-med2d_b.pth"]
    for name, var in [("checkpoint_final.pth", "MODEL_WEIGHTS_FILE_ODAC_HASH"),
                      ("sam-med2d_b.pth", "MODEL_WEIGHTS_FILE_SAMMED2D_HASH")]:
        assert _sha((weights / name).read_bytes()) == digests[var]
    # Outside MediSwarm/: the Dockerfile copies them into their own layer below the code.
    assert not (target / "MediSwarm").exists()
    assert (target / "torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth").is_file()
    assert not (target / "torch_home_cache/hub/facebookresearch_dinov2_main/docs").exists()


def test_hf_cache_is_world_writable_in_the_context(cache, tmp_path):
    """Replaces the Dockerfile's 'RUN chmod -R a+rwX /huggingface_home', which stored the tree twice."""
    cache_dir, digests = cache
    target = tmp_path / "context"
    assert run_steps(ALL_STEPS, cache_dir, target, digests).returncode == 0
    hf = target / "hf_home_cache"
    for path in [hf, *hf.rglob("*")]:
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode & 0o666 == 0o666, (path, oct(mode))
        if path.is_dir():
            assert mode & 0o111 == 0o111, (path, oct(mode))


def test_release_build_without_sammed2d_stops_before_downloading(cache, tmp_path):
    cache_dir, digests = cache
    (cache_dir / "pretrained_weights/sam-med2d_b.pth").unlink()
    result = run_steps("check_sammed2d; echo reached-downloads", cache_dir, tmp_path / "context", digests,
                       {"MEDISWARM_IMAGE_VERSION": "1.9.0"})
    assert result.returncode == 1
    assert "reached-downloads" not in result.stdout
    assert "sam-med2d_b.pth" in result.stderr and "MEDISWARM_REQUIRE_SAMMED2D=0" in result.stderr


@pytest.mark.parametrize("env", [
    {},                                                         # plain developer build
    {"MEDISWARM_IMAGE_VERSION": "1.9.0-dev.261014.f3dfd0f7"},   # what CI exports (#428)
    {"MEDISWARM_IMAGE_VERSION": "1.9.0", "MEDISWARM_REQUIRE_SAMMED2D": "0"},  # deliberate opt-out
], ids=["dev", "ci", "opt-out"])
def test_builds_without_sammed2d_warn_and_skip_it(cache, tmp_path, env):
    cache_dir, digests = cache
    (cache_dir / "pretrained_weights/sam-med2d_b.pth").unlink()
    target = tmp_path / "context"
    result = run_steps(ALL_STEPS, cache_dir, target, digests, env)
    assert result.returncode == 0, result.stderr
    assert "WARNING" in result.stderr and "never publish it as a release" in result.stderr
    assert not (target / "pretrained_weights/sam-med2d_b.pth").exists()
    assert (target / "pretrained_weights/checkpoint_final.pth").exists()


def test_require_switch_makes_any_build_need_sammed2d(cache, tmp_path):
    cache_dir, digests = cache
    (cache_dir / "pretrained_weights/sam-med2d_b.pth").unlink()
    result = run_steps("check_sammed2d", cache_dir, tmp_path / "context", digests,
                       {"MEDISWARM_REQUIRE_SAMMED2D": "1"})
    assert result.returncode == 1
    assert "SAM-Med2D weights missing" in result.stderr


def test_malformed_require_switch_is_rejected(cache, tmp_path):
    cache_dir, digests = cache
    result = run_steps("check_sammed2d", cache_dir, tmp_path / "context", digests,
                       {"MEDISWARM_REQUIRE_SAMMED2D": "yes"})
    assert result.returncode == 1
    assert "must be 0 or 1" in result.stderr


def test_truncated_sammed2d_fails_even_in_a_dev_build(cache, tmp_path):
    """A present file is always checked, here against the script's real SAM-Med2D digest."""
    cache_dir, digests = cache
    result = run_steps("check_sammed2d", cache_dir, tmp_path / "context", digests,
                       keep_real=("MODEL_WEIGHTS_FILE_SAMMED2D_HASH",))
    assert result.returncode == 1
    assert "unexpected SHA-256" in result.stderr and SAM_SHA256 in result.stderr


def test_script_pins_the_validated_sammed2d_digest():
    assert f"MODEL_WEIGHTS_FILE_SAMMED2D_HASH={SAM_SHA256}" in SCRIPT.read_text()


@pytest.mark.parametrize("var", sorted(v for v in WEIGHTS if v != "MODEL_WEIGHTS_FILE_SAMMED2D"))
def test_any_corrupt_cached_weight_fails_before_anything_is_copied(cache, tmp_path, var):
    cache_dir, digests = cache
    rel, hash_var = WEIGHTS[var]
    if rel is None:
        rel = ("hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/blobs/"
               + digests["MODEL_WEIGHTS_FILE_RESNETEIGHTEEN_HASH"])
    (cache_dir / rel).write_bytes(b"truncated")
    target = tmp_path / "context"
    result = run_steps(ALL_STEPS, cache_dir, target, digests)
    assert result.returncode == 1
    assert "unexpected SHA-256" in result.stderr
    assert not (target / "pretrained_weights").exists()


def test_resnet18_ref_is_rewritten_without_a_trailing_newline(cache, tmp_path):
    """huggingface_hub reads refs/main verbatim; '<hash>\\n' made the baked cache unusable offline."""
    cache_dir, digests = cache
    ref = cache_dir / "hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text(RESNET_COMMIT + "\n")  # what the old script's `echo` left in persistent caches
    target = tmp_path / "context"
    result = run_steps(ALL_STEPS, cache_dir, target, digests)
    assert result.returncode == 0, result.stderr
    assert ref.read_bytes() == RESNET_COMMIT.encode()
    shipped = target / "hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/refs/main"
    assert shipped.read_bytes() == RESNET_COMMIT.encode()


def test_ref_with_a_newline_fails_verification(cache, tmp_path):
    cache_dir, digests = cache
    ref = cache_dir / "hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text(RESNET_COMMIT + "\n")
    result = run_steps("verify_files", cache_dir, tmp_path / "context", digests)
    assert result.returncode == 1
    assert "no newline" in result.stderr


def test_main_runs_end_to_end_on_a_filled_cache_without_downloading(cache, tmp_path):
    """main(): no file is missing, so nothing is fetched (wget is not even on PATH here)."""
    cache_dir, digests = cache
    target = tmp_path / "context"
    target.mkdir()
    overrides = "".join(f"{var}={digest}; " for var, digest in digests.items())
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "MEDISWARM_BUILD_CACHE_DIR": str(cache_dir),
           "MEDISWARM_IMAGE_VERSION": "1.9.0"}
    result = subprocess.run(["bash", "-c", f'source "$1"; {overrides}set -e; main "$2"',
                             "weights-under-test", str(SCRIPT), str(target)],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert (target / "pretrained_weights/sam-med2d_b.pth").is_file()
    assert cache_dir.is_dir(), "a persistent cache must survive the build"


@pytest.mark.parametrize("args, env, message", [
    ([], {}, "Usage"),
    (["ctx"], {"MEDISWARM_BUILD_CACHE_DIR": "relative/cache"}, "must be an absolute path"),
])
def test_bad_invocations_are_refused(tmp_path, args, env, message):
    result = subprocess.run(["bash", str(SCRIPT), *args], cwd=tmp_path,
                            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), **env},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert message in result.stderr
