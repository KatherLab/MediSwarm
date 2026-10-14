"""Static guards on Dockerfile_ODELIA and the CVE allowlist for the final 1.9.0 image.

What they protect:
- Layering: the multi-GB static content sits in its own COPY layers BELOW everything that changes
  per commit, so a code-only rebuild with the docker cache does not ship ~3.7 GB again.
- Frozen stack: torch (base digest), numpy, monai, Lightning and the NVFlare wire stack stay exactly
  as in 1.8.1. Sites move to a new image at different times and one swarm mixes both images.
- Removed packages stay removed (wandb and its bundled binary alone carried 56 HIGH/CRITICAL CVEs).
- .trivyignore stays reviewable: every accepted finding carries a reason and an expiry date.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker_config" / "Dockerfile_ODELIA"
TRIVYIGNORE = REPO_ROOT / ".trivyignore"


def instructions():
    """Dockerfile instructions with continuation lines joined and comments dropped."""
    out, current = [], ""
    for raw in DOCKERFILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        current = f"{current} {line}" if current else line
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        out.append(current)
        current = ""
    return out


def index_of(prefix):
    for i, instr in enumerate(instructions()):
        if instr.startswith(prefix):
            return i
    raise AssertionError(f"no instruction starts with {prefix!r}")


def pinned():
    pins = {}
    for instr in instructions():
        if instr.startswith("RUN python3 -m pip install"):
            for name, version in re.findall(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-]+)", instr):
                pins[re.sub(r"[-_.]+", "-", name).lower()] = version
    return pins


def test_static_content_is_layered_below_everything_that_changes():
    first_run = min(i for i, instr in enumerate(instructions()) if instr.startswith("RUN "))
    for source, dest in [("./pretrained_weights", "/MediSwarm/pretrained_weights"),
                         ("./torch_home_cache", "/torch_home"),
                         ("./hf_home_cache", "/huggingface_home"),
                         ("./nvflare_source", "/MediSwarm/docker_config/NVFlare")]:
        assert index_of(f"COPY {source} {dest}") < first_run, source
    assert index_of("COPY ./MediSwarm/docker_config/master_template.yml") < index_of("COPY ./MediSwarm /MediSwarm")


def test_hf_cache_is_not_stored_twice():
    """chmod after COPY duplicated the 130 MB cache in its own layer; the context carries the modes now."""
    assert not any(instr.startswith("RUN chmod") and "/huggingface_home" in instr for instr in instructions())


def test_nvflare_is_installed_from_the_shipped_source():
    install = [instr for instr in instructions() if "pip install --no-cache-dir /workspace/nvflare" in instr]
    assert len(install) == 1
    assert install[0].startswith("RUN cp -r /MediSwarm/docker_config/NVFlare /workspace/nvflare")
    assert "rm -rf /workspace/nvflare" in install[0]


def test_frozen_training_and_wire_stack():
    text = DOCKERFILE.read_text()
    assert "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime@sha256:" \
           "923f687790bec78081c357e71dcd5dcef80b0cc00f6c34484902a5e83362c854" in text
    assert "ENV KMP_DUPLICATE_LIB_OK=TRUE" in text
    pins = pinned()
    for name, version in {"numpy": "1.26.4", "monai": "1.4.0", "pytorch-lightning": "2.4.0",
                          "torchvision": "0.17.2", "torchaudio": "2.2.2", "grpcio": "1.62.1",
                          "protobuf": "4.24.4", "msgpack": "1.1.0", "aiohttp": "3.11.18",
                          "websockets": "15.0.1", "flask": "3.0.2"}.items():
        assert pins.get(name) == version, (name, pins.get(name))


def test_unused_packages_stay_out():
    pins = pinned()
    for name in ["wandb", "datasets", "gitpython", "gitdb", "smmap", "docker-pycreds", "sentry-sdk",
                 "setproctitle", "pyarrow", "multiprocess", "dill", "xxhash"]:
        assert name not in pins, name


def test_fixed_versions_of_flagged_packages():
    pins = pinned()
    assert pins["pyjwt"] == "2.14.0"
    assert pins["werkzeug"] == "3.0.6"
    assert pins["setuptools"] == "80.10.2"


def test_trivyignore_entries_have_a_reason_and_an_expiry():
    lines = TRIVYIGNORE.read_text().splitlines()
    ids = []
    for i, line in enumerate(lines):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        assert re.fullmatch(r"(CVE-\d{4}-\d+|GHSA(-[0-9a-z]{4}){3})", fields[0]), line
        assert any(re.fullmatch(r"exp:\d{4}-\d{2}-\d{2}", f) for f in fields[1:]), f"no expiry: {line}"
        assert i > 0 and lines[i - 1].startswith("# ") and len(lines[i - 1]) > 20, f"no reason above: {line}"
        ids.append(fields[0])
    assert ids and len(ids) == len(set(ids))
    # Never wave the removed or bumped packages' findings through.
    assert "CVE-2026-78676" not in ids  # GitPython (removed with wandb)
    assert "CVE-2026-102268" not in ids  # PyJWT (bumped)
