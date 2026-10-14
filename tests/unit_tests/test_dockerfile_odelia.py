"""Static guards on Dockerfile_ODELIA.

Layering: the multi-GB static content sits in its own COPY layers BELOW everything that changes per
commit, so a code-only rebuild with the docker cache does not ship ~3.7 GB again.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker_config" / "Dockerfile_ODELIA"


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

