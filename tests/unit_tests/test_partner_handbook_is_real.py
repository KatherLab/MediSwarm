"""Partner-facing docs must only tell sites to run things that exist.

Partner instructions previously drifted from the code -- e.g. three docs promised a
pre-run check block that one kit never had, and two sites were told to set a variable
in a way that guarantees the run aborts. A guide that lies is worse than no guide,
because a site cannot tell which half to trust.

So: every docker.sh flag and every repository script these docs name must actually exist.

Until #565 this file read docs/consortium/PARTNER_HANDBOOK.md, which was never in the
repository (the handbook is kept outside it), and skipped when it was absent: all its tests
skipped on every run and CI stayed green while nothing was checked. It now reads the
partner-facing docs that are tracked, and a missing doc FAILS, never skips. Moving or
renaming one of them means updating PARTNER_DOCS here.
docs/BREASTDIVIDER_SITE_GUIDE.md has its own test (test_breastdivider_site_guide.py).
"""

import re
from pathlib import Path

import pytest
import yaml  # imported, not importorskip'ed: a missing dependency must fail, not skip (#565)

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_GUIDE = "docs/KIT_1.9.0_SITE_GUIDE.md"

# doc -> (kit template whose docker.sh it quotes, whether a flag quoted alone in inline code,
# e.g. `--status`, also counts as a docker.sh flag). The two runbooks also name flags of docker
# and of the build/deploy scripts in prose, so for them only docker.sh command lines count.
PARTNER_DOCS = {
    SITE_GUIDE: ("master_template.yml", True),                               # moving to the 1.9.0 kit
    "assets/readme/README.participant.md": ("master_template.yml", True),    # ODELIA participant guide
    "assets/readme/README.participant.STAMP.md": ("master_template_STAMP.yml", True),
    "docs/EXTERNAL_VALIDATION.md": ("master_template.yml", True),            # predict-only sites
    "docs/SWARM_FAILURE_MODES.md": ("master_template.yml", False),           # fixes sites are told to run
    "docs/RELEASE_RUNBOOK.md": ("master_template.yml", False),               # commands sent to sites
}

FLAG = r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*"


def _cli_options(script):
    """The options a docker.sh accepts: the labels of its argument loop (its first `case $1 in`)."""
    loop = re.search(r"case \$1 in\n(.*?)\n\s*esac", script, re.S)
    assert loop, "docker.sh has no `case $1 in ... esac` argument loop any more; update this test"
    return set(re.findall(r"^\s*(--[A-Za-z0-9_-]+)\)", loop.group(1), re.M))


def _kit_options(template):
    scripts = yaml.safe_load((REPO_ROOT / "docker_config" / template).read_text())
    return {"client": _cli_options(scripts["docker_cln_sh"]), "server": _cli_options(scripts["docker_svr_sh"])}


def _docker_sh_commands(text):
    """Each docker.sh invocation: the rest of its line or inline code span, "\\" continuations
    followed, without quoted values, comments and anything after a pipe or command separator."""
    commands = []
    for raw in re.findall(r"docker\.sh(?:[^\n`\\]|\\\n)*", text):
        command = re.sub(r"\"[^\"]*\"|'[^']*'", "", raw)
        command = re.split(r"\|\||&&|[|;]|(?:^|\s)#", command)[0]
        commands.append(command)
    return commands


def _read(name):
    path = REPO_ROOT / name
    assert path.is_file(), (
        f"{name} is gone: it is a partner-facing doc this test guards. Point PARTNER_DOCS "
        f"at its replacement instead of letting the check go quiet (#565)."
    )
    return path.read_text()


@pytest.fixture(scope="module", params=sorted(PARTNER_DOCS))
def doc(request):
    return request.param, _read(request.param)


def test_kit_option_parsing_finds_the_known_options():
    """Guards the parser above: if it broke, every doc would look wrong (or nothing would)."""
    odelia = _kit_options("master_template.yml")
    assert {"--data_dir", "--scratch_dir", "--GPU", "--start_client", "--migrate_from", "--status"} <= odelia["client"]
    assert "--start_server" in odelia["server"] and "--start_server" not in odelia["client"]
    assert {"--data_dir", "--start_client"} <= _kit_options("master_template_STAMP.yml")["client"]


def test_every_docker_sh_flag_in_the_doc_exists(doc):
    name, text = doc
    template, prose_flags_count = PARTNER_DOCS[name]
    options = _kit_options(template)
    used, missing = set(), set()
    for command in _docker_sh_commands(text):
        kit_script = "server" if "--start_server" in command else "client"
        for flag in re.findall(FLAG, command):
            used.add(flag)
            if flag not in options[kit_script]:
                missing.add(f"{flag} ({kit_script} docker.sh)")
    if prose_flags_count:
        for flag in re.findall(r"`(--[A-Za-z][A-Za-z0-9_-]*)`", text):
            used.add(flag)
            if flag not in options["client"]:
                missing.add(f"{flag} (client docker.sh, quoted in the text)")
    assert used, f"{name} quotes no docker.sh flag any more: drop it from PARTNER_DOCS or fix the parsing"
    assert not missing, f"{name} tells sites to use docker.sh flags that do not exist: {sorted(missing)}"


def test_the_site_guide_covers_the_move():
    """The 1.9.0 guide exists to walk a site through these; losing one means the guide changed shape."""
    flags = {flag for command in _docker_sh_commands(_read(SITE_GUIDE)) for flag in re.findall(FLAG, command)}
    assert {"--migrate_from", "--yes", "--start", "--status", "--start_client"} <= flags


def test_every_repository_script_in_the_doc_exists(doc):
    name, text = doc
    # repository-relative paths only ("scripts/..." or "./scripts/..."), not "<kit>/scripts/..."
    paths = set(re.findall(r"(?<![\w./~-])(?:\./)?(scripts/(?:[\w.-]+/)*[\w.-]*)", text))
    missing = sorted(p for p in paths if not (REPO_ROOT / p.rstrip(".")).exists())
    assert not missing, f"{name} references scripts that do not exist: {missing}"


def test_doc_does_not_tell_sites_to_set_the_cache_to_a_host_path(doc):
    """The mistake that has already cost two sites a failed run."""
    name, text = doc
    assert not re.search(r'ODELIA_PREPROCESS_CACHE_DIR\s*=\s*"?\$SCRATCHDIR', text), (
        f"{name}: ODELIA_PREPROCESS_CACHE_DIR is a CONTAINER path; interpolating the host "
        "$SCRATCHDIR lands it under the read-only /data mount and aborts the run"
    )
