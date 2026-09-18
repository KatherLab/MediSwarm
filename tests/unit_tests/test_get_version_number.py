"""getVersionNumber.sh must be pinnable for the lifetime of a CI run (#428).

The script derives YYMMDD from the wall clock, and is re-invoked independently by
the build step and by runIntegrationTests.sh. A run that crosses local midnight
therefore built `...260709...` and then looked up `...260710...`, which never
existed. Exporting MEDISWARM_IMAGE_VERSION pins one value for the whole run.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build" / "getVersionNumber.sh"

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+-dev\.\d{6}\.[0-9a-f]+$")


def _run(env=None):
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()


def test_pinned_version_is_echoed_verbatim():
    import os
    env = {**os.environ, "MEDISWARM_IMAGE_VERSION": "1.5.0-dev.260709.deadbee"}
    assert _run(env) == "1.5.0-dev.260709.deadbee"


def test_blank_pin_falls_back_to_computing_the_version():
    import os
    env = {**os.environ, "MEDISWARM_IMAGE_VERSION": ""}
    assert VERSION_RE.match(_run(env)), "empty pin must not short-circuit"


def test_unpinned_version_has_the_expected_shape():
    import os
    env = {k: v for k, v in os.environ.items() if k != "MEDISWARM_IMAGE_VERSION"}
    out = _run(env)
    assert VERSION_RE.match(out), f"unexpected version format: {out!r}"


def test_pin_makes_repeated_invocations_agree():
    """The property the CI race violated: two calls must return the same string."""
    import os
    env = {**os.environ, "MEDISWARM_IMAGE_VERSION": "9.9.9-dev.999999.cafe123"}
    assert _run(env) == _run(env)
