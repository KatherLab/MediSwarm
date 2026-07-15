"""The launcher's KEY_METRIC prefix must equal the job's IntimeModelSelector key_metric (#409).

The training script runs as a SubprocessLauncher child and cannot read the NVFlare
component config, so `key_metric` is forwarded through the launcher command. That
leaves two places holding the same value. This is a pure text check -- no torch,
no NVFlare -- so it always runs in CI and fails the moment the two drift apart.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = REPO_ROOT / "application" / "jobs"
SHARED_CUSTOM = (JOBS_DIR / "_shared" / "custom").resolve()

KEY_METRIC_RE = re.compile(r'key_metric\s*=\s*"([^"]+)"')
# anchored so it cannot match `app_script = "..."`
SCRIPT_LINE_RE = re.compile(r'^\s*script\s*=\s*"([^"]*)"', re.M)
ENV_TOKEN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(\S+)$")


def _shared_training_jobs():
    """Jobs whose app/custom symlinks to _shared/custom (i.e. run threedcnn_ptl)."""
    jobs = []
    if not JOBS_DIR.is_dir():
        return jobs
    for job in sorted(JOBS_DIR.iterdir()):
        custom = job / "app" / "custom"
        conf = job / "app" / "config" / "config_fed_client.conf"
        if custom.is_symlink() and conf.is_file() and custom.resolve() == SHARED_CUSTOM:
            jobs.append((job.name, conf))
    return jobs


def _launcher_env(script_value):
    """Leading KEY=VALUE tokens, mirroring SubprocessLauncher's strip loop."""
    env = {}
    for token in script_value.split():
        match = ENV_TOKEN_RE.match(token)
        if not match:
            break
        env[match.group(1)] = match.group(2)
    return env


JOBS = _shared_training_jobs()


def test_shared_training_jobs_are_discovered():
    assert JOBS, "no job symlinks app/custom -> _shared/custom"
    assert len(JOBS) >= 6, f"expected >=6 shared-training jobs, found {len(JOBS)}"


@pytest.mark.parametrize("job_name,conf_path", JOBS, ids=[name for name, _ in JOBS])
def test_launcher_key_metric_matches_selector(job_name, conf_path):
    text = conf_path.read_text()

    selector = KEY_METRIC_RE.findall(text)
    assert selector, f"{job_name}: no key_metric in config_fed_client.conf"
    selector_metric = selector[0]

    scripts = SCRIPT_LINE_RE.findall(text)
    assert len(scripts) == 1, f"{job_name}: expected 1 launcher script line, got {len(scripts)}"
    env = _launcher_env(scripts[0])

    assert "KEY_METRIC" in env, (
        f"{job_name}: launcher must forward KEY_METRIC={selector_metric} so the local "
        f"ModelCheckpoint monitors the same metric as IntimeModelSelector (#409)"
    )
    assert env["KEY_METRIC"] == selector_metric, (
        f"{job_name}: launcher KEY_METRIC={env['KEY_METRIC']} != selector key_metric={selector_metric}"
    )
