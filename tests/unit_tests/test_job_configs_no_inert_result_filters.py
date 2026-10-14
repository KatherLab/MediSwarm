"""Swarm job configs must not wire task_result_filters on the learn task (#593).

CCWF swarm (SwarmClientController, FaultTolerantSwarmClientController) dispatches only "swarm_*" task names
and runs the learn task through a direct call, ``self.learn_executor.execute(...)`` in
``nvflare/app_common/ccwf/client_ctl.py`` (execute_learn_task). NVFlare applies task filters only to tasks
dispatched by name (ClientRunner._do_process_task, TaskController.broadcast_and_wait), with the exact key
"<task>/<direction>", so a result filter keyed on "train" never runs in a swarm job. PercentilePrivacy sat in
every job config until 1.9.0 and never executed: no log line of it in 2,655 mirrored training rounds. A
filter that reads like privacy protection but never runs is worse than none; client-level DP noise is the
dp_noise round hook (docs/DIFFERENTIAL_PRIVACY.md). These tests keep the dead configuration from returning.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = REPO_ROOT / "application" / "jobs"
CLIENT_CONFS = sorted(JOBS_DIR.glob("*/app/config/config_fed_client.conf"))
SWARM_CONTROLLER_RE = re.compile(r'path\s*=\s*"[\w.]*SwarmClientController"')


def _job(path: Path) -> str:
    return path.parents[2].name


def test_client_configs_are_found():
    assert len(CLIENT_CONFS) >= 8, [str(p) for p in CLIENT_CONFS]


@pytest.mark.parametrize("conf_path", CLIENT_CONFS, ids=_job)
def test_no_percentile_privacy_in_swarm_jobs(conf_path):
    """Plain-text check, runs without any parser."""
    text = conf_path.read_text()
    if not SWARM_CONTROLLER_RE.search(text):
        pytest.skip("not a swarm job")
    assert "percentile_privacy" not in text, f"{_job(conf_path)}: PercentilePrivacy never runs in swarm mode"


@pytest.mark.parametrize("conf_path", CLIENT_CONFS, ids=_job)
def test_swarm_jobs_wire_no_result_filter_on_the_learn_task(conf_path):
    pyhocon = pytest.importorskip("pyhocon")  # ships with the NVFlare fork (CI installs it)
    conf = pyhocon.ConfigFactory.parse_file(str(conf_path))
    learn_tasks = [
        executor["executor"].get("args", {}).get("learn_task_name", "train")
        for executor in conf.get("executors", [])
        if str(executor.get("executor", {}).get("path", "")).endswith("SwarmClientController")
    ]
    if not learn_tasks:
        pytest.skip("not a swarm job")
    for chain in conf.get("task_result_filters", []):
        wired = sorted(set(chain.get("tasks", [])) & set(learn_tasks))
        assert not wired, (
            f"{_job(conf_path)}: task_result_filters on {wired} never run in CCWF swarm (the learn task bypasses "
            f"task filters); use a round hook inside the training subprocess instead (see dp_noise.py)"
        )
