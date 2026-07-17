"""Unit tests for the #398 swarm round-progress parser in server_tools/app.py.

parse_swarm_progress() is pure (reads a file, returns a dict), so it's tested
here against synthetic FL-server nohup.out content. Requires fastapi (app.py
imports it at module load); skipped if unavailable.
"""

import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")  # app.py imports fastapi at module level

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_TOOLS = REPO_ROOT / "server_tools"


@pytest.fixture(scope="module")
def app():
    p = str(SERVER_TOOLS)
    if p not in sys.path:
        sys.path.insert(0, p)
    import app as app_module  # noqa: E402
    return app_module


def _write(tmp_path, text, *, mtime=None):
    f = tmp_path / "nohup.out"
    f.write_text(text)
    if mtime is not None:
        import os
        os.utime(f, (mtime, mtime))
    return str(f)


_UPPER = """\
2026-07-06 09:37:20 - Ctl - INFO - [run=a370564e-d30d-4831-85a0-6f83e64ee6c5, wf=swarm_controller] - Workflow Config: {'num_rounds': 20}
2026-07-06 09:59:30 - Ctl - WARNING - [run=a370564e-d30d-4831-85a0-6f83e64ee6c5, peer=USZ_1] - FaultTolerant: client USZ_1 reported error 'EXECUTION_EXCEPTION'; pruning and continuing with 5 active clients (min_clients=4)
2026-07-06 10:20:31 - Ctl - INFO - [run=a370564e-d30d-4831-85a0-6f83e64ee6c5, peer=RSH_1] - updated status of client RSH_1 on round 0: action=finished_learn_task, all_done=False
"""

# Real shape from the 2026-07-07 Duke run: LOWERCASE client names (node_A/node_B).
_LOWER = """\
2026-07-07 13:08:00 - Ctl - INFO - [run=769e80a5-cd51-48ae-ac31-06f82055344e, wf=swarm_controller] - Workflow Config: {'num_rounds': 3}
2026-07-07 13:40:17 - Ctl - INFO - [run=769e80a5-cd51-48ae-ac31-06f82055344e, peer=node_A] - updated status of client node_A on round 1: action=finished_learn_task, all_done=False
2026-07-07 13:49:47 - Ctl - INFO - [run=769e80a5-cd51-48ae-ac31-06f82055344e, peer=node_B] - updated status of client node_B on round 1: action=start_learn_task, all_done=False
"""


def test_uppercase_clients_and_prune(app, tmp_path):
    p = app.parse_swarm_progress(_write(tmp_path, _UPPER, mtime=time.time()))
    assert p["job_id"].startswith("a370564e")
    assert p["round"] == 0 and p["num_rounds"] == 20
    assert p["clients"] == {"RSH_1": "finished_learn_task"}
    assert [x["client"] for x in p["prunes"]] == ["USZ_1"]
    assert p["prunes"][0]["active"] == 5


def test_lowercase_clients_are_parsed(app, tmp_path):
    # Regression: node_A/node_B (lowercase) were dropped by an [A-Z] regex.
    p = app.parse_swarm_progress(_write(tmp_path, _LOWER, mtime=time.time()))
    assert p["round"] == 1 and p["num_rounds"] == 3
    assert p["clients"] == {"node_A": "finished_learn_task", "node_B": "start_learn_task"}
    assert p["status"] == "training"


def test_stalled_when_log_old(app, tmp_path):
    p = app.parse_swarm_progress(_write(tmp_path, _LOWER, mtime=time.time() - 3600))
    assert p["status"] == "stalled"


def test_missing_and_idle(app, tmp_path):
    assert app.parse_swarm_progress("") is None
    assert app.parse_swarm_progress(str(tmp_path / "nope.out")) is None
    idle = _write(tmp_path, "2026-07-07 12:00:00 - ServerRunner - INFO - waiting\n", mtime=time.time())
    assert app.parse_swarm_progress(idle)["status"] == "idle"
