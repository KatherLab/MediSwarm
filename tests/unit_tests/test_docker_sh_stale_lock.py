"""A client or server container that restarts after a crash must not be blocked by its own lock.

docker.sh starts the container with --restart=unless-stopped and the command runs
start.sh, which refuses to start while ../daemon_pid.fl exists. After a power loss the
lock survives in the bind-mounted kit, the container comes back "Up", and the node stays
offline until someone deletes the file (coordinator 30 Sep, USZ 3 Oct, VHIO 6 Oct 2026).
The container command therefore clears the lock before start.sh: it runs only on a
(re)start, when no daemon of that container can be alive.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = [REPO_ROOT / "docker_config" / "master_template.yml",
             REPO_ROOT / "docker_config" / "master_template_STAMP.yml"]


@pytest.mark.parametrize("template_path", TEMPLATES, ids=lambda p: p.name)
@pytest.mark.parametrize("key", ["docker_cln_sh", "docker_svr_sh"])
def test_container_command_clears_stale_lock_before_start(template_path, key):
    script = yaml.safe_load(template_path.read_text())[key]
    starts = [line for line in script.splitlines() if "nohup ./start.sh" in line]
    assert starts, f"no start.sh launch found in {key} of {template_path.name}"
    for line in starts:
        lock = line.find("rm -f ../daemon_pid.fl")
        assert lock != -1 and lock < line.find("nohup ./start.sh"), line.strip()
