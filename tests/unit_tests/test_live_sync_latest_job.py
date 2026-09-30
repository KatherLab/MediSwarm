"""live_sync.sh must follow the newest job directory, not the alphabetically last one.

Job ids are random UUIDs. Sorting them by name made every site upload a new job's
logs under the previous job's id (30 Sep 2026: 79dfd8ce sorts after 644a5ee5).
The test extracts find_latest_job_id from the script and runs it on a fake kit root.
"""

import os
import re
import subprocess
import time
from pathlib import Path

LIVE_SYNC = Path(__file__).resolve().parents[2] / "kit_live_sync" / "live_sync.sh"


def _function_source(name: str) -> str:
    text = LIVE_SYNC.read_text()
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert match, f"{name} not found in {LIVE_SYNC}"
    return match.group(0)


def _latest_job(kit_root: Path) -> str:
    script = f'set -euo pipefail\nKIT_ROOT="{kit_root}"\n{_function_source("find_latest_job_id")}\nfind_latest_job_id\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()


def test_newest_job_wins_even_if_it_sorts_first(tmp_path):
    for name in ("startup", "local", "transfer"):
        (tmp_path / name).mkdir()
    old = tmp_path / "79dfd8ce-c371-4c7f-80b5-93f4bc226141"
    new = tmp_path / "644a5ee5-c998-4ced-9d78-66a898d98c26"
    old.mkdir()
    new.mkdir()
    now = time.time()
    os.utime(old, (now - 3600, now - 3600))
    os.utime(new, (now, now))
    assert _latest_job(tmp_path) == new.name


def test_ignores_non_job_directories(tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "some-notes").mkdir()
    assert _latest_job(tmp_path) == ""
