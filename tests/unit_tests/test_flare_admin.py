"""scripts/admin/flare_admin.py: one admin command per call, through the FLARE API.

The FLARE API session is replaced by a fake that records the calls, so these tests need no
nvflare and no server. The fake's answers copy the shapes the real API returned on a DECADE
server in November 2026 (status dicts, job dicts, the read-only signing error).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load():
    name = "_test_admin_flare_admin"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "admin" / "flare_admin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeSession:
    def __init__(self, submit_error=None):
        self.calls = []
        self.closed = False
        self.submit_error = submit_error

    def check_status(self, target):
        self.calls.append(("check_status", target))
        if target == "server":
            return {"status": "ok", "server_status": "stopped", "jobs": []}
        return {"client_status": [{"client_name": "SITE_A", "status": "no_jobs"}]}

    def list_jobs(self, detailed=False):
        self.calls.append(("list_jobs", detailed))
        return [{"job_id": "fbe92494", "status": "FINISHED:EXECUTION_EXCEPTION"}]

    def get_job_meta(self, job_id):
        self.calls.append(("get_job_meta", job_id))
        return {"job_id": job_id}

    def submit_job(self, job_dir):
        self.calls.append(("submit_job", job_dir))
        if self.submit_error:
            raise RuntimeError(self.submit_error)
        return "new-job-id"

    def abort_job(self, job_id):
        self.calls.append(("abort_job", job_id))
        return "aborted"

    def download_job_result(self, job_id):
        self.calls.append(("download_job_result", job_id))
        return f"/fl_admin/transfer/{job_id}"

    def grep_target(self, target, options=None, pattern=None, file=None):
        self.calls.append(("grep_target", target, options, pattern, file))
        return "110: ERROR - External process has not called flare.init within timeout: 600"

    def tail_target(self, target, options=None, file=None):
        self.calls.append(("tail_target", target, options, file))
        return "last lines"

    def ls_target(self, target, options=None, path=None):
        self.calls.append(("ls_target", target, options, path))
        return "log.txt"

    def close(self):
        self.closed = True


def _run(argv, sess):
    return _load().main(argv, session_factory=lambda args: sess)


def test_status_asks_server_and_clients(capsys):
    sess = FakeSession()
    assert _run(["status"], sess) == 0
    assert sess.calls == [("check_status", "server"), ("check_status", "client")]
    out = capsys.readouterr().out
    assert "server_status" in out and "no_jobs" in out
    assert sess.closed


def test_list_prints_one_job_per_line(capsys):
    sess = FakeSession()
    assert _run(["list"], sess) == 0
    assert "FINISHED:EXECUTION_EXCEPTION" in capsys.readouterr().out


def test_submit_prints_the_job_id(capsys):
    sess = FakeSession()
    assert _run(["submit", "/jobs/STAMP_job"], sess) == 0
    assert sess.calls == [("submit_job", "/jobs/STAMP_job")]
    assert capsys.readouterr().out.strip() == "JOB_ID new-job-id"


def test_submit_from_a_read_only_mount_explains_the_signing(capsys):
    err = ("APIStatus.ERROR_RUNTIME: Failed to sign job folder: [Errno 30] Read-only file system: "
           "'/jobs/STAMP_job/.__nvfl_sig.json'")
    sess = FakeSession(submit_error=err)
    assert _run(["submit", "/jobs/STAMP_job"], sess) == 1
    msg = capsys.readouterr().err
    assert "signs a job folder in place" in msg and "writable" in msg
    assert sess.closed


def test_abort_needs_yes(capsys):
    sess = FakeSession()
    assert _run(["abort", "fbe92494"], sess) == 2
    assert sess.calls == []
    assert "--yes" in capsys.readouterr().err


def test_abort_with_yes():
    sess = FakeSession()
    assert _run(["abort", "fbe92494", "--yes"], sess) == 0
    assert sess.calls == [("abort_job", "fbe92494")]


def test_grep_passes_site_pattern_and_file():
    sess = FakeSession()
    assert _run(["grep", "SITE_A", "ERROR", "fbe92494/log.txt"], sess) == 0
    assert sess.calls == [("grep_target", "SITE_A", "-n", "ERROR", "fbe92494/log.txt")]


def test_tail_builds_the_line_option():
    sess = FakeSession()
    assert _run(["tail", "SITE_A", "startup/.mediswarm_sync/live_sync_daemon.log", "-n", "25"], sess) == 0
    assert sess.calls == [("tail_target", "SITE_A", "-n 25", "startup/.mediswarm_sync/live_sync_daemon.log")]


def test_ls_without_path():
    sess = FakeSession()
    assert _run(["ls", "SITE_A"], sess) == 0
    assert sess.calls == [("ls_target", "SITE_A", None, None)]


def test_download_and_meta(capsys):
    sess = FakeSession()
    assert _run(["download", "fbe92494"], sess) == 0
    assert _run(["meta", "fbe92494"], sess) == 0
    assert sess.calls == [("download_job_result", "fbe92494"), ("get_job_meta", "fbe92494")]
    assert "/fl_admin/transfer/fbe92494" in capsys.readouterr().out


def test_username_comes_from_the_kit(tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "startup" / "fed_admin.json").write_text(json.dumps({"admin": {"username": "admin@example.org"}}))
    assert _load().admin_username(str(tmp_path)) == "admin@example.org"


def test_username_missing_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="pass --username"):
        _load().admin_username(str(tmp_path))
