"""HTTP Basic login of the live monitor (server_tools/app.py).

The monitor shows every site's logs and per-case CSVs and serves checkpoints; deployed on the
consortium VPN it must not be open to every node. It fails closed (review finding PS-2): without
MEDISWARM_MONITOR_USERS it refuses to start, unless MEDISWARM_MONITOR_ALLOW_OPEN=1 opens it on
purpose for local testing.
"""

import base64
import hashlib
import importlib
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # fastapi.testclient needs it

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "server_tools"))

DIGEST = hashlib.sha256(b"s3cret-pass").hexdigest()


def _header(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def _load_app(monkeypatch, tmp_path, users_text=None, allow_open=None, users_path=None):
    monkeypatch.setenv("MEDISWARM_LIVE_BASE", str(tmp_path / "live"))
    (tmp_path / "live").mkdir(exist_ok=True)
    monkeypatch.delenv("MEDISWARM_MONITOR_USERS", raising=False)
    monkeypatch.delenv("MEDISWARM_MONITOR_ALLOW_OPEN", raising=False)
    if users_text is not None:
        users_path = tmp_path / "users"
        users_path.write_text(users_text)
    if users_path is not None:
        monkeypatch.setenv("MEDISWARM_MONITOR_USERS", str(users_path))
    if allow_open is not None:
        monkeypatch.setenv("MEDISWARM_MONITOR_ALLOW_OPEN", allow_open)
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_refuses_to_start_without_users_file(monkeypatch, tmp_path):
    """The default, no variable set, must not serve anything."""
    with pytest.raises(RuntimeError, match="MEDISWARM_MONITOR_USERS.*MEDISWARM_MONITOR_ALLOW_OPEN=1"):
        _load_app(monkeypatch, tmp_path)


@pytest.mark.parametrize("value", ["", "0", "true", "yes", " 1"])
def test_only_allow_open_1_opens_it(monkeypatch, tmp_path, value):
    with pytest.raises(RuntimeError):
        _load_app(monkeypatch, tmp_path, allow_open=value)


def test_open_only_on_explicit_opt_in_and_says_so(monkeypatch, tmp_path, caplog):
    from fastapi.testclient import TestClient
    with caplog.at_level(logging.WARNING, logger="mediswarm.monitor"):
        app_module = _load_app(monkeypatch, tmp_path, allow_open="1")
    assert TestClient(app_module.app).get("/api/runs").status_code == 200
    assert any("WITHOUT LOGIN" in r.getMessage() for r in caplog.records)


def test_login_required_and_checked(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    app_module = _load_app(monkeypatch, tmp_path, f"# operators\nalice:{DIGEST}\n")
    client = TestClient(app_module.app)
    r = client.get("/api/runs")
    assert r.status_code == 401 and "Basic" in r.headers["www-authenticate"]
    assert client.get("/api/runs", headers=_header("alice", "wrong")).status_code == 401
    assert client.get("/api/runs", headers=_header("bob", "s3cret-pass")).status_code == 401
    assert client.get("/api/runs", headers=_header("alice", "s3cret-pass")).status_code == 200


def test_users_file_wins_over_allow_open(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    app_module = _load_app(monkeypatch, tmp_path, f"alice:{DIGEST}\n", allow_open="1")
    client = TestClient(app_module.app)
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs", headers=_header("alice", "s3cret-pass")).status_code == 200


@pytest.mark.parametrize("text", ["", "# only a comment\n", "alice:not-a-hash\n", "no-separator\n"])
def test_bad_users_file_fails_closed(monkeypatch, tmp_path, text):
    with pytest.raises(ValueError):
        _load_app(monkeypatch, tmp_path, text)


@pytest.mark.parametrize("allow_open", [None, "1"])
def test_missing_users_file_fails_closed(monkeypatch, tmp_path, allow_open):
    """A configured but missing file is a broken deployment, not a request to run open."""
    with pytest.raises(ValueError, match="no such file"):
        _load_app(monkeypatch, tmp_path, users_path=tmp_path / "absent", allow_open=allow_open)
