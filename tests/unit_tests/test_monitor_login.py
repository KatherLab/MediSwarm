"""Optional HTTP Basic login of the live monitor (server_tools/app.py).

The monitor shows every site's logs and serves files; deployed on the consortium VPN
it must not be open to every node. Login is off unless MEDISWARM_MONITOR_USERS is set.
"""

import base64
import hashlib
import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # fastapi.testclient needs it

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "server_tools"))


def _header(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def _load_app(monkeypatch, tmp_path, users_text=None):
    monkeypatch.setenv("MEDISWARM_LIVE_BASE", str(tmp_path / "live"))
    (tmp_path / "live").mkdir(exist_ok=True)
    if users_text is None:
        monkeypatch.delenv("MEDISWARM_MONITOR_USERS", raising=False)
    else:
        f = tmp_path / "users"
        f.write_text(users_text)
        monkeypatch.setenv("MEDISWARM_MONITOR_USERS", str(f))
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_open_without_users_file(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    app_module = _load_app(monkeypatch, tmp_path)
    assert TestClient(app_module.app).get("/api/runs").status_code == 200


def test_login_required_and_checked(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    digest = hashlib.sha256(b"s3cret-pass").hexdigest()
    app_module = _load_app(monkeypatch, tmp_path, f"# operators\nalice:{digest}\n")
    client = TestClient(app_module.app)
    r = client.get("/api/runs")
    assert r.status_code == 401 and "Basic" in r.headers["www-authenticate"]
    assert client.get("/api/runs", headers=_header("alice", "wrong")).status_code == 401
    assert client.get("/api/runs", headers=_header("bob", "s3cret-pass")).status_code == 401
    assert client.get("/api/runs", headers=_header("alice", "s3cret-pass")).status_code == 200


@pytest.mark.parametrize("text", ["", "# only a comment\n", "alice:not-a-hash\n", "no-separator\n"])
def test_bad_users_file_fails_closed(monkeypatch, tmp_path, text):
    with pytest.raises(ValueError):
        _load_app(monkeypatch, tmp_path, text)
