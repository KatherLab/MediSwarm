"""The mediswarm-vpn service scripts (#620): config generation, failure reasons, refusal, health monitor.

At CAM and RUMC the service started OpenVPN but never got a 172.24.4.x address, while
setup_vpntunnel.sh reported success and the health timer restarted OpenVPN every 30 s. These tests run
the real shell functions (scripts/client_node_setup/mediswarm_vpn_lib.sh) and scripts with fake
``ip``/``systemctl``/``journalctl``/``ping`` on PATH, so they need bash but neither root nor a tunnel.
The end-to-end reproduction against an OpenVPN gateway is tests/vpn_service/run_vpn_service_repro.sh.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SETUP_DIR = REPO / "scripts" / "client_node_setup"
LIB = SETUP_DIR / "mediswarm_vpn_lib.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

CRLF_OVPN = (
    "client\r\ndev tun\r\nproto udp\r\nremote gw.example 1194\r\nresolv-retry infinite\r\nnobind\r\n"
    "persist-key\r\npersist-tun\r\nkey-direction 1\r\nverb 3\r\nauth-user-pass\r\n#tuning\r\n\r\npull\r\n"
    "<ca>\r\n-----BEGIN CERTIFICATE-----\nCA\n-----END CERTIFICATE-----\n</ca>\r\n"
    "<tls-auth>\r\n-----BEGIN OpenVPN Static key V1-----\nTA\n-----END OpenVPN Static key V1-----\n</tls-auth>\r\n\r\n"
)


def _lib(script: str, cwd=None, env=None, stdin=None):
    """Runs ``script`` in bash after sourcing the library."""
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(["bash", "-c", f'source "{LIB}"\n{script}'], cwd=cwd, env=full_env, input=stdin,
                          capture_output=True, text=True, timeout=60)


def _directives(conf: Path):
    """Directive lines outside inline blocks."""
    out, block = [], None
    for line in conf.read_text().splitlines():
        if block:
            if line.startswith(f"</{block}>"):
                block = None
            continue
        if line.startswith("<") and not line.startswith("</"):
            block = line[1:line.index(">")]
            continue
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _prepare(tmp_path, ovpn_text, name="site.ovpn"):
    src = tmp_path / name
    src.write_bytes(ovpn_text.encode())
    dest, auth, files = tmp_path / "mediswarm.conf", tmp_path / "mediswarm.auth", tmp_path / "mediswarm.d"
    res = _lib(f'msvpn_prepare_config "{src}" "{dest}" "{auth}" "{files}"')
    return res, dest, auth, files


def test_goodaccess_download_gets_one_auth_line_and_keeps_inline_blocks(tmp_path):
    res, dest, auth, _ = _prepare(tmp_path, CRLF_OVPN)
    assert res.returncode == 0, res.stderr
    text = dest.read_text()
    assert "\r" not in text and text.endswith("\n")
    d = _directives(dest)
    assert [x for x in d if x.startswith("auth-user-pass")] == [f"auth-user-pass {auth}"]
    for item in ("auth-nocache", "persist-key", "persist-tun", "keepalive 10 60"):
        assert sum(1 for x in d if x.split()[0] == item.split()[0]) == 1, item
    assert "-----BEGIN OpenVPN Static key V1-----\nTA\n-----END OpenVPN Static key V1-----\n</tls-auth>" in text
    assert oct(dest.stat().st_mode & 0o777) == "0o600"


def test_ovpn_without_final_newline_keeps_auth_line_separate(tmp_path):
    no_newline = CRLF_OVPN[:-4]
    assert no_newline.endswith("</tls-auth>")
    # What the old script did: "echo auth-user-pass ... | tee -a" glues the directive to the closing
    # tag, where OpenVPN ignores the rest of the line, so no credentials are sent (AUTH_FAILED).
    old = no_newline.replace("auth-user-pass\r\n", "") + "auth-user-pass /etc/openvpn/client/mediswarm.auth\n"
    assert "</tls-auth>auth-user-pass" in old
    res, dest, auth, _ = _prepare(tmp_path, no_newline)
    assert res.returncode == 0, res.stderr
    lines = dest.read_text().splitlines()
    assert "</tls-auth>" in lines and f"auth-user-pass {auth}" in lines


def test_auth_path_from_our_old_guide_is_replaced(tmp_path):
    # assets/VPN setup guide(CLI).md and the good_access README told sites to write this path, which
    # setup -n -s never creates: OpenVPN exited at start with "No such file or directory".
    res, dest, auth, _ = _prepare(tmp_path, CRLF_OVPN.replace("auth-user-pass\r\n",
                                                               "auth-user-pass /etc/openvpn/credentials\r\n"))
    assert res.returncode == 0, res.stderr
    assert [x for x in _directives(dest) if x.startswith("auth-user-pass")] == [f"auth-user-pass {auth}"]
    assert "replaced" in res.stderr


def test_referenced_key_files_are_copied_out_of_home(tmp_path):
    # NetworkManager export: certificates are files under the user's ~/.cert/nm-openvpn/, which the
    # service (ProtectHome=true) cannot see. One relative path and one quoted path with a space.
    home = tmp_path / "home" / "siteuser" / ".cert" / "nm-openvpn"
    home.mkdir(parents=True)
    for name in ("ca.pem", "cert.pem", "tls auth.pem"):
        (home / name).write_text(name)
    (tmp_path / "client.key").write_text("key")
    ovpn = (f"client\nremote gw.example 1194 udp\nca {home}/ca.pem\ncert {home}/cert.pem\nkey client.key\n"
            f'tls-auth "{home}/tls auth.pem" 1\nauth-user-pass\n')
    res, dest, _, files = _prepare(tmp_path, ovpn)
    assert res.returncode == 0, res.stderr
    d = {x.split()[0]: x.split()[1:] for x in _directives(dest)}
    assert d["ca"] == [f"{files}/ca-ca.pem"] and d["key"] == [f"{files}/key-client.key"]
    assert d["tls-auth"] == [f"{files}/tls-auth-tls_auth.pem", "1"]
    assert (files / "tls-auth-tls_auth.pem").read_text() == "tls auth.pem"
    assert all(oct(f.stat().st_mode & 0o777) == "0o600" for f in files.iterdir())


def test_missing_referenced_file_is_an_error(tmp_path):
    res, dest, _, _ = _prepare(tmp_path, "client\nremote gw.example 1194\nca /nonexistent/ca.pem\nauth-user-pass\n")
    assert res.returncode == 1 and "/nonexistent/ca.pem" in res.stderr
    assert not dest.exists()


def test_daemon_log_user_group_are_dropped(tmp_path):
    ovpn = "client\nremote gw.example 1194\nauth-user-pass\ndaemon\nlog /var/log/x.log\nuser nobody\ngroup nogroup\n"
    res, dest, _, _ = _prepare(tmp_path, ovpn)
    assert res.returncode == 0, res.stderr
    assert not {"daemon", "log", "user", "group"} & {x.split()[0] for x in _directives(dest)}


def test_file_without_remote_is_rejected(tmp_path):
    res, dest, _, _ = _prepare(tmp_path, "dev tun\nauth-user-pass\n")
    assert res.returncode == 1 and "no remote" in res.stderr


def test_lint_flags_the_old_failure_modes(tmp_path):
    glued = tmp_path / "glued.conf"
    glued.write_text("client\nremote gw 1194\n<tls-auth>\nTA\n</tls-auth>auth-user-pass /etc/x.auth\n")
    assert "no auth-user-pass" in _lib(f'msvpn_lint_config "{glued}"').stdout
    home = tmp_path / "home.conf"
    home.write_text("client\nremote gw 1194\nca /home/siteuser/.cert/nm-openvpn/ca.pem\nauth-user-pass /etc/openvpn/credentials\n")
    out = _lib(f'msvpn_lint_config "{home}"').stdout
    assert "invisible to the service" in out and "/etc/openvpn/credentials does not exist" in out


@pytest.mark.parametrize("log, code, definitive", [
    ("AUTH: Received control message: AUTH_FAILED\nSIGTERM[soft,auth-failure] received, process exiting\n",
     "auth_failed", False),
    ("AUTH: Received control message: AUTH_FAILED\n" * 3, "auth_failed", True),
    ("Options error: --auth-user-pass fails with '/etc/openvpn/credentials': No such file or directory (errno=2)\n"
     "Exiting due to fatal error\n", "auth_file", True),
    # OpenVPN 2.5 as the service logged it at a site that followed the old guide (reproduced offline).
    ("WARNING: cannot stat file '/etc/openvpn/credentials': No such file or directory (errno=2)\n"
     "Options error: --auth-user-pass fails with '/etc/openvpn/credentials': No such file or directory (errno=2)\n"
     "Options error: Please correct these errors.\nUse --help for more information.\n", "auth_file", True),
    ("Options error: --ca fails with '/home/u/.cert/ca.pem': No such file or directory (errno=2)\n",
     "options_error", True),
    ("Cannot pre-load keyfile (/home/siteuser/.cert/nm-openvpn/tls-auth.pem)\nExiting due to fatal error\n",
     "key_file", True),
    ("TLS Error: TLS key negotiation failed to occur within 60 seconds (check your network connectivity)\n",
     "tls_timeout", False),
    ("[vpn-gw.test] Inactivity timeout (--ping-restart), restarting\n", "tls_timeout", False),
    ("RESOLVE: Cannot resolve host address: gw.example:1194 (Temporary failure in name resolution)\n",
     "resolve", False),
    ("setgid('nogroup') failed: Operation not permitted (errno=1)\nExiting due to fatal error\n",
     "permission", True),
    ("refusing to start: another GoodAccess tunnel already holds a VPN address (tun0 172.24.4.69).\n",
     "other_tunnel", True),
    ("Initialization Sequence Completed\n", "connected", False),
    ("Initialization Sequence Completed\n[gw] Inactivity timeout (--ping-restart), restarting\n", "ping_restart", False),
    ("SIGUSR1[soft,server-pushed-connection-reset] received, process restarting\n", "kicked", False),
    ("", "none", False),
])
def test_classify_openvpn_log(log, code, definitive):
    res = _lib('v="$(msvpn_classify_log)"; printf "%s\\n" "$v"; msvpn_is_definitive "$v" && echo DEFINITIVE || true',
               stdin=log)
    assert res.returncode == 0, res.stderr
    assert res.stdout.split("\t")[0] == code
    assert ("DEFINITIVE" in res.stdout) == definitive


def test_vpn_addresses_are_found_on_any_interface():
    sample = ("1: lo    inet 127.0.0.1/8 scope host lo\n"
              "3: br-1    inet 172.24.0.1/16 brd 172.24.255.255 scope global br-1\n"
              "7: tun1    inet 172.24.4.69/22 scope global tun1\n"
              "8: tun0    inet 10.8.0.6 peer 10.8.0.5/32 scope global tun0\n")
    assert _lib("msvpn_vpn_addrs_from", stdin=sample).stdout == "tun1 172.24.4.69\n"


def test_password_with_backslash_survives_the_prompt(tmp_path):
    auth = tmp_path / "auth"
    res = _lib(f'msvpn_prompt_credentials && msvpn_write_auth_file "{auth}" "$MSVPN_ACCOUNT" "$MSVPN_PASSWORD"',
               stdin="cam\n  Cam\\Pass-2026 \n")
    assert res.returncode == 0, res.stderr
    assert auth.read_text() == "cam\nCam\\Pass-2026\n"
    assert oct(auth.stat().st_mode & 0o777) == "0o600"
    # The old prompt (plain read) dropped the backslash.
    old = subprocess.run(["bash", "-c", 'read p; printf "%s" "$p"'], input="Cam\\Pass\n", capture_output=True, text=True)
    assert old.stdout == "CamPass"
    assert _lib(f'msvpn_check_auth_file "{auth}"').stdout == ""
    (tmp_path / "half").write_text("cam\n")
    assert "password" in _lib(f'msvpn_check_auth_file "{tmp_path / "half"}"').stdout


# ---------------------------------------------------------------------------
# Scripts with fake system commands
# ---------------------------------------------------------------------------

def _fake_bin(tmp_path, scripts: dict) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    for name, body in scripts.items():
        f = bindir / name
        f.write_text("#!/bin/bash\n" + body)
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    return bindir


def _ip_fake(addr_file: Path) -> str:
    return f'[[ "$*" == *"addr show"* ]] && cat "{addr_file}" 2>/dev/null; exit 0\n'


def test_start_wrapper_refuses_next_to_another_tunnel(tmp_path):
    addrs = tmp_path / "addrs"
    addrs.write_text("5: tun0    inet 172.24.4.69/22 scope global tun0\n")
    conf, auth = tmp_path / "mediswarm.conf", tmp_path / "mediswarm.auth"
    auth.write_text("cam\npw\n")
    conf.write_text(f"client\nremote gw 1194\nauth-user-pass {auth}\n")
    bindir = _fake_bin(tmp_path, {"ip": _ip_fake(addrs), "pgrep": 'echo "4242 /usr/sbin/openvpn --config hand.ovpn"\n',
                                  "nmcli": "exit 0\n", "openvpn": 'echo "openvpn $*"\n'})
    env = {"PATH": f"{bindir}:{os.environ['PATH']}", "MSVPN_CONF": str(conf), "MSVPN_AUTH": str(auth),
           "OPENVPN_BIN": str(bindir / "openvpn")}
    run = lambda: subprocess.run(["bash", str(SETUP_DIR / "vpn_service_start.sh")], env={**os.environ, **env},
                                 capture_output=True, text=True, timeout=30)
    res = run()
    assert res.returncode == 3 and "refusing to start" in res.stdout and "4242" in res.stdout
    addrs.write_text("")
    res = run()
    assert res.returncode == 0 and res.stdout.strip() == f"openvpn --suppress-timestamps --config {conf}"
    auth.write_text("cam\n")
    res = run()
    assert res.returncode == 4 and "not configured" in res.stdout


@pytest.fixture
def monitor(tmp_path):
    """vpn_health_monitor.sh with fake systemctl/ip/ping/journalctl; returns (run, state dir)."""
    st = tmp_path / "st"
    st.mkdir()
    (st / "addrs").write_text("")
    (st / "enabled").write_text("enabled")
    (st / "ActiveState").write_text("failed")
    (st / "InvocationID").write_text("inv1")
    (st / "journal").write_text("AUTH: Received control message: AUTH_FAILED\n")
    bindir = _fake_bin(tmp_path, {
        "ip": _ip_fake(st / "addrs"),
        "systemctl": (f'S="{st}"\n'
                      'case "$1" in\n'
                      '  is-enabled) cat "$S/enabled" ;;\n'
                      '  show) p="$3"; [[ "$p" == ExecMainStartTimestampMonotonic ]] && '
                      '{ up=$(cut -d. -f1 /proc/uptime); echo $(( (up - $(cat "$S/age")) * 1000000 )); exit 0; }; '
                      'cat "$S/$p" 2>/dev/null ;;\n'
                      '  restart) echo "$*" >> "$S/restarts" ;;\n'
                      'esac\n'),
        "ping": f'[[ -e "{st}/ping_ok" ]]\n',
        "journalctl": f'cat "{st}/journal"\n',
        "logger": "exit 0\n",
    })
    (st / "age").write_text("600")

    def run(**extra):
        env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "STATE_FILE": str(st / "health.state"),
               **{k: str(v) for k, v in extra.items()}}
        res = subprocess.run(["bash", str(SETUP_DIR / "vpn_health_monitor.sh")], env=env, capture_output=True,
                             text=True, timeout=30)
        assert res.returncode == 0, res.stderr
        return res.stdout

    def restarts():
        f = st / "restarts"
        return len(f.read_text().splitlines()) if f.exists() else 0

    run.restarts = restarts
    return run, st


def test_monitor_leaves_a_disabled_service_alone(monitor):
    run, st = monitor
    (st / "enabled").write_text("disabled")
    for _ in range(5):
        out = run()
    assert run.restarts() == 0 and "not monitoring" not in out  # logged once, then quiet


def test_monitor_waits_while_openvpn_is_still_connecting(monitor):
    run, st = monitor
    (st / "ActiveState").write_text("activating")
    (st / "age").write_text("40")
    for _ in range(5):
        run()
    assert run.restarts() == 0


def test_monitor_restarts_after_threshold_then_backs_off_and_logs_the_reason(monitor):
    run, st = monitor
    outs = [run(BACKOFF_BASE=1000) for _ in range(3)]
    assert run.restarts() == 1
    assert "AUTH_FAILED" in outs[-1]
    for _ in range(6):
        run(BACKOFF_BASE=1000)
    assert run.restarts() == 1  # next restart only after the backoff


def test_monitor_does_not_fight_another_tunnel(monitor):
    run, st = monitor
    (st / "addrs").write_text("5: tun0    inet 172.24.4.69/22 scope global tun0\n")
    (st / "ActiveState").write_text("failed")
    for _ in range(5):
        run()
    assert run.restarts() == 0


def test_monitor_judges_pings_only_after_the_gateway_answered_once(monitor):
    run, st = monitor
    (st / "addrs").write_text("5: tun0    inet 172.24.4.69/22 scope global tun0\n")
    (st / "ActiveState").write_text("active")
    for _ in range(4):
        run()
    assert run.restarts() == 0  # gateway never answered ICMP: not a tunnel failure
    (st / "ping_ok").write_text("")
    run()
    (st / "ping_ok").unlink()
    for _ in range(3):
        run()
    assert run.restarts() == 1


def test_unit_matches_the_scripts():
    unit = (SETUP_DIR / "mediswarm-vpn.service").read_text()
    keys = dict(line.split("=", 1) for line in unit.splitlines() if "=" in line and not line.startswith("#"))
    assert keys["Type"] == "notify"
    assert keys["ExecStart"] == "/usr/local/lib/mediswarm-vpn/vpn_service_start.sh"
    assert set(keys["RestartPreventExitStatus"].split()) == {"3", "4"}
    assert {"CAP_NET_ADMIN", "CAP_SETUID", "CAP_SETGID"} <= set(keys["CapabilityBoundingSet"].split())
    lib = LIB.read_text()
    assert "MSVPN_EXIT_OTHER_TUNNEL=3" in lib and "MSVPN_EXIT_NOT_CONFIGURED=4" in lib
    assert 'MSVPN_LIBEXEC="${MSVPN_LIBEXEC:-/usr/local/lib/mediswarm-vpn}"' in lib
    runtime = _lib('printf "%s\\n" "${MSVPN_RUNTIME_FILES[@]}"').stdout.split()
    for name in runtime:
        assert os.access(SETUP_DIR / name, os.X_OK), name
