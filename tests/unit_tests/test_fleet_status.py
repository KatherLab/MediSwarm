"""scripts/admin/fleet_status.py: per-site kit/image readiness from the uploaded heartbeats.

The fixture tree mirrors /srv/mediswarm/live: the eight ODELIA sites in every state the
migration week can show them in (migrated and idle, migrated on the old image, not migrated,
stale, missing, malformed), next to folders that are not ODELIA sites. Heartbeat fields and
id forms are those of real heartbeats (1.8.1: sha256:cbf6ba7aa574, the classic-store config
digest); host names and IPs are placeholders that must never reach the output.
"""

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_TOOLS = REPO_ROOT / "server_tools"


def _load():
    name = "_test_admin_fleet_status"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "admin" / "fleet_status.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fs = _load()
if str(SERVER_TOOLS) not in sys.path:
    sys.path.insert(0, str(SERVER_TOOLS))
import heartbeat_io  # noqa: E402

# 1.9.1 as the registry and the two Docker image stores name it
CONFIG = "sha256:c527c8c509aa4173f6523f82f59c2d8247afabe5bea08c97af195d12a2c58f51"
INDEX = "sha256:8df5e80d3182955010a26a20b77fe9366ef4d99635b4c8e7cbefbcdfb4dbcba4"
MANIFEST = "sha256:c92290a0bce628d16e2d368215672841cb35ea26ac7db8564d0f6cfb6f94f177"
OLD_181 = "sha256:cbf6ba7aa574"
HOST = "node7.hospital.example"
IP = "192.0.2.77"   # documentation range (RFC 5737)


def short(digest):
    return digest[:19]


def iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def beat(root, site, run, *, age=10, mode="swarm", name="heartbeat.json", kit="1.9.0",
         image_id=short(CONFIG), status="running", upload_age=None, raw=None, **extra):
    """Write one heartbeat as live_sync uploads it; mtime stands in for the upload time."""
    now = time.time()
    data = {"site_name": site, "mode": mode, "job_id": "" if run == "_active" else run,
            "timestamp": iso(now - age), "status": status, "kit_version": kit,
            "image_ref": "jefftud/odelia:current" if image_id else "", "image_id": image_id,
            "sync_status": "ok", "sync_error": "", "hostname": HOST, "ip_address": IP,
            "sync_version": "3" if kit == "1.9.0" else "2"}
    if kit == "1.9.0":
        data.update({"live_sync_version": "3", "client_state": "running"})
    data.update(extra)
    path = Path(root) / site / mode / run / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw if raw is not None else json.dumps(data, indent=2))
    up = now - (age if upload_age is None else upload_age)
    os.utime(path, (up, up))
    return path


@pytest.fixture(autouse=True)
def upload_time_is_mtime(monkeypatch):
    # The real upload time is the file's ctime, which a test cannot set.
    monkeypatch.setattr(fs, "_upload_epoch", lambda p: os.stat(p).st_mtime)


@pytest.fixture(autouse=True)
def no_docker(monkeypatch):
    def refuse(args):
        raise AssertionError(f"unexpected docker call {args}")
    monkeypatch.setattr(fs, "_docker_json", refuse)


@pytest.fixture
def fleet(tmp_path):
    """The eight ODELIA sites in eight states, plus folders that are not sites."""
    r = tmp_path / "live"
    # CAM_1: migrated, idles without a job (swarm/_active, status waiting); the old 1.8.1 job
    # heartbeat next to it stopped when --migrate_from retired the old daemon
    beat(r, "CAM_1", "_active", status="waiting", age=20)
    beat(r, "CAM_1", "d67132d0-abf5-4a6f-b15d-0093f2a0267e", kit="1.8.1", image_id=OLD_181, age=7200)
    # MHA_1: migrated and ran a job; its Docker uses the containerd store -> index digest
    beat(r, "MHA_1", "7a678d9a-5fec-473f-9b23-e43c41322f1f", image_id=short(INDEX), age=40)
    # RSH_1: not migrated, 1.8.1 client up
    beat(r, "RSH_1", "d67132d0-abf5-4a6f-b15d-0093f2a0267e", kit="1.8.1", image_id=OLD_181, age=30)
    # RUMC_1: migrated, but started before :current moved -> still the 1.8.1 image; clock 2 h ahead
    beat(r, "RUMC_1", "_active", status="waiting", image_id=OLD_181, age=-7200, upload_age=25)
    # UKA_1: migrated client, but the 1.8.1 kit's live_sync still uploads (F16)
    beat(r, "UKA_1", "_active", status="waiting", age=15)
    beat(r, "UKA_1", "d67132d0-abf5-4a6f-b15d-0093f2a0267e", kit="1.8.1", image_id=OLD_181, age=45)
    # UMCU_1: migrated, then the client stopped; the 1.9.0 daemon's last heartbeat
    beat(r, "UMCU_1", "_active", status="stopped", client_state="stopped", image_id="", age=120)
    # USZ_1: newest heartbeat from July, an old kit (salvaged doubled quote)
    old = beat(r, "USZ_1", "f7824f34-187f-4eb7-9460-5b71dcf72851", kit="1.5.0-dev.260702.be9ef04",
               image_id="", age=86 * 86400)
    old.write_text(old.read_text().replace('"1.5.0-dev.260702.be9ef04"', '"1.5.0-dev.260702.be9ef04""'))
    os.utime(old, (time.time() - 86 * 86400,) * 2)
    # VHIO_1: never uploaded (no folder)
    # Not ODELIA sites: must not appear, must not change the exit status
    for other in ["_audit", "_probe", "ODELIA_MODELS", "client_A", "node_A", "Mainz_1", "UKB_1", "UKHD_1"]:
        beat(r, other, "a1b2c3d4-0000-0000-0000-000000000000", kit="1.4.1", image_id="", age=5)
    return r


def run(capsys, *args):
    code = fs.main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def run_json(capsys, root, *args):
    code, out, err = run(capsys, "--live-root", str(root), "--json", *args)
    assert not err, err
    doc = json.loads(out)
    return code, doc, {s["site"]: s for s in doc["sites"]}


# ------------------------------------------------------------------- the fleet table

def test_every_state_of_the_migration_week(fleet, capsys):
    code, doc, sites = run_json(capsys, fleet, "--expect-image-id", CONFIG, "--expect-image-id", INDEX)
    assert code == fs.EXIT_NOT_READY and doc["ready"] is False
    assert list(sites) == fs.ODELIA_SITES, "default = the eight ODELIA sites, in order, nothing else"
    verdicts = {s: v["verdict"] for s, v in sites.items()}
    assert verdicts == {
        "CAM_1": "READY", "MHA_1": "READY", "RSH_1": "OLD KIT", "RUMC_1": "OLD IMAGE",
        "UKA_1": "OLD KIT", "UMCU_1": "STOPPED", "USZ_1": "STALE", "VHIO_1": "NO HEARTBEAT",
    }
    assert doc["ready_sites"] == 2


def test_idle_migrated_client_is_ready_from_its_active_heartbeat(fleet, capsys):
    _, _, sites = run_json(capsys, fleet, "--expect-image-id", CONFIG)
    cam = sites["CAM_1"]
    assert cam["heartbeat"] == "CAM_1/swarm/_active/heartbeat.json" and cam["status"] == "waiting"
    assert cam["image_expected"] is True and cam["notes"] == []


def test_config_and_index_digest_both_identify_the_expected_image(fleet, capsys):
    _, _, only_config = run_json(capsys, fleet, "--expect-image-id", CONFIG)
    assert only_config["MHA_1"]["verdict"] == "OLD IMAGE", "the index digest alone is not the config digest"
    _, _, both = run_json(capsys, fleet, "--expect-image-id", CONFIG, "--expect-image-id", short(INDEX))
    assert both["MHA_1"]["verdict"] == "READY" and both["CAM_1"]["verdict"] == "READY"


def test_not_migrated_site_says_so(fleet, capsys):
    _, _, sites = run_json(capsys, fleet)
    assert sites["RSH_1"]["reason"] == "kit 1.8.1, expected 1.9.0: not migrated"
    assert sites["RSH_1"]["kit_version"] == "1.8.1"


def test_old_image_names_what_runs_and_how_to_fix_it(fleet, capsys):
    _, _, sites = run_json(capsys, fleet, "--expect-image-id", CONFIG)
    reason = sites["RUMC_1"]["reason"]
    assert OLD_181 in reason and ":current" in reason and "restart the client" in reason


def test_site_clock_ahead_uses_the_upload_time(fleet, capsys):
    _, _, sites = run_json(capsys, fleet)
    rumc = sites["RUMC_1"]
    assert rumc["verdict"] != "STALE" and 0 <= rumc["age_seconds"] < 120
    assert any("clock" in n and "ahead" in n for n in rumc["notes"])


def test_old_kit_still_uploading_blocks_a_migrated_site(fleet, capsys):
    _, _, sites = run_json(capsys, fleet, "--expect-image-id", CONFIG)
    uka = sites["UKA_1"]
    assert uka["kit_version"] == "1.9.0", "the verdict is about the 1.9.0 heartbeat"
    assert "1.8.1 still uploads" in uka["reason"] and "--migrate_from" in uka["reason"]


def test_retired_old_daemon_final_heartbeat_does_not_block(tmp_path, capsys):
    root = tmp_path / "live"
    beat(root, "UKA_1", "_active", status="waiting", age=15)
    beat(root, "UKA_1", "d67132d0-abf5-4a6f-b15d-0093f2a0267e", name="heartbeat_final.json", kit="1.8.1",
         image_id=OLD_181, status="finished", age=60)
    code, _, sites = run_json(capsys, root, "--sites", "UKA_1", "--expect-image-id", CONFIG)
    assert sites["UKA_1"]["verdict"] == "READY" and code == fs.EXIT_READY


def test_stopped_client(fleet, capsys):
    _, _, sites = run_json(capsys, fleet)
    assert sites["UMCU_1"]["verdict"] == "STOPPED"
    assert "client container" in sites["UMCU_1"]["reason"]


@pytest.mark.parametrize("fields,word", [
    ({"client_state": "stopped"}, "client_state stopped"),
    ({"image_id": ""}, "no image id"),
    ({"status": "finished"}, "live_sync ended"),
])
def test_fresh_heartbeat_that_says_the_client_is_down(tmp_path, capsys, fields, word):
    root = tmp_path / "live"
    beat(root, "CAM_1", "_active", **{"status": "waiting", "age": 10, **fields})
    _, _, sites = run_json(capsys, root, "--sites", "CAM_1")
    assert sites["CAM_1"]["verdict"] == "STOPPED" and word in sites["CAM_1"]["reason"]


def test_stale_old_kit_proves_nothing_about_the_migration(fleet, capsys):
    _, _, sites = run_json(capsys, fleet)
    usz = sites["USZ_1"]
    assert usz["verdict"] == "STALE" and "proves nothing about the migration" in usz["reason"]
    assert usz["kit_version"] == "1.5.0-dev.260702.be9ef04", "doubled closing quote salvaged"
    _, _, since = run_json(capsys, fleet, "--since", iso(time.time() - 3 * 86400)[:10])
    assert "predates" in since["USZ_1"]["reason"]


def test_stale_migrated_site_means_its_client_is_down(tmp_path, capsys):
    root = tmp_path / "live"
    beat(root, "CAM_1", "_active", status="waiting", age=3600)
    code, _, sites = run_json(capsys, root, "--sites", "CAM_1")
    assert code == fs.EXIT_NOT_READY
    assert sites["CAM_1"]["verdict"] == "STALE" and "not running" in sites["CAM_1"]["reason"]
    code, _, sites = run_json(capsys, root, "--sites", "CAM_1", "--max-age", "2h")
    assert sites["CAM_1"]["verdict"] == "READY" and code == fs.EXIT_READY


def test_missing_site_and_empty_site(tmp_path, capsys):
    root = tmp_path / "live"
    (root / "MHA_1" / "swarm").mkdir(parents=True)
    _, _, sites = run_json(capsys, root, "--sites", "MHA_1,VHIO_1")
    assert sites["VHIO_1"]["verdict"] == sites["MHA_1"]["verdict"] == "NO HEARTBEAT"
    assert "never uploaded" in sites["VHIO_1"]["reason"]
    assert "no heartbeat under MHA_1/swarm" in sites["MHA_1"]["reason"]


def test_malformed_heartbeats(tmp_path, capsys):
    root = tmp_path / "live"
    beat(root, "RSH_1", "aaaaaaaa-0000-0000-0000-000000000000", raw="{not json", age=5)
    beat(root, "CAM_1", "_active", status="waiting", age=60)
    beat(root, "CAM_1", "bbbbbbbb-0000-0000-0000-000000000000", raw='{"kit_version": "1.9.0", "status": ', age=5)
    # ANSI colour codes from the run name: salvaged, as the monitor does
    beat(root, "MHA_1", "cccccccc-0000-0000-0000-000000000000", age=5,
         raw=json.dumps({"kit_version": "1.9.0", "timestamp": iso(time.time() - 5), "status": "running",
                         "image_id": short(CONFIG), "client_state": "running"}).replace(
             '"status"', '"run_name": "\x1b[32mrun\x1b[0m", "status"'))
    _, _, sites = run_json(capsys, root, "--sites", "RSH_1,CAM_1,MHA_1")
    assert sites["RSH_1"]["verdict"] == "NO HEARTBEAT" and "unreadable" in sites["RSH_1"]["reason"]
    assert sites["CAM_1"]["verdict"] == "READY"
    assert any("newer heartbeat file is unreadable" in n for n in sites["CAM_1"]["notes"])
    assert sites["MHA_1"]["verdict"] == "READY"


def test_non_site_folders_are_ignored_and_all_ready_exits_zero(fleet, capsys):
    code, doc, sites = run_json(capsys, fleet, "--sites", "CAM_1", "--sites", "MHA_1",
                                "--expect-image-id", CONFIG, "--expect-image-id", INDEX)
    assert code == fs.EXIT_READY and doc["ready"] is True and list(sites) == ["CAM_1", "MHA_1"]


def test_local_heartbeats_count_only_when_asked(tmp_path, capsys):
    root = tmp_path / "live"
    beat(root, "CAM_1", "20261015_run", mode="local", age=10)
    _, _, sites = run_json(capsys, root, "--sites", "CAM_1")
    assert sites["CAM_1"]["verdict"] == "NO HEARTBEAT"
    _, _, sites = run_json(capsys, root, "--sites", "CAM_1", "--include-local")
    assert sites["CAM_1"]["verdict"] == "READY" and sites["CAM_1"]["mode"] == "local"


def test_table_output_has_every_site_and_no_host_identity(fleet, capsys):
    code, out, _ = run(capsys, "--live-root", str(fleet), "--expect-image-id", CONFIG, "--expect-image-id", INDEX)
    assert code == fs.EXIT_NOT_READY
    lines = out.splitlines()
    header = next(line for line in lines if line.startswith("SITE"))
    for column in ["KIT", "IMAGE", "ID", "MATCH", "AGE", "STATUS", "JOB", "VERDICT", "REASON"]:
        assert column in header
    for site in fs.ODELIA_SITES:
        assert any(line.startswith(site + " ") for line in lines), site
    for other in ["_audit", "_probe", "ODELIA_MODELS", "client_A", "node_A", "Mainz_1"]:
        assert other not in out
    assert "2/8 READY" in out
    cam = next(line for line in lines if line.startswith("CAM_1 "))
    assert "waiting" in cam and "c527c8c509aa" in cam and "READY" in cam
    _, js, _ = run(capsys, "--live-root", str(fleet), "--json")
    for text in (out, js):
        assert HOST not in text and IP not in text


def test_usage_errors_exit_two(tmp_path, capsys):
    assert run(capsys, "--live-root", str(tmp_path / "nope"))[0] == fs.EXIT_USAGE
    (tmp_path / "live").mkdir()
    assert run(capsys, "--live-root", str(tmp_path / "live"), "--max-age", "soon")[0] == fs.EXIT_USAGE
    assert run(capsys, "--live-root", str(tmp_path / "live"), "--sites", "../etc")[0] == fs.EXIT_USAGE
    assert run(capsys, "--live-root", str(tmp_path / "live"), "--expect-image-id", "sha256:abc")[0] == fs.EXIT_USAGE


# ------------------------------------------------------------------- image ids

def fake_docker(monkeypatch, store):
    """docker image inspect as a classic-store or a containerd-store host answers it."""
    def answer(args):
        args = list(args)
        if args[-1] != "jefftud/odelia:1.9.1":
            return None
        if store == "classic":
            return [{"Id": CONFIG, "RepoDigests": ["jefftud/odelia@" + INDEX]}]
        if "--platform" in args:
            return [{"Id": MANIFEST, "RepoDigests": ["jefftud/odelia@" + INDEX],
                     "Descriptor": {"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": MANIFEST}}]
        return [{"Id": INDEX, "RepoDigests": ["jefftud/odelia@" + INDEX],
                 "Descriptor": {"mediaType": "application/vnd.oci.image.index.v1+json", "digest": INDEX}}]
    monkeypatch.setattr(fs, "_docker_json", answer)


def test_classic_store_host_resolves_config_and_repo_digest(monkeypatch):
    fake_docker(monkeypatch, "classic")
    exp = fs.ImageExpectation()
    fs.resolve_image("jefftud/odelia:1.9.1", exp, use_registry=False)
    assert exp.warnings == []
    assert exp.matches(short(CONFIG)) and exp.matches(short(INDEX)) and not exp.matches(OLD_181)


def test_containerd_store_host_warns_that_the_config_digest_is_unknown(monkeypatch):
    fake_docker(monkeypatch, "containerd")
    exp = fs.ImageExpectation()
    fs.resolve_image("jefftud/odelia:1.9.1", exp, use_registry=False)
    assert exp.matches(short(INDEX)) and exp.matches(short(MANIFEST)) and not exp.matches(short(CONFIG))
    assert len(exp.warnings) == 1 and "--expect-image-id" in exp.warnings[0]


def test_registry_fills_in_the_config_digest_only_when_asked(monkeypatch):
    fake_docker(monkeypatch, "containerd")
    asked = []
    monkeypatch.setattr(fs, "_registry_config_digest", lambda ref: asked.append(ref) or CONFIG)
    exp = fs.ImageExpectation()
    fs.resolve_image("jefftud/odelia:1.9.1", exp, use_registry=True)
    assert asked == ["jefftud/odelia:1.9.1"] and exp.matches(short(CONFIG)) and exp.warnings == []


def test_expect_image_end_to_end(fleet, monkeypatch, capsys):
    fake_docker(monkeypatch, "containerd")
    code, doc, sites = run_json(capsys, fleet, "--expect-image", "jefftud/odelia:1.9.1",
                                "--expect-image-id", short(CONFIG), "--sites", "CAM_1,MHA_1")
    assert code == fs.EXIT_READY and doc["expect_image"]["warnings"] == []
    assert {i["source"].split(",")[0] for i in doc["expect_image"]["ids"]} == {"--expect-image-id", "index",
                                                                               "linux/amd64 manifest"}


def test_unresolvable_image_is_a_usage_error_not_a_silent_skip(fleet, monkeypatch, capsys):
    monkeypatch.setattr(fs, "_docker_json", lambda args: None)
    code, _, err = run(capsys, "--live-root", str(fleet), "--expect-image", "jefftud/odelia:9.9.9")
    assert code == fs.EXIT_USAGE and "--expect-image-id" in err


def test_without_an_image_expectation_the_table_says_so(fleet, capsys):
    _, out, _ = run(capsys, "--live-root", str(fleet), "--sites", "CAM_1")
    assert "NOT CHECKED" in out


def test_id_matching_rules():
    assert fs.ids_match("cbf6ba7aa574", "cbf6ba7aa57472be76a0e6ece62088bffc591b405d51cc16afce6b57dbc36b81")
    assert not fs.ids_match("cbf6ba7a", "cbf6ba7aa574"), "fewer than 12 hex never match"
    assert fs.id_hex(" 'sha256:CBF6BA7AA574\" ") == "cbf6ba7aa574"
    assert fs.id_hex("jefftud/odelia:current") == ""
    assert [fs.parse_duration(x) for x in ("900", "30s", "15m", "2h", "1d")] == [900, 30, 900, 7200, 86400]


# ------------------------------------------------------------------- shared reader

def test_shared_reader_salvages_the_known_defects():
    assert heartbeat_io.parse_heartbeat_text('{"kit_version": "1.6.0"", "x": ""}') == {"kit_version": "1.6.0", "x": ""}
    assert heartbeat_io.parse_heartbeat_text('{"run_name": "\x1b[32mr\x1b[0m"}') == {"run_name": "r"}
    with pytest.raises(ValueError):
        heartbeat_io.parse_heartbeat_text("[1, 2]")
    assert heartbeat_io.clean_value(' "1.8.1"" ') == "1.8.1"


def test_monitor_reads_heartbeats_through_the_shared_reader(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("MEDISWARM_MONITOR_ALLOW_OPEN", "1")
    import app
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "heartbeat.json").write_text('{"kit_version": "1.6.0"", "status": "running"}')
    assert app._read_heartbeat(run_dir) == {"kit_version": "1.6.0", "status": "running"}
    (run_dir / "heartbeat_final.json").write_text("{broken")
    assert app._read_heartbeat(run_dir)["_parse_error_file"].endswith("heartbeat_final.json")


def test_monitor_deploy_instructions_copy_the_shared_reader():
    readme = (SERVER_TOOLS / "README.md").read_text()
    copies = [line for line in readme.splitlines() if line.startswith("cp ") and "app.py" in line]
    assert copies and all("heartbeat_io.py" in line for line in copies)
