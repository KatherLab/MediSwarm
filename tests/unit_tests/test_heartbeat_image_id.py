"""scripts/build/heartbeat_image_id.py: the id a site's heartbeat reports for a pushed image.

After a release, the only proof that a site restarted onto it is the heartbeat's image_id, which is
the amd64 CONFIG digest -- not the tag's manifest digest Docker Hub shows. The registry is faked.
"""

import importlib.util
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("heartbeat_image_id",
                                              REPO_ROOT / "scripts" / "build" / "heartbeat_image_id.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)

INDEX = {"mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [
    {"digest": "sha256:" + "a" * 64, "platform": {"architecture": "amd64", "os": "linux"}},
    {"digest": "sha256:" + "b" * 64, "platform": {"architecture": "unknown", "os": "unknown"},
     "annotations": {"vnd.docker.reference.type": "attestation-manifest"}},
]}
MANIFEST = {"config": {"digest": "sha256:" + "c" * 64}}


def test_repository_strips_tag_and_digest_but_not_a_registry_port():
    assert hb.repository("jefftud/odelia:1.9.0") == "jefftud/odelia"
    assert hb.repository("jefftud/odelia@sha256:" + "a" * 64) == "jefftud/odelia"
    assert hb.repository("localhost:5000/odelia:1.9.0") == "localhost:5000/odelia"
    assert hb.repository("localhost:5000/odelia") == "localhost:5000/odelia"


def test_index_resolves_to_the_amd64_config_digest(monkeypatch):
    seen = []

    def fake_raw(ref):
        seen.append(ref)
        return INDEX if ref == "jefftud/odelia:1.9.0" else MANIFEST

    monkeypatch.setattr(hb, "raw", fake_raw)
    assert hb.config_digest("jefftud/odelia:1.9.0") == "sha256:" + "c" * 64
    assert seen[1] == "jefftud/odelia@sha256:" + "a" * 64, "the attestation manifest must be skipped"


def test_newest_heartbeat_per_site_is_compared(tmp_path, monkeypatch, capsys):
    for site, job, image_id, age in [("UKA_1", "old", "sha256:cbf6ba7aa574", 100),
                                     ("UKA_1", "new", "sha256:cccccccccccc", 0),
                                     ("CAM_1", "only", "sha256:cbf6ba7aa574", 0)]:
        beat = tmp_path / site / "swarm" / job / "heartbeat.json"
        beat.parent.mkdir(parents=True)
        beat.write_text(json.dumps({"image_id": image_id, "timestamp": "t", "kit_version": "1.8.1"}))
        os.utime(beat, (1_000_000 - age, 1_000_000 - age))
    monkeypatch.setattr(hb, "config_digest", lambda ref: "sha256:" + "c" * 64)
    assert hb.main(["--live-root", str(tmp_path), "jefftud/odelia:1.9.0"]) == 0
    out = capsys.readouterr().out
    assert "OK  UKA_1" in out and "OLD CAM_1" in out
