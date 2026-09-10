"""Provenance guarding for the shared warm-start mirror (#535).

All six ODELIA jobs read and write one checkpoint path, and five different
architectures can land there. Loading another model's weights is E2 in
docs/EVALUATION_PITFALLS.md. These tests pin the guard, and equally pin that it
does not break warm-continue for the mirrors written before it existed.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import SHARED_CUSTOM_DIR, import_module_from_path  # noqa: E402

pytest.importorskip("nvflare")
MODULE_PATH = SHARED_CUSTOM_DIR / "warm_continue.py"


@pytest.fixture(scope="module")
def mod():
    return import_module_from_path("_warm_continue_prov", MODULE_PATH)


def test_sidecar_sits_beside_the_checkpoint(mod):
    assert mod.sidecar_path("/scratch/x.pt") == "/scratch/x.pt.provenance.json"


def test_roundtrip_records_the_model(mod, tmp_path):
    ckpt = tmp_path / "g.pt"
    ckpt.write_bytes(b"weights")
    mod.write_provenance(str(ckpt), "1DivideAndConquer", "job-1", "abc")
    got = mod.read_provenance(str(ckpt))
    assert got["model_name"] == "1DivideAndConquer"
    assert got["job_id"] == "job-1"


def test_missing_sidecar_reads_as_none_not_an_error(mod, tmp_path):
    ckpt = tmp_path / "g.pt"
    ckpt.write_bytes(b"weights")
    assert mod.read_provenance(str(ckpt)) is None


def test_corrupt_sidecar_reads_as_none_not_an_exception(mod, tmp_path):
    """A damaged sidecar must degrade to 'unknown', never take down a run."""
    ckpt = tmp_path / "g.pt"
    ckpt.write_bytes(b"weights")
    Path(mod.sidecar_path(str(ckpt))).write_text("{not json")
    assert mod.read_provenance(str(ckpt)) is None


def test_digest_is_stable_and_content_sensitive(mod, tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"same"); b.write_bytes(b"same")
    assert mod.file_digest(str(a)) == mod.file_digest(str(b))
    b.write_bytes(b"different")
    assert mod.file_digest(str(a)) != mod.file_digest(str(b))


def test_digest_of_unreadable_file_is_empty(mod, tmp_path):
    assert mod.file_digest(str(tmp_path / "nope.pt")) == ""


def test_current_model_name_reads_the_env(mod, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "Swin3D")
    assert mod.current_model_name() == "Swin3D"
    monkeypatch.delenv("MODEL_NAME", raising=False)
    assert mod.current_model_name() == "unknown"


def test_write_provenance_never_raises_on_a_bad_path(mod):
    """Provenance is a guard, not a feature -- it must not fail a run."""
    mod.write_provenance("/nonexistent-dir-xyz/g.pt", "MST")   # must not raise


def test_digest_handles_a_file_larger_than_one_block(mod, tmp_path):
    """Checkpoints are ~700 MB; the digest must stream, not slurp."""
    import hashlib
    big = tmp_path / "big.pt"
    payload = os.urandom(1024 * 1024 * 3 + 17)      # >3 blocks, non-aligned
    big.write_bytes(payload)
    assert mod.file_digest(str(big)) == hashlib.sha256(payload).hexdigest()


def test_a_failed_provenance_write_does_not_mask_a_successful_mirror(mod, tmp_path, monkeypatch):
    """Regression: provenance sat inside the copy's try block, so a failure there
    made a successful mirror log as a failure. The copy is the important part."""
    import types

    class _Logger:
        def __init__(self): self.messages = []

    persistor = mod.WarmStartablePTFileModelPersistor.__new__(mod.WarmStartablePTFileModelPersistor)
    persistor.latest_global_path = str(tmp_path / "mirror.pt")
    persistor.logger = _Logger()
    persistor.log_info = lambda ctx, m, **k: persistor.logger.messages.append(("info", m))
    persistor.log_warning = lambda ctx, m, **k: persistor.logger.messages.append(("warning", m))

    src = tmp_path / "src.pt"
    src.write_bytes(b"weights")

    # force the provenance step to blow up
    monkeypatch.setattr(mod, "write_provenance", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    persistor._mirror_checkpoint(str(src), types.SimpleNamespace(), "latest")

    kinds = [k for k, _ in persistor.logger.messages]
    texts = " ".join(m for _, m in persistor.logger.messages)
    assert os.path.exists(persistor.latest_global_path), "the checkpoint must still be mirrored"
    assert "mirrored latest global" in texts, "success must still be reported"
    assert "could not record provenance" in texts, "the provenance failure is reported separately"
    assert kinds.count("warning") == 1
