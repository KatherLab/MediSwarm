"""MST_SAMMed2D — the MST slice-fusion classifier on the SAM-Med2D ViT-B encoder.

Runs where torch, einops and x_transformers are installed (the ODELIA image); skipped
elsewhere. No pretrained weights are needed: the backbone falls back to random init and
logs a warning, which is exactly the path a preflight check on a site without the
checkpoint takes.
"""
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("x_transformers")

CUSTOM = Path(__file__).resolve().parents[2] / "application" / "jobs" / "ODELIA_ternary_classification" / "app" / "custom"


@pytest.fixture(scope="module")
def mst_module():
    sys.path.insert(0, str(CUSTOM))
    from models import mst  # noqa: E402
    return mst


def test_checkpoint_resolution_prefers_env(tmp_path, monkeypatch, mst_module):
    f = tmp_path / "sam-med2d_b.pth"
    f.write_bytes(b"x")
    monkeypatch.setenv("SAM_MED2D_CHECKPOINT", str(f))
    assert mst_module.resolve_sam_med2d_checkpoint() == str(f)


def test_checkpoint_env_must_exist(tmp_path, monkeypatch, mst_module):
    monkeypatch.setenv("SAM_MED2D_CHECKPOINT", str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError):
        mst_module.resolve_sam_med2d_checkpoint()


def test_sammed2d_backbone_forward_shape(monkeypatch, mst_module, caplog):
    """A 224-pixel volume goes in, one logit per class comes out; random init warns."""
    monkeypatch.delenv("SAM_MED2D_CHECKPOINT", raising=False)
    monkeypatch.setattr(mst_module, "SAM_MED2D_CHECKPOINT_CANDIDATES", ())
    with caplog.at_level("WARNING"):
        net = mst_module._MST(out_ch=3, backbone_type="sammed2d", slice_fusion_type="transformer")
    assert any("RANDOM weights" in r.getMessage() for r in caplog.records)
    net.eval()
    x = torch.randn(1, 1, 4, 224, 224)  # (B, C, D, H, W) as the ODELIA dataset delivers it
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 3)
    assert net.emb_ch == 256


def test_unknown_backbone_still_rejected(mst_module):
    with pytest.raises(ValueError, match="Unknown backbone_type"):
        mst_module._MST(out_ch=3, backbone_type="nope")
