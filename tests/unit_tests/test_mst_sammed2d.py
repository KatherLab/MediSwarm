"""MST_SAMMed2D — the MST slice-fusion classifier on the SAM-Med2D ViT-B encoder.

Runs where torch, einops and x_transformers are installed (the ODELIA image); skipped
elsewhere. No pretrained weights are needed, and the 267 M-parameter model is never built with
real weights: the guard tests swap in a tiny encoder of the same architecture (and its own fake
checkpoint), and the one full-size forward pass uses the test-only random-init override.
"""
import logging
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


@pytest.fixture
def no_weights(monkeypatch, tmp_path, mst_module):
    """No SAM_MED2D_CHECKPOINT, nothing at the default locations, no override."""
    monkeypatch.delenv("SAM_MED2D_CHECKPOINT", raising=False)
    monkeypatch.delenv("SAM_MED2D_ALLOW_RANDOM_INIT", raising=False)
    candidates = (str(tmp_path / "image" / "sam-med2d_b.pth"), str(tmp_path / "hub" / "sam-med2d_b.pth"))
    monkeypatch.setattr(mst_module, "SAM_MED2D_CHECKPOINT_CANDIDATES", candidates)
    return candidates


def _tiny_encoder():
    """The SAM-Med2D image encoder class (adapters, relative positions, windowed and global
    attention), sized down so a test can build it in milliseconds."""
    from models.segment_anything.modeling import ImageEncoderViT

    torch.manual_seed(0)
    return ImageEncoderViT(
        img_size=32, patch_size=16, embed_dim=16, depth=2, num_heads=2, out_chans=8,
        use_rel_pos=True, window_size=1, global_attn_indexes=(1,), adapter_train=True,
    )


@pytest.fixture
def tiny(monkeypatch, mst_module):
    """Replace the 267 M-parameter encoder with the tiny one wherever the guard builds it."""
    monkeypatch.setattr(mst_module, "_new_sam_med2d_encoder", _tiny_encoder)


def _fake_checkpoint(path, wrap=True, drop=(), add=(), reshape=()):
    """A SAM-Med2D-style checkpoint for the tiny encoder: {'model': sam_state, 'optimizer': ...}.

    Its values differ from the tiny encoder's own (seeded) initialisation, so a test can tell
    loaded weights from random ones."""
    shapes = _tiny_encoder().state_dict()
    torch.manual_seed(1)
    state = {f"image_encoder.{k}": torch.randn_like(v) for k, v in shapes.items()}
    state["mask_decoder.iou_token.weight"] = torch.randn(1, 8)  # parts of SAM the classifier drops
    state["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"] = torch.randn(2, 4)
    for key in drop:
        del state[f"image_encoder.{key}"]
    for key in add:
        state[f"image_encoder.{key}"] = torch.zeros(3)
    for key in reshape:
        state[f"image_encoder.{key}"] = torch.zeros(1, 2, 3)
    torch.save({"model": state, "optimizer": {"state": {}, "param_groups": []}} if wrap else state, path)
    return {k[len("image_encoder."):]: v for k, v in state.items() if k.startswith("image_encoder.")}


def test_checkpoint_resolution_prefers_env(tmp_path, monkeypatch, mst_module):
    f = tmp_path / "sam-med2d_b.pth"
    f.write_bytes(b"x")
    monkeypatch.setenv("SAM_MED2D_CHECKPOINT", str(f))
    assert mst_module.resolve_sam_med2d_checkpoint() == str(f)


def test_checkpoint_env_must_exist(tmp_path, monkeypatch, mst_module):
    monkeypatch.setenv("SAM_MED2D_CHECKPOINT", str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError, match="missing.pth"):
        mst_module.resolve_sam_med2d_checkpoint()


def test_missing_weights_refuse_to_start(no_weights, monkeypatch, mst_module):
    """No weights anywhere: refuse before building anything, naming every place searched and the fix."""
    def must_not_build():
        raise AssertionError("the encoder must not be built before the weights are found")

    monkeypatch.setattr(mst_module, "_new_sam_med2d_encoder", must_not_build)
    with pytest.raises(RuntimeError, match="MST_SAMMed2D refuses to start") as excinfo:
        mst_module.build_sam_med2d_encoder()
    message = str(excinfo.value)
    for searched in ("SAM_MED2D_CHECKPOINT (not set)", *no_weights):
        assert searched in message
    assert "SAM_MED2D_ALLOW_RANDOM_INIT=1" in message  # the test-only escape hatch is named
    assert mst_module.SAM_MED2D_RELEASE_SHA256 in message  # so is the file to get


@pytest.mark.parametrize("value", ["0", "", "no", "false"])
def test_only_an_explicit_override_allows_random_weights(no_weights, monkeypatch, tiny, mst_module, value):
    monkeypatch.setenv("SAM_MED2D_ALLOW_RANDOM_INIT", value)
    with pytest.raises(RuntimeError, match="refuses to start"):
        mst_module.build_sam_med2d_encoder()


def test_override_allows_random_init_loudly(no_weights, monkeypatch, tiny, mst_module, caplog):
    monkeypatch.setenv("SAM_MED2D_ALLOW_RANDOM_INIT", "1")
    with caplog.at_level(logging.WARNING):
        encoder = mst_module.build_sam_med2d_encoder()
    assert isinstance(encoder, torch.nn.Module)
    warning = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "RANDOM weights" in warning and "SAM_MED2D_ALLOW_RANDOM_INIT=1" in warning
    assert "were not found" in warning  # the reason travels with the warning


@pytest.mark.parametrize("wrap", [True, False], ids=["sam_med2d_release_format", "plain_state_dict"])
def test_complete_checkpoint_loads_and_is_logged(tmp_path, monkeypatch, tiny, mst_module, caplog, wrap):
    path = tmp_path / "sam-med2d_b.pth"
    expected = _fake_checkpoint(path, wrap=wrap)
    monkeypatch.setenv("SAM_MED2D_CHECKPOINT", str(path))
    with caplog.at_level(logging.INFO):
        encoder = mst_module.build_sam_med2d_encoder()
    loaded = encoder.state_dict()
    assert set(loaded) == set(expected)
    assert all(torch.equal(loaded[k], expected[k]) for k in expected)
    info = " ".join(r.getMessage() for r in caplog.records)
    assert f"SAM-Med2D backbone weights: {path}" in info  # the line operators grep for
    assert f"{os.path.getsize(path):,} bytes" in info
    assert f"loaded {len(expected)}/{len(expected)} encoder tensors" in info
    assert "not the published release file" in info  # a fake file is not the release


@pytest.mark.parametrize(
    "damage, expected_text",
    [
        (dict(drop=["blocks.0.Adapter.channel.0.weight"]), "1 of"),
        (dict(add=["blocks.9.extra.weight"]), "unexpected encoder tensors (blocks.9.extra.weight)"),
        (dict(reshape=["pos_embed"]), "another shape (pos_embed (file (1, 2, 3), model (1, 2, 2, 16)))"),
    ],
    ids=["missing_adapter_tensor", "unexpected_tensor", "reshaped_tensor"],
)
def test_incomplete_checkpoint_refuses_to_start(tmp_path, monkeypatch, tiny, mst_module, damage, expected_text):
    """The old loader used load_state_dict(strict=False): a file without the adapter weights (e.g. the
    original SAM ViT-B) loaded 'successfully' with random adapters."""
    path = tmp_path / "partial.pth"
    _fake_checkpoint(path, **damage)
    monkeypatch.setenv("SAM_MED2D_CHECKPOINT", str(path))
    with pytest.raises(RuntimeError, match="does not fit the SAM-Med2D ViT-B encoder") as excinfo:
        mst_module.build_sam_med2d_encoder()
    assert str(path) in str(excinfo.value)
    assert expected_text in str(excinfo.value)


def test_unreadable_checkpoint_refuses_to_start(tmp_path, monkeypatch, tiny, mst_module):
    path = tmp_path / "sam-med2d_b.pth"
    _fake_checkpoint(path)
    path.write_bytes(path.read_bytes()[:200])  # a truncated copy
    monkeypatch.setenv("SAM_MED2D_CHECKPOINT", str(path))
    with pytest.raises(RuntimeError, match="cannot be read as a SAM-Med2D checkpoint") as excinfo:
        mst_module.build_sam_med2d_encoder()
    assert "200 bytes" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


def test_override_never_returns_a_partially_loaded_encoder(tmp_path, monkeypatch, tiny, mst_module, caplog):
    path = tmp_path / "partial.pth"
    from_file = _fake_checkpoint(path, drop=["neck.0.weight"])
    monkeypatch.setenv("SAM_MED2D_CHECKPOINT", str(path))
    monkeypatch.setenv("SAM_MED2D_ALLOW_RANDOM_INIT", "1")
    with caplog.at_level(logging.WARNING):
        encoder = mst_module.build_sam_med2d_encoder()
    random_init = _tiny_encoder().state_dict()
    loaded = encoder.state_dict()
    assert all(torch.equal(loaded[k], random_init[k]) for k in random_init)
    assert not torch.equal(loaded["patch_embed.proj.weight"], from_file["patch_embed.proj.weight"])
    assert any("RANDOM weights" in r.getMessage() for r in caplog.records)


def test_architecture_only_skips_the_weights(no_weights, monkeypatch, tiny, mst_module, caplog):
    """predict.py builds the architecture and loads every weight from the evaluated checkpoint."""
    monkeypatch.setenv("SAM_MED2D_CHECKPOINT", "/does/not/exist.pth")  # not even looked at
    with caplog.at_level(logging.INFO):
        with mst_module.sam_med2d_architecture_only("loading a full checkpoint next"):
            encoder = mst_module.build_sam_med2d_encoder()
    assert isinstance(encoder, torch.nn.Module)
    assert any("loading a full checkpoint next" in r.getMessage() for r in caplog.records)
    assert not any("RANDOM" in r.getMessage() for r in caplog.records)
    monkeypatch.delenv("SAM_MED2D_CHECKPOINT")
    with pytest.raises(RuntimeError, match="refuses to start"):  # the context does not leak
        mst_module.build_sam_med2d_encoder()


def test_sammed2d_backbone_forward_shape(no_weights, monkeypatch, mst_module, caplog):
    """A 224-pixel volume goes in, one logit per class comes out (random init via the test override).

    Also pins down the module audit behind #618: the model has no BatchNorm, no active dropout and no
    drop-path, so its train-mode and eval-mode outputs agree.
    """
    monkeypatch.setenv("SAM_MED2D_ALLOW_RANDOM_INIT", "1")
    with caplog.at_level("WARNING"):
        net = mst_module._MST(out_ch=3, backbone_type="sammed2d", slice_fusion_type="transformer")
    assert any("RANDOM weights" in r.getMessage() for r in caplog.records)
    net.eval()
    x = torch.randn(1, 1, 4, 224, 224)  # (B, C, D, H, W) as the ODELIA dataset delivers it
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 3)
    assert net.emb_ch == 256
    for name, module in net.named_modules():
        assert not isinstance(module, torch.nn.modules.batchnorm._BatchNorm), name
        assert not (isinstance(module, torch.nn.Dropout) and module.p > 0), name
        assert "droppath" not in type(module).__name__.lower(), name


def test_unknown_backbone_still_rejected(mst_module):
    with pytest.raises(ValueError, match="Unknown backbone_type"):
        mst_module._MST(out_ch=3, backbone_type="nope")
