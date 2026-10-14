"""Unit tests for application/jobs/_shared/custom/state_digest.py (release 1.9.0 contract 3.4).

The digest must identify the weights, not their container: torch vs numpy, CPU vs CUDA, contiguous vs not,
dict order and the per-site class-weight buffers must not change it; dtype, shape, values and names must.
checkpoint_state_digest must hash what the persistor will send (the 'model' entry, bfloat16 widened).
"""

import hashlib

import pytest

from conftest import SHARED_CUSTOM_DIR, import_module_from_path

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def sd():
    return import_module_from_path("state_digest_under_test", SHARED_CUSTOM_DIR / "state_digest.py")


def _state():
    generator = torch.Generator().manual_seed(0)
    return {
        "encoder.weight": torch.randn(4, 3, generator=generator),
        "encoder.bias": torch.randn(4, generator=generator),
        "head.weight": torch.randn(2, 4, generator=generator),
        "bn.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
    }


def test_digest_is_independent_of_representation(sd):
    state = _state()
    reference = sd.state_digest(state)
    assert len(reference) == 64

    reordered = dict(reversed(list(state.items())))
    as_numpy = {k: v.numpy() for k, v in state.items()}
    non_contiguous = dict(state)
    non_contiguous["head.weight"] = state["head.weight"].t().contiguous().t()
    assert not non_contiguous["head.weight"].is_contiguous()
    requires_grad = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in state.items()}
    big_endian = {k: v.numpy().astype(v.numpy().dtype.newbyteorder(">")) for k, v in state.items()}

    for variant in (reordered, as_numpy, non_contiguous, requires_grad, big_endian):
        assert sd.state_digest(variant) == reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_digest_is_the_same_on_cuda(sd):
    state = _state()
    on_gpu = {k: v.cuda() for k, v in state.items()}
    assert sd.state_digest(on_gpu) == sd.state_digest(state)


def test_digest_changes_with_dtype_shape_values_and_names(sd):
    state = _state()
    reference = sd.state_digest(state)

    half = dict(state, **{"encoder.weight": state["encoder.weight"].half()})
    double = dict(state, **{"encoder.bias": state["encoder.bias"].double()})
    reshaped = dict(state, **{"encoder.weight": state["encoder.weight"].reshape(3, 4)})
    nudged = dict(state, **{"head.weight": state["head.weight"] + 1e-7})
    renamed = {("decoder.bias" if k == "encoder.bias" else k): v for k, v in state.items()}
    missing = {k: v for k, v in state.items() if k != "mask"}

    digests = {sd.state_digest(v) for v in (half, double, reshaped, nudged, renamed, missing)}
    assert reference not in digests
    assert len(digests) == 6


def test_per_site_class_weight_buffers_are_ignored(sd):
    state = _state()
    reference = sd.state_digest(state)
    local = dict(state, _class_weight=torch.tensor([1.0, 2.5, 7.0]), **{"loss.weight": torch.tensor([1.0, 2.5, 7.0])})
    nested = dict(state, **{"model._class_weight": torch.tensor([3.0])})
    assert sd.state_digest(local) == reference
    assert sd.state_digest(nested) == reference
    assert sd.is_ignored_key("_class_weight") and sd.is_ignored_key("backbone.loss.weight")
    assert not sd.is_ignored_key("my_class_weight") and not sd.is_ignored_key("head.weight")


def test_documented_format(sd):
    """A reimplementation from the module docstring must give the same digest."""
    state = {"b": torch.tensor([1.5, -2.0]), "a": torch.tensor(3, dtype=torch.int64), "c": torch.zeros(2, 0)}
    expected = hashlib.sha256()
    expected.update(b"a|int64||")
    expected.update(np.array(3, dtype="<i8").tobytes())
    expected.update(b"b|float32|2|")
    expected.update(np.array([1.5, -2.0], dtype="<f4").tobytes())
    expected.update(b"c|float32|2,0|")
    assert sd.state_digest(state) == expected.hexdigest()


def test_bfloat16_and_python_numbers_are_supported(sd):
    bf16 = {"w": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    assert sd.state_digest(bf16) != sd.state_digest({"w": torch.tensor([1.0, 2.0])})
    assert sd.state_digest({"n": 3}) == sd.state_digest({"n": np.array(3, dtype=np.int64)})
    with pytest.raises(TypeError):
        sd.state_digest({"s": "not a tensor"})
    with pytest.raises(TypeError):
        sd.state_digest([("w", torch.ones(1))])


def test_checkpoint_digest_matches_what_the_persistor_sends(sd, tmp_path):
    state = _state()
    wrapped = tmp_path / "global.pt"
    torch.save({"model": state, "train_conf": {"train": {"model": "Tiny"}}}, wrapped)
    plain = tmp_path / "plain.pt"
    torch.save(state, plain)
    assert sd.checkpoint_state_digest(str(wrapped)) == sd.state_digest(state)
    assert sd.checkpoint_state_digest(str(plain)) == sd.state_digest(state)

    # The persistor widens bfloat16 to float32 before shipping (PTModelPersistenceFormatManager).
    bf16_file = tmp_path / "bf16.pt"
    torch.save({"model": {"w": torch.tensor([1.0, 0.5], dtype=torch.bfloat16)}}, bf16_file)
    assert sd.checkpoint_state_digest(str(bf16_file)) == sd.state_digest({"w": torch.tensor([1.0, 0.5])})


def test_normalise_hex_digest(sd):
    value = "AB" * 32
    assert sd.normalise_hex_digest(f"  {value} ") == value.lower()
    assert sd.normalise_hex_digest("") is None and sd.normalise_hex_digest(None) is None
    for bad in ("abc", "g" * 64, "a" * 63, "a" * 65):
        with pytest.raises(ValueError):
            sd.normalise_hex_digest(bad)
