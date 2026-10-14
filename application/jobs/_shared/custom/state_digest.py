"""Canonical SHA-256 digest of a model state dict (release 1.9.0 contract 3.4).

Why this exists
---------------
A warm-started swarm run must start from exactly the intended weights. The file digest of the
checkpoint (``sha256sum``) is checked by the starting client's persistor, but what every site actually
trains from is the round-0 global that arrives over the wire: a state dict of tensors (or numpy arrays),
with no file anywhere near it. This digest identifies those weights independently of how they are held:

* keys are visited in sorted order, so dict insertion order does not matter;
* each value contributes ``name|dtype|shape|`` followed by its raw bytes (C-contiguous, little-endian,
  on the CPU), so a CUDA tensor, a non-contiguous view, a torch tensor and the numpy array the persistor
  ships all give the same digest, while a different dtype (float16 vs float32) or shape gives another;
* per-site buffers that are not learned parameters are ignored (``IGNORED_KEYS``): ``_class_weight`` is
  computed from each site's own label counts, and ``loss.weight`` is the same tensor injected into the
  loss. Neither is part of the global, but a local model's state dict carries them.

The job builder (scripts/admin) records ``checkpoint_state_digest(path)`` of the warm-start file and
passes it to every site as ``ODELIA_WARM_START_SHA256``; the reproducibility round hook recomputes it from the
received round-0 global and stops the job on a mismatch. ``checkpoint_state_dict`` mirrors what NVFlare's
PTFileModelPersistor turns a checkpoint file into (the ``model`` entry if present, bfloat16 widened to
float32), so both sides hash the same tensors.

Format (version 1), for a reimplementation: for each kept key in ``sorted(keys)``::

    update(f"{key}|{dtype}|{shape}|".encode("utf-8")); update(raw bytes)

with ``dtype`` the numpy-style name (``float32``, ``int64``, ``bool``, ``bfloat16``) and ``shape`` the
dimensions joined by ``,`` (empty for a scalar).
"""

import hashlib
from typing import Iterable, Mapping, Optional

DIGEST_VERSION = 1
IGNORED_KEYS = ("_class_weight", "loss.weight")


def is_ignored_key(key: str, ignored: Iterable[str] = IGNORED_KEYS) -> bool:
    """True for ``_class_weight``/``loss.weight`` at any module depth (e.g. ``model._class_weight``)."""
    return any(key == name or key.endswith("." + name) for name in ignored)


def _shape_text(shape) -> str:
    return ",".join(str(int(d)) for d in shape)


def _torch_view(value):
    """(dtype name, shape, flat uint8 numpy view) of a torch tensor, without copying a CPU contiguous tensor."""
    import torch

    tensor = value.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    dtype_name = str(tensor.dtype).replace("torch.", "")
    shape = tuple(tensor.shape)
    flat = tensor.contiguous().reshape(-1)
    if tensor.dtype == torch.bfloat16:
        flat = flat.view(torch.int16)  # numpy has no bfloat16; same bytes reinterpreted
    try:
        array = flat.numpy()
    except TypeError as exc:
        raise TypeError(f"state_digest: unsupported tensor dtype {tensor.dtype}") from exc
    return dtype_name, shape, array.view("uint8")


def _numpy_view(value):
    """(dtype name, shape, flat uint8 view) of anything numpy can hold as a plain numeric array."""
    import numpy as np

    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"state_digest: cannot digest a value of dtype {array.dtype} (expected numbers)")
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and np.little_endian is False):
        array = array.astype(array.dtype.newbyteorder("<"))
    shape = tuple(array.shape)
    array = np.ascontiguousarray(array).reshape(-1)
    return array.dtype.name, shape, array.view("uint8")


def _canonical(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return _torch_view(value)
    except ImportError:
        pass
    return _numpy_view(value)


def state_digest(state: Mapping, ignored: Iterable[str] = IGNORED_KEYS) -> str:
    """Hex SHA-256 of ``state`` (a state dict of tensors/arrays/numbers) in the format described above."""
    if not isinstance(state, Mapping):
        raise TypeError(f"state_digest expects a mapping of parameter names to tensors, got {type(state).__name__}")
    ignored = tuple(ignored)
    digest = hashlib.sha256()
    for key in sorted(state):
        if not isinstance(key, str):
            raise TypeError(f"state_digest: parameter names must be strings, got {key!r}")
        if is_ignored_key(key, ignored):
            continue
        dtype_name, shape, raw = _canonical(state[key])
        digest.update(f"{key}|{dtype_name}|{_shape_text(shape)}|".encode("utf-8"))
        digest.update(raw)
    return digest.hexdigest()


def checkpoint_state_dict(path: str, weights_only: bool = True) -> dict:
    """The state dict NVFlare's PTFileModelPersistor would send as the global for checkpoint ``path``.

    Same rules as PTModelPersistenceFormatManager: a dict with a ``model`` entry contributes that entry,
    any other dict is taken as the weights themselves; bfloat16 tensors are widened to float32, as the
    persistor does before converting to numpy. ``weights_only`` matches the persistor's default.
    """
    import torch

    data = torch.load(path, map_location="cpu", weights_only=weights_only)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a checkpoint dict, got {type(data).__name__}")
    weights = data["model"] if "model" in data else data
    result = {}
    for key, value in weights.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        result[key] = value
    return result


def checkpoint_state_digest(path: str, weights_only: bool = True, ignored: Iterable[str] = IGNORED_KEYS) -> str:
    """state_digest of the global a persistor seeded from checkpoint ``path`` sends in round 0."""
    return state_digest(checkpoint_state_dict(path, weights_only=weights_only), ignored)


def normalise_hex_digest(value: Optional[str], what: str = "digest") -> Optional[str]:
    """Lower-case a 64-character hex SHA-256; None/empty stays None; anything else raises ValueError."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{what} must be a 64-character hex SHA-256, got {value!r}")
    return text


if __name__ == "__main__":  # pragma: no cover - convenience for operators: python3 state_digest.py ckpt.pt
    import sys

    for checkpoint in sys.argv[1:]:
        print(f"{checkpoint_state_digest(checkpoint)}  {checkpoint}")
