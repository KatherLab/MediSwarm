#!/usr/bin/env python3
"""Raise STAMP's *readable* feature-extraction version ceiling.

Why
---
STAMP refuses to read a feature H5 whose ``stamp_version`` attribute is newer
than the installed STAMP::

    RuntimeError: features were extracted with a newer version of stamp,
    please update your stamp to at least version 2.5.0.

DECADE sites extract features with STAMP 2.5.0, but STAMP 2.5.0 hard-requires
Python 3.13 (``requires-python = ">=3.13,<3.14"``) while our PyTorch/NVFlare
stack runs Python 3.11 (NVFlare declares support only through 3.12). So the
training image stays on STAMP 2.4.0.

Reading 2.5.0-extracted features on 2.4.0 is safe. Between the 2.4.0 and 2.5.0
tags of KatherLab/STAMP:

* the feature H5 layout is unchanged — datasets ``feats`` / ``coords`` and attrs
  ``stamp_version, extractor, unit, tile_size_um, tile_size_px, code_hash,
  feat_type`` are written identically;
* ``preprocessing/extractor/uni.py`` is byte-identical, so UNI features are
  numerically the same;
* the only ``preprocessing/tiling.py`` changes are bug fixes (eager PIL decode,
  a None-guard on MPP metadata) that do not alter tiles, coords or features.

The upstream check is therefore a conservative forward-compat guard, not a
format break. This patch raises only the *readable* ceiling to
``MAX_READABLE_EXTRACTION_VERSION``; features newer than that still raise.

Safety
------
Fails loudly (non-zero exit) if the expected upstream code is not found, so a
future STAMP version bump cannot silently skip this patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Highest extraction version we have verified this loader can read.
MAX_READABLE_EXTRACTION_VERSION = "2.5.0"

# The exact upstream comparison (stamp/modeling/data.py), which must appear once.
TARGET = ") > Version(stamp.__version__):"
REPLACEMENT = (
    f') > max(Version(stamp.__version__), Version("{MAX_READABLE_EXTRACTION_VERSION}")):'
    f"  # MediSwarm: allow reading features from STAMP <= {MAX_READABLE_EXTRACTION_VERSION}"
)


def _data_module_path() -> Path:
    import stamp.modeling.data as data_mod  # noqa: PLC0415 — needs installed stamp

    return Path(data_mod.__file__)


def main() -> int:
    path = _data_module_path()
    source = path.read_text(encoding="utf-8")

    if REPLACEMENT in source:
        print(f"[patch_stamp] already patched: {path}")
        return 0

    count = source.count(TARGET)
    if count != 1:
        print(
            f"[patch_stamp] ERROR: expected exactly 1 occurrence of\n"
            f"  {TARGET!r}\n"
            f"in {path}, found {count}. Upstream STAMP changed — review this patch "
            f"before bumping the STAMP pin.",
            file=sys.stderr,
        )
        return 1

    path.write_text(source.replace(TARGET, REPLACEMENT), encoding="utf-8")

    # Verify the patched module still imports and the ceiling behaves as intended.
    verify = path.read_text(encoding="utf-8")
    if REPLACEMENT not in verify:
        print("[patch_stamp] ERROR: replacement did not persist", file=sys.stderr)
        return 1

    import importlib

    importlib.invalidate_caches()
    importlib.import_module("stamp.modeling.data")

    print(
        f"[patch_stamp] patched {path}: readable extraction version ceiling "
        f"raised to {MAX_READABLE_EXTRACTION_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
