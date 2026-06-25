#!/usr/bin/env python3
"""Patch an NVFlare job copy with an explicit warm-start mode."""

import argparse
import re
from pathlib import Path

CONFIG_RELATIVE_PATH = Path("app/config/config_fed_client.conf")
PERSISTOR_PATH = 'path = "warm_continue.WarmStartablePTFileModelPersistor"'

MODE_AUTO = "auto"
MODE_FRESH = "fresh"
MODE_REQUIRE = "require"
MODE_ALIASES = {
    MODE_AUTO: MODE_AUTO,
    MODE_FRESH: MODE_FRESH,
    MODE_REQUIRE: MODE_REQUIRE,
    "continue": MODE_REQUIRE,
}


def normalize_warm_start_mode(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    if normalized not in MODE_ALIASES:
        expected = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(f"Invalid warm-start mode '{mode}'. Expected one of: {expected}")
    return MODE_ALIASES[normalized]


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""


def patch_config_text(config_text: str, mode: str) -> str:
    internal_mode = normalize_warm_start_mode(mode)
    lines = config_text.splitlines(keepends=True)

    try:
        persistor_index = next(i for i, line in enumerate(lines) if PERSISTOR_PATH in line)
    except StopIteration as exc:
        raise ValueError(f"Config does not reference {PERSISTOR_PATH}") from exc

    source_re = re.compile(r"^(\s*)source_ckpt_file_full_name\s*=")
    mode_re = re.compile(r"^(\s*)warm_start_mode\s*=")

    source_index = None
    for i in range(persistor_index + 1, len(lines)):
        if source_re.match(lines[i]):
            source_index = i
            break

    if source_index is None:
        raise ValueError("Config does not contain source_ckpt_file_full_name after the warm-start persistor")

    replacement_index = None
    for i in range(persistor_index + 1, source_index + 1):
        if mode_re.match(lines[i]):
            replacement_index = i
            break

    if replacement_index is not None:
        indent = mode_re.match(lines[replacement_index]).group(1)
        lines[replacement_index] = f'{indent}warm_start_mode = "{internal_mode}"{_line_ending(lines[replacement_index])}'
    else:
        indent = source_re.match(lines[source_index]).group(1)
        eol = _line_ending(lines[source_index]) or "\n"
        lines.insert(source_index, f'{indent}warm_start_mode = "{internal_mode}"{eol}')

    return "".join(lines)


def patch_job_dir(job_dir: Path, mode: str) -> Path:
    config_path = job_dir / CONFIG_RELATIVE_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing job config: {config_path}")

    config_text = config_path.read_text()
    config_path.write_text(patch_config_text(config_text, mode))
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path, help="Path to the copied NVFlare job directory")
    parser.add_argument(
        "--mode",
        required=True,
        help="Warm-start mode. Use fresh, continue, auto, or internal require.",
    )
    args = parser.parse_args()

    internal_mode = normalize_warm_start_mode(args.mode)
    config_path = patch_job_dir(args.job_dir, internal_mode)
    print(f'Patched {config_path}: warm_start_mode = "{internal_mode}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
