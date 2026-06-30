#!/usr/bin/env python3
"""Patch an NVFlare job copy for admin-controlled warm-start tests."""

import argparse
import re
from pathlib import Path

CONFIG_RELATIVE_PATH = Path("app/config/config_fed_client.conf")
SERVER_CONFIG_RELATIVE_PATH = Path("app/config/config_fed_server.conf")
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


def _positive_int_or_none(value, name: str):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def patch_numeric_assignment(config_text: str, key: str, value: int) -> str:
    value = _positive_int_or_none(value, key)
    pattern = re.compile(
        rf"^(\s*{re.escape(key)}\s*=\s*)\d+([^\S\r\n]*(?:#.*)?(?:\r?\n|$))",
        re.MULTILINE,
    )

    def replace(match):
        return f"{match.group(1)}{value}{match.group(2)}"

    patched, count = pattern.subn(replace, config_text, count=1)
    if count != 1:
        raise ValueError(f"Config does not contain numeric assignment for {key}")
    return patched


def upsert_numeric_assignment(config_text: str, key: str, value: int, after_key: str) -> str:
    try:
        return patch_numeric_assignment(config_text, key, value)
    except ValueError:
        pass

    value = _positive_int_or_none(value, key)
    pattern = re.compile(
        rf"^(\s*){re.escape(after_key)}\s*=\s*\d+[^\S\r\n]*(?:#.*)?(?:\r?\n|$)",
        re.MULTILINE,
    )
    match = pattern.search(config_text)
    if not match:
        raise ValueError(f"Config does not contain numeric assignment for {after_key}")

    line_end = match.group(0)
    eol = "\r\n" if line_end.endswith("\r\n") else "\n"
    insert = f"{match.group(1)}{key} = {value}{eol}"
    return config_text[: match.end()] + insert + config_text[match.end() :]


def patch_config_text(config_text: str, mode: str, min_responses_required=None) -> str:
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

    patched = "".join(lines)
    min_responses_required = _positive_int_or_none(min_responses_required, "min_responses_required")
    if min_responses_required is not None:
        patched = patch_numeric_assignment(patched, "min_responses_required", min_responses_required)

    return patched


def patch_server_config_text(config_text: str, num_rounds=None, min_clients=None, configure_min_clients=None) -> str:
    patched = config_text
    num_rounds = _positive_int_or_none(num_rounds, "num_rounds")
    min_clients = _positive_int_or_none(min_clients, "min_clients")
    configure_min_clients = _positive_int_or_none(configure_min_clients, "configure_min_clients")
    if num_rounds is not None:
        patched = patch_numeric_assignment(patched, "num_rounds", num_rounds)
    if min_clients is not None:
        patched = patch_numeric_assignment(patched, "min_clients", min_clients)
    if configure_min_clients is not None:
        patched = upsert_numeric_assignment(
            patched,
            "configure_min_clients",
            configure_min_clients,
            after_key="min_clients",
        )
    return patched


def patch_job_dir(
    job_dir: Path,
    mode: str,
    num_rounds=None,
    min_clients=None,
    configure_min_clients=None,
    min_responses_required=None,
) -> Path:
    config_path = job_dir / CONFIG_RELATIVE_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing job config: {config_path}")

    config_text = config_path.read_text()
    config_path.write_text(patch_config_text(config_text, mode, min_responses_required=min_responses_required))

    if num_rounds is not None or min_clients is not None or configure_min_clients is not None:
        server_config_path = job_dir / SERVER_CONFIG_RELATIVE_PATH
        if not server_config_path.is_file():
            raise FileNotFoundError(f"Missing job server config: {server_config_path}")
        server_config_text = server_config_path.read_text()
        server_config_path.write_text(
            patch_server_config_text(
                server_config_text,
                num_rounds=num_rounds,
                min_clients=min_clients,
                configure_min_clients=configure_min_clients,
            )
        )

    return config_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path, help="Path to the copied NVFlare job directory")
    parser.add_argument(
        "--mode",
        required=True,
        help="Warm-start mode. Use fresh, continue, auto, or internal require.",
    )
    parser.add_argument("--num-rounds", type=int, help="Patch app/config/config_fed_server.conf num_rounds")
    parser.add_argument("--min-clients", type=int, help="Patch app/config/config_fed_server.conf min_clients")
    parser.add_argument(
        "--configure-min-clients",
        type=int,
        help="Patch app/config/config_fed_server.conf configure_min_clients",
    )
    parser.add_argument(
        "--min-responses",
        type=int,
        help="Patch app/config/config_fed_client.conf min_responses_required",
    )
    args = parser.parse_args()

    internal_mode = normalize_warm_start_mode(args.mode)
    config_path = patch_job_dir(
        args.job_dir,
        internal_mode,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
        configure_min_clients=args.configure_min_clients,
        min_responses_required=args.min_responses,
    )
    print(f'Patched {config_path}: warm_start_mode = "{internal_mode}"')
    if args.num_rounds is not None:
        print(f"Patched {args.job_dir / SERVER_CONFIG_RELATIVE_PATH}: num_rounds = {args.num_rounds}")
    if args.min_clients is not None:
        print(f"Patched {args.job_dir / SERVER_CONFIG_RELATIVE_PATH}: min_clients = {args.min_clients}")
    if args.configure_min_clients is not None:
        print(
            f"Patched {args.job_dir / SERVER_CONFIG_RELATIVE_PATH}: "
            f"configure_min_clients = {args.configure_min_clients}"
        )
    if args.min_responses is not None:
        print(f"Patched {config_path}: min_responses_required = {args.min_responses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
