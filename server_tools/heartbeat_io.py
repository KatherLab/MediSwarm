"""Read the heartbeats that live_sync uploads (standard library only).

One reader for every consumer: the live monitor (``server_tools/app.py``) and the
coordinator's fleet check (``scripts/admin/fleet_status.py``). Deployed next to ``app.py``
(``/srv/mediswarm/heartbeat_io.py``, see server_tools/README.md).

A heartbeat is ``<live root>/<SITE>/<swarm|local>/<job id, run name or _active>/heartbeat.json``
(and ``heartbeat_final.json`` once live_sync ended a run), written by
``kit_live_sync/build_heartbeat.sh`` at the site.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

HEARTBEAT_FILES = ("heartbeat_final.json", "heartbeat.json")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# A doubled trailing quote after a non-empty value: "1.6.0"" -> "1.6.0". Older
# build_heartbeat.sh versions extracted version strings with their closing quote. An empty
# string "" is valid JSON and must not match, hence the word character in front.
_DOUBLED_QUOTE_RE = re.compile(r'(\w)""(\s*[,}\n])')


def parse_heartbeat_text(raw: str) -> dict[str, Any]:
    """Parse a heartbeat, salvaging the two known defects of older kits.

    Older kits wrote ANSI colour codes from the run name and doubled closing quotes on
    version strings into the file, both invalid JSON. Raises ValueError when the text is
    still not a JSON object after the repair.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        fixed = _ANSI_RE.sub("", raw)
        fixed = _DOUBLED_QUOTE_RE.sub(r'\1"\2', fixed)
        data = json.loads(fixed)
    if not isinstance(data, dict):
        raise ValueError(f"a heartbeat is a JSON object, not {type(data).__name__}")
    return data


def read_heartbeat_file(path: Path) -> dict[str, Any]:
    """The heartbeat in ``path``, or ``{"_parse_error": ..., "_parse_error_file": ...}``."""
    try:
        return parse_heartbeat_text(Path(path).read_text())
    except Exception as exc:
        return {
            "_parse_error": f"{type(exc).__name__}: {exc}",
            "_parse_error_file": str(path),
        }


def read_run_heartbeat(run_dir: Path) -> dict[str, Any]:
    """A run folder's heartbeat: heartbeat_final.json if present, else heartbeat.json, else {}."""
    for name in HEARTBEAT_FILES:
        p = Path(run_dir) / name
        if p.exists():
            return read_heartbeat_file(p)
    return {}


def clean_value(value: Any) -> str:
    """A heartbeat string field without surrounding blanks and stray quotes (``'1.8.1"'`` -> ``1.8.1``)."""
    if value is None:
        return ""
    return str(value).strip().strip("\"'").strip()


def parse_timestamp(value: Any) -> Optional[datetime]:
    """The heartbeat's ``timestamp`` (``2026-10-15T19:12:59Z``, UTC) as an aware datetime, or None."""
    text = clean_value(value)
    if not text:
        return None
    try:
        ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
