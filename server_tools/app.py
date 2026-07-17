"""MediSwarm Live Monitor — enhanced dashboard for training runs.

Serves a styled web UI that displays live training status, metrics charts,
and artifact links for all sites synced by live_sync to /srv/mediswarm/live/.

Features:
- Filters by site, mode, status, job_id
- Default sort by timestamp (newest first)
- Status inference (stale running → stale, very old → presumed finished)
- Server-side file paths with download links
- Job grouping for swarm runs
- Kit version column
- TensorBoard metric parsing (via tbparse)
- Enriched detail page with full artifact inventory
"""

from pathlib import Path
import csv
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from html import escape as html_escape

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse

# Optional: TensorBoard event parsing via tbparse
try:
    from tbparse import SummaryReader

    HAS_TBPARSE = True
except ImportError:
    HAS_TBPARSE = False

BASE = Path(os.environ.get("MEDISWARM_LIVE_BASE", "/srv/mediswarm/live"))
ROSTER_PATH = Path(
    os.environ.get("MEDISWARM_MONITOR_SITES", str(BASE / "monitor_sites.json"))
)
STALE_AFTER_SECONDS = int(os.environ.get("MEDISWARM_STALE_AFTER_SECONDS", "300"))
ROWS_CACHE_TTL_SECONDS = float(os.environ.get("MEDISWARM_MONITOR_CACHE_TTL", "5"))
ERROR_TAIL_BYTES = int(os.environ.get("MEDISWARM_ERROR_TAIL_BYTES", "120000"))
MAX_ERROR_SCAN_FILES = int(os.environ.get("MEDISWARM_MAX_ERROR_SCAN_FILES", "40"))

# Path to the FL server's nohup.out — the authoritative source of swarm round
# progress (start/finished_learn per round, prunes, aborts). The CCWF workflow
# runs the rounds peer-to-peer on the clients, so the server log is the one place
# that sees the whole run; without this the monitor can't tell "training" from
# "stalled". Empty -> the swarm-progress panel is simply omitted. (#398)
SERVER_LOG = os.environ.get("MEDISWARM_SERVER_LOG", "")
# A run with no new server-log line for this long (and not terminal) is "stalled".
SWARM_STALL_SECONDS = int(os.environ.get("MEDISWARM_SWARM_STALL_SECONDS", "600"))

app = FastAPI(title="MediSwarm Live Monitor")

_ROWS_CACHE: dict[str, Any] = {"expires_at": 0.0, "value": None}

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._\-]*$")


def _safe_segment(value: str) -> str:
    if not value or not _SAFE_SEGMENT_RE.match(value) or ".." in value:
        raise HTTPException(status_code=400, detail="Invalid path segment")
    return value


def _resolve_run_dir(site: str, mode: str, run_id: str) -> Path:
    site = _safe_segment(site)
    mode = _safe_segment(mode)
    run_id = _safe_segment(run_id)
    run_dir = (BASE / site / mode / run_id).resolve()
    base_resolved = BASE.resolve()
    try:
        common = os.path.commonpath([str(base_resolved), str(run_dir)])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    if common != str(base_resolved):
        raise HTTPException(status_code=400, detail="Invalid path")
    return run_dir


# ---------------------------------------------------------------------------
# CSS & HTML helpers
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #f5f6fa;
  --card: #ffffff;
  --header-bg: #1a1a2e;
  --header-fg: #eaeaea;
  --accent: #0f3460;
  --green: #27ae60;
  --blue: #2980b9;
  --orange: #e67e22;
  --gray: #95a5a6;
  --red: #c0392b;
  --purple: #8e44ad;
  --border: #dfe6e9;
  --text: #2d3436;
  --text-light: #636e72;
  --mono: 'SF Mono', 'Fira Code', 'Consolas', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--text); }
header { background: var(--header-bg); color: var(--header-fg); padding: 1rem 2rem;
         display: flex; justify-content: space-between; align-items: center; }
header h1 { font-size: 1.4rem; font-weight: 600; }
header .meta { font-size: 0.82rem; color: var(--gray); }
header .meta a { color: var(--gray); text-decoration: underline; margin-left: 1rem; }
main { max-width: 1600px; margin: 1.5rem auto; padding: 0 1rem; }

/* Filter bar */
.filter-bar { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 1.2rem;
              align-items: center; }
.filter-bar label { font-size: 0.82rem; font-weight: 600; color: var(--text-light); }
.filter-bar select, .filter-bar input { font-size: 0.82rem; padding: 4px 10px;
  border: 1px solid var(--border); border-radius: 6px; background: var(--card); }
.filter-bar .filter-group { display: flex; align-items: center; gap: 0.3rem; }
.filter-bar .btn-small { display: inline-block; padding: 4px 12px; border-radius: 6px;
  background: var(--accent); color: #fff; font-size: 0.8rem; text-decoration: none;
  cursor: pointer; border: none; }
.filter-bar .btn-small:hover { background: #16213e; }

table { width: 100%; border-collapse: collapse; background: var(--card);
        border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
th { background: var(--accent); color: #fff; text-align: left;
     padding: 0.7rem 0.9rem; font-size: 0.78rem; text-transform: uppercase;
     letter-spacing: 0.04em; cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { background: #16213e; }
th .sort-arrow { font-size: 0.7rem; margin-left: 0.3rem; }
td { padding: 0.55rem 0.9rem; border-bottom: 1px solid var(--border);
     font-size: 0.85rem; vertical-align: top; }
tr:nth-child(even) td { background: #f9fafb; }
tr:hover td { background: #eef2f7; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 0.75rem; font-weight: 600; color: #fff; }
.badge-running { background: var(--green); }
.badge-waiting { background: var(--purple); }
.badge-finished { background: var(--blue); }
.badge-unknown { background: var(--gray); }
.badge-error { background: var(--red); }
.badge-stale { background: var(--orange); }
.badge-missing { background: var(--red); }
.status-reason { font-size: 0.72rem; color: var(--red); font-style: italic;
                 display: inline-block; max-width: 300px; vertical-align: middle; }
.artifact { font-size: 0.78rem; color: var(--text-light); }
.artifact .yes { color: var(--green); font-weight: 600; }
.artifact .no { color: var(--gray); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.links a { margin-right: 0.5rem; font-size: 0.8rem; }
.run-id { font-family: var(--mono); font-size: 0.75rem; word-break: break-all; }
.run-name { font-weight: 500; }
.age-stale { color: var(--orange); }
.age-dead { color: var(--red); }
.empty { text-align: center; padding: 3rem; color: var(--gray); }
.version, .ip-address { font-family: var(--mono); font-size: 0.75rem; color: var(--text-light); }
.job-id { font-family: var(--mono); font-size: 0.72rem; color: var(--purple); }
.quick-links { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-left: auto; }
.quick-links a { display: inline-block; padding: 3px 9px; border-radius: 999px;
  background: #eef2f7; color: var(--accent); font-size: 0.76rem; text-decoration: none; }
.quick-links a:hover { background: #dfe6e9; text-decoration: none; }
.error-summary { color: var(--red); font-size: 0.78rem; margin-top: 0.25rem; }
.error-source { font-family: var(--mono); font-size: 0.75rem; color: var(--text-light); }
.error-excerpt { white-space: pre-wrap; background: #2d1115 !important; color: #ffd6d6 !important; }

/* Job group header */
.job-group-row td { background: #e8e4f0 !important; font-weight: 600;
  font-size: 0.82rem; color: var(--purple); padding: 0.5rem 0.9rem; }

/* Summary stats */
.stats-bar { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.stat-card { background: var(--card); border-radius: 8px; padding: 0.6rem 1.2rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06); display: flex; align-items: center; gap: 0.5rem; }
.stat-card .stat-num { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
.stat-card .stat-label { font-size: 0.78rem; color: var(--text-light); }
/* Swarm round-progress panel (#398) */
.swarm-panel { background: var(--card); border-radius: 8px; padding: 0.7rem 1rem;
  margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  border-left: 4px solid var(--text-light); }
.swarm-panel.swarm-training { border-left-color: var(--green); }
.swarm-panel.swarm-stalled  { border-left-color: var(--orange, #e67e22); }
.swarm-panel.swarm-aborted, .swarm-panel.swarm-fatal { border-left-color: var(--red, #e74c3c); }
.swarm-panel.swarm-finished { border-left-color: var(--accent); }
.swarm-head { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.swarm-title { font-weight: 700; }
.swarm-detail { color: var(--text-light); font-size: 0.85rem; }
.swarm-chips { margin-top: 0.5rem; display: flex; gap: 0.35rem; flex-wrap: wrap; }
.swarm-chip { font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;
  background: #dfe4ea; color: #2f3542; font-family: monospace; }
.swarm-chip.chip-done { background: var(--green); color: #fff; }
.swarm-chip.chip-run  { background: #f1c40f; color: #2f3542; }
.swarm-chip.chip-idle { background: #dfe4ea; color: #2f3542; }
.swarm-prunes { margin-top: 0.4rem; font-size: 0.8rem; color: var(--red, #e74c3c); }
.swarm-stall { margin-top: 0.4rem; font-size: 0.85rem; font-weight: 600;
  color: var(--orange, #e67e22); }

/* Detail page */
.detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.2rem; }
@media (max-width: 900px) { .detail-grid { grid-template-columns: 1fr; } }
.card { background: var(--card); border-radius: 8px; padding: 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.card h2 { font-size: 1rem; margin-bottom: 0.8rem; color: var(--accent); }
.card pre { background: #1a1a2e; color: #dfe6e9; padding: 1rem; border-radius: 6px;
            overflow-x: auto; font-size: 0.78rem; line-height: 1.5; max-height: 400px;
            overflow-y: auto; }
.card table { box-shadow: none; }
.card table th { background: var(--accent); cursor: default; }
.kv-table td:first-child { font-weight: 600; white-space: nowrap; width: 180px; }
.kv-table td:last-child { font-family: var(--mono); font-size: 0.82rem; word-break: break-all; }
.btn { display: inline-block; padding: 6px 14px; border-radius: 6px;
       background: var(--accent); color: #fff; font-size: 0.82rem;
       text-decoration: none; margin-right: 0.5rem; margin-bottom: 0.3rem; }
.btn:hover { background: #16213e; text-decoration: none; }
.btn-download { background: var(--green); }
.btn-download:hover { background: #1e8449; }
.chart-container { position: relative; width: 100%; height: 350px; }
.breadcrumb { font-size: 0.85rem; margin-bottom: 1rem; color: var(--text-light); }
.breadcrumb a { color: var(--accent); }

/* File list in detail */
.file-list { list-style: none; }
.file-list li { padding: 0.3rem 0; border-bottom: 1px solid #f0f0f0; font-size: 0.82rem;
  display: flex; align-items: center; gap: 0.5rem; }
.file-list li:last-child { border-bottom: none; }
.file-icon { font-size: 0.9rem; }
.file-path { font-family: var(--mono); font-size: 0.78rem; color: var(--text-light);
  word-break: break-all; }
.file-size { font-size: 0.75rem; color: var(--gray); white-space: nowrap; }
"""


def _status_badge(status: str, reason: str = "") -> str:
    cls = "badge-unknown"
    if status == "running":
        cls = "badge-running"
    elif status == "waiting":
        cls = "badge-waiting"
    elif status == "finished":
        cls = "badge-finished"
    elif status in ("error", "failed"):
        cls = "badge-error"
    elif status == "stale":
        cls = "badge-stale"
    elif status == "missing":
        cls = "badge-missing"
    title = f' title="{html_escape(reason)}"' if reason else ""
    badge = f'<span class="badge {cls}"{title}>{html_escape(status)}</span>'
    # Show reason text next to badge for error/stale states
    if reason and status in ("error", "failed", "stale", "missing"):
        badge += f' <span class="status-reason">{html_escape(reason)}</span>'
    return badge


def _age_class(age_str: str) -> str:
    try:
        secs = int(age_str.rstrip("s"))
    except (ValueError, AttributeError):
        return ""
    if secs > 600:
        return "age-dead"
    if secs > 120:
        return "age-stale"
    return ""


def _html_page(title: str, body: str, *, refresh: int = 0, extra_head: str = "") -> str:
    refresh_tag = (
        f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    )
    safe_title = html_escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_tag}
  <title>{safe_title}</title>
  <style>{CSS}</style>
  {extra_head}
</head>
<body>
{body}
</body>
</html>"""


def compute_version_skew(rows):
    """Sites whose running image disagrees with the rest of the swarm (or the roster).

    All clients in a federated run must execute the same code. Nothing checked this
    before: `expected_version` was only ever used as a display fallback, never
    compared, so a site running a stale image was invisible.

    Returns {site_name: reason} for every site that is out of line.
    """
    live = [r for r in rows if r.get("image_id") or r.get("kit_version")]
    if len(live) < 2:
        return {}

    def identity(row):
        # image_id is the definitive answer (a tag can move); fall back to what we have.
        return row.get("image_id") or row.get("image_ref") or row.get("kit_version") or ""

    counts = {}
    for row in live:
        key = identity(row)
        if key:
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return {}
    majority = max(counts, key=lambda k: counts[k])

    skew = {}
    for row in live:
        key = identity(row)
        if key and key != majority:
            skew[row["site"]] = f"running {row.get('image_ref') or key}, swarm majority is on a different image"
            continue
        expected = row.get("expected_version") or ""
        seen = row.get("kit_version") or ""
        if expected and seen and expected != seen:
            skew[row["site"]] = f"kit {seen}, expected {expected}"
    return skew


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def read_text(p: Path, limit: int = 50_000) -> str:
    if not p.exists():
        return ""
    return p.read_text(errors="replace")[-limit:]


def parse_age(ts: str) -> str:
    if not ts:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        secs = int((datetime.now(timezone.utc) - dt).total_seconds())
        if secs < -60:
            skew = abs(secs)
            if skew < 3600:
                return f"clock +{skew // 60}m"
            return f"clock +{skew // 3600}h {(skew % 3600) // 60}m"
        if secs < 0:
            return "0s"
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        if secs < 86400:
            return f"{secs // 3600}h {(secs % 3600) // 60}m"
        return f"{secs // 86400}d {(secs % 86400) // 3600}h"
    except Exception:
        return "unknown"


def _age_seconds(ts: str) -> int:
    """Return age in seconds from an ISO timestamp, or -1 if unparseable."""
    if not ts:
        return -1
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return int((datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return -1


def _read_heartbeat(run_dir: Path) -> dict[str, Any]:
    for name in ["heartbeat_final.json", "heartbeat.json"]:
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError as exc:
                # Try to salvage malformed JSON (e.g. doubled quotes from
                # build_heartbeat.sh extracting version strings with trailing ")
                try:
                    raw = p.read_text()
                    # Strip ANSI escape codes first
                    fixed = re.sub(r'\x1b\[[0-9;]*m', '', raw)
                    # Fix doubled trailing quote on non-empty values:
                    # "some_value"" -> "some_value"
                    # Must NOT match empty strings "" which are valid JSON.
                    # Pattern: a word char followed by ""[,}\n] means extra quote
                    fixed = re.sub(r'(\w)""(\s*[,}\n])', r'\1"\2', fixed)
                    return json.loads(fixed)
                except Exception:
                    return {
                        "_parse_error": f"{type(exc).__name__}: {exc}",
                        "_parse_error_file": str(p),
                    }
            except Exception as exc:
                return {
                    "_parse_error": f"{type(exc).__name__}: {exc}",
                    "_parse_error_file": str(p),
                }
    return {}


def _read_expected_sites() -> dict[str, dict[str, Any]]:
    """Read the expected collaborator roster.

    The file may be either a list of site objects or {"sites": [...]}.
    Disabled sites are ignored. Unknown/partial entries are skipped.
    """
    if not ROSTER_PATH.exists():
        return {}
    try:
        raw = json.loads(ROSTER_PATH.read_text())
    except Exception:
        return {}
    entries = raw.get("sites", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return {}

    roster: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        if item.get("enabled", True) is False:
            continue
        site = str(item.get("site_name") or item.get("site") or "").strip()
        if not site:
            continue
        roster[site] = {
            "site_name": site,
            "display_name": item.get("display_name", ""),
            "ip_address": item.get("ip_address", ""),
            "institution": item.get("institution", ""),
            "expected_version": item.get("expected_version", ""),
        }
    return roster


_ERROR_PATTERNS: list[dict[str, str]] = [
    {
        "category": "nvflare",
        "pattern": r"TaskCompletionStatus\.TIMEOUT|learn_task.*timed?\s*out",
        "summary": "NVFlare training task timed out",
    },
    {
        "category": "nvflare",
        "pattern": r"progress_timeout.*exceeded|no progress.*timeout",
        "summary": "NVFlare progress timeout",
    },
    {
        "category": "nvflare",
        "pattern": r"peer_read_timeout.*exceeded|peer.*read.*timed?\s*out|CommunicateError|CommError|ConnectionRefused|connection.*refused",
        "summary": "NVFlare communication or peer transfer failure",
    },
    {
        "category": "nvflare",
        "pattern": r"FATAL_SYSTEM_ERROR|EXECUTION_EXCEPTION.*abort|ABORT_RUN|asked to abort.*abort_signal|Server runner failed",
        "summary": "NVFlare fatal error or aborted run",
    },
    {
        "category": "deep_learning",
        "pattern": r"CUDA out of memory|OutOfMemoryError|RuntimeError:.*CUDA|CUDA error|NCCL error",
        "summary": "CUDA/GPU runtime error",
    },
    {
        "category": "deep_learning",
        "pattern": r"Traceback \(most recent call last\)|RuntimeError:|ValueError:|KeyError:|FileNotFoundError:|ModuleNotFoundError:",
        "summary": "Python/deep-learning exception",
    },
    {
        "category": "deep_learning",
        "pattern": r"failed to load.*checkpoint|size mismatch|Missing key\(s\)|Unexpected key\(s\)|No such file or directory",
        "summary": "Model/data/checkpoint loading failure",
    },
    {
        "category": "live_sync",
        "pattern": r"Live-sync: cannot connect|Permission denied|rsync.*failed|ssh:.*Could not|sync\.conf not found|Host key verification failed",
        "summary": "Live-sync SSH/rsync failure",
    },
    {
        "category": "system",
        "pattern": r"Killed|oom-kill|Out of memory|No space left on device|Device or resource busy",
        "summary": "System resource failure",
    },
]


def _error_log_candidates(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for name in [
        "nohup.out",
        "local_training_console_output.txt",
        "log.txt",
        "live_sync_daemon.log",
    ]:
        p = run_dir / name
        if p.exists() and p.is_file():
            candidates.append(p)

    rd = run_dir / "run_dir"
    if rd.exists():
        for p in sorted(rd.rglob("*")):
            if len(candidates) >= MAX_ERROR_SCAN_FILES:
                break
            if p.is_file() and (
                p.name.endswith((".log", ".txt", ".out"))
                or p.name in {"stderr", "stdout"}
            ):
                candidates.append(p)
    return candidates


def _extract_error_evidence(
    run_dir: Path, hb: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Return structured fatal evidence if logs/heartbeat indicate failure."""
    hb = hb or {}
    if hb.get("_parse_error"):
        return {
            "category": "heartbeat",
            "summary": "Heartbeat JSON parse error",
            "source": hb.get("_parse_error_file", "heartbeat"),
            "line": "",
            "excerpt": hb.get("_parse_error", ""),
            "pattern": "json_parse_error",
        }

    sync_status = str(hb.get("sync_status", "")).lower()
    sync_error = str(hb.get("sync_error", "")).strip()
    if sync_status == "error" or sync_error:
        return {
            "category": "live_sync",
            "summary": sync_error or "Live-sync reported an error",
            "source": "heartbeat",
            "line": sync_error,
            "excerpt": sync_error,
            "pattern": "heartbeat_sync_error",
        }

    for p in _error_log_candidates(run_dir):
        try:
            text = p.read_text(errors="replace")[-ERROR_TAIL_BYTES:]
        except Exception:
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            for spec in _ERROR_PATTERNS:
                if re.search(spec["pattern"], line, flags=re.IGNORECASE):
                    start = max(0, idx - 4)
                    end = min(len(lines), idx + 5)
                    try:
                        source = str(p.relative_to(run_dir))
                    except Exception:
                        source = str(p)
                    return {
                        "category": spec["category"],
                        "summary": spec["summary"],
                        "source": source,
                        "line": line.strip(),
                        "excerpt": "\n".join(lines[start:end]),
                        "pattern": spec["pattern"],
                    }
    return None


def _check_console_for_errors(run_dir: Path) -> str:
    evidence = _extract_error_evidence(run_dir)
    return evidence["summary"] if evidence else ""


def _infer_status(hb: dict[str, Any], run_dir: Path) -> tuple[str, str]:
    """Infer the effective status from heartbeat + file system state.

    Returns (status, reason) where reason is a human-readable explanation
    of the status, especially useful for error/stale states.

    Rules:
    - Fatal heartbeat/log evidence always wins -> "error"
    - Old running/waiting heartbeats become "stale", never presumed finished
    - "finished" requires a final heartbeat or explicit completion evidence
    - Swarm active heartbeats without a job_id are "waiting"
    """
    evidence = _extract_error_evidence(run_dir, hb)
    if evidence:
        return "error", evidence["summary"]

    has_final = (run_dir / "heartbeat_final.json").exists()
    raw_status = str(hb.get("status", "unknown") or "unknown").lower()

    if has_final:
        try:
            final = json.loads((run_dir / "heartbeat_final.json").read_text())
            final_status = str(final.get("status", raw_status) or raw_status).lower()
            if final_status in {"error", "failed", "failure"}:
                return "error", str(final.get("status_reason", "Final heartbeat reported error"))
            if final_status in {"finished", "complete", "completed"}:
                return "finished", ""
            return final_status, ""
        except Exception:
            pass

    age = _age_seconds(hb.get("timestamp", ""))
    if raw_status in {"running", "waiting", "pending"} and age > STALE_AFTER_SECONDS:
        minutes = max(1, age // 60)
        return "stale", f"No heartbeat for {minutes}min"

    if raw_status in {"error", "failed", "failure"}:
        return "error", str(hb.get("status_reason", "Heartbeat reported error"))

    if raw_status in {"finished", "complete", "completed"}:
        return "finished", ""

    if raw_status in {"waiting", "pending"}:
        return "waiting", ""

    if raw_status == "running" and run_dir.name == "_active" and not hb.get("job_id"):
        return "waiting", ""

    # Completion evidence for older uploads that predate heartbeat_final.json.
    completion_text = read_text(run_dir / "nohup.out", limit=ERROR_TAIL_BYTES) + "\n" + read_text(run_dir / "log.txt", limit=ERROR_TAIL_BYTES)
    if re.search(r"Server runner finished|job .*finished|run .*finished|finished all rounds", completion_text, flags=re.IGNORECASE):
        return "finished", ""

    return raw_status or "unknown", ""


def _find_csv_files(run_dir: Path) -> list[str]:
    rd = run_dir / "run_dir"
    if not rd.exists():
        return []
    return sorted(
        p.name for p in rd.rglob("*_model_gt_and_classprob_*.csv") if p.is_file()
    )


def _find_tb_events(run_dir: Path) -> list[Path]:
    rd = run_dir / "run_dir"
    if not rd.exists():
        return []
    return sorted(rd.rglob("events.out.tfevents*"))


def _find_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    """Find all checkpoint files under run_dir/run_dir/."""
    rd = run_dir / "run_dir"
    results = []
    if not rd.exists():
        return results
    for p in sorted(rd.rglob("*.ckpt")):
        results.append({
            "name": p.name,
            "rel_path": str(p.relative_to(run_dir)),
            "size": p.stat().st_size if p.exists() else 0,
            "server_path": str(p),
        })
    return results


def _find_all_files(run_dir: Path) -> list[dict[str, Any]]:
    """Find all files in the run directory with metadata."""
    results = []
    if not run_dir.exists():
        return results
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            try:
                stat = p.stat()
                results.append({
                    "name": p.name,
                    "rel_path": str(p.relative_to(run_dir)),
                    "size": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "server_path": str(p),
                })
            except Exception:
                pass
    return results


def _collect_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    roster = _read_expected_sites()
    if not BASE.exists():
        return [_missing_row(site, data) for site, data in roster.items()]

    for site_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
        for mode_dir in sorted(p for p in site_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
                hb = _read_heartbeat(run_dir)
                ts = hb.get("timestamp", "")
                age = parse_age(ts)
                status, status_reason = _infer_status(hb, run_dir)
                error_evidence = _extract_error_evidence(run_dir, hb)
                csv_files = _find_csv_files(run_dir)
                tb_events = _find_tb_events(run_dir)
                checkpoints = _find_checkpoints(run_dir)
                roster_item = roster.get(site_dir.name, {})

                # Count total files and size
                total_files = 0
                total_size = 0
                rd = run_dir / "run_dir"
                if rd.exists():
                    for f in rd.rglob("*"):
                        if f.is_file():
                            total_files += 1
                            try:
                                total_size += f.stat().st_size
                            except Exception:
                                pass

                out.append(
                    {
                        "site": site_dir.name,
                        "mode": mode_dir.name,
                        "run_id": run_dir.name,
                        "run_name": hb.get("run_name", ""),
                        "job_id": hb.get("job_id", ""),
                        "status": status,
                        "status_reason": status_reason,
                        "raw_status": hb.get("status", "unknown"),
                        "timestamp": ts,
                        "age": age,
                        "age_seconds": _age_seconds(ts),
                        "kit_version": hb.get("kit_version", "")
                        or roster_item.get("expected_version", ""),
                        # What the site is ACTUALLY running. kit_version is only what
                        # the kit claims (scraped from docker.sh); once the image can be
                        # overridden per site, the two can differ -- and a silent version
                        # skew across sites corrupts a federated run.
                        "image_ref": hb.get("image_ref", ""),
                        "image_id": hb.get("image_id", ""),
                        "expected_version": roster_item.get("expected_version", ""),
                        "hostname": hb.get("hostname", ""),
                        "ip_address": hb.get("ip_address", "")
                        or roster_item.get("ip_address", ""),
                        "display_name": roster_item.get("display_name", ""),
                        "institution": roster_item.get("institution", ""),
                        "is_expected": site_dir.name in roster,
                        "is_synthetic": False,
                        "is_active_shadow": run_dir.name == "_active",
                        "error_evidence": error_evidence or {},
                        "has_console": (run_dir / "nohup.out").exists()
                        or (run_dir / "local_training_console_output.txt").exists(),
                        "has_log": (run_dir / "log.txt").exists(),
                        "has_sync_log": (run_dir / "live_sync_daemon.log").exists(),
                        "has_global_model": (run_dir / "FL_global_model.pt").exists(),
                        "has_best_model": (run_dir / "best_FL_global_model.pt").exists(),
                        "checkpoints": len(checkpoints),
                        "csv_files": csv_files,
                        "tb_events": len(tb_events),
                        "total_files": total_files,
                        "total_size": total_size,
                        "server_path": str(run_dir),
                    }
                )

    seen_sites = {x["site"] for x in out}
    for site, data in roster.items():
        if site not in seen_sites:
            out.append(_missing_row(site, data))

    # Sort by timestamp descending (newest first)
    out.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return out


def _missing_row(site: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "site": site,
        "mode": "swarm",
        "run_id": "_missing",
        "run_name": "",
        "job_id": "",
        "status": "missing",
        "status_reason": "No heartbeat uploaded for this site",
        "raw_status": "missing",
        "timestamp": "",
        "age": "unknown",
        "age_seconds": -1,
        "kit_version": data.get("expected_version", ""),
        "hostname": "",
        "ip_address": data.get("ip_address", ""),
        "display_name": data.get("display_name", ""),
        "institution": data.get("institution", ""),
        "is_expected": True,
        "is_synthetic": True,
        "is_active_shadow": False,
        "error_evidence": {},
        "has_console": False,
        "has_log": False,
        "has_sync_log": False,
        "has_global_model": False,
        "has_best_model": False,
        "checkpoints": 0,
        "csv_files": [],
        "tb_events": 0,
        "total_files": 0,
        "total_size": 0,
        "server_path": str(BASE / site),
    }


def rows(*, force: bool = False) -> list[dict[str, Any]]:
    now = time.monotonic()
    if (
        not force
        and _ROWS_CACHE["value"] is not None
        and now < float(_ROWS_CACHE["expires_at"])
    ):
        return list(_ROWS_CACHE["value"])
    value = _collect_rows()
    # Stamp version skew here -- the single choke point every view goes through.
    skew = compute_version_skew(value)
    for row in value:
        row["version_skew"] = skew.get(row["site"], "")
    _ROWS_CACHE["value"] = value
    _ROWS_CACHE["expires_at"] = now + ROWS_CACHE_TTL_SECONDS
    return list(value)


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*-\s*(\w+)\s+ACC:\s*([\d.]+),\s*AUC_ROC:\s*([\d.]+)"
)

# NVFlare P2P transfer log lines:
#   [client] download ref=xxx done: elapsed=227.47s size=688.8MB (722,310,270 bytes)
#   [server] download tx xxx done: status=finished elapsed=230.22s size=688.8MB (722,310,270 bytes)
_DOWNLOAD_RE = re.compile(
    r"\[(?P<role>client|server)\]\s+download\s+"
    r"(?:ref|tx)[= ](?P<ref>\S+)\s+done:\s+"
    r"(?:status=\S+\s+)?"
    r"elapsed=(?P<elapsed>[\d.]+)s\s+"
    r"size=(?P<size_human>\S+)\s+"
    r"\((?P<size_bytes>[\d,]+)\s*bytes\)"
)

# task train sent to peer — logs elapsed time for sending model weights via P2P
_TASK_SENT_RE = re.compile(
    r"task\s+(?P<task>\w+)\s+sent\s+to\s+peer\s+in\s+(?P<elapsed>[\d.]+)\s+secs"
)

# result ACK'd by CJ — shows the subprocess result acknowledgement
_RESULT_ACK_RE = re.compile(
    r"result\s+ACK'd\s+by\s+CJ\s+in\s+(?P<elapsed>[\d.]+)s"
)


def parse_console_metrics(text: str) -> dict[str, Any]:
    data: dict[str, dict[int, dict[str, float]]] = {}
    for m in _EPOCH_RE.finditer(text):
        epoch = int(m.group(1))
        phase = m.group(2)
        acc = float(m.group(3))
        auc = float(m.group(4))
        data.setdefault(phase, {})[epoch] = {"acc": acc, "auc_roc": auc}

    if not data:
        return {"epochs": [], "series": {}}

    all_epochs = sorted({e for phase_data in data.values() for e in phase_data})
    series: dict[str, Any] = {}
    for phase, epoch_map in sorted(data.items()):
        series[f"{phase}_acc"] = [epoch_map.get(e, {}).get("acc") for e in all_epochs]
        series[f"{phase}_auc_roc"] = [
            epoch_map.get(e, {}).get("auc_roc") for e in all_epochs
        ]

    return {"epochs": all_epochs, "series": series}


def _get_console_text(site: str, mode: str, run_id: str) -> str:
    run_dir = _resolve_run_dir(site, mode, run_id)
    for name in ["nohup.out", "local_training_console_output.txt"]:
        p = run_dir / name
        if p.exists():
            # Read up to 50 MB so dataset-setup output (label distribution,
            # printed once at the start of the run) and the full epoch history
            # are both parseable. Long swarm runs can produce ~16 MB logs;
            # the previous 500 KB tail dropped everything except the last
            # handful of epochs.
            return read_text(p, limit=50_000_000)
    return ""


def _extract_training_summary(text: str) -> dict[str, Any]:
    """Extract a training summary from console output."""
    summary: dict[str, Any] = {}

    # Total epochs
    epochs = _EPOCH_RE.findall(text)
    if epochs:
        all_epochs = [int(e[0]) for e in epochs]
        summary["total_epochs"] = max(all_epochs) + 1
        summary["last_epoch"] = max(all_epochs)

    # Best checkpoint
    best_match = re.findall(
        r"Epoch\s+(\d+)\s*-\s*val\s+ACC:\s*([\d.]+),\s*AUC_ROC:\s*([\d.]+)", text
    )
    if best_match:
        best_auc = max(best_match, key=lambda x: float(x[2]))
        summary["best_val_epoch"] = int(best_auc[0])
        summary["best_val_acc"] = float(best_auc[1])
        summary["best_val_auc_roc"] = float(best_auc[2])

    # Final train metrics
    train_match = re.findall(
        r"Epoch\s+(\d+)\s*-\s*train\s+ACC:\s*([\d.]+),\s*AUC_ROC:\s*([\d.]+)", text
    )
    if train_match:
        last_train = train_match[-1]
        summary["final_train_acc"] = float(last_train[1])
        summary["final_train_auc_roc"] = float(last_train[2])

    # NVFlare round info (swarm mode)
    round_matches = re.findall(r"(?:Round|round)\s+(\d+)", text)
    if round_matches:
        summary["total_rounds"] = max(int(r) for r in round_matches)

    # P2P transfer summary
    dl_matches = list(_DOWNLOAD_RE.finditer(text))
    if dl_matches:
        total_bytes = sum(
            int(m.group("size_bytes").replace(",", "")) for m in dl_matches
        )
        total_elapsed = sum(float(m.group("elapsed")) for m in dl_matches)
        avg_speed = (
            (total_bytes / (1024 * 1024)) / total_elapsed
            if total_elapsed > 0
            else 0
        )
        summary["p2p_downloads"] = len(dl_matches)
        summary["p2p_total_bytes"] = total_bytes
        summary["p2p_avg_speed_MBs"] = round(avg_speed, 2)

    return summary


def _parse_label_distribution(text: str) -> dict[str, Any]:
    """Parse label distribution from console output.

    Looks for lines like:
      Total samples in training set: 22
      Samples in training set of class 0: 15 (68.2%)
      Samples in training set of class 1: 5 (22.7%)
      Samples in training set of class 2: 2 (9.1%)
    """
    dist: dict[str, dict[str, int | float]] = {}
    # Total samples per split
    for m in re.finditer(
        r"Total samples in (\w+) set:\s*(\d+)", text
    ):
        split = m.group(1)  # training, validation, test
        dist.setdefault(split, {})["total"] = int(m.group(2))

    # Per-class counts
    for m in re.finditer(
        r"Samples in (\w+) set of class (\d+):\s*(\d+)\s*\(([\d.]+)%\)", text
    ):
        split = m.group(1)
        cls = int(m.group(2))
        count = int(m.group(3))
        dist.setdefault(split, {})[f"class_{cls}"] = count

    if not dist:
        return {}

    # Build structured output: {splits: [...], classes: [...], counts: {split: [count_per_class]}}
    splits = sorted(dist.keys(), key=lambda s: {"training": 0, "validation": 1, "test": 2}.get(s, 9))
    # Determine class list from all splits
    all_classes = sorted({
        k for split_data in dist.values() for k in split_data if k.startswith("class_")
    })
    result: dict[str, Any] = {
        "splits": splits,
        "classes": [c.replace("class_", "Class ") for c in all_classes],
        "counts": {},
        "totals": {},
    }
    for split in splits:
        split_data = dist[split]
        result["counts"][split] = [split_data.get(c, 0) for c in all_classes]
        result["totals"][split] = split_data.get("total", sum(result["counts"][split]))

    return result


def _parse_p2p_transfers(text: str) -> list[dict[str, Any]]:
    """Parse P2P model transfer events from NVFlare console output.

    Returns a list of transfer dicts with keys:
      role, ref, elapsed_s, size_human, size_bytes, speed_MBs, line
    Sorted by elapsed time (largest first) so the slowest transfers
    are most visible.
    """
    transfers: list[dict[str, Any]] = []

    for m in _DOWNLOAD_RE.finditer(text):
        size_bytes = int(m.group("size_bytes").replace(",", ""))
        elapsed = float(m.group("elapsed"))
        speed = (size_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
        transfers.append({
            "type": "download",
            "role": m.group("role"),
            "ref": m.group("ref")[:12],
            "elapsed_s": round(elapsed, 1),
            "size_human": m.group("size_human"),
            "size_bytes": size_bytes,
            "speed_MBs": round(speed, 2),
        })

    for m in _TASK_SENT_RE.finditer(text):
        transfers.append({
            "type": "task_sent",
            "role": "client",
            "ref": m.group("task"),
            "elapsed_s": round(float(m.group("elapsed")), 1),
            "size_human": "-",
            "size_bytes": 0,
            "speed_MBs": 0,
        })

    for m in _RESULT_ACK_RE.finditer(text):
        transfers.append({
            "type": "result_ack",
            "role": "client",
            "ref": "CJ-ack",
            "elapsed_s": round(float(m.group("elapsed")), 1),
            "size_human": "-",
            "size_bytes": 0,
            "speed_MBs": 0,
        })

    # Sort: downloads first (by elapsed desc), then task_sent, then ack
    type_order = {"download": 0, "task_sent": 1, "result_ack": 2}
    transfers.sort(key=lambda t: (type_order.get(t["type"], 9), -t["elapsed_s"]))
    return transfers


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Swarm round-progress panel (#398) — parses the FL server's nohup.out, the one
# log that sees the whole CCWF run (rounds happen peer-to-peer on the clients).
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_JOBID_RE = re.compile(r"run=([0-9a-fA-F-]{36})")
_ROUND_RE = re.compile(r"on round (\d+)")
_PRUNE_RE = re.compile(r"client ([A-Za-z0-9_]+) reported error '([^']+)'.*?continuing with (\d+)")


def _read_tail(path: Path, max_bytes: int = 500_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
    except OSError:
        return ""
    return _ANSI_RE.sub("", data.decode("utf-8", "replace"))


def parse_swarm_progress(log_path: str, now: "float | None" = None):
    """Extract the latest job's round progress from the server nohup.out.

    Returns a dict (job_id, status, round, num_rounds, clients, prunes, log_age,
    finished) or None if there is no readable server log. Pure/​testable: pass a
    synthetic file path. status ∈ {training, stalled, finished, aborted, fatal, idle}.
    """
    if not log_path:
        return None
    p = Path(log_path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    now = now if now is not None else time.time()
    text = _read_tail(p)
    if not text:
        return None
    lines = text.splitlines()
    jobs = _JOBID_RE.findall(text)
    log_age = now - mtime
    if not jobs:
        return {"job_id": None, "status": "idle", "round": None, "num_rounds": None,
                "clients": {}, "prunes": [], "log_age": log_age, "finished": 0}
    jid = jobs[-1]
    jlines = [ln for ln in lines if jid in ln]
    # num_rounds from THIS job's Workflow Config line (scan jlines, not the whole
    # tail, so an older job's config in the tail can't leak the wrong count).
    num_rounds = None
    for ln in jlines:
        mm = re.search(r"'num_rounds':\s*(\d+)", ln)
        if mm:
            num_rounds = int(mm.group(1))
    rounds = [int(x) for ln in jlines for x in _ROUND_RE.findall(ln)]
    highest = max(rounds) if rounds else None
    clients: "dict[str, str]" = {}
    if highest is not None:
        rtag = f"on round {highest}"
        for ln in jlines:
            if rtag not in ln:
                continue
            mp = re.search(r"peer=([A-Za-z0-9_]+)", ln)
            ma = re.search(r"action=(\w+)", ln)
            if mp and ma:
                clients[mp.group(1)] = ma.group(1)
    prunes = []
    for ln in jlines:
        mp = _PRUNE_RE.search(ln)
        if mp:
            prunes.append({"client": mp.group(1), "error": mp.group(2), "active": int(mp.group(3))})
    terminal = None
    for ln in jlines:
        if "Server runner finished" in ln:
            terminal = "finished"
        elif "ABORTED" in ln or "Try to abort" in ln:
            terminal = "aborted"
        elif "FATAL" in ln:
            terminal = "fatal"
    if terminal:
        status = terminal
    elif log_age > SWARM_STALL_SECONDS:
        status = "stalled"
    else:
        status = "training"
    return {"job_id": jid, "status": status, "round": highest, "num_rounds": num_rounds,
            "clients": clients, "prunes": prunes, "log_age": log_age,
            "finished": sum(1 for a in clients.values() if a == "finished_learn_task")}


def render_swarm_progress_panel() -> str:
    prog = parse_swarm_progress(SERVER_LOG)
    if prog is None:
        return ""
    badge_map = {"training": "running", "stalled": "stale", "aborted": "error",
                 "fatal": "error", "finished": "finished", "idle": "waiting"}
    badge = _status_badge(badge_map.get(prog["status"], "unknown"))
    age = int(prog["log_age"])
    age_txt = f"{age}s ago" if age < 120 else f"{age // 60}m ago"
    if prog["job_id"] is None:
        return (f'<div class="swarm-panel swarm-idle"><div class="swarm-head">'
                f'<span class="swarm-title">Swarm run</span> {badge} '
                f'<span class="swarm-detail">no active job · log {age_txt}</span></div></div>')
    rnd, nr = prog["round"], prog["num_rounds"]
    if rnd is None:
        round_txt = "configuring"
    else:
        round_txt = f"round {rnd}" + (f" / {nr}" if nr else "")
    action_cls = {"finished_learn_task": "chip-done", "start_learn_task": "chip-run"}
    chips = "".join(
        f'<span class="swarm-chip {action_cls.get(prog["clients"][c], "chip-idle")}" '
        f'title="{html_escape(prog["clients"][c])}">{html_escape(c)}</span>'
        for c in sorted(prog["clients"])
    )
    pruned = ""
    if prog["prunes"]:
        items = ", ".join(f'{html_escape(x["client"])} ({html_escape(x["error"])})' for x in prog["prunes"])
        pruned = (f'<div class="swarm-prunes">pruned: {items} → '
                  f'{prog["prunes"][-1]["active"]} active</div>')
    stall = ""
    if prog["status"] == "stalled":
        stall = f'<div class="swarm-stall">⚠ no server-log activity for {age_txt.replace(" ago","")} — possible stall</div>'
    return f"""
<div class="swarm-panel swarm-{prog['status']}">
  <div class="swarm-head"><span class="swarm-title">Swarm run</span> {badge}
    <span class="swarm-detail">job {html_escape(prog['job_id'][:8])} · {round_txt}
    · {prog['finished']} finished this round · log {age_txt}</span></div>
  <div class="swarm-chips">{chips or '<span class="swarm-detail">no per-client status yet</span>'}</div>
  {pruned}{stall}
</div>"""


@app.get("/", response_class=HTMLResponse)
def index(
    site_filter: str = Query("", alias="site"),
    mode_filter: str = Query("", alias="mode"),
    status_filter: str = Query("", alias="status"),
    job_filter: str = Query("", alias="job"),
    version_filter: str = Query("", alias="version"),
    group_by_job: bool = Query(False, alias="group"),
):
    r = rows()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Collect unique values for filters
    all_sites = sorted({x["site"] for x in r})
    all_modes = sorted({x["mode"] for x in r})
    all_statuses = sorted({x["status"] for x in r})
    all_jobs = sorted({x["job_id"] for x in r if x["job_id"]})
    all_versions = sorted({x["kit_version"] for x in r if x.get("kit_version")})

    if not r:
        body = f"""
<header>
  <h1>MediSwarm Live Monitor</h1>
  <div class="meta">Refreshed {now_str}
    <a href="/" title="Refresh now">Refresh</a></div>
</header>
<main><div class="empty">No training runs found under {BASE}</div></main>"""
        return _html_page("MediSwarm Monitor", body, refresh=30)

    # Apply filters
    filtered = r
    if site_filter:
        filtered = [x for x in filtered if x["site"] == site_filter]
    if mode_filter:
        filtered = [x for x in filtered if x["mode"] == mode_filter]
    if status_filter:
        filtered = [x for x in filtered if x["status"] == status_filter]
    if job_filter:
        filtered = [x for x in filtered if x["job_id"] == job_filter]
    if version_filter:
        filtered = [x for x in filtered if x.get("kit_version") == version_filter]

    # Stats
    n_total = len(filtered)
    n_expected = len({x["site"] for x in filtered if x.get("is_expected")})
    n_seen = len({x["site"] for x in filtered if not x.get("is_synthetic")})
    n_running = sum(1 for x in filtered if x["status"] == "running")
    n_finished = sum(1 for x in filtered if x["status"] == "finished")
    n_stale = sum(1 for x in filtered if x["status"] == "stale")
    n_error = sum(1 for x in filtered if x["status"] in {"error", "failed"})
    n_missing = sum(1 for x in filtered if x["status"] == "missing")

    stats_html = f"""
<div class="stats-bar">
  <div class="stat-card"><span class="stat-num">{n_total}</span><span class="stat-label">Total Runs</span></div>
  <div class="stat-card"><span class="stat-num">{n_expected}</span><span class="stat-label">Expected</span></div>
  <div class="stat-card"><span class="stat-num">{n_seen}</span><span class="stat-label">Seen</span></div>
  <div class="stat-card"><span class="stat-num">{n_running}</span><span class="stat-label">Running</span></div>
  <div class="stat-card"><span class="stat-num">{n_missing}</span><span class="stat-label">Missing</span></div>
  <div class="stat-card"><span class="stat-num">{n_error}</span><span class="stat-label">Errors</span></div>
  <div class="stat-card"><span class="stat-num">{n_finished}</span><span class="stat-label">Finished</span></div>
  <div class="stat-card"><span class="stat-num">{n_stale}</span><span class="stat-label">Stale</span></div>
</div>"""

    # Filter bar
    def _select_opts(name: str, values: list[str], current: str) -> str:
        opts = '<option value="">All</option>'
        for v in values:
            sel = " selected" if v == current else ""
            opts += f'<option value="{html_escape(v)}"{sel}>{html_escape(v)}</option>'
        return f'<select name="{name}" onchange="this.form.submit()">{opts}</select>'

    group_checked = " checked" if group_by_job else ""
    filter_html = f"""
<form class="filter-bar" method="get" action="/">
  <div class="filter-group">
    <label>Site:</label> {_select_opts("site", all_sites, site_filter)}
  </div>
  <div class="filter-group">
    <label>Mode:</label> {_select_opts("mode", all_modes, mode_filter)}
  </div>
  <div class="filter-group">
    <label>Status:</label> {_select_opts("status", all_statuses, status_filter)}
  </div>
  <div class="filter-group">
    <label>Job:</label> {_select_opts("job", all_jobs, job_filter)}
  </div>
  <div class="filter-group">
    <label>Version:</label> {_select_opts("version", all_versions, version_filter)}
  </div>
  <div class="filter-group">
    <label><input type="checkbox" name="group" value="true"{group_checked}
      onchange="this.form.submit()"> Group by Job</label>
  </div>
  <div class="quick-links">
    <a href="/?status=error">Errors</a>
    <a href="/?status=missing">Missing</a>
    <a href="/?status=stale">Stale</a>
    {f'<a href="/?version={html_escape(all_versions[-1])}">Latest version</a>' if all_versions else ''}
  </div>
  <a href="/" class="btn-small">Clear Filters</a>
</form>"""

    # Build table rows
    table_rows = []

    if group_by_job and not job_filter:
        # Group swarm runs by job_id, local runs standalone
        from collections import OrderedDict

        job_groups: OrderedDict[str, list[dict]] = OrderedDict()
        standalone: list[dict] = []

        for x in filtered:
            if x["job_id"]:
                job_groups.setdefault(x["job_id"], []).append(x)
            else:
                standalone.append(x)

        for job_id, items in job_groups.items():
            n_items = len(items)
            sites = ", ".join(sorted({x["site"] for x in items}))
            statuses = {x["status"] for x in items}
            expected_sites = set(_read_expected_sites())
            seen_sites = {x["site"] for x in items if not x.get("is_synthetic")}
            missing_sites = sorted(expected_sites - seen_sites)
            if "error" in statuses or "failed" in statuses:
                job_status = "error"
            elif missing_sites:
                job_status = "missing"
            elif "stale" in statuses:
                job_status = "stale"
            elif "running" in statuses:
                job_status = "running"
            elif "waiting" in statuses:
                job_status = "waiting"
            elif statuses == {"finished"}:
                job_status = "finished"
            else:
                job_status = "unknown"
            reasons = [
                f"{x['site']}: {x['status_reason']}"
                for x in items if x.get("status_reason")
            ]
            if missing_sites:
                reasons.append("missing: " + ", ".join(missing_sites))
            job_reason = "; ".join(reasons)
            run_name = items[0].get("run_name", "") if items else ""
            coverage = (
                f"{len(seen_sites)}/{len(expected_sites)} clients seen"
                if expected_sites
                else f"{n_items} client(s)"
            )
            table_rows.append(
                f"""<tr class="job-group-row"><td colspan="12">
                Job: <span class="job-id">{html_escape(job_id)}</span>
                &nbsp;&middot;&nbsp; {html_escape(coverage)}: {html_escape(sites)}
                &nbsp;&middot;&nbsp; {_status_badge(job_status, job_reason)}
                {f' &middot; {html_escape(run_name)}' if run_name else ''}
                </td></tr>"""
            )
            for x in items:
                table_rows.append(_build_table_row(x))

        for x in standalone:
            table_rows.append(_build_table_row(x))
    else:
        for x in filtered:
            table_rows.append(_build_table_row(x))

    body = f"""
<header>
  <h1>MediSwarm Live Monitor</h1>
  <div class="meta"><span id="refresh-status">Refreshed {now_str}</span>
    <a href="/" title="Refresh now">Refresh</a>
    &middot; <a href="/api/runs">API</a></div>
</header>
<main>
{render_swarm_progress_panel()}
{stats_html}
{filter_html}
<table>
<thead><tr>
  <th>Site</th><th>Host</th><th>IP Address</th><th>Mode</th><th>Run</th><th>Status</th><th>Age</th>
  <th>Version</th><th>Artifacts</th><th>Size</th><th>Server Path</th><th>Links</th>
</tr></thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</main>
<script>
// Client-side age ticking: update all age cells every second
function tickAges() {{
  document.querySelectorAll('[data-timestamp]').forEach(el => {{
    const ts = el.getAttribute('data-timestamp');
    if (!ts) return;
    try {{
      const dt = new Date(ts.replace('Z', '+00:00'));
      const secs = Math.floor((Date.now() - dt.getTime()) / 1000);
      if (secs < 0) {{ el.textContent = '0s'; return; }}
      if (secs < 60) el.textContent = secs + 's';
      else if (secs < 3600) el.textContent = Math.floor(secs/60) + 'm ' + (secs%60) + 's';
      else if (secs < 86400) el.textContent = Math.floor(secs/3600) + 'h ' + Math.floor((secs%3600)/60) + 'm';
      else el.textContent = Math.floor(secs/86400) + 'd ' + Math.floor((secs%86400)/3600) + 'h';
      // Update age class
      el.className = '';
      if (secs > 600) el.className = 'age-dead';
      else if (secs > 120) el.className = 'age-stale';
    }} catch(e) {{}}
  }});
}}
setInterval(tickAges, 1000);
tickAges();

// Auto-refresh page every 30 seconds via full reload (preserves filter state)
setTimeout(() => location.reload(), 30000);
</script>"""
    return _html_page("MediSwarm Monitor", body)


def _build_table_row(x: dict[str, Any]) -> str:
    """Build a single <tr> for the index table."""
    # Links
    links = []
    if not x.get("is_synthetic"):
        detail = f"/detail/{x['site']}/{x['mode']}/{x['run_id']}"
        links.append(f'<a class="btn" href="{detail}">Details</a>')
    if x["has_console"]:
        label = "nohup" if x["mode"] == "swarm" else "console"
        links.append(
            f"<a href=\"/console/{x['site']}/{x['mode']}/{x['run_id']}\">{label}</a>"
        )
    if x["has_log"]:
        links.append(
            f"<a href=\"/log/{x['site']}/{x['mode']}/{x['run_id']}\">log</a>"
        )
    if x.get("has_sync_log"):
        links.append(
            f"<a href=\"/download/{x['site']}/{x['mode']}/{x['run_id']}/live_sync_daemon.log\">sync</a>"
        )
    if not links:
        links.append('<span class="no">-</span>')

    # Artifacts
    arts = []
    if x["checkpoints"]:
        arts.append(f'<span class="yes">{x["checkpoints"]} ckpt</span>')
    if x["has_global_model"]:
        arts.append('<span class="yes">FL_global</span>')
    if x["has_best_model"]:
        arts.append('<span class="yes">best_FL</span>')
    if x["csv_files"]:
        arts.append(f'<span class="yes">{len(x["csv_files"])} CSV</span>')
    if x["tb_events"]:
        arts.append(f'<span class="yes">{x["tb_events"]} TFE</span>')
    if not arts:
        arts.append('<span class="no">none</span>')

    # Run display
    run_display = ""
    if x["run_name"]:
        run_display = (
            f'<span class="run-name">{html_escape(x["run_name"])}</span><br>'
        )
    run_display += f'<span class="run-id">{html_escape(x["run_id"])}</span>'
    if x["job_id"]:
        run_display += f'<br><span class="job-id">job: {html_escape(x["job_id"][:8])}...</span>'
    if x.get("is_active_shadow"):
        run_display += '<br><span class="job-id">active heartbeat</span>'

    # Age: use data-timestamp for client-side ticking
    ts_attr = html_escape(x.get("timestamp", ""))
    age_cls = _age_class(x["age"])
    age_td = (
        f'<span data-timestamp="{ts_attr}" class="{age_cls}">{x["age"]}</span>'
        if ts_attr
        else x["age"]
    )

    # Hostname
    hostname = x.get("hostname", "")
    hostname_td = (
        f'<span class="version" title="{html_escape(hostname)}">'
        f'{html_escape(hostname[:20])}</span>'
        if hostname
        else '<span class="no">-</span>'
    )
    ip_address = x.get("ip_address", "")
    ip_td = (
        f'<span class="ip-address">{html_escape(ip_address)}</span>'
        if ip_address
        else '<span class="no">-</span>'
    )

    skew_reason = x.get("version_skew", "")
    if skew_reason:
        version = (
            f'<span class="version" style="background:#b91c1c;color:#fff" '
            f'title="{html_escape(skew_reason)}">&#9888; {html_escape(x["kit_version"] or "?")}</span>'
        )
    else:
        version = (
            f'<span class="version">{html_escape(x["kit_version"])}</span>'
            if x["kit_version"]
            else '<span class="no">-</span>'
        )

    size_str = _format_size(x["total_size"]) if x["total_size"] else "-"
    server_path = (
        f'<span class="file-path" title="{html_escape(x["server_path"])}">'
        f'{html_escape(x["server_path"][-40:])}</span>'
    )
    error_summary = ""
    evidence = x.get("error_evidence") or {}
    if evidence:
        source = evidence.get("source", "")
        error_summary = (
            f'<div class="error-summary">{html_escape(evidence.get("category", "error"))}: '
            f'{html_escape(evidence.get("summary", ""))}'
            f'{f" ({html_escape(source)})" if source else ""}</div>'
        )

    return f"""<tr>
  <td>{html_escape(x['site'])}</td>
  <td>{hostname_td}</td>
  <td>{ip_td}</td>
  <td>{html_escape(x['mode'])}</td>
  <td>{run_display}</td>
  <td>{_status_badge(x['status'], x.get('status_reason', ''))}{error_summary}</td>
  <td>{age_td}</td>
  <td>{version}</td>
  <td class="artifact">{' &middot; '.join(arts)}</td>
  <td>{size_str}</td>
  <td>{server_path}</td>
  <td class="links">{' '.join(links)}</td>
</tr>"""


# ---------------------------------------------------------------------------
# Detail page
# ---------------------------------------------------------------------------


@app.get("/detail/{site}/{mode}/{run_id}", response_class=HTMLResponse)
def detail(site: str, mode: str, run_id: str):
    run_dir = _resolve_run_dir(site, mode, run_id)
    hb = _read_heartbeat(run_dir)
    status, status_reason = _infer_status(hb, run_dir)
    error_evidence = _extract_error_evidence(run_dir, hb)
    console_text = _get_console_text(site, mode, run_id)
    metrics = parse_console_metrics(console_text)
    csv_files = _find_csv_files(run_dir)
    tb_events = _find_tb_events(run_dir)
    checkpoints = _find_checkpoints(run_dir)
    all_files = _find_all_files(run_dir)
    training_summary = _extract_training_summary(console_text)
    label_dist = _parse_label_distribution(console_text)
    p2p_transfers = _parse_p2p_transfers(console_text)

    # -- Heartbeat info card --
    hb_display_keys = [
        ("site_name", "Site Name"),
        ("mode", "Mode"),
        ("job_id", "Job ID"),
        ("run_name", "Run Name"),
        ("timestamp", "Last Heartbeat"),
        ("status", "Raw Status"),
        ("kit_version", "Kit Version"),
        ("ip_address", "IP Address"),
        ("sync_version", "Live Sync Version"),
        ("sync_status", "Live Sync Status"),
        ("sync_error", "Live Sync Error"),
        ("job_dir_seen", "Job Dir Seen"),
        ("run_dir_seen", "Run Dir Seen"),
        ("kit_root", "Kit Root (client)"),
        ("run_dir", "Run Dir (client)"),
        ("log_file", "Log File (client)"),
        ("console_file", "Console File (client)"),
        ("global_model", "Global Model (client)"),
        ("best_global_model", "Best Global Model (client)"),
        ("last_ckpt", "Last Checkpoint (client)"),
        ("epoch_ckpt", "Epoch Checkpoint (client)"),
        ("tb_file", "TensorBoard File (client)"),
        ("hostname", "Hostname"),
    ]
    hb_rows = ""
    # Add inferred status first
    hb_rows += f"<tr><td>Effective Status</td><td>{_status_badge(status, status_reason)}</td></tr>\n"
    for key, label in hb_display_keys:
        val = hb.get(key, "")
        if val:
            hb_rows += f"<tr><td>{html_escape(label)}</td><td>{html_escape(str(val))}</td></tr>\n"
    # Add age
    ts = hb.get("timestamp", "")
    if ts:
        hb_rows += f"<tr><td>Heartbeat Age</td><td>{parse_age(ts)}</td></tr>\n"
    if not hb_rows:
        hb_rows = '<tr><td colspan="2">No heartbeat data available</td></tr>'

    error_html = ""
    if error_evidence:
        source = html_escape(str(error_evidence.get("source", "")))
        line = html_escape(str(error_evidence.get("line", "")))
        excerpt = html_escape(str(error_evidence.get("excerpt", "")))
        error_html = f"""
<div class="card" style="grid-column: 1 / -1;">
  <h2>Error Evidence</h2>
  <table class="kv-table"><tbody>
    <tr><td>Category</td><td>{html_escape(str(error_evidence.get("category", "")))}</td></tr>
    <tr><td>Summary</td><td>{html_escape(str(error_evidence.get("summary", "")))}</td></tr>
    <tr><td>Source</td><td>{source}</td></tr>
    <tr><td>Matched Line</td><td>{line}</td></tr>
  </tbody></table>
  <pre class="error-excerpt">{excerpt}</pre>
</div>"""

    # -- Training summary card --
    summary_html = ""
    if training_summary:
        summary_rows = ""
        if "total_epochs" in training_summary:
            summary_rows += (
                f"<tr><td>Total Epochs</td><td>{training_summary['total_epochs']}</td></tr>"
            )
        if "best_val_epoch" in training_summary:
            summary_rows += (
                f"<tr><td>Best Validation</td>"
                f"<td>Epoch {training_summary['best_val_epoch']} &mdash; "
                f"ACC: {training_summary['best_val_acc']:.4f}, "
                f"AUC_ROC: {training_summary['best_val_auc_roc']:.4f}</td></tr>"
            )
        if "final_train_acc" in training_summary:
            summary_rows += (
                f"<tr><td>Final Training</td>"
                f"<td>ACC: {training_summary['final_train_acc']:.4f}, "
                f"AUC_ROC: {training_summary['final_train_auc_roc']:.4f}</td></tr>"
            )
        if "total_rounds" in training_summary:
            summary_rows += (
                f"<tr><td>FL Rounds</td><td>{training_summary['total_rounds']}</td></tr>"
            )
        if "p2p_downloads" in training_summary:
            p2p_size = training_summary["p2p_total_bytes"]
            p2p_size_str = (
                f"{p2p_size / (1024**3):.2f} GB"
                if p2p_size >= 1024**3
                else f"{p2p_size / (1024**2):.1f} MB"
            )
            summary_rows += (
                f"<tr><td>P2P Transfers</td>"
                f"<td>{training_summary['p2p_downloads']} download(s), "
                f"{p2p_size_str} total, "
                f"avg {training_summary['p2p_avg_speed_MBs']:.2f} MB/s</td></tr>"
            )
        if summary_rows:
            summary_html = f"""
<div class="card">
  <h2>Training Summary</h2>
  <table class="kv-table"><tbody>{summary_rows}</tbody></table>
</div>"""

    # -- P2P Transfers card --
    p2p_html = ""
    if p2p_transfers:
        downloads = [t for t in p2p_transfers if t["type"] == "download"]
        tasks_sent = [t for t in p2p_transfers if t["type"] == "task_sent"]
        acks = [t for t in p2p_transfers if t["type"] == "result_ack"]

        # Summary stats
        dl_count = len(downloads)
        total_bytes = sum(t["size_bytes"] for t in downloads)
        total_elapsed = sum(t["elapsed_s"] for t in downloads)
        avg_speed = (
            (total_bytes / (1024 * 1024)) / total_elapsed
            if total_elapsed > 0
            else 0
        )
        total_size_human = (
            f"{total_bytes / (1024**3):.2f} GB"
            if total_bytes >= 1024**3
            else f"{total_bytes / (1024**2):.1f} MB"
        )

        p2p_summary = (
            f"<div style='margin-bottom:0.8rem;font-size:0.85rem;'>"
            f"<strong>{dl_count}</strong> model download(s) | "
            f"<strong>{total_size_human}</strong> total transferred | "
            f"avg <strong>{avg_speed:.2f} MB/s</strong>"
        )
        if tasks_sent:
            p2p_summary += f" | <strong>{len(tasks_sent)}</strong> task(s) sent to peers"
        if acks:
            p2p_summary += f" | <strong>{len(acks)}</strong> result ACK(s)"
        p2p_summary += "</div>"

        # Transfer rows
        p2p_rows = ""
        for t in p2p_transfers:
            type_badge = {
                "download": '<span style="color:#2980b9;font-weight:600;">download</span>',
                "task_sent": '<span style="color:#27ae60;font-weight:600;">task sent</span>',
                "result_ack": '<span style="color:#8e44ad;font-weight:600;">result ACK</span>',
            }.get(t["type"], t["type"])
            speed_str = (
                f'{t["speed_MBs"]:.2f} MB/s' if t["speed_MBs"] > 0 else "-"
            )
            p2p_rows += (
                f"<tr>"
                f"<td>{type_badge}</td>"
                f"<td>{html_escape(t['role'])}</td>"
                f"<td><code>{html_escape(t['ref'])}</code></td>"
                f'<td style="text-align:right;">{t["elapsed_s"]:.1f}s</td>'
                f"<td>{html_escape(t['size_human'])}</td>"
                f'<td style="text-align:right;">{speed_str}</td>'
                f"</tr>"
            )

        p2p_html = f"""
<div class="card" style="grid-column: 1 / -1;">
  <h2>P2P Model Transfers</h2>
  {p2p_summary}
  <table class="kv-table" style="width:100%;">
    <thead><tr>
      <th>Type</th><th>Role</th><th>Ref</th>
      <th style="text-align:right;">Elapsed</th><th>Size</th>
      <th style="text-align:right;">Speed</th>
    </tr></thead>
    <tbody>{p2p_rows}</tbody>
  </table>
</div>"""

    # -- Server-side files card --
    file_list_html = ""
    if all_files:
        items = ""
        for f in all_files:
            size = _format_size(f["size"])
            dl_url = f"/download/{site}/{mode}/{run_id}/{f['rel_path']}"
            icon = "📄"
            if f["name"].endswith(".csv"):
                icon = "📊"
            elif f["name"].endswith(".ckpt") or f["name"].endswith(".pt"):
                icon = "📦"
            elif f["name"].endswith(".json") or f["name"].endswith(".yaml"):
                icon = "📝"
            elif "tfevents" in f["name"]:
                icon = "📈"
            items += (
                f'<li><span class="file-icon">{icon}</span>'
                f'<span style="flex:1;">'
                f'<a href="{dl_url}" title="Download">{html_escape(f["rel_path"])}</a>'
                f'<br><span class="file-path">{html_escape(f["server_path"])}</span>'
                f'</span>'
                f'<span class="file-size">{size}</span>'
                f'<a class="btn btn-download" href="{dl_url}" '
                f'style="font-size:0.72rem;padding:2px 8px;">Download</a>'
                f'</li>'
            )
        total_size = _format_size(sum(f["size"] for f in all_files))
        file_list_html = f"""
<div class="card" style="grid-column: 1 / -1;">
  <h2>Server Files ({len(all_files)} files, {total_size} total)</h2>
  <p style="margin-bottom:0.5rem;font-size:0.82rem;color:var(--text-light);">
    Server directory: <code>{html_escape(str(run_dir))}</code></p>
  <ul class="file-list">{items}</ul>
</div>"""

    # -- CSV links card --
    csv_links_html = ""
    if csv_files:
        csv_items = "".join(
            f'<li><a href="/csv/{site}/{mode}/{run_id}/{f}">{html_escape(f)}</a>'
            f' &middot; <a class="btn btn-download" '
            f'href="/download/{site}/{mode}/{run_id}/run_dir/{f}" '
            f'style="font-size:0.72rem;padding:2px 8px;">Download</a></li>'
            for f in csv_files
        )
        csv_links_html = f"""
<div class="card">
  <h2>Result CSVs</h2>
  <ul class="file-list">{csv_items}</ul>
</div>"""

    # -- Checkpoints card --
    ckpt_html = ""
    if checkpoints:
        ckpt_items = "".join(
            f'<li><span class="file-icon">📦</span>'
            f'<span style="flex:1;">{html_escape(c["name"])}'
            f'<br><span class="file-path">{html_escape(c["server_path"])}</span></span>'
            f'<span class="file-size">{_format_size(c["size"])}</span>'
            f'<a class="btn btn-download" '
            f'href="/download/{site}/{mode}/{run_id}/{c["rel_path"]}" '
            f'style="font-size:0.72rem;padding:2px 8px;">Download</a>'
            f'</li>'
            for c in checkpoints
        )
        ckpt_html = f"""
<div class="card">
  <h2>Checkpoints ({len(checkpoints)})</h2>
  <ul class="file-list">{ckpt_items}</ul>
</div>"""

    # -- Console snippet (last 300 lines) --
    console_lines = console_text.strip().split("\n")
    console_tail = "\n".join(console_lines[-300:]) if console_lines else "No output."
    console_tail_escaped = (
        console_tail.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    console_len = len(console_lines)

    # -- Chart section --
    chart_html = ""
    if metrics["epochs"]:
        chart_html = f"""
<div class="card" style="grid-column: 1 / -1;">
  <h2>Training Metrics (from console)</h2>
  <div class="chart-container"><canvas id="metricsChart"></canvas></div>
</div>
<script>
const metricsData = {json.dumps(metrics)};
const ctx = document.getElementById('metricsChart').getContext('2d');
const colors = {{
  'train_acc': '#27ae60', 'val_acc': '#2980b9',
  'test_acc': '#8e44ad',
  'train_auc_roc': '#e67e22', 'val_auc_roc': '#c0392b',
  'test_auc_roc': '#f39c12'
}};
const defaultVisible = ['train_acc','val_acc','train_auc_roc','val_auc_roc'];
const datasets = [];
for (const [key, values] of Object.entries(metricsData.series)) {{
  datasets.push({{
    label: key.replace('_', ' '),
    data: values,
    borderColor: colors[key] || '#636e72',
    backgroundColor: 'transparent',
    tension: 0.3,
    pointRadius: 2,
    borderWidth: 2,
    hidden: !defaultVisible.includes(key)
  }});
}}
new Chart(ctx, {{
  type: 'line',
  data: {{ labels: metricsData.epochs, datasets: datasets }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    scales: {{
      x: {{ title: {{ display: true, text: 'Epoch' }} }},
      y: {{ title: {{ display: true, text: 'Value' }}, min: 0, max: 1 }}
    }},
    plugins: {{
      legend: {{ position: 'top' }}
    }}
  }}
}});
</script>"""

    # -- TensorBoard metrics --
    tb_html = ""
    if tb_events and HAS_TBPARSE:
        # Parse and display inline
        try:
            reader = SummaryReader(str(tb_events[0].parent))
            df = reader.scalars
            tags = sorted(df["tag"].unique()) if len(df) > 0 else []
            tb_data: dict[str, Any] = {}
            for tag in tags:
                subset = df[df["tag"] == tag].sort_values("step")
                tb_data[tag] = {
                    "steps": subset["step"].tolist(),
                    "values": subset["value"].tolist(),
                }
            if tb_data:
                tb_html = f"""
<div class="card" style="grid-column: 1 / -1;">
  <h2>TensorBoard Metrics ({len(tags)} tags)</h2>
  <div class="chart-container"><canvas id="tbChart"></canvas></div>
  <div style="margin-top:0.5rem;">
    <a class="btn" href="/tb_metrics/{site}/{mode}/{run_id}">Raw JSON</a>
  </div>
</div>
<script>
const tbData = {json.dumps(tb_data)};
const tbCtx = document.getElementById('tbChart').getContext('2d');
const tbColors = ['#27ae60','#2980b9','#8e44ad','#e67e22','#c0392b',
                  '#f39c12','#1abc9c','#e74c3c','#3498db','#9b59b6'];
const tbDatasets = [];
let colorIdx = 0;
for (const [tag, data] of Object.entries(tbData)) {{
  const tagLower = tag.toLowerCase();
  const isDefault = tagLower.includes('acc') || tagLower.includes('auc_roc')
                    || tagLower.includes('auroc') || tagLower.includes('auc-roc');
  tbDatasets.push({{
    label: tag,
    data: data.steps.map((s, i) => ({{ x: s, y: data.values[i] }})),
    borderColor: tbColors[colorIdx % tbColors.length],
    backgroundColor: 'transparent',
    tension: 0.3,
    pointRadius: 1,
    borderWidth: 2,
    showLine: true,
    hidden: !isDefault
  }});
  colorIdx++;
}}
new Chart(tbCtx, {{
  type: 'scatter',
  data: {{ datasets: tbDatasets }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    scales: {{
      x: {{ title: {{ display: true, text: 'Step' }} }},
      y: {{ title: {{ display: true, text: 'Value' }} }}
    }},
    plugins: {{
      legend: {{ position: 'top' }}
    }}
  }}
}});
</script>"""
        except Exception as e:
            tb_html = f"""
<div class="card">
  <h2>TensorBoard Metrics</h2>
  <p>Error parsing TensorBoard events: {html_escape(str(e))}</p>
  <a class="btn" href="/tb_metrics/{site}/{mode}/{run_id}">Try raw JSON</a>
</div>"""
    elif tb_events:
        tb_html = f"""
<div class="card">
  <h2>TensorBoard Metrics</h2>
  <p>Found {len(tb_events)} TensorBoard event file(s).</p>
  <p><code>tbparse</code> is {'installed' if HAS_TBPARSE else
  'not installed &mdash; install with <code>pip install tbparse</code> to enable parsing'}.</p>
</div>"""

    # -- Models card --
    model_html = ""
    model_files = []
    for mname in [
        "FL_global_model.pt",
        "best_FL_global_model.pt",
        "last_global_model.ckpt",
    ]:
        mp = run_dir / mname
        if not mp.exists():
            mp = run_dir / "run_dir" / mname
        if mp.exists():
            model_files.append(
                {
                    "name": mname,
                    "size": mp.stat().st_size,
                    "path": str(mp),
                    "rel_path": str(mp.relative_to(run_dir)),
                }
            )
    if model_files:
        model_items = "".join(
            f'<li><span class="file-icon">🧠</span>'
            f'<span style="flex:1;">{html_escape(m["name"])}'
            f'<br><span class="file-path">{html_escape(m["path"])}</span></span>'
            f'<span class="file-size">{_format_size(m["size"])}</span>'
            f'<a class="btn btn-download" '
            f'href="/download/{site}/{mode}/{run_id}/{m["rel_path"]}" '
            f'style="font-size:0.72rem;padding:2px 8px;">Download</a>'
            f"</li>"
            for m in model_files
        )
        model_html = f"""
<div class="card">
  <h2>Models</h2>
  <ul class="file-list">{model_items}</ul>
</div>"""

    # -- Label distribution chart --
    label_dist_html = ""
    if label_dist:
        bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                      '#1abc9c', '#e67e22', '#34495e']
        label_dist_html = f"""
<div class="card" style="grid-column: 1 / -1;">
  <h2>Label Distribution</h2>
  <div class="chart-container" style="max-height:350px;"><canvas id="labelDistChart"></canvas></div>
</div>
<script>
const ldData = {json.dumps(label_dist)};
const ldCtx = document.getElementById('labelDistChart').getContext('2d');
const barColors = {json.dumps(bar_colors)};
const ldDatasets = ldData.classes.map((cls, i) => ({{
  label: cls,
  data: ldData.splits.map(split => ldData.counts[split][i] || 0),
  backgroundColor: barColors[i % barColors.length],
  borderColor: barColors[i % barColors.length],
  borderWidth: 1
}}));
new Chart(ldCtx, {{
  type: 'bar',
  data: {{ labels: ldData.splits, datasets: ldDatasets }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{
        callbacks: {{
          afterBody: function(items) {{
            const split = items[0].label;
            const total = ldData.totals[split] || 0;
            return 'Total: ' + total;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Dataset Split' }} }},
      y: {{ title: {{ display: true, text: 'Sample Count' }}, beginAtZero: true }}
    }}
  }}
}});
</script>"""

    log_btn = ""
    if (run_dir / "log.txt").exists():
        log_btn = f'<a class="btn" href="/log/{site}/{mode}/{run_id}">Full NVFlare Log</a>'

    body = f"""
<header>
  <h1>MediSwarm Live Monitor</h1>
  <div class="meta"><a href="/">Back to overview</a></div>
</header>
<main>
<div class="breadcrumb">
  <a href="/">Home</a> &rsaquo; {html_escape(site)} &rsaquo; {html_escape(mode)} &rsaquo; {html_escape(run_id)}
</div>

<div class="detail-grid">
  <div class="card">
    <h2>Heartbeat &amp; Status</h2>
    <table class="kv-table"><tbody>{hb_rows}</tbody></table>
    <div style="margin-top:0.8rem;">
      <a class="btn" href="/heartbeat/{site}/{mode}/{run_id}">Raw Heartbeat JSON</a>
    </div>
  </div>

  {summary_html}
  {error_html}
  {p2p_html}
  {csv_links_html}
  {ckpt_html}
  {model_html}

  {chart_html}
  {tb_html}
  {label_dist_html}
  {file_list_html}

  <div class="card" style="grid-column: 1 / -1;">
    <h2>Console Output (last 300 of {console_len} lines)</h2>
    <div style="margin-bottom:0.5rem;">
      <a class="btn" href="/console/{site}/{mode}/{run_id}">Full Console Output</a>
      {log_btn}
    </div>
    <pre>{console_tail_escaped}</pre>
  </div>
</div>
</main>"""
    # Include Chart.js if any chart is rendered
    needs_chartjs = bool(chart_html) or bool(tb_html and HAS_TBPARSE and tb_events) or bool(label_dist_html)
    chartjs_head = '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>' if needs_chartjs else ""

    return _html_page(
        f"{html_escape(site)}/{html_escape(mode)}/{html_escape(run_id)} — MediSwarm",
        body,
        extra_head=chartjs_head,
    )


# ---------------------------------------------------------------------------
# File download endpoint
# ---------------------------------------------------------------------------


@app.get("/download/{site}/{mode}/{run_id}/{file_path:path}")
def download_file(site: str, mode: str, run_id: str, file_path: str):
    """Download any file from a run directory."""
    run_dir = _resolve_run_dir(site, mode, run_id)

    # Prevent traversal in file_path
    if ".." in file_path:
        raise HTTPException(status_code=400, detail="Invalid file path")

    target = (run_dir / file_path).resolve()

    # Ensure target is under run_dir
    try:
        common = os.path.commonpath([str(run_dir.resolve()), str(target)])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    if common != str(run_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(target),
        filename=target.name,
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Existing endpoints (preserved)
# ---------------------------------------------------------------------------


@app.get("/heartbeat/{site}/{mode}/{run_id}", response_class=PlainTextResponse)
def heartbeat(site: str, mode: str, run_id: str):
    run_dir = _resolve_run_dir(site, mode, run_id)
    for name in ["heartbeat_final.json", "heartbeat.json"]:
        p = run_dir / name
        if p.exists():
            return read_text(p)
    return ""


@app.get("/console/{site}/{mode}/{run_id}", response_class=PlainTextResponse)
def console(site: str, mode: str, run_id: str):
    return _get_console_text(site, mode, run_id)


@app.get("/log/{site}/{mode}/{run_id}", response_class=PlainTextResponse)
def log(site: str, mode: str, run_id: str):
    run_dir = _resolve_run_dir(site, mode, run_id)
    return read_text(run_dir / "log.txt")


# ---------------------------------------------------------------------------
# New endpoints
# ---------------------------------------------------------------------------


@app.get("/metrics/{site}/{mode}/{run_id}", response_class=JSONResponse)
def metrics(site: str, mode: str, run_id: str):
    text = _get_console_text(site, mode, run_id)
    return parse_console_metrics(text)


@app.get("/tb_metrics/{site}/{mode}/{run_id}", response_class=JSONResponse)
def tb_metrics(site: str, mode: str, run_id: str):
    if not HAS_TBPARSE:
        return JSONResponse(
            {"error": "tbparse is not installed"}, status_code=501
        )

    validated_dir = _resolve_run_dir(site, mode, run_id)
    run_dir = validated_dir / "run_dir"
    events = (
        sorted(run_dir.rglob("events.out.tfevents*")) if run_dir.exists() else []
    )
    if not events:
        return {"scalars": []}

    try:
        reader = SummaryReader(str(events[0].parent))
        df = reader.scalars
        result: dict[str, Any] = {"tags": {}}
        for tag in df["tag"].unique():
            subset = df[df["tag"] == tag].sort_values("step")
            result["tags"][tag] = {
                "steps": subset["step"].tolist(),
                "values": subset["value"].tolist(),
            }
        return result
    except Exception:
        return JSONResponse(
            {"error": "Failed to parse TensorBoard events"}, status_code=500
        )


@app.get("/csv/{site}/{mode}/{run_id}/{filename}", response_class=HTMLResponse)
def csv_view(site: str, mode: str, run_id: str, filename: str):
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename or ".." in filename or "/" in filename:
        return HTMLResponse("<p>Invalid filename</p>", status_code=400)

    validated_dir = _resolve_run_dir(site, mode, run_id)
    rd = validated_dir / "run_dir"
    matches = list(rd.rglob(safe_name)) if rd.exists() else []
    if not matches:
        return HTMLResponse("<p>File not found</p>", status_code=404)

    csv_path = matches[0]
    text = csv_path.read_text(errors="replace")

    reader = csv.reader(io.StringIO(text))
    all_rows = list(reader)
    if not all_rows:
        return HTMLResponse("<p>Empty CSV file</p>")

    headers = all_rows[0]
    data_rows = all_rows[1:]

    th = "".join(f"<th>{html_escape(h)}</th>" for h in headers)
    trs = ""
    for row in data_rows[:500]:
        tds = "".join(f"<td>{html_escape(cell)}</td>" for cell in row)
        trs += f"<tr>{tds}</tr>\n"

    truncated = (
        f"<p><em>Showing first 500 of {len(data_rows)} rows.</em></p>"
        if len(data_rows) > 500
        else ""
    )

    dl_url = f"/download/{site}/{mode}/{run_id}/run_dir/{safe_name}"

    body = f"""
<header>
  <h1>MediSwarm Live Monitor</h1>
  <div class="meta"><a href="/detail/{site}/{mode}/{run_id}">Back to detail</a></div>
</header>
<main>
<div class="breadcrumb">
  <a href="/">Home</a> &rsaquo;
  <a href="/detail/{site}/{mode}/{run_id}">{html_escape(site)}/{html_escape(mode)}/{html_escape(run_id)}</a> &rsaquo;
  {html_escape(safe_name)}
</div>
<div class="card">
  <h2>{html_escape(safe_name)}</h2>
  <div style="margin-bottom:0.5rem;">
    <a class="btn btn-download" href="{dl_url}">Download CSV</a>
    <span class="file-path">Server: {html_escape(str(csv_path))}</span>
  </div>
  {truncated}
  <div style="overflow-x:auto;">
  <table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>
  </div>
</div>
</main>"""
    return _html_page(f"{html_escape(safe_name)} — MediSwarm", body)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@app.get("/api/runs", response_class=JSONResponse)
def api_runs():
    return rows()


@app.get("/api/metrics/{site}/{mode}/{run_id}", response_class=JSONResponse)
def api_metrics(site: str, mode: str, run_id: str):
    text = _get_console_text(site, mode, run_id)
    return parse_console_metrics(text)


@app.get("/api/heartbeat/{site}/{mode}/{run_id}", response_class=JSONResponse)
def api_heartbeat(site: str, mode: str, run_id: str):
    run_dir = _resolve_run_dir(site, mode, run_id)
    return _read_heartbeat(run_dir)


@app.get("/api/errors/{site}/{mode}/{run_id}", response_class=JSONResponse)
def api_errors(site: str, mode: str, run_id: str):
    run_dir = _resolve_run_dir(site, mode, run_id)
    hb = _read_heartbeat(run_dir)
    return _extract_error_evidence(run_dir, hb) or {}


@app.get("/api/files/{site}/{mode}/{run_id}", response_class=JSONResponse)
def api_files(site: str, mode: str, run_id: str):
    """Return all files in a run directory as JSON."""
    run_dir = _resolve_run_dir(site, mode, run_id)
    return _find_all_files(run_dir)


@app.get("/api/summary/{site}/{mode}/{run_id}", response_class=JSONResponse)
def api_summary(site: str, mode: str, run_id: str):
    """Return training summary extracted from console output."""
    text = _get_console_text(site, mode, run_id)
    return _extract_training_summary(text)
