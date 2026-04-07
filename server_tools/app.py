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

BASE = Path("/srv/mediswarm/live")
app = FastAPI(title="MediSwarm Live Monitor")

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")


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
.badge-finished { background: var(--blue); }
.badge-unknown { background: var(--gray); }
.badge-error { background: var(--red); }
.badge-stale { background: var(--orange); }
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
.version { font-family: var(--mono); font-size: 0.75rem; color: var(--text-light); }
.job-id { font-family: var(--mono); font-size: 0.72rem; color: var(--purple); }

/* Job group header */
.job-group-row td { background: #e8e4f0 !important; font-weight: 600;
  font-size: 0.82rem; color: var(--purple); padding: 0.5rem 0.9rem; }

/* Summary stats */
.stats-bar { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.stat-card { background: var(--card); border-radius: 8px; padding: 0.6rem 1.2rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06); display: flex; align-items: center; gap: 0.5rem; }
.stat-card .stat-num { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
.stat-card .stat-label { font-size: 0.78rem; color: var(--text-light); }

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


def _status_badge(status: str) -> str:
    cls = "badge-unknown"
    if status == "running":
        cls = "badge-running"
    elif status == "finished":
        cls = "badge-finished"
    elif status in ("error", "failed"):
        cls = "badge-error"
    elif status == "stale":
        cls = "badge-stale"
    return f'<span class="badge {cls}">{html_escape(status)}</span>'


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
            except Exception:
                pass
    return {}


def _infer_status(hb: dict[str, Any], run_dir: Path) -> str:
    """Infer the effective status from heartbeat + file system state.

    Rules:
    - If heartbeat_final.json exists -> use its status (typically "finished")
    - If status is "running" but heartbeat is >5 min old -> "stale"
    - If status is "running" but heartbeat is >1 hour old -> "finished" (presumed)
    - Otherwise use heartbeat status as-is
    """
    has_final = (run_dir / "heartbeat_final.json").exists()
    raw_status = hb.get("status", "unknown")

    if has_final:
        try:
            final = json.loads((run_dir / "heartbeat_final.json").read_text())
            return final.get("status", raw_status)
        except Exception:
            pass

    if raw_status == "running":
        age = _age_seconds(hb.get("timestamp", ""))
        if age > 3600:
            return "finished"
        if age > 300:
            return "stale"

    return raw_status


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


def rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not BASE.exists():
        return out

    for site_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
        for mode_dir in sorted(p for p in site_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
                hb = _read_heartbeat(run_dir)
                ts = hb.get("timestamp", "")
                age = parse_age(ts)
                status = _infer_status(hb, run_dir)
                csv_files = _find_csv_files(run_dir)
                tb_events = _find_tb_events(run_dir)
                checkpoints = _find_checkpoints(run_dir)

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
                        "raw_status": hb.get("status", "unknown"),
                        "timestamp": ts,
                        "age": age,
                        "age_seconds": _age_seconds(ts),
                        "kit_version": hb.get("kit_version", ""),
                        "has_console": (run_dir / "nohup.out").exists()
                        or (run_dir / "local_training_console_output.txt").exists(),
                        "has_log": (run_dir / "log.txt").exists(),
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

    # Sort by timestamp descending (newest first)
    out.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return out


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*-\s*(\w+)\s+ACC:\s*([\d.]+),\s*AUC_ROC:\s*([\d.]+)"
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
            return read_text(p, limit=500_000)
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

    return summary


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(
    site_filter: str = Query("", alias="site"),
    mode_filter: str = Query("", alias="mode"),
    status_filter: str = Query("", alias="status"),
    job_filter: str = Query("", alias="job"),
    group_by_job: bool = Query(False, alias="group"),
):
    r = rows()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Collect unique values for filters
    all_sites = sorted({x["site"] for x in r})
    all_modes = sorted({x["mode"] for x in r})
    all_statuses = sorted({x["status"] for x in r})
    all_jobs = sorted({x["job_id"] for x in r if x["job_id"]})

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

    # Stats
    n_total = len(filtered)
    n_running = sum(1 for x in filtered if x["status"] == "running")
    n_finished = sum(1 for x in filtered if x["status"] == "finished")
    n_stale = sum(1 for x in filtered if x["status"] == "stale")
    n_sites = len({x["site"] for x in filtered})

    stats_html = f"""
<div class="stats-bar">
  <div class="stat-card"><span class="stat-num">{n_total}</span><span class="stat-label">Total Runs</span></div>
  <div class="stat-card"><span class="stat-num">{n_running}</span><span class="stat-label">Running</span></div>
  <div class="stat-card"><span class="stat-num">{n_finished}</span><span class="stat-label">Finished</span></div>
  <div class="stat-card"><span class="stat-num">{n_stale}</span><span class="stat-label">Stale</span></div>
  <div class="stat-card"><span class="stat-num">{n_sites}</span><span class="stat-label">Sites</span></div>
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
    <label><input type="checkbox" name="group" value="true"{group_checked}
      onchange="this.form.submit()"> Group by Job</label>
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
            job_status = (
                "running"
                if "running" in statuses
                else ("stale" if "stale" in statuses else "finished")
            )
            run_name = items[0].get("run_name", "") if items else ""
            table_rows.append(
                f"""<tr class="job-group-row"><td colspan="10">
                Job: <span class="job-id">{html_escape(job_id)}</span>
                &nbsp;&middot;&nbsp; {n_items} client(s): {html_escape(sites)}
                &nbsp;&middot;&nbsp; {_status_badge(job_status)}
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
  <div class="meta">Refreshed {now_str}
    <a href="/" title="Refresh now">Refresh</a>
    &middot; <a href="/api/runs">API</a></div>
</header>
<main>
{stats_html}
{filter_html}
<table>
<thead><tr>
  <th>Site</th><th>Mode</th><th>Run</th><th>Status</th><th>Age</th>
  <th>Version</th><th>Artifacts</th><th>Size</th><th>Server Path</th><th>Links</th>
</tr></thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</main>"""
    return _html_page("MediSwarm Monitor", body, refresh=30)


def _build_table_row(x: dict[str, Any]) -> str:
    """Build a single <tr> for the index table."""
    # Links
    links = []
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

    age_cls = _age_class(x["age"])
    age_td = (
        f'<span class="{age_cls}">{x["age"]}</span>' if age_cls else x["age"]
    )

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

    return f"""<tr>
  <td>{html_escape(x['site'])}</td>
  <td>{html_escape(x['mode'])}</td>
  <td>{run_display}</td>
  <td>{_status_badge(x['status'])}</td>
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
    status = _infer_status(hb, run_dir)
    console_text = _get_console_text(site, mode, run_id)
    metrics = parse_console_metrics(console_text)
    csv_files = _find_csv_files(run_dir)
    tb_events = _find_tb_events(run_dir)
    checkpoints = _find_checkpoints(run_dir)
    all_files = _find_all_files(run_dir)
    training_summary = _extract_training_summary(console_text)

    # -- Heartbeat info card --
    hb_display_keys = [
        ("site_name", "Site Name"),
        ("mode", "Mode"),
        ("job_id", "Job ID"),
        ("run_name", "Run Name"),
        ("timestamp", "Last Heartbeat"),
        ("status", "Raw Status"),
        ("kit_version", "Kit Version"),
        ("kit_root", "Kit Root (client)"),
        ("run_dir", "Run Dir (client)"),
        ("log_file", "Log File (client)"),
        ("console_file", "Console File (client)"),
        ("global_model", "Global Model (client)"),
        ("best_global_model", "Best Global Model (client)"),
        ("last_ckpt", "Last Checkpoint (client)"),
        ("epoch_ckpt", "Epoch Checkpoint (client)"),
        ("tb_file", "TensorBoard File (client)"),
    ]
    hb_rows = ""
    # Add inferred status first
    hb_rows += f"<tr><td>Effective Status</td><td>{_status_badge(status)}</td></tr>\n"
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
        if summary_rows:
            summary_html = f"""
<div class="card">
  <h2>Training Summary</h2>
  <table class="kv-table"><tbody>{summary_rows}</tbody></table>
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
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const metricsData = {json.dumps(metrics)};
const ctx = document.getElementById('metricsChart').getContext('2d');
const colors = {{
  'train_acc': '#27ae60', 'val_acc': '#2980b9',
  'test_acc': '#8e44ad',
  'train_auc_roc': '#e67e22', 'val_auc_roc': '#c0392b',
  'test_auc_roc': '#f39c12'
}};
const datasets = [];
for (const [key, values] of Object.entries(metricsData.series)) {{
  datasets.push({{
    label: key.replace('_', ' '),
    data: values,
    borderColor: colors[key] || '#636e72',
    backgroundColor: 'transparent',
    tension: 0.3,
    pointRadius: 2,
    borderWidth: 2
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
  tbDatasets.push({{
    label: tag,
    data: data.steps.map((s, i) => ({{ x: s, y: data.values[i] }})),
    borderColor: tbColors[colorIdx % tbColors.length],
    backgroundColor: 'transparent',
    tension: 0.3,
    pointRadius: 1,
    borderWidth: 2,
    showLine: true
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
  {csv_links_html}
  {ckpt_html}
  {model_html}

  {chart_html}
  {tb_html}
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
    return _html_page(
        f"{html_escape(site)}/{html_escape(mode)}/{html_escape(run_id)} — MediSwarm",
        body,
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
