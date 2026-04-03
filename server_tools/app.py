"""MediSwarm Live Monitor — enhanced dashboard for training runs.

Serves a styled web UI that displays live training status, metrics charts,
and artifact links for all sites synced by live_sync to /srv/mediswarm/live/.
"""

from pathlib import Path
import csv
import io
import json
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# Optional: TensorBoard event parsing via tbparse
try:
    from tbparse import SummaryReader

    HAS_TBPARSE = True
except ImportError:
    HAS_TBPARSE = False

BASE = Path("/srv/mediswarm/live")
app = FastAPI(title="MediSwarm Live Monitor")

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
main { max-width: 1400px; margin: 1.5rem auto; padding: 0 1rem; }
table { width: 100%; border-collapse: collapse; background: var(--card);
        border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
th { background: var(--accent); color: #fff; text-align: left;
     padding: 0.7rem 0.9rem; font-size: 0.82rem; text-transform: uppercase;
     letter-spacing: 0.04em; }
td { padding: 0.65rem 0.9rem; border-bottom: 1px solid var(--border);
     font-size: 0.88rem; vertical-align: top; }
tr:nth-child(even) td { background: #f9fafb; }
tr:hover td { background: #eef2f7; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 0.78rem; font-weight: 600; color: #fff; }
.badge-running { background: var(--green); }
.badge-finished { background: var(--blue); }
.badge-unknown { background: var(--gray); }
.badge-error { background: var(--red); }
.artifact { font-size: 0.8rem; color: var(--text-light); }
.artifact .yes { color: var(--green); font-weight: 600; }
.artifact .no { color: var(--gray); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.links a { margin-right: 0.7rem; font-size: 0.82rem; }
.run-id { font-family: var(--mono); font-size: 0.78rem; word-break: break-all; }
.run-name { font-weight: 500; }
.age-stale { color: var(--orange); }
.age-dead { color: var(--red); }
.empty { text-align: center; padding: 3rem; color: var(--gray); }

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
.card table th { background: var(--accent); }
.kv-table td:first-child { font-weight: 600; white-space: nowrap; width: 160px; }
.btn { display: inline-block; padding: 6px 14px; border-radius: 6px;
       background: var(--accent); color: #fff; font-size: 0.82rem;
       text-decoration: none; margin-right: 0.5rem; }
.btn:hover { background: #16213e; text-decoration: none; }
.chart-container { position: relative; width: 100%; height: 320px; }
.breadcrumb { font-size: 0.85rem; margin-bottom: 1rem; color: var(--text-light); }
.breadcrumb a { color: var(--accent); }
"""


def _status_badge(status: str) -> str:
    cls = "badge-unknown"
    if status == "running":
        cls = "badge-running"
    elif status == "finished":
        cls = "badge-finished"
    elif status in ("error", "failed"):
        cls = "badge-error"
    return f'<span class="badge {cls}">{status}</span>'


def _age_class(age_str: str) -> str:
    """Return a CSS class for stale/dead heartbeats."""
    try:
        secs = int(age_str.rstrip("s"))
    except (ValueError, AttributeError):
        return ""
    if secs > 600:
        return "age-dead"
    if secs > 120:
        return "age-stale"
    return ""


def _html_page(title: str, body: str, *, refresh: int = 0) -> str:
    refresh_tag = (
        f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_tag}
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


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
        return f"{secs // 3600}h {(secs % 3600) // 60}m"
    except Exception:
        return "unknown"


def _read_heartbeat(run_dir: Path) -> dict[str, Any]:
    """Read the best available heartbeat file (prefer final over live)."""
    for name in ["heartbeat_final.json", "heartbeat.json"]:
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


def _find_csv_files(run_dir: Path) -> list[str]:
    """Find class-probability CSV files under run_dir/."""
    rd = run_dir / "run_dir"
    if not rd.exists():
        return []
    return sorted(
        p.name for p in rd.rglob("*_model_gt_and_classprob_*.csv") if p.is_file()
    )


def _find_tb_events(run_dir: Path) -> list[Path]:
    """Find TensorBoard event files under run_dir/."""
    rd = run_dir / "run_dir"
    if not rd.exists():
        return []
    return sorted(rd.rglob("events.out.tfevents*"))


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

                out.append(
                    {
                        "site": site_dir.name,
                        "mode": mode_dir.name,
                        "run_id": run_dir.name,
                        "run_name": hb.get("run_name", ""),
                        "job_id": hb.get("job_id", ""),
                        "status": hb.get("status", "unknown"),
                        "timestamp": ts,
                        "age": age,
                        "has_console": (run_dir / "nohup.out").exists()
                        or (run_dir / "local_training_console_output.txt").exists(),
                        "has_log": (run_dir / "log.txt").exists(),
                        "last_ckpt": bool(hb.get("last_ckpt")),
                        "epoch_ckpt": bool(hb.get("epoch_ckpt")),
                        "global_model": bool(hb.get("global_model")),
                        "best_global_model": bool(hb.get("best_global_model")),
                        "csv_files": _find_csv_files(run_dir),
                        "tb_events": bool(_find_tb_events(run_dir)),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*-\s*(\w+)\s+ACC:\s*([\d.]+),\s*AUC_ROC:\s*([\d.]+)"
)


def parse_console_metrics(text: str) -> dict[str, Any]:
    """Extract epoch-level ACC and AUC_ROC from console output."""
    data: dict[str, dict[int, dict[str, float]]] = {}
    for m in _EPOCH_RE.finditer(text):
        epoch = int(m.group(1))
        phase = m.group(2)  # train / val / test
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
    for name in ["nohup.out", "local_training_console_output.txt"]:
        p = BASE / site / mode / run_id / name
        if p.exists():
            return read_text(p, limit=500_000)
    return ""


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index():
    r = rows()
    now_str = datetime.now().strftime("%H:%M:%S")

    if not r:
        body = f"""
<header>
  <h1>MediSwarm Live Monitor</h1>
  <div class="meta">Refreshed {now_str}
    <a href="/" title="Refresh now">Refresh</a></div>
</header>
<main><div class="empty">No training runs found under {BASE}</div></main>"""
        return _html_page("MediSwarm Monitor", body, refresh=30)

    table_rows = []
    for x in r:
        # Links
        links = []
        detail = f"/detail/{x['site']}/{x['mode']}/{x['run_id']}"
        links.append(f'<a class="btn" href="{detail}">Details</a>')
        links.append(
            f"<a href=\"/heartbeat/{x['site']}/{x['mode']}/{x['run_id']}\">heartbeat</a>"
        )
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
        if x["last_ckpt"]:
            arts.append('<span class="yes">last.ckpt</span>')
        if x["epoch_ckpt"]:
            arts.append('<span class="yes">epoch.ckpt</span>')
        if x["global_model"]:
            arts.append('<span class="yes">FL_global</span>')
        if x["best_global_model"]:
            arts.append('<span class="yes">best_FL</span>')
        if x["csv_files"]:
            arts.append(f'<span class="yes">{len(x["csv_files"])} CSV</span>')
        if x["tb_events"]:
            arts.append('<span class="yes">TFEvents</span>')
        if not arts:
            arts.append('<span class="no">none</span>')

        # Run display
        run_display = ""
        if x["run_name"]:
            run_display = f'<span class="run-name">{x["run_name"]}</span><br>'
        run_display += f'<span class="run-id">{x["run_id"]}</span>'

        age_cls = _age_class(x["age"])
        age_td = (
            f'<span class="{age_cls}">{x["age"]}</span>' if age_cls else x["age"]
        )

        table_rows.append(
            f"""<tr>
  <td>{x['site']}</td>
  <td>{x['mode']}</td>
  <td>{run_display}</td>
  <td>{_status_badge(x['status'])}</td>
  <td>{age_td}</td>
  <td class="artifact">{' &middot; '.join(arts)}</td>
  <td class="links">{' '.join(links)}</td>
</tr>"""
        )

    body = f"""
<header>
  <h1>MediSwarm Live Monitor</h1>
  <div class="meta">Refreshed {now_str}
    <a href="/" title="Refresh now">Refresh</a>
    &middot; <a href="/api/runs">API</a></div>
</header>
<main>
<table>
<thead><tr>
  <th>Site</th><th>Mode</th><th>Run</th><th>Status</th><th>Age</th>
  <th>Artifacts</th><th>Links</th>
</tr></thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</main>"""
    return _html_page("MediSwarm Monitor", body, refresh=30)


# ---------------------------------------------------------------------------
# Detail page
# ---------------------------------------------------------------------------


@app.get("/detail/{site}/{mode}/{run_id}", response_class=HTMLResponse)
def detail(site: str, mode: str, run_id: str):
    run_dir = BASE / site / mode / run_id
    hb = _read_heartbeat(run_dir)
    console_text = _get_console_text(site, mode, run_id)
    metrics = parse_console_metrics(console_text)
    csv_files = _find_csv_files(run_dir)
    has_tb = bool(_find_tb_events(run_dir))

    # Heartbeat info card
    hb_rows = ""
    for key in [
        "site_name",
        "mode",
        "job_id",
        "run_name",
        "timestamp",
        "status",
        "run_dir",
        "last_ckpt",
        "epoch_ckpt",
        "global_model",
        "best_global_model",
        "tb_file",
    ]:
        val = hb.get(key, "")
        if val:
            hb_rows += f"<tr><td>{key}</td><td>{val}</td></tr>\n"
    if not hb_rows:
        hb_rows = '<tr><td colspan="2">No heartbeat data available</td></tr>'

    # CSV links
    csv_links = ""
    if csv_files:
        csv_items = "".join(
            f'<li><a href="/csv/{site}/{mode}/{run_id}/{f}">{f}</a></li>'
            for f in csv_files
        )
        csv_links = f"<ul>{csv_items}</ul>"
    else:
        csv_links = "<p>No CSV result files found.</p>"

    # Console snippet (last 200 lines)
    console_lines = console_text.strip().split("\n")
    console_tail = "\n".join(console_lines[-200:]) if console_lines else "No output."
    # Escape HTML in console output
    console_tail = (
        console_tail.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )

    # Chart section
    chart_html = ""
    if metrics["epochs"]:
        chart_html = f"""
<div class="card" style="grid-column: 1 / -1;">
  <h2>Training Metrics</h2>
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
    pointRadius: 3,
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

    # TensorBoard metrics link
    tb_html = ""
    if has_tb and HAS_TBPARSE:
        tb_html = f"""
<div class="card">
  <h2>TensorBoard Metrics</h2>
  <p>TensorBoard events available.
     <a class="btn" href="/tb_metrics/{site}/{mode}/{run_id}">View raw JSON</a></p>
</div>"""
    elif has_tb:
        tb_html = """
<div class="card">
  <h2>TensorBoard Metrics</h2>
  <p>TensorBoard events found but <code>tbparse</code> is not installed.
     Install with <code>pip install tbparse</code> to enable parsing.</p>
</div>"""

    body = f"""
<header>
  <h1>MediSwarm Live Monitor</h1>
  <div class="meta"><a href="/">Back to overview</a></div>
</header>
<main>
<div class="breadcrumb">
  <a href="/">Home</a> &rsaquo; {site} &rsaquo; {mode} &rsaquo; {run_id}
</div>

<div class="detail-grid">
  <div class="card">
    <h2>Heartbeat</h2>
    <table class="kv-table"><tbody>{hb_rows}</tbody></table>
  </div>

  <div class="card">
    <h2>Artifacts &amp; CSVs</h2>
    {csv_links}
    <div style="margin-top:0.8rem;">
      <a class="btn" href="/heartbeat/{site}/{mode}/{run_id}">Raw heartbeat</a>
      <a class="btn" href="/console/{site}/{mode}/{run_id}">Full console</a>
      {"<a class='btn' href='/log/" + site + "/" + mode + "/" + run_id + "'>Full log</a>" if (run_dir / "log.txt").exists() else ""}
    </div>
  </div>

  {chart_html}
  {tb_html}

  <div class="card" style="grid-column: 1 / -1;">
    <h2>Console Output (last 200 lines)</h2>
    <pre>{console_tail}</pre>
  </div>
</div>
</main>"""
    return _html_page(f"{site}/{mode}/{run_id} — MediSwarm", body)


# ---------------------------------------------------------------------------
# Existing endpoints (preserved)
# ---------------------------------------------------------------------------


@app.get("/heartbeat/{site}/{mode}/{run_id}", response_class=PlainTextResponse)
def heartbeat(site: str, mode: str, run_id: str):
    for name in ["heartbeat_final.json", "heartbeat.json"]:
        p = BASE / site / mode / run_id / name
        if p.exists():
            return read_text(p)
    return ""


@app.get("/console/{site}/{mode}/{run_id}", response_class=PlainTextResponse)
def console(site: str, mode: str, run_id: str):
    return _get_console_text(site, mode, run_id)


@app.get("/log/{site}/{mode}/{run_id}", response_class=PlainTextResponse)
def log(site: str, mode: str, run_id: str):
    return read_text(BASE / site / mode / run_id / "log.txt")


# ---------------------------------------------------------------------------
# New endpoints
# ---------------------------------------------------------------------------


@app.get("/metrics/{site}/{mode}/{run_id}", response_class=JSONResponse)
def metrics(site: str, mode: str, run_id: str):
    """Return parsed training metrics from console output as JSON."""
    text = _get_console_text(site, mode, run_id)
    return parse_console_metrics(text)


@app.get("/tb_metrics/{site}/{mode}/{run_id}", response_class=JSONResponse)
def tb_metrics(site: str, mode: str, run_id: str):
    """Return TensorBoard scalar metrics as JSON (requires tbparse)."""
    if not HAS_TBPARSE:
        return JSONResponse(
            {"error": "tbparse is not installed"}, status_code=501
        )

    run_dir = BASE / site / mode / run_id / "run_dir"
    events = sorted(run_dir.rglob("events.out.tfevents*")) if run_dir.exists() else []
    if not events:
        return {"scalars": []}

    # Parse the directory containing events
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
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/csv/{site}/{mode}/{run_id}/{filename}", response_class=HTMLResponse)
def csv_view(site: str, mode: str, run_id: str, filename: str):
    """Render a CSV file as a styled HTML table."""
    # Sanitize filename to prevent directory traversal
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename or ".." in filename or "/" in filename:
        return HTMLResponse("<p>Invalid filename</p>", status_code=400)

    rd = BASE / site / mode / run_id / "run_dir"
    # Search recursively for the file
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

    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for row in data_rows[:500]:  # limit display to 500 rows
        tds = "".join(f"<td>{cell}</td>" for cell in row)
        trs += f"<tr>{tds}</tr>\n"

    truncated = (
        f"<p><em>Showing first 500 of {len(data_rows)} rows.</em></p>"
        if len(data_rows) > 500
        else ""
    )

    body = f"""
<header>
  <h1>MediSwarm Live Monitor</h1>
  <div class="meta"><a href="/detail/{site}/{mode}/{run_id}">Back to detail</a></div>
</header>
<main>
<div class="breadcrumb">
  <a href="/">Home</a> &rsaquo;
  <a href="/detail/{site}/{mode}/{run_id}">{site}/{mode}/{run_id}</a> &rsaquo;
  {safe_name}
</div>
<div class="card">
  <h2>{safe_name}</h2>
  {truncated}
  <div style="overflow-x:auto;">
  <table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>
  </div>
</div>
</main>"""
    return _html_page(f"{safe_name} — MediSwarm", body)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@app.get("/api/runs", response_class=JSONResponse)
def api_runs():
    """Return all runs as JSON."""
    return rows()


@app.get("/api/metrics/{site}/{mode}/{run_id}", response_class=JSONResponse)
def api_metrics(site: str, mode: str, run_id: str):
    """Return parsed training metrics as JSON (alias for /metrics/)."""
    text = _get_console_text(site, mode, run_id)
    return parse_console_metrics(text)


@app.get("/api/heartbeat/{site}/{mode}/{run_id}", response_class=JSONResponse)
def api_heartbeat(site: str, mode: str, run_id: str):
    """Return heartbeat JSON directly."""
    run_dir = BASE / site / mode / run_id
    return _read_heartbeat(run_dir)
