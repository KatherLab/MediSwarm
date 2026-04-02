from pathlib import Path
import json
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse

BASE = Path("/srv/mediswarm/live")
app = FastAPI()

def read_text(p: Path, limit: int = 50000) -> str:
    if not p.exists():
        return ""
    return p.read_text(errors="replace")[-limit:]

def parse_age(ts: str) -> str:
    if not ts:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        secs = int((datetime.now(timezone.utc) - dt).total_seconds())
        return f"{secs}s"
    except Exception:
        return "unknown"

def rows():
    out = []
    if not BASE.exists():
        return out

    for site_dir in sorted([p for p in BASE.iterdir() if p.is_dir()]):
        for mode_dir in sorted([p for p in site_dir.iterdir() if p.is_dir()]):
            for run_dir in sorted([p for p in mode_dir.iterdir() if p.is_dir()]):
                hb = {}
                for hb_name in ["heartbeat.json", "heartbeat_final.json"]:
                    hb_path = run_dir / hb_name
                    if hb_path.exists():
                        try:
                            hb = json.loads(hb_path.read_text())
                            break
                        except Exception:
                            pass

                out.append({
                    "site": site_dir.name,
                    "mode": mode_dir.name,
                    "run_id": run_dir.name,
                    "status": hb.get("status", "unknown"),
                    "timestamp": hb.get("timestamp", ""),
                    "age": parse_age(hb.get("timestamp", "")),
                    "has_console": (run_dir / "nohup.out").exists() or (run_dir / "local_training_console_output.txt").exists(),
                    "has_log": (run_dir / "log.txt").exists(),
                    "last_ckpt": hb.get("last_ckpt", ""),
                    "epoch_ckpt": hb.get("epoch_ckpt", ""),
                    "global_model": hb.get("global_model", ""),
                })
    return out

@app.get("/", response_class=HTMLResponse)
def index():
    r = rows()
    html = [
        "<html><body>",
        "<h1>MediSwarm Live Monitor</h1>",
        "<table border='1' cellpadding='6' cellspacing='0'>",
        "<tr><th>Site</th><th>Mode</th><th>Run</th><th>Status</th><th>Timestamp</th><th>Age</th><th>Artifacts</th><th>Links</th></tr>",
    ]
    for x in r:
        links = [
            f"<a href='/heartbeat/{x['site']}/{x['mode']}/{x['run_id']}'>heartbeat</a>"
        ]
        if x["mode"] == "swarm":
            links.append(f"<a href='/console/{x['site']}/{x['mode']}/{x['run_id']}'>nohup</a>")
            links.append(f"<a href='/log/{x['site']}/{x['mode']}/{x['run_id']}'>log</a>")
        else:
            links.append(f"<a href='/console/{x['site']}/{x['mode']}/{x['run_id']}'>local_console</a>")

        artifacts = (
            f"last.ckpt={bool(x['last_ckpt'])}<br>"
            f"epoch.ckpt={bool(x['epoch_ckpt'])}<br>"
            f"FL_global_model.pt={bool(x['global_model'])}"
        )

        html.append(
            f"<tr><td>{x['site']}</td><td>{x['mode']}</td><td>{x['run_id']}</td>"
            f"<td>{x['status']}</td><td>{x['timestamp']}</td><td>{x['age']}</td>"
            f"<td>{artifacts}</td><td>{' | '.join(links)}</td></tr>"
        )
    html.append("</table></body></html>")
    return "".join(html)

@app.get("/heartbeat/{site}/{mode}/{run_id}", response_class=PlainTextResponse)
def heartbeat(site: str, mode: str, run_id: str):
    for name in ["heartbeat.json", "heartbeat_final.json"]:
        p = BASE / site / mode / run_id / name
        if p.exists():
            return read_text(p)
    return ""

@app.get("/console/{site}/{mode}/{run_id}", response_class=PlainTextResponse)
def console(site: str, mode: str, run_id: str):
    p1 = BASE / site / mode / run_id / "nohup.out"
    p2 = BASE / site / mode / run_id / "local_training_console_output.txt"
    if p1.exists():
        return read_text(p1)
    if p2.exists():
        return read_text(p2)
    return ""

@app.get("/log/{site}/{mode}/{run_id}", response_class=PlainTextResponse)
def log(site: str, mode: str, run_id: str):
    return read_text(BASE / site / mode / run_id / "log.txt")