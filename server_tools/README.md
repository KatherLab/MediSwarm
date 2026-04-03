# Simple version
Run it on your server:

sudo mkdir -p /srv/mediswarm/live
sudo chown -R mediswarm-upload:mediswarm-upload /srv/mediswarm/live

python3 -m venv /srv/mediswarm/venv
/srv/mediswarm/venv/bin/pip install fastapi uvicorn

cp server_tools/app.py /srv/mediswarm/app.py
/srv/mediswarm/venv/bin/uvicorn app:app --app-dir /srv/mediswarm --host 0.0.0.0 --port 8080

Then open:
http://172.24.4.65:8080/

# Detailed version: MediSwarm Live Monitor: installation and usage

This document explains how to install and run the FastAPI-based monitor for uploaded MediSwarm logs and artifacts.

It assumes uploaded files are stored under:

```text
/srv/mediswarm/live
```

and that the application code is saved as:

```text
/srv/mediswarm/app.py
```

---

## 1) What this monitor does

The monitor scans the uploaded artifact tree:

```text
/srv/mediswarm/live/<SITE_NAME>/<MODE>/<RUN_OR_JOB_ID>/
```

and shows:

- site name
- mode (`local` or `swarm`)
- run or job ID
- latest heartbeat status
- heartbeat age
- whether console/log files exist
- whether key artifacts exist:
    - `last.ckpt`
    - `epoch=*.ckpt`
    - `FL_global_model.pt`

It also provides links to view:

- `heartbeat.json`
- `nohup.out` or `local_training_console_output.txt`
- `log.txt`

---

## 2) Save the application file

Save your Python code to:

```bash
sudo mkdir -p /srv/mediswarm
sudo nano /srv/mediswarm/app.py
```

Paste the FastAPI code into that file and save it.

---

## 3) Prepare the upload directory

Make sure the upload directory exists and is readable by the web app:

```bash
sudo mkdir -p /srv/mediswarm/live
sudo chown -R mediswarm-upload:mediswarm-upload /srv/mediswarm/live
sudo chmod -R 775 /srv/mediswarm/live
```

You can verify the directory exists with:

```bash
ls -ld /srv/mediswarm/live
```

---

## 4) Create a Python virtual environment

Install a virtual environment for the monitor:

```bash
python3 -m venv /srv/mediswarm/venv
```

Activate it:

```bash
source /srv/mediswarm/venv/bin/activate
```

Upgrade `pip`:

```bash
pip install --upgrade pip
```

Install dependencies:

```bash
pip install fastapi uvicorn
```

---

## 5) Run the monitor manually

Start the FastAPI server with:

```bash
/srv/mediswarm/venv/bin/uvicorn app:app --app-dir /srv/mediswarm --host 0.0.0.0 --port 8080
```

If it starts successfully, open:

```text
http://172.24.4.65:8080/
```

If you are testing on the server itself, you can also open:

```text
http://localhost:8080/
```

---

## 6) What you should see

The main page shows a table with columns:

- Site
- Mode
- Run
- Status
- Timestamp
- Age
- Artifacts
- Links

Typical entries look like:

- `MHA_1`
- `swarm`
- `db98789c-746b-4be3-a1b6-c50473b42ed8`

The links open:
- heartbeat JSON
- console output
- log output

---

## 7) Uploaded directory layout expected by the monitor

The `live_sync` daemon on each training site uploads artifacts here via rsync.
All training jobs (ODELIA and challenge models) write results to `$SCRATCHDIR/runs/$SITE_NAME/<RUN_NAME>/` on the host,
and `live_sync` uploads them to the `run_dir/` subdirectory below.

The monitor expects uploads in this structure:

```text
/srv/mediswarm/live/MHA_1/swarm/db98789c-746b-4be3-a1b6-c50473b42ed8/
    heartbeat.json
    heartbeat_final.json
    nohup.out
    log.txt
    FL_global_model.pt
    best_FL_global_model.pt
    run_dir/
```

and for local runs:

```text
/srv/mediswarm/live/MHA_1/local/<RUN_NAME>/
    heartbeat.json
    heartbeat_final.json
    local_training_console_output.txt
    run_dir/
```

---

## 8) Test with uploaded files

To confirm the monitor is reading the directory correctly:

```bash
find /srv/mediswarm/live -maxdepth 5 | sort | head -200
```

Then restart the monitor and refresh the page.

---

## 9) Run it in the background

A simple way to keep it running in the background:

```bash
nohup /srv/mediswarm/venv/bin/uvicorn app:app --app-dir /srv/mediswarm --host 0.0.0.0 --port 8080 > /srv/mediswarm/monitor.log 2>&1 &
```

Check that it is running:

```bash
ps aux | grep uvicorn
```

Read the log:

```bash
tail -f /srv/mediswarm/monitor.log
```

---

## 10) Recommended systemd service

For a cleaner deployment, create a systemd service.

Create:

```bash
sudo nano /etc/systemd/system/mediswarm-monitor.service
```

Paste:

```ini
[Unit]
Description=MediSwarm Live Monitor
After=network.target

[Service]
User=jeff
Group=jeff
WorkingDirectory=/srv/mediswarm
ExecStart=/srv/mediswarm/venv/bin/uvicorn app:app --app-dir /srv/mediswarm --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Reload systemd:

```bash
sudo systemctl daemon-reload
```

Enable the service:

```bash
sudo systemctl enable mediswarm-monitor
```

Start the service:

```bash
sudo systemctl start mediswarm-monitor
```

Check status:

```bash
sudo systemctl status mediswarm-monitor
```

Read logs:

```bash
journalctl -u mediswarm-monitor -f
```

---

## 11) Firewall / network

If the page does not open from another machine, make sure port `8080` is allowed.

For example, with UFW:

```bash
sudo ufw allow 8080/tcp
```

Then verify from another machine:

```bash
curl http://172.24.4.65:8080/
```

---

## 12) How to update the app later

If you change `/srv/mediswarm/app.py`:

### If running manually
Stop the old process and start it again.

### If running as a systemd service
Restart it with:

```bash
sudo systemctl restart mediswarm-monitor
```

---

## 13) Troubleshooting

### The page opens but shows no rows

Check that files actually exist:

```bash
find /srv/mediswarm/live -maxdepth 6 -type f | sort | head -200
```

### The app does not start

Check Python dependencies:

```bash
source /srv/mediswarm/venv/bin/activate
python -c "import fastapi, uvicorn; print('ok')"
```

### Permission denied reading files

Check ownership and permissions:

```bash
ls -ld /srv/mediswarm /srv/mediswarm/live
find /srv/mediswarm/live -maxdepth 3 -ls | head -50
```

### Port already in use

Run on a different port, for example `8090`:

```bash
/srv/mediswarm/venv/bin/uvicorn app:app --app-dir /srv/mediswarm --host 0.0.0.0 --port 8090
```

---

## 14) Minimal install summary

```bash
sudo mkdir -p /srv/mediswarm/live
sudo chown -R mediswarm-upload:mediswarm-upload /srv/mediswarm/live
sudo chmod -R 775 /srv/mediswarm/live

python3 -m venv /srv/mediswarm/venv
source /srv/mediswarm/venv/bin/activate
pip install fastapi uvicorn

/srv/mediswarm/venv/bin/uvicorn app:app --app-dir /srv/mediswarm --host 0.0.0.0 --port 8080
```

Then open:

```text
http://172.24.4.65:8080/
```
