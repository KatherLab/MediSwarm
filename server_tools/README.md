Run it on your server:

sudo mkdir -p /srv/mediswarm/live
sudo chown -R mediswarm-upload:mediswarm-upload /srv/mediswarm/live

python3 -m venv /srv/mediswarm/venv
/srv/mediswarm/venv/bin/pip install fastapi uvicorn

cp server_tools/app.py /srv/mediswarm/app.py
/srv/mediswarm/venv/bin/uvicorn app:app --app-dir /srv/mediswarm --host 0.0.0.0 --port 8080