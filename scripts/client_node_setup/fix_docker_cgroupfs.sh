#!/usr/bin/env bash
# Fix the recurring "Failed to initialize NVML: Unknown Error" that strips the GPU
# from already-running containers after a `systemctl daemon-reload` (e.g. the daily
# apt auto-upgrade) on hosts using Docker's *systemd* cgroup driver on cgroup v2.
#
# It switches Docker to the cgroupfs cgroup driver (daemon-reload-safe). This
# restarts the Docker daemon, which briefly bounces ALL containers on the host
# (they recover per their restart policy). The change auto-reverts if Docker fails
# to come back. See docs/SWARM_FAILURE_MODES.md (failure mode F2).
#
#   Run as root:   sudo bash fix_docker_cgroupfs.sh
set -u

if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root, e.g.:  sudo bash $0"
    exit 1
fi

D=/etc/docker/daemon.json

echo "[1/4] Current Docker cgroup driver…"
cur=$(docker info 2>/dev/null | sed -n 's/.*Cgroup Driver: //p' | head -1)
echo "      driver = ${cur:-unknown}"
if [ "$cur" = "cgroupfs" ]; then
    echo "      Already on cgroupfs — nothing to do."
    exit 0
fi

echo "[2/4] Backing up and updating $D (merging native.cgroupdriver=cgroupfs)…"
BK=""
if [ -f "$D" ]; then BK="$D.bak.$(date +%s)"; cp -a "$D" "$BK"; echo "      backup -> $BK"; fi
python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"; cfg = {}
if os.path.exists(p):
    try: cfg = json.load(open(p))
    except Exception: cfg = {}
opts = [o for o in cfg.get("exec-opts", []) if not o.startswith("native.cgroupdriver=")]
opts.append("native.cgroupdriver=cgroupfs"); cfg["exec-opts"] = opts
json.dump(cfg, open(p, "w"), indent=4)
print("      new daemon.json:")
print(open(p).read())
PY

echo "[3/4] Restarting Docker (briefly bounces all containers on this host)…"
systemctl restart docker; sleep 10
if ! docker info >/dev/null 2>&1; then
    echo "[!!] Docker did not come back — reverting daemon.json."
    if [ -n "$BK" ]; then cp -a "$BK" "$D"; else rm -f "$D"; fi
    systemctl restart docker; sleep 8
    docker info >/dev/null 2>&1 && echo "      Reverted; Docker is back up (still on the old driver)." \
                                || echo "      Docker STILL DOWN — investigate manually."
    exit 2
fi
echo "      Docker OK. driver = $(docker info 2>/dev/null | sed -n 's/.*Cgroup Driver: //p' | head -1)"

echo "[4/4] Validating that a daemon-reload no longer strips the GPU…"
C=$(docker ps --filter name=odelia_swarm_client --format '{{.Names}}' | head -1)
if [ -n "$C" ]; then
    systemctl daemon-reload; sleep 3
    if docker exec "$C" nvidia-smi -L >/dev/null 2>&1; then
        echo "      PASS: GPU still visible in $C after daemon-reload."
    else
        echo "      NOTE: GPU not visible in $C now — recreate the client:"
        echo "            docker rm -f $C ; remove the kit's daemon_pid.fl ; ./docker.sh --start_client"
    fi
else
    echo "      (No running odelia_swarm_client container to validate against — that's fine.)"
fi

echo
echo "Done. The cgroupfs change persists across reboots; the swarm client does NOT —"
echo "after a host reboot, relaunch it with ./docker.sh --start_client (clear daemon_pid.fl first)."
