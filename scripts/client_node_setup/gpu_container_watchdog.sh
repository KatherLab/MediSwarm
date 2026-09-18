#!/usr/bin/env bash
# =============================================================================
# MediSwarm GPU-container watchdog (#343)
# =============================================================================
# Restart a swarm client container that has LOST GPU access -- but only when it
# is NOT running a training job.
#
# Background: on cgroup v2 + Docker's `systemd` cgroup driver, a daily
# `systemctl daemon-reload` (e.g. from apt-daily-upgrade) strips the GPU from an
# already-running container. The client's HEALTHCHECK (`nvidia-smi`) then marks
# the container `unhealthy`; a job that already grabbed the GPU keeps running on
# its open device handles, but newly started jobs fail (`NVML: Unknown Error`).
#
# The durable fix is the cgroupfs driver
# (scripts/client_node_setup/fix_docker_cgroupfs.sh). This watchdog is the
# fallback for hosts that stay on the systemd driver: it auto-recovers an *idle*
# GPU-less container (a `docker restart` re-applies the device cgroup and
# restores the GPU) without disrupting a running job.
#
# Decision each tick:
#   healthy / starting / no healthcheck      -> nothing to do
#   unhealthy + GPU compute process running  -> a job grabbed the GPU before the
#                                               loss; LEAVE IT (restart = lost job)
#   unhealthy + no GPU compute process        -> idle; `docker restart` it
#
# Usage:
#   gpu_container_watchdog.sh check        # one tick (used by the systemd timer)
#   gpu_container_watchdog.sh install      # install a 2-min systemd timer (run as root)
#   gpu_container_watchdog.sh uninstall
#
# Environment (optional):
#   CLIENT_CONTAINER   container name (default: first running container matching 'client')
# =============================================================================

set -uo pipefail

CLIENT_CONTAINER="${CLIENT_CONTAINER:-}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [gpu-watchdog] $*"; logger -t mediswarm-gpu-watchdog "$*" 2>/dev/null || true; }

resolve_container() {
    if [ -n "$CLIENT_CONTAINER" ]; then echo "$CLIENT_CONTAINER"; return; fi
    docker ps --format '{{.Names}}' 2>/dev/null | grep -i client | head -1
}

# number of processes currently using the GPU(s) on this host
gpu_compute_app_count() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo 0
}

check() {
    local container; container="$(resolve_container)"
    if [ -z "$container" ]; then
        log "no client container running -- nothing to do"
        return 0
    fi

    local health
    health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container" 2>/dev/null)"
    if [ "$health" != "unhealthy" ]; then
        return 0  # healthy / starting / no healthcheck
    fi

    local apps; apps="$(gpu_compute_app_count)"
    if [ "${apps:-0}" -gt 0 ]; then
        log "container $container is GPU-unhealthy but $apps GPU process(es) are running -- a job is active; NOT restarting"
        return 0
    fi

    log "container $container lost its GPU and no job is running -- restarting it"
    if docker restart "$container" >/dev/null 2>&1; then
        log "restarted $container (a cgroupfs driver would prevent the loss: see fix_docker_cgroupfs.sh)"
    else
        log "ERROR: failed to restart $container"
    fi
}

install_timer() {
    local script_path; script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    log "installing systemd timer for the GPU-container watchdog..."
    sudo tee /etc/systemd/system/mediswarm-gpu-watchdog.service >/dev/null <<EOF
[Unit]
Description=MediSwarm GPU-container watchdog (restart a GPU-less idle client)
After=docker.service
Wants=docker.service

[Service]
Type=oneshot
ExecStart=$script_path check
EOF
    sudo tee /etc/systemd/system/mediswarm-gpu-watchdog.timer >/dev/null <<EOF
[Unit]
Description=Run the MediSwarm GPU-container watchdog every 2 minutes

[Timer]
OnBootSec=180
OnUnitActiveSec=120
AccuracySec=10s

[Install]
WantedBy=timers.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable --now mediswarm-gpu-watchdog.timer
    log "installed. Check with: systemctl status mediswarm-gpu-watchdog.timer"
}

uninstall_timer() {
    sudo systemctl disable --now mediswarm-gpu-watchdog.timer 2>/dev/null || true
    sudo rm -f /etc/systemd/system/mediswarm-gpu-watchdog.service /etc/systemd/system/mediswarm-gpu-watchdog.timer
    sudo systemctl daemon-reload
    log "uninstalled."
}

case "${1:-check}" in
    check)     check ;;
    install)   install_timer ;;
    uninstall) uninstall_timer ;;
    *) echo "Usage: $0 {check|install|uninstall}"; exit 1 ;;
esac
