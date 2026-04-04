#!/usr/bin/env bash
# =============================================================================
# MediSwarm VPN Health Monitor
# =============================================================================
# Pings the VPN gateway at regular intervals and restarts the systemd VPN
# service after consecutive failures.
#
# Usage:
#   ./vpn_health_monitor.sh                  # run once (suitable for cron/systemd timer)
#   ./vpn_health_monitor.sh --daemon         # run in a loop (every INTERVAL seconds)
#   ./vpn_health_monitor.sh --install-timer  # install a systemd timer that runs every minute
#
# Environment variables (all optional):
#   VPN_GATEWAY        IP to ping              (default: 172.24.4.1)
#   VPN_SERVICE        systemd service name    (default: mediswarm-vpn)
#   FAIL_THRESHOLD     consecutive fails before restart (default: 3)
#   PING_INTERVAL      seconds between checks in daemon mode (default: 60)
#   STATE_FILE         path for failure counter (default: /tmp/mediswarm_vpn_health.state)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment)
# ---------------------------------------------------------------------------
VPN_GATEWAY="${VPN_GATEWAY:-172.24.4.1}"
VPN_SERVICE="${VPN_SERVICE:-mediswarm-vpn}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"
PING_INTERVAL="${PING_INTERVAL:-60}"
STATE_FILE="${STATE_FILE:-/tmp/mediswarm_vpn_health.state}"

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
log() {
    local level="$1"; shift
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $*"
    logger -t mediswarm-vpn-health -p "daemon.$level" "$*" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Read / write failure counter
# ---------------------------------------------------------------------------
read_failures() {
    if [[ -f "$STATE_FILE" ]]; then
        cat "$STATE_FILE"
    else
        echo 0
    fi
}

write_failures() {
    echo "$1" > "$STATE_FILE"
}

# ---------------------------------------------------------------------------
# Single health check
# ---------------------------------------------------------------------------
check_vpn() {
    local failures
    failures=$(read_failures)

    if ping -c 2 -W 5 "$VPN_GATEWAY" &>/dev/null; then
        if (( failures > 0 )); then
            log info "VPN gateway $VPN_GATEWAY reachable again (was at $failures consecutive failures)"
        fi
        write_failures 0
        return 0
    fi

    failures=$((failures + 1))
    write_failures "$failures"
    log warning "VPN gateway $VPN_GATEWAY unreachable ($failures/$FAIL_THRESHOLD consecutive failures)"

    if (( failures >= FAIL_THRESHOLD )); then
        log err "Failure threshold reached — restarting $VPN_SERVICE"
        if sudo systemctl restart "$VPN_SERVICE"; then
            log info "$VPN_SERVICE restarted successfully"
        else
            log err "Failed to restart $VPN_SERVICE (exit code $?)"
        fi
        write_failures 0
    fi
}

# ---------------------------------------------------------------------------
# Install a systemd timer for this script
# ---------------------------------------------------------------------------
install_timer() {
    local script_path
    script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

    log info "Installing systemd timer for VPN health monitor..."

    sudo tee /etc/systemd/system/mediswarm-vpn-health.service >/dev/null <<EOF
[Unit]
Description=MediSwarm VPN Health Check
After=network-online.target

[Service]
Type=oneshot
ExecStart=$script_path
Environment=VPN_GATEWAY=$VPN_GATEWAY
Environment=VPN_SERVICE=$VPN_SERVICE
Environment=FAIL_THRESHOLD=$FAIL_THRESHOLD
EOF

    sudo tee /etc/systemd/system/mediswarm-vpn-health.timer >/dev/null <<EOF
[Unit]
Description=Run MediSwarm VPN health check every minute

[Timer]
OnBootSec=120
OnUnitActiveSec=60
AccuracySec=5s

[Install]
WantedBy=timers.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable --now mediswarm-vpn-health.timer
    log info "Timer installed and started. Check with: systemctl status mediswarm-vpn-health.timer"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    case "${1:-}" in
        --daemon)
            log info "Starting VPN health monitor daemon (gateway=$VPN_GATEWAY, interval=${PING_INTERVAL}s, threshold=$FAIL_THRESHOLD)"
            while true; do
                check_vpn
                sleep "$PING_INTERVAL"
            done
            ;;
        --install-timer)
            install_timer
            ;;
        *)
            check_vpn
            ;;
    esac
}

main "$@"
