#!/usr/bin/env bash
# =============================================================================
# MediSwarm VPN Health Monitor
# =============================================================================
# Checks the GoodAccess tunnel of mediswarm-vpn.service and restarts the service when that can help,
# logging why the tunnel is down (the last OpenVPN error from the journal) before each restart.
#
# Usage:
#   ./vpn_health_monitor.sh                  # run once (what the systemd timer does)
#   ./vpn_health_monitor.sh --daemon         # run in a loop (every PING_INTERVAL seconds)
#   ./vpn_health_monitor.sh --install-timer  # install a systemd timer that runs every 30 seconds
#
# Rules (the old monitor restarted the service whenever tun0 had no address, every 30 s, so a
# connection that needed longer never completed, and it judged tun0 although that can belong to
# another tunnel, #620):
#   - does nothing while the service is disabled: a site that switched it off keeps its own tunnel;
#   - does nothing when another tunnel (NetworkManager, hand-started) holds the VPN address and the
#     service is not running: a second session with the same account cannot work;
#   - never restarts a connection attempt younger than CONNECT_GRACE seconds (state "activating":
#     OpenVPN has not completed its handshake; systemd's TimeoutStartSec ends a stuck attempt);
#   - a missing address or an unreachable gateway must persist for FAIL_THRESHOLD checks;
#   - gateway pings count only after the gateway answered once since the service started, so a
#     gateway that never answers ICMP does not cause restarts;
#   - consecutive restarts that do not bring the tunnel back are spaced BACKOFF_BASE * 2^n seconds
#     apart, at most BACKOFF_MAX.
#
# Environment variables (all optional):
#   VPN_GATEWAY      IP to ping                                   (default: 172.24.4.1)
#   VPN_SERVICE      systemd service name                         (default: mediswarm-vpn)
#   FAIL_THRESHOLD   consecutive bad checks before a restart      (default: 3)
#   CONNECT_GRACE    seconds a connection attempt may take        (default: 180)
#   BACKOFF_BASE     first spacing between restarts, seconds      (default: 60)
#   BACKOFF_MAX      largest spacing between restarts, seconds    (default: 1800)
#   PING_INTERVAL    seconds between checks in daemon mode        (default: 60)
#   STATE_FILE       state of the monitor                         (default: /run/mediswarm-vpn-health.state)
#   VPN_IFACE        ignored (the tunnel is found by its 172.24.4.x address); kept so timer units
#                    installed by older versions still start
# =============================================================================

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mediswarm_vpn_lib.sh
source "$HERE/mediswarm_vpn_lib.sh"

VPN_GATEWAY="${VPN_GATEWAY:-172.24.4.1}"
VPN_SERVICE="${VPN_SERVICE:-mediswarm-vpn}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"
CONNECT_GRACE="${CONNECT_GRACE:-180}"
BACKOFF_BASE="${BACKOFF_BASE:-60}"
BACKOFF_MAX="${BACKOFF_MAX:-1800}"
PING_INTERVAL="${PING_INTERVAL:-60}"
STATE_FILE="${STATE_FILE:-/run/mediswarm-vpn-health.state}"
# shellcheck disable=SC2034  # read by the library functions
MSVPN_UNIT="$VPN_SERVICE"

log() {
    local level="$1"; shift
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $*"
    logger -t mediswarm-vpn-health -p "daemon.$level" "$*" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# State: consecutive bad checks, consecutive restarts, time of the last restart, the service
# invocation in which the gateway last answered a ping, and the last reported status (status lines
# are logged only when they change, so a site's journal is not flooded every 30 s).
# ---------------------------------------------------------------------------
load_state() {
    failures=0; restarts=0; last_restart=0; ping_ok_invocation=""; last_status=""
    [[ -r "$STATE_FILE" ]] || return 0
    local k v
    while IFS='=' read -r k v; do
        case "$k" in
            failures|restarts|last_restart) [[ "$v" =~ ^[0-9]+$ ]] && printf -v "$k" '%s' "$v" ;;
            ping_ok_invocation|last_status) printf -v "$k" '%s' "$v" ;;
        esac
    done < "$STATE_FILE"
}

save_state() {
    local tmp="$STATE_FILE.$$"
    printf 'failures=%s\nrestarts=%s\nlast_restart=%s\nping_ok_invocation=%s\nlast_status=%s\n' \
        "$failures" "$restarts" "$last_restart" "$ping_ok_invocation" "$last_status" > "$tmp" 2>/dev/null \
        && mv -f "$tmp" "$STATE_FILE" 2>/dev/null || rm -f "$tmp" 2>/dev/null
}

status() {   # status <key> <level> <message>: logs only when the status key changes
    if [[ "$1" != "$last_status" ]]; then
        log "$2" "$3"
        last_status="$1"
    fi
}

# Restart the service unless the backoff says to wait. Logs why, with OpenVPN's last error.
restart_service() {
    local why="$1" now wait verdict start age
    now="$(date +%s)"
    wait=$(( BACKOFF_BASE * (1 << (restarts < 10 ? restarts : 10)) ))
    (( wait > BACKOFF_MAX )) && wait="$BACKOFF_MAX"
    if (( restarts > 0 && now - last_restart < wait )); then
        status "backoff-$restarts" info "$why; $restarts restart(s) did not help, next restart in $(( wait - (now - last_restart) ))s at the earliest"
        return 0
    fi
    # OpenVPN's messages since the current attempt started (or since our last restart).
    age="$(msvpn_service_age_s "$VPN_SERVICE")"
    start=$(( now - ${age:-600} ))
    (( restarts == 0 )) || start="$last_restart"
    verdict="$(msvpn_journal_since "$start" "$VPN_SERVICE" | msvpn_classify_log)"
    log err "restarting $VPN_SERVICE: $why. OpenVPN: $(cut -f2 <<< "$verdict") [$(cut -f3 <<< "$verdict" | cut -c1-200)]"
    if systemctl restart --no-block "$VPN_SERVICE"; then
        restarts=$(( restarts + 1 )); last_restart="$now"; failures=0; last_status="restarted"
    else
        log err "Failed to restart $VPN_SERVICE"
    fi
}

# ---------------------------------------------------------------------------
# Single health check
# ---------------------------------------------------------------------------
check_vpn() {
    load_state
    local enabled active age addrs inv
    enabled="$(systemctl is-enabled "$VPN_SERVICE" 2>/dev/null || true)"
    if [[ "$enabled" != enabled* ]]; then
        status disabled info "$VPN_SERVICE is ${enabled:-not installed}: not monitoring it (enable it with: sudo systemctl enable --now $VPN_SERVICE)"
        failures=0; save_state; return 0
    fi
    active="$(msvpn_unit_prop ActiveState "$VPN_SERVICE")"
    age="$(msvpn_service_age_s "$VPN_SERVICE")"
    addrs="$(msvpn_vpn_addrs | tr '\n' ' ' | sed 's/ $//')"

    if [[ -n "$addrs" ]]; then
        if [[ "$active" != active && "$active" != reloading ]]; then
            status other info "VPN address ($addrs) is held by another tunnel and $VPN_SERVICE is $active: leaving both alone (one session per GoodAccess account)"
            failures=0; restarts=0; save_state; return 0
        fi
        inv="$(msvpn_unit_prop InvocationID "$VPN_SERVICE")"
        if ping -c 2 -W 5 "$VPN_GATEWAY" >/dev/null 2>&1; then
            (( failures > 0 )) && log info "VPN gateway $VPN_GATEWAY reachable again (was at $failures consecutive failures)"
            status ok info "VPN up ($addrs), gateway $VPN_GATEWAY answers"
            failures=0; restarts=0; ping_ok_invocation="$inv"; save_state; return 0
        fi
        if [[ "$ping_ok_invocation" != "$inv" ]]; then
            status noping info "VPN up ($addrs); gateway $VPN_GATEWAY has not answered a ping since $VPN_SERVICE started, so pings are not used to judge the tunnel"
            failures=0; restarts=0; save_state; return 0
        fi
        failures=$(( failures + 1 ))
        status pingfail warning "VPN gateway $VPN_GATEWAY unreachable (restart after $FAIL_THRESHOLD consecutive failed checks)"
        (( failures >= FAIL_THRESHOLD )) && restart_service "gateway $VPN_GATEWAY unreachable for $failures checks"
        save_state; return 0
    fi

    # No VPN address on this host.
    if [[ "$active" == activating || "$active" == active ]] && [[ -n "$age" ]] && (( age < CONNECT_GRACE )); then
        status "connecting" info "$VPN_SERVICE is connecting (${age}s, grace ${CONNECT_GRACE}s)"
        failures=0; save_state; return 0
    fi
    failures=$(( failures + 1 ))
    status "noaddr-$active" warning "no VPN address; $VPN_SERVICE is ${active:-unknown}$( [[ -n "$age" ]] && echo ", process started ${age}s ago") (restart after $FAIL_THRESHOLD consecutive checks)"
    (( failures >= FAIL_THRESHOLD )) && restart_service "no VPN address for $failures checks"
    save_state
}

# ---------------------------------------------------------------------------
# Install a systemd timer for this script. The script and its library are copied to
# /usr/local/lib/mediswarm-vpn/ first, so the timer survives a replaced or deleted kit folder.
# ---------------------------------------------------------------------------
install_timer() {
    log info "Installing systemd timer for VPN health monitor..."
    if [[ "$HERE" != "$MSVPN_LIBEXEC" ]]; then
        sudo mkdir -p "$MSVPN_LIBEXEC"
        local f
        for f in "${MSVPN_RUNTIME_FILES[@]}"; do
            sudo install -m 755 "$HERE/$f" "$MSVPN_LIBEXEC/$f"
        done
    fi
    # Only variables set explicitly at install time are pinned in the unit; the rest follow the
    # script's defaults.
    local env_lines="" var
    for var in VPN_GATEWAY VPN_SERVICE FAIL_THRESHOLD CONNECT_GRACE BACKOFF_BASE BACKOFF_MAX; do
        [[ -n "$(printenv "$var" 2>/dev/null)" ]] && env_lines+="Environment=$var=${!var}"$'\n'
    done

    sudo tee /etc/systemd/system/mediswarm-vpn-health.service >/dev/null <<EOF
[Unit]
Description=MediSwarm VPN Health Check
After=network-online.target

[Service]
Type=oneshot
ExecStart=$MSVPN_LIBEXEC/vpn_health_monitor.sh
${env_lines}
EOF

    sudo tee /etc/systemd/system/mediswarm-vpn-health.timer >/dev/null <<EOF
[Unit]
Description=Run MediSwarm VPN health check every 30 seconds

[Timer]
OnBootSec=120
OnUnitActiveSec=30
AccuracySec=5s

[Install]
WantedBy=timers.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable --now mediswarm-vpn-health.timer
    log info "Timer installed and started. Check with: systemctl status mediswarm-vpn-health.timer; journalctl -t mediswarm-vpn-health"
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
