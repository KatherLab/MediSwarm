#!/usr/bin/env bash
# =============================================================================
# MediSwarm VPN Tunnel Setup
# =============================================================================
# Sets up and launches an OpenVPN connection using GoodAccess .ovpn configs.
#
# Modes:
#   Legacy (default):  Launches OpenVPN via nohup in the background.
#   Systemd (-s):      Installs the mediswarm-vpn systemd service (auto-reconnect, boot persistence),
#                      starts it and waits until the tunnel has its 172.24.4.x address. If it does not
#                      come up, prints why (from OpenVPN's log) and exits non-zero.
#   Diagnose:          --diagnose prints the state of tunnels, service and configuration for the
#                      swarm operator. It changes nothing.
#
# Usage:
#   setup_vpntunnel.sh -d <host_index> | -c <file.ovpn> [-n] [-s]
#   setup_vpntunnel.sh --diagnose
#
#   -d <host_index>  One of: TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich.
#                     The config is assets/openvpn_configs/good_access/<host_index>.ovpn.
#   -c <file.ovpn>   Use this .ovpn file instead (e.g. when running from a startup kit).
#   -n               First-time setup: installs OpenVPN and prompts for VPN credentials
#   -s               Use systemd service mode (recommended for production).
#   --diagnose       Read-only report: tunnels, service state, config checks, OpenVPN's last error.
#   -h               Show this help message
#
# Only one GoodAccess tunnel per account can work at a time. Both modes refuse to start a tunnel
# while another one (NetworkManager, hand-started, the service) already holds a 172.24.4.x address.
#
# Exit status: 0 = tunnel up (or legacy mode started), 1 = usage or configuration error,
#   3 = another tunnel is already up (nothing started), 4 = OpenVPN failed (reason printed, service
#   stopped and disabled), 5 = no address within VPN_WAIT_SECONDS (default 90; the service keeps trying).
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
# shellcheck source=mediswarm_vpn_lib.sh
source "$SCRIPT_DIR/mediswarm_vpn_lib.sh"

ACTION="nochange"
SYSTEMD_MODE=""
OVPN_FILE=""
host_index=""
ORIG_ARGS=("$@")

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d)  host_index="$2"; shift ;;
        -c)  OVPN_FILE="$2"; shift ;;
        -n)  ACTION="new";;
        -s)  SYSTEMD_MODE="1";;
        --diagnose) ACTION="diagnose";;
        -h|--help)  ACTION="help";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ $ACTION = "help" ]]; then
   echo "Usage: setup_vpntunnel.sh -d <host_index> | -c <file.ovpn> [-n] [-s]"
   echo "       setup_vpntunnel.sh --diagnose"
   echo "       -d   host index: TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich"
   echo "       -c   path of your .ovpn file (instead of -d)"
   echo "       -n   one-time setup (install OpenVPN + configure credentials)"
   echo "       -s   use systemd service mode (recommended for production)"
   echo "       --diagnose  print tunnel/service/config state for the swarm operator (changes nothing)"
   echo ""
   exit 0
fi

# ---------------------------------------------------------------------------
# --diagnose: everything the swarm operator needs to see why no tunnel address comes up. Prints no
# key material and no password.
# ---------------------------------------------------------------------------
diagnose() {
    local p
    echo "=== MediSwarm VPN diagnostics: $(hostname) $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
    echo "-- OpenVPN: $(openvpn --version 2>/dev/null | head -1 || true)"
    msvpn_openvpn_has_systemd && echo "   built with systemd support" || echo "   NOT built with systemd support (or not installed)"
    echo "-- GoodAccess addresses (172.24.4.0/22) on this host:"
    msvpn_vpn_addrs | sed 's/^/  /' | grep . || echo "  none"
    echo "-- tunnels:"
    msvpn_describe_holders
    echo "-- service $MSVPN_UNIT:"
    for p in UnitFileState ActiveState SubState Result NRestarts MainPID ExecMainStartTimestamp ExecMainStatus; do
        echo "  $p=$(msvpn_unit_prop "$p")"
    done
    echo "  ExecStart=$(msvpn_unit_prop ExecStart | grep -o 'path=[^ ;]*' | head -1)"
    echo "-- health timer: $(systemctl is-enabled mediswarm-vpn-health.timer 2>/dev/null || true)," \
         "$(systemctl is-active mediswarm-vpn-health.timer 2>/dev/null || true);" \
         "$(msvpn_unit_prop ExecStart mediswarm-vpn-health.service | grep -o 'path=[^ ;]*' | head -1)"
    echo "-- config $MSVPN_CONF:"
    if [[ -r "$MSVPN_CONF" ]]; then
        msvpn_lint_config "$MSVPN_CONF" | sed 's/^/  problem: /' | grep . || echo "  no problems found"
        echo "  directives: $(awk '{ sub(/\r$/, "") } /^<[A-Za-z0-9-]+>/ { b = 1 } /^<\// { b = 0; next }
            !b && $1 !~ /^(#|;|$)/ { printf "%s; ", ($1 == "remote" ? "remote <gateway>" : $0) }' "$MSVPN_CONF")"
    else
        echo "  missing"
    fi
    echo "-- credentials $MSVPN_AUTH:"
    msvpn_check_auth_file "$MSVPN_AUTH" | sed 's/^/  problem: /' | grep . \
        || echo "  present, two non-empty lines, mode $(stat -c '%a %U' "$MSVPN_AUTH" 2>/dev/null)"
    echo "-- OpenVPN status since the service last started:"
    local age since
    age="$(msvpn_service_age_s)"
    since=$(( $(date +%s) - ${age:-86400} ))
    msvpn_journal_since "$since" | msvpn_classify_log | awk -F'\t' '{ print "  " $1 ": " $2; if ($3 != "") print "  evidence: " $3 }'
    echo "-- last 40 service log lines:"
    journalctl -u "$MSVPN_UNIT" -n 40 -o short-iso --no-pager 2>/dev/null | sed 's/^/  /' || true
    echo "-- route to the gateway: $(ip route get 172.24.4.1 2>/dev/null | head -1)"
    echo "-- default routes: $(ip route show default 2>/dev/null | tr '\n' ';')"
    echo "=== end ==="
}

if [[ $ACTION = "diagnose" ]]; then
    set +e
    if [[ $EUID -ne 0 ]]; then
        echo "(not root: journal and credentials checks may be incomplete; run with sudo)"
    fi
    diagnose
    exit 0
fi

if [[ -z "$OVPN_FILE" ]]; then
    if [ -z "${host_index:-}" ]; then
       echo "Please specify your host index via option -d <host_index> (or the .ovpn file via -c <file>)"
       echo "Host index should be chosen from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich]"
       exit 1
    fi
    OVPN_FILE="$REPO_ROOT/assets/openvpn_configs/good_access/$host_index.ovpn"
fi
if [[ ! -f "$OVPN_FILE" ]]; then
    echo "Configuration file $OVPN_FILE not found"
    echo "Place your GoodAccess .ovpn there, or pass its path with -c <file.ovpn>."
    exit 1
fi

# The systemd mode writes under /etc and talks to systemd throughout: run it as root.
if [[ -n "$SYSTEMD_MODE" && $EUID -ne 0 ]]; then
    exec sudo -- "$0" "${ORIG_ARGS[@]}"
fi

refuse_second_tunnel() {   # prints why and returns 0 when another tunnel holds a GoodAccess address
    local others
    others="$(msvpn_vpn_addrs)"
    [[ -n "$others" ]] || return 1
    echo ""
    echo "Another GoodAccess tunnel is already up on this host:"
    sed 's/^/  interface /' <<< "$others"
    msvpn_describe_holders
    echo ""
    echo "GoodAccess allows one session per account: a second tunnel with the same account is refused"
    echo "by the gateway or knocks the first one off, so it cannot get an address while this one runs."
    echo "Nothing was started. Keep the tunnel that works. To move to the service later (between runs):"
    echo "stop the other tunnel, rerun this script with -s, and if it reports a problem start the other"
    echo "tunnel again. Details: docs/VPN_SERVICE.md"
    return 0
}

# ---------------------------------------------------------------------------
# First-time setup (-n)
# ---------------------------------------------------------------------------
if [ "$ACTION" = "new" ]; then
    echo "Setting up VPN tunnel for swarm learning ..."
    sudo apt-get update -qq
    sudo apt-get -y install openvpn

    echo "Please enter your VPN credentials (ask TUD maintainer for the account and password if you don't have them yet)"
    msvpn_prompt_credentials

    if [[ -n "$SYSTEMD_MODE" ]]; then
        # For systemd mode, store credentials in /etc/openvpn/client/
        msvpn_write_auth_file "$MSVPN_AUTH" "$MSVPN_ACCOUNT" "$MSVPN_PASSWORD"
        chown root:root "$MSVPN_AUTH"
        echo "Credentials saved to $MSVPN_AUTH"
    else
        # Legacy mode: store in /etc/openvpn/credentials
        sudo mkdir -p /etc/openvpn
        printf '%s\n' "$MSVPN_ACCOUNT" "$MSVPN_PASSWORD" | sudo tee /etc/openvpn/credentials > /dev/null
        sudo chmod 600 /etc/openvpn/credentials
        echo "Credentials saved to /etc/openvpn/credentials"
    fi
    unset MSVPN_PASSWORD
fi

# ---------------------------------------------------------------------------
# Systemd service mode (-s)
# ---------------------------------------------------------------------------
if [[ -n "$SYSTEMD_MODE" ]]; then
    echo "=== Installing MediSwarm VPN as systemd service ==="

    # Credentials: the service always uses its own file. A site that set up the legacy mode before
    # has /etc/openvpn/credentials; reuse it instead of failing.
    if [[ ! -f "$MSVPN_AUTH" && -f /etc/openvpn/credentials ]]; then
        mkdir -p "$(dirname "$MSVPN_AUTH")"
        install -m 600 -o root -g root /etc/openvpn/credentials "$MSVPN_AUTH"
        echo "Using the credentials of the legacy mode (/etc/openvpn/credentials) for the service."
    fi
    if [[ ! -f "$MSVPN_AUTH" ]]; then
        echo "Credentials file $MSVPN_AUTH not found."
        echo "Please run with -n -s to set up credentials first."
        exit 1
    fi
    auth_problems="$(msvpn_check_auth_file "$MSVPN_AUTH")"
    if [[ -n "$auth_problems" ]]; then
        echo "$auth_problems"
        echo "Please run with -n -s to enter the credentials again."
        exit 1
    fi

    echo "Writing $MSVPN_CONF from $OVPN_FILE ..."
    mkdir -p "$(dirname "$MSVPN_CONF")"
    [[ -f "$MSVPN_CONF" ]] && cp -p "$MSVPN_CONF" "$MSVPN_CONF.bak"
    if ! msvpn_prepare_config "$OVPN_FILE" "$MSVPN_CONF" "$MSVPN_AUTH" "$MSVPN_FILES_DIR"; then
        echo "Cannot use $OVPN_FILE (see above); nothing was changed in the service."
        exit 1
    fi

    echo "Installing systemd service ..."
    msvpn_install_runtime "$SCRIPT_DIR"
    install -m 644 "$SCRIPT_DIR/mediswarm-vpn.service" /etc/systemd/system/mediswarm-vpn.service
    if ! msvpn_openvpn_has_systemd; then
        sed -i 's/^Type=notify$/Type=simple/' /etc/systemd/system/mediswarm-vpn.service
        echo "  Note: this OpenVPN has no systemd support; the service uses Type=simple."
    fi
    # A health timer installed by an older version runs a monitor that restarts the service even
    # when it is disabled and judges tun0 only. Point it at the new monitor; its enabled/disabled
    # state stays as the site left it.
    health_unit=/etc/systemd/system/mediswarm-vpn-health.service
    if [[ -f "$health_unit" ]] && ! grep -qx "ExecStart=$MSVPN_LIBEXEC/vpn_health_monitor.sh" "$health_unit"; then
        sed -i "s#^ExecStart=.*#ExecStart=$MSVPN_LIBEXEC/vpn_health_monitor.sh#" "$health_unit"
        echo "  Updated $health_unit to the new health monitor."
    fi
    systemctl daemon-reload

    # A rerun replaces a running service of ours; any address left after stopping it belongs to
    # another tunnel.
    systemctl stop mediswarm-vpn.service 2>/dev/null || true
    if refuse_second_tunnel; then
        systemctl disable mediswarm-vpn.service 2>/dev/null || true
        echo "(The service is installed but disabled, so it does not start next to that tunnel at boot.)"
        exit "$MSVPN_EXIT_OTHER_TUNNEL"
    fi

    systemctl enable mediswarm-vpn.service
    t0="$(date +%s)"
    systemctl restart --no-block mediswarm-vpn.service
    wait_s="${VPN_WAIT_SECONDS:-90}"
    echo "Waiting up to ${wait_s}s for the tunnel address ..."
    rc=0
    msvpn_wait_for_tunnel "$t0" "$wait_s" || rc=$?

    if (( rc == 0 )); then
        echo ""
        echo "=== MediSwarm VPN service installed: tunnel up after $(( $(date +%s) - t0 ))s ==="
        sed 's/^/  interface /' <<< "$MSVPN_TUNNEL_ADDRS"
        echo ""
        echo "Useful commands:"
        echo "  sudo systemctl status mediswarm-vpn      # check status"
        echo "  sudo journalctl -u mediswarm-vpn -f       # follow logs"
        echo "  sudo systemctl restart mediswarm-vpn      # manual restart"
        echo "  sudo systemctl stop mediswarm-vpn         # stop VPN"
        echo "  sudo $0 --diagnose   # report for the swarm operator"
        echo ""

        # Optionally install the health monitor timer
        echo "Do you want to install the VPN health monitor timer? (recommended)"
        read -r -p "Install health monitor? [Y/n]: " install_health || true
        if [[ "${install_health:-Y}" =~ ^[Yy]$ ]]; then
            "$SCRIPT_DIR/vpn_health_monitor.sh" --install-timer
        fi
        exit 0
    fi

    echo ""
    echo "The tunnel did not come up: $(cut -f2 <<< "$MSVPN_VERDICT")"
    evidence="$(cut -f3 <<< "$MSVPN_VERDICT")"
    [[ -n "$evidence" ]] && echo "  OpenVPN: $evidence"
    echo "Last OpenVPN messages:"
    msvpn_journal_since "$t0" | grep -v '^[[:space:]]*$' | tail -n 15 | sed 's/^/    /'
    lint="$(msvpn_lint_config "$MSVPN_CONF")"
    if [[ -n "$lint" ]]; then
        echo "Problems in $MSVPN_CONF:"
        sed 's/^/    /' <<< "$lint"
    fi
    echo ""
    if (( rc == 4 )); then
        systemctl stop mediswarm-vpn.service 2>/dev/null || true
        systemctl disable mediswarm-vpn.service 2>/dev/null || true
        echo "The service was stopped and disabled so that it does not keep retrying with this problem."
        echo "Fix the cause and run this script again (wrong account or password: rerun with -n -s)."
    else
        echo "The service keeps trying in the background (sudo journalctl -u mediswarm-vpn -f)."
        echo "If no 172.24.4.x address appears, stop it with: sudo systemctl disable --now mediswarm-vpn"
    fi
    echo "Report for the swarm operator: sudo $0 --diagnose"
    exit "$rc"
fi

# ---------------------------------------------------------------------------
# Legacy mode (nohup, backward compatible)
# ---------------------------------------------------------------------------
if [[ ! -f /etc/openvpn/credentials ]]; then
    echo "Credentials file /etc/openvpn/credentials not found, please use option -n to create it"
    exit 1
fi
if refuse_second_tunnel; then
    exit "$MSVPN_EXIT_OTHER_TUNNEL"
fi

echo "Starting VPN tunnel for swarm learning (legacy nohup mode) ..."
echo "TIP: For production use, re-run with -s for systemd service mode with auto-reconnect."
echo ""
# --auth-user-pass after --config overrides the .ovpn's own line, so the GoodAccess file works unchanged.
sudo nohup openvpn --config "$OVPN_FILE" --auth-user-pass /etc/openvpn/credentials &
sleep 3
sudo chmod a+r nohup.out 2>/dev/null || true

if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above or nohup.out for more details."
    exit 1
fi

hostname -I

echo "You should see an IP address of the form 172.24.4.x in the output above."
echo "If it does not appear, try 'hostname -I' again in 10 or 20 seconds."
