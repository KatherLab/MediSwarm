#!/usr/bin/env bash
# =============================================================================
# MediSwarm VPN Tunnel Setup
# =============================================================================
# Sets up and launches an OpenVPN connection using GoodAccess .ovpn configs.
#
# Modes:
#   Legacy (default):  Launches OpenVPN via nohup in the background.
#   Systemd (-s):      Installs a systemd service with auto-reconnect,
#                       keepalive, and boot persistence.
#
# Usage:
#   setup_vpntunnel.sh -d <host_index> [-n] [-s]
#
#   -d <host_index>  Required. One of: TUD, Ribera, VHIO, Radboud, UKA,
#                     Utrecht, Mitera, Cambridge, Zurich
#   -n               First-time setup: installs OpenVPN and prompts for
#                     VPN credentials
#   -s               Use systemd service mode (recommended for production).
#                     Installs mediswarm-vpn.service with auto-reconnect,
#                     keepalive, and boot persistence.
#   -h               Show this help message
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

ACTION="nochange"
SYSTEMD_MODE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d)  host_index="$2"; shift ;;
        -n)  ACTION="new";;
        -s)  SYSTEMD_MODE="1";;
        -h)  ACTION="help";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ $ACTION = "help" ]]; then
   echo "Usage: setup_vpntunnel.sh -d <host_index> [-n] [-s]"
   echo "       -d   host index: TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich"
   echo "       -n   one-time setup (install OpenVPN + configure credentials)"
   echo "       -s   use systemd service mode (recommended for production)"
   echo ""
   exit 0
fi

if [ -z "${host_index:-}" ]; then
   echo "Please specify your host index via option -d <host_index>"
   echo "Host index should be chosen from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich]"
   exit 1
fi

# Locate the .ovpn file
OVPN_FILE="$REPO_ROOT/assets/openvpn_configs/good_access/$host_index.ovpn"
if [[ ! -f "$OVPN_FILE" ]]; then
    echo "Configuration file $OVPN_FILE not found"
    exit 1
fi

# ---------------------------------------------------------------------------
# First-time setup (-n)
# ---------------------------------------------------------------------------
if [ "$ACTION" = "new" ]; then
    echo "Setting up VPN tunnel for swarm learning ..."
    sudo apt-get update -qq
    sudo apt-get -y install openvpn

    echo "Please enter your VPN credentials (ask TUD maintainer for the account and password if you don't have them yet)"
    read -p "VPN account: " vpn_account
    stty -echo
    read -p "VPN password: " vpn_password
    stty echo
    echo ""

    if [[ -n "$SYSTEMD_MODE" ]]; then
        # For systemd mode, store credentials in /etc/openvpn/client/
        sudo mkdir -p /etc/openvpn/client
        printf '%s\n' "$vpn_account" "$vpn_password" | sudo tee /etc/openvpn/client/mediswarm.auth > /dev/null
        sudo chmod 600 /etc/openvpn/client/mediswarm.auth
        sudo chown root:root /etc/openvpn/client/mediswarm.auth
        echo "Credentials saved to /etc/openvpn/client/mediswarm.auth"
    else
        # Legacy mode: store in /etc/openvpn/credentials
        sudo mkdir -p /etc/openvpn
        printf '%s\n' "$vpn_account" "$vpn_password" | sudo tee /etc/openvpn/credentials > /dev/null
        sudo chmod 600 /etc/openvpn/credentials
        echo "Credentials saved to /etc/openvpn/credentials"
    fi
fi

# ---------------------------------------------------------------------------
# Systemd service mode (-s)
# ---------------------------------------------------------------------------
if [[ -n "$SYSTEMD_MODE" ]]; then
    echo "=== Installing MediSwarm VPN as systemd service ==="

    # Verify credentials exist
    if [[ ! -f /etc/openvpn/client/mediswarm.auth ]]; then
        echo "Credentials file /etc/openvpn/client/mediswarm.auth not found."
        echo "Please run with -n -s to set up credentials first."
        exit 1
    fi

    # Copy .ovpn to /etc/openvpn/client/mediswarm.conf and inject settings
    echo "Copying VPN config to /etc/openvpn/client/mediswarm.conf ..."
    sudo cp "$OVPN_FILE" /etc/openvpn/client/mediswarm.conf

    # Inject keepalive and credential settings if not already present
    inject_if_missing() {
        local setting="$1"
        local value="$2"
        local file="/etc/openvpn/client/mediswarm.conf"
        if ! sudo grep -q "^$setting" "$file"; then
            echo "$value" | sudo tee -a "$file" > /dev/null
            echo "  Injected: $value"
        else
            echo "  Already present: $setting"
        fi
    }

    echo "Injecting keepalive and credential settings ..."
    inject_if_missing "auth-user-pass"  "auth-user-pass /etc/openvpn/client/mediswarm.auth"
    inject_if_missing "auth-nocache"    "auth-nocache"
    inject_if_missing "persist-key"     "persist-key"
    inject_if_missing "persist-tun"     "persist-tun"
    inject_if_missing "resolv-retry"    "resolv-retry infinite"
    inject_if_missing "keepalive"       "keepalive 10 60"

    # Install systemd service
    echo "Installing systemd service ..."
    sudo cp "$SCRIPT_DIR/mediswarm-vpn.service" /etc/systemd/system/mediswarm-vpn.service
    sudo systemctl daemon-reload
    sudo systemctl enable mediswarm-vpn.service
    sudo systemctl restart mediswarm-vpn.service

    echo ""
    echo "=== MediSwarm VPN service installed ==="
    echo ""
    echo "Useful commands:"
    echo "  sudo systemctl status mediswarm-vpn      # check status"
    echo "  sudo journalctl -u mediswarm-vpn -f       # follow logs"
    echo "  sudo systemctl restart mediswarm-vpn      # manual restart"
    echo "  sudo systemctl stop mediswarm-vpn         # stop VPN"
    echo ""

    # Optionally install the health monitor timer
    echo "Do you want to install the VPN health monitor timer? (recommended)"
    read -p "Install health monitor? [Y/n]: " install_health
    if [[ "${install_health:-Y}" =~ ^[Yy]$ ]]; then
        "$SCRIPT_DIR/vpn_health_monitor.sh" --install-timer
    fi

    # Wait and show VPN IP
    sleep 5
    echo ""
    echo "Current IP addresses:"
    hostname -I
    echo ""
    echo "You should see an IP address of the form 172.24.4.x above."
    echo "If not, check: sudo journalctl -u mediswarm-vpn -f"
    exit 0
fi

# ---------------------------------------------------------------------------
# Legacy mode (nohup, backward compatible)
# ---------------------------------------------------------------------------
if [[ ! -f /etc/openvpn/credentials ]]; then
    echo "Credentials file /etc/openvpn/credentials not found, please use option -n to create it"
    exit 1
fi

echo "Starting VPN tunnel for swarm learning (legacy nohup mode) ..."
echo "TIP: For production use, re-run with -s for systemd service mode with auto-reconnect."
echo ""
sudo nohup openvpn --config "$OVPN_FILE" &
sleep 3
sudo chmod a+r nohup.out 2>/dev/null || true

if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above or nohup.out for more details."
    exit 1
fi

hostname -I

echo "You should see an IP address of the form 172.24.4.x in the output above."
echo "If it does not appear, try 'hostname -I' again in 10 or 20 seconds."
