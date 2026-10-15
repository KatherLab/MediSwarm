#!/usr/bin/env bash
# =============================================================================
# ExecStart of mediswarm-vpn.service (installed to /usr/local/lib/mediswarm-vpn/ by setup_vpntunnel.sh -s)
# =============================================================================
# Checks that starting OpenVPN can work, then replaces itself with OpenVPN. exec keeps the PID, so with
# Type=notify systemd receives OpenVPN's READY=1 from the main process and reports the unit "active"
# only once the tunnel is up.
#
# Refuses (exit 3) when another tunnel already holds a GoodAccess address: GoodAccess allows one
# session per account, so a second tunnel with the same account is refused by the gateway or replaces
# the first one, and the two keep knocking each other off (#620: the service ran next to CAM's
# NetworkManager tunnel). Exit 4 = config or credentials missing. Both are in RestartPreventExitStatus=.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mediswarm_vpn_lib.sh
source "$HERE/mediswarm_vpn_lib.sh"

others="$(msvpn_vpn_addrs)"
if [[ -n "$others" ]]; then
    echo "refusing to start: another GoodAccess tunnel already holds a VPN address ($(tr '\n' ' ' <<< "$others" | sed 's/ $//'))."
    echo "GoodAccess allows one session per account: a second tunnel with the same account is refused or replaces the first one."
    echo "Keep the working tunnel and disable this service (sudo systemctl disable --now $MSVPN_UNIT), or stop the other tunnel first."
    msvpn_describe_holders
    exit "$MSVPN_EXIT_OTHER_TUNNEL"
fi

if [[ ! -r "$MSVPN_CONF" ]]; then
    echo "not configured: $MSVPN_CONF is missing. Run setup_vpntunnel.sh -d <site> -n -s"
    exit "$MSVPN_EXIT_NOT_CONFIGURED"
fi
# The credentials file the config actually names (the last auth-user-pass outside the inline blocks).
auth_file="$(awk '{ sub(/\r$/, "") } /^<[A-Za-z0-9-]+>/ { b = 1 } /^<\// { b = 0; next } !b && $1 == "auth-user-pass" && NF > 1 { f = $2 } END { print f }' "$MSVPN_CONF")"
problems="$(msvpn_check_auth_file "${auth_file:-$MSVPN_AUTH}")"
if [[ -n "$problems" ]]; then
    echo "not configured: credentials file problem: $problems. Run setup_vpntunnel.sh -d <site> -n -s"
    exit "$MSVPN_EXIT_NOT_CONFIGURED"
fi

OPENVPN_BIN="${OPENVPN_BIN:-$(command -v openvpn || echo /usr/sbin/openvpn)}"
exec "$OPENVPN_BIN" --suppress-timestamps --config "$MSVPN_CONF"
