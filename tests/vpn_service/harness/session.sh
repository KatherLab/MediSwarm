#!/bin/bash
# client-connect / client-disconnect hook of the test gateway: keeps one marker file per connected
# member (used by session_policy=reject) and logs connects/disconnects for the harness.
VPN=/srv/vpn
user="${common_name:-unknown}"
case "${script_type:-}" in
    client-connect)
        touch "$VPN/sessions/$user"
        printf '%s connect user=%s ip=%s\n' "$(date +%s)" "$user" "${ifconfig_pool_remote_ip:-?}" >> "$VPN/sessions.log"
        ;;
    client-disconnect)
        rm -f "$VPN/sessions/$user"
        printf '%s disconnect user=%s\n' "$(date +%s)" "$user" >> "$VPN/sessions.log"
        ;;
esac
exit 0
