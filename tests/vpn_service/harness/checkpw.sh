#!/bin/bash
# auth-user-pass-verify (via-file) for the test gateway. $1 = file with username and password.
# Logs a one-line verdict to /srv/vpn/auth.log so the harness can count attempts per scenario.
VPN=/srv/vpn
user="$(sed -n 1p "$1")"
pass="$(sed -n 2p "$1")"

delay="$(cat "$VPN/auth_delay" 2>/dev/null || echo 0)"
[[ "$delay" =~ ^[0-9]+$ ]] && (( delay > 0 )) && sleep "$delay"

verdict="reject:bad-credentials"
while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "$user:$pass" ]]; then verdict="accept"; break; fi
done < "$VPN/users"

if [[ "$verdict" == accept && "$(cat "$VPN/session_policy" 2>/dev/null)" == reject && -e "$VPN/sessions/$user" ]]; then
    verdict="reject:already-connected"
fi

printf '%s user=%q verdict=%s\n' "$(date +%s)" "$user" "$verdict" >> "$VPN/auth.log"
[[ "$verdict" == accept ]]
