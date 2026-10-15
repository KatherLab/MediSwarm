#!/usr/bin/env bash
# Entry point of the test gateway container: an OpenVPN server that behaves like the GoodAccess
# gateway as far as a site can observe it (taken from the TUD client log, PUSH_REPLY of Oct 2026):
#   - username/password on top of one shared client certificate, tls-auth with key-direction,
#   - one static address per member in 172.24.4.0/22 (gateway 172.24.4.1),
#   - pushes "redirect-gateway def1 bypass-dhcp", a DNS server, "explicit-exit-notify 3",
#     "ping 10" and "ping-restart 120".
#
# Knobs (files under /srv/vpn, so a running server can be switched between scenarios):
#   users            "name:password" lines
#   auth_delay       seconds the password check takes (emulates a slow/deferred gateway auth)
#   session_policy   "kick"   (OpenVPN default: a new login of the same member drops the old session)
#                    "reject" (a second concurrent login of the same member gets AUTH_FAILED)
set -euo pipefail

PKI=/srv/pki
VPN=/srv/vpn
mkdir -p "$VPN/ccd" "$VPN/sessions"
touch "$VPN/users"
[[ -f "$VPN/session_policy" ]] || echo kick > "$VPN/session_policy"
[[ -f "$VPN/auth_delay" ]] || echo 0 > "$VPN/auth_delay"
cp /repro/harness/checkpw.sh /repro/harness/session.sh /usr/local/bin/
chmod 755 /usr/local/bin/checkpw.sh /usr/local/bin/session.sh
chmod 777 "$VPN/sessions"

cat > /etc/openvpn/server.conf <<EOF
port 1194
proto udp
dev tun
topology subnet
server 172.24.4.0 255.255.252.0
ca $PKI/ca.crt
cert $PKI/server.crt
key $PKI/server.key
dh none
tls-auth $PKI/ta.key 0
auth SHA256
cipher AES-256-GCM
data-ciphers AES-256-GCM
keepalive 10 120
push "redirect-gateway def1 bypass-dhcp"
push "dhcp-option DNS 172.24.4.1"
push "explicit-exit-notify 3"
client-config-dir $VPN/ccd
username-as-common-name
verify-client-cert require
script-security 2
auth-user-pass-verify /usr/local/bin/checkpw.sh via-file
client-connect /usr/local/bin/session.sh
client-disconnect /usr/local/bin/session.sh
status /run/openvpn-server.status 2
verb 3
EOF

exec openvpn --config /etc/openvpn/server.conf
