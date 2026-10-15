#!/usr/bin/env bash
# Throwaway PKI and GoodAccess-shaped client configs for the offline VPN reproduction.
#
# Runs inside the test image (needs openssl + openvpn), writes everything to $1 (default /out).
# Nothing here is a real credential: the CA lives for the duration of one test run.
#
# The client config mirrors the byte layout of a real GoodAccess download (checked against the
# TUD gateway config, secrets excluded): directive lines and the inline-block tags end in CRLF,
# the PEM bodies in LF, a bare "auth-user-pass", the file ends with "</tls-auth>" CRLF CRLF.
set -euo pipefail

OUT="${1:-/out}"
GW_NAME="${GW_NAME:-vpn-gw.test}"
mkdir -p "$OUT"
cd "$OUT"

days=7
openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days "$days" -keyout ca.key -out ca.crt \
    -subj "/CN=MediSwarm VPN repro CA" \
    -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign" 2>/dev/null

issue() {   # issue <name> <CN> <eku>
    local name="$1" cn="$2" eku="$3"
    openssl req -newkey rsa:2048 -nodes -sha256 -keyout "$name.key" -out "$name.csr" -subj "/CN=$cn" 2>/dev/null
    printf 'basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=%s\n' \
        "$eku" > "$name.ext"
    openssl x509 -req -in "$name.csr" -CA ca.crt -CAkey ca.key -CAcreateserial -days "$days" -sha256 \
        -extfile "$name.ext" -out "$name.crt" 2>/dev/null
    rm -f "$name.csr" "$name.ext"
}
issue server "$GW_NAME" serverAuth
# GoodAccess ships one client certificate per gateway; members are told apart by username/password.
issue client "goodaccess-member" clientAuth

openvpn --genkey secret ta.key 2>/dev/null || openvpn --genkey --secret ta.key

crlf() { printf '%s\r\n' "$@"; }
inline() {   # inline <tag> <file>: CRLF tags, LF body (as in the GoodAccess download)
    crlf "<$1>"
    cat "$2"
    crlf "</$1>"
}
header() {
    crlf "client" "dev tun" "proto udp" "remote $GW_NAME 1194" "resolv-retry infinite" "nobind" \
         "persist-key" "persist-tun" "tls-client" "verify-x509-name $GW_NAME name" "remote-cert-tls server" \
         "key-direction 1" "auth SHA256" "cipher AES-256-GCM" "verb 3" "auth-user-pass" "#tuning" "" "" \
         "fast-io" "allow-pull-fqdn" "pull"
}

{ header; inline ca ca.crt; inline cert client.crt; inline key client.key; inline tls-auth ta.key; crlf ""; } \
    > goodaccess.ovpn

# Same file without any trailing newline (what an editor or a copy/paste can leave behind).
head -c -4 goodaccess.ovpn > goodaccess_nonl.ovpn       # drops the final "\r\n\r\n"
tail -c 12 goodaccess_nonl.ovpn | grep -q '</tls-auth>$'

# As our own CLI guide and assets/openvpn_configs/good_access/README.md told sites to edit it.
sed 's#^auth-user-pass\r$#auth-user-pass /etc/openvpn/credentials\r#' goodaccess.ovpn > docs_auth.ovpn
grep -q '^auth-user-pass /etc/openvpn/credentials' docs_auth.ovpn

# A hand-made config with the usual privilege drop.
sed 's#^pull\r$#pull\r\nuser nobody\r\ngroup nogroup\r#' goodaccess.ovpn > user_group.ovpn
grep -q '^group nogroup' user_group.ovpn

# A config of another gateway/PKI (stale download): other CA, client certificate and tls-auth key.
mkdir -p stale && (
    cd stale
    openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days "$days" -keyout ca.key -out ca.crt \
        -subj "/CN=Old gateway CA" 2>/dev/null
    openssl req -newkey rsa:2048 -nodes -sha256 -keyout client.key -out client.csr -subj "/CN=old-member" 2>/dev/null
    printf 'extendedKeyUsage=clientAuth\n' > client.ext
    openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days "$days" -sha256 \
        -extfile client.ext -out client.crt 2>/dev/null
    openvpn --genkey secret ta.key 2>/dev/null || openvpn --genkey --secret ta.key
)
{ header; inline ca stale/ca.crt; inline cert stale/client.crt; inline key stale/client.key; inline tls-auth stale/ta.key; crlf ""; } \
    > stale.ovpn

# NetworkManager export style: certificates as separate files under the importing user's home
# (NM extracts inline blocks to ~/.cert/nm-openvpn/ on import).
NMDIR=/home/siteuser/.cert/nm-openvpn
{
    printf '%s\n' "client" "remote $GW_NAME 1194 udp" "dev tun" "nobind" "persist-key" "persist-tun" \
        "remote-cert-tls server" "verify-x509-name $GW_NAME name" "auth SHA256" "cipher AES-256-GCM" \
        "ca $NMDIR/repro-ca.pem" "cert $NMDIR/repro-cert.pem" "key $NMDIR/repro-key.pem" \
        "tls-auth $NMDIR/repro-tls-auth.pem 1" "auth-user-pass"
} > nm_export.ovpn
mkdir -p nm-openvpn
cp ca.crt nm-openvpn/repro-ca.pem
cp client.crt nm-openvpn/repro-cert.pem
cp client.key nm-openvpn/repro-key.pem
cp ta.key nm-openvpn/repro-tls-auth.pem

chmod -R a+rX "$OUT"
echo "PKI and client configs written to $OUT"
