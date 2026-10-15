#!/usr/bin/env bash
# =============================================================================
# MediSwarm VPN service: shared helpers (sourced, not executed)
# =============================================================================
# Used by setup_vpntunnel.sh, vpn_service_start.sh (ExecStart of mediswarm-vpn.service) and
# vpn_health_monitor.sh. setup_vpntunnel.sh -s copies these scripts to /usr/local/lib/mediswarm-vpn/:
# the service runs with ProtectHome=true and cannot see a repository or kit under /home, and the
# health timer must keep working after a kit folder is replaced or deleted.
#
# The functions only read system state (ip, systemctl, journalctl) or write the files they are given,
# so tests/unit_tests/test_vpn_service_scripts.py runs them without root.
# =============================================================================

MSVPN_UNIT="${MSVPN_UNIT:-mediswarm-vpn}"
MSVPN_CONF="${MSVPN_CONF:-/etc/openvpn/client/mediswarm.conf}"
MSVPN_AUTH="${MSVPN_AUTH:-/etc/openvpn/client/mediswarm.auth}"
MSVPN_FILES_DIR="${MSVPN_FILES_DIR:-/etc/openvpn/client/mediswarm.d}"
MSVPN_LIBEXEC="${MSVPN_LIBEXEC:-/usr/local/lib/mediswarm-vpn}"
# GoodAccess gives every member a fixed address in 172.24.4.0/22 (gateway 172.24.4.1). [.] instead of
# \. because awk -v would turn "\." into a warning.
MSVPN_ADDR_REGEX="${MSVPN_ADDR_REGEX:-^172[.]24[.][4-7][.][0-9]+$}"
# Exit codes of vpn_service_start.sh. The unit lists both in RestartPreventExitStatus=: restarting every
# 15 s cannot fix either, the journal says why, and the health monitor retries with backoff.
# shellcheck disable=SC2034  # used by the scripts that source this file
MSVPN_EXIT_OTHER_TUNNEL=3
# shellcheck disable=SC2034
MSVPN_EXIT_NOT_CONFIGURED=4
# Files the runtime needs (installed by msvpn_install_runtime).
MSVPN_RUNTIME_FILES=(mediswarm_vpn_lib.sh vpn_service_start.sh vpn_health_monitor.sh)

# ---------------------------------------------------------------------------
# Tunnel state
# ---------------------------------------------------------------------------

# "iface address" for every GoodAccess address in `ip -o -4 addr show` output read from stdin.
msvpn_vpn_addrs_from() {
    awk -v re="$MSVPN_ADDR_REGEX" '$3 == "inet" { split($4, a, "/"); if (a[1] ~ re) print $2, a[1] }'
}

msvpn_vpn_addrs() {
    ip -o -4 addr show 2>/dev/null | msvpn_vpn_addrs_from
}

# Who holds a tunnel: OpenVPN processes and active NetworkManager VPN connections (informational).
msvpn_describe_holders() {
    local found=""
    if command -v pgrep >/dev/null 2>&1; then
        local procs
        procs="$(pgrep -a -x openvpn 2>/dev/null | cut -c1-220)"
        if [[ -n "$procs" ]]; then
            sed 's/^/  openvpn process: /' <<< "$procs"
            found=1
        fi
    fi
    if command -v nmcli >/dev/null 2>&1; then
        local nm
        nm="$(nmcli -t -f NAME,UUID,TYPE connection show --active 2>/dev/null \
            | awk -F: '$3 == "vpn" { printf "  NetworkManager VPN connection: %s (UUID %s)\n", $1, $2 }')"
        if [[ -n "$nm" ]]; then
            printf '%s\n' "$nm"
            found=1
        fi
    fi
    [[ -n "$found" ]] || echo "  no OpenVPN process and no NetworkManager VPN connection found:" \
        "the interface may be left over from a killed tunnel (remove it with: sudo ip link delete <interface>)"
}

msvpn_unit_prop() {   # msvpn_unit_prop <property> [unit]
    systemctl show -p "$1" --value "${2:-$MSVPN_UNIT}" 2>/dev/null || true
}

# Seconds since the service's current (or last) OpenVPN process started; empty when it never ran.
msvpn_service_age_s() {
    local mono up
    mono="$(msvpn_unit_prop ExecMainStartTimestampMonotonic "${1:-$MSVPN_UNIT}")"
    [[ "$mono" =~ ^[0-9]+$ ]] && (( mono > 0 )) || return 0
    up="$(cut -d. -f1 /proc/uptime)"
    echo $(( up - mono / 1000000 ))
}

msvpn_journal_since() {   # msvpn_journal_since <epoch> [unit]: OpenVPN's messages since then
    journalctl -u "${2:-$MSVPN_UNIT}" --since "@$1" -o cat --no-pager 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Why is there no tunnel? Classifies OpenVPN/service output read from stdin.
# Prints "code<TAB>explanation<TAB>evidence line". The latest decisive event wins; a generic
# "fatal error" does not hide the specific cause logged just before it.
# ---------------------------------------------------------------------------
msvpn_classify_log() {
    awk '
    function set(c, e) {
        generic = (c == "fatal" || c == "route_error" || c == "ping_restart")
        if (generic && code != "" && code != "connected" && code != "fatal" && code != "route_error" && code != "ping_restart")
            return
        code = c; evidence = e
    }
    { sub(/\r$/, "") }
    /Please correct these errors|Use --help for more information/          { next }
    /Initialization Sequence Completed/                                    { up = 1; set("connected", $0); next }
    /refusing to start: another GoodAccess tunnel/                         { set("other_tunnel", $0); next }
    /not configured:/                                                      { set("not_configured", $0); next }
    /AUTH_FAILED/                                                          { n_auth++; set("auth_failed", $0); next }
    /auth-user-pass fails with|Error opening .Auth. file|could not read Auth username/ { set("auth_file", $0); next }
    /Enter Auth Username|Need .* password|systemd-ask-password/            { set("prompt", $0); next }
    /Cannot pre-load keyfile|Cannot load (inline )?(CA )?certificate|Cannot load private key|Cannot load certificate/ { set("key_file", $0); next }
    /Options error|Unrecognized option|option .* is not allowed/            { set("options_error", $0); next }
    /VERIFY ERROR|VERIFY X509NAME ERROR|VERIFY KU ERROR|VERIFY EKU ERROR|certificate verify failed/ { set("tls_verify", $0); next }
    /TLS key negotiation failed|TLS handshake failed/                        { set("tls_timeout", $0); next }
    /Cannot resolve host address|RESOLVE: Cannot resolve/                   { set("resolve", $0); next }
    /Cannot open TUN\/TAP|Cannot ioctl TUNSETIFF|ERROR: Cannot open TUN/     { set("tun_device", $0); next }
    /set[gu]id\(.*failed|setgroups.*failed|Operation not permitted/         { set("permission", $0); next }
    /server-pushed-connection-reset/                                        { set("kicked", $0); next }
    # Before the first connection a ping timeout means the gateway never answered the handshake.
    /Inactivity timeout \(--ping-restart\)/                                 { set(up ? "ping_restart" : "tls_timeout", $0); next }
    /route add command failed|sitnl_send: rtnl: generic error/              { set("route_error", $0); next }
    /Exiting due to fatal error/                                            { set("fatal", $0); next }
    END {
        if (code == "") { code = "none"; evidence = "" }
        x["connected"] = "the tunnel came up (Initialization Sequence Completed)"
        x["other_tunnel"] = "the service refused to start because another GoodAccess tunnel already holds the VPN address"
        x["not_configured"] = "the service is not configured (config or credentials file missing)"
        x["auth_failed"] = "the gateway rejected the login (AUTH_FAILED, " n_auth "x): wrong account or password, or the account is connected from another tunnel or machine (GoodAccess allows one session per account)"
        x["auth_file"] = "OpenVPN cannot read the credentials file named in auth-user-pass"
        x["prompt"] = "OpenVPN waits for an interactive username/password prompt (no credentials file configured)"
        x["key_file"] = "OpenVPN cannot read a key or certificate file named in the config (a file under /home is invisible to the service)"
        x["options_error"] = "OpenVPN rejected the configuration (a file it names is missing or unreadable, or a directive is invalid)"
        x["tls_verify"] = "certificate check failed: the .ovpn does not belong to this gateway (stale or wrong file)"
        x["tls_timeout"] = "no TLS handshake with the gateway within 60 s: the gateway dropped the session without a reason (it does so when no username/password is sent, or for a .ovpn whose keys belong to another gateway), or UDP 1194 is blocked on the way"
        x["resolve"] = "the gateway name does not resolve (DNS)"
        x["tun_device"] = "cannot create the tunnel device (/dev/net/tun missing or not permitted)"
        x["permission"] = "an operation was not permitted (user/group directives or missing capabilities)"
        x["kicked"] = "the gateway reset the session, usually because the same account connected from another tunnel"
        x["ping_restart"] = "the connection to the gateway was lost (ping timeout); OpenVPN is reconnecting"
        x["route_error"] = "OpenVPN could not set a route"
        x["fatal"] = "OpenVPN stopped with a fatal error"
        x["none"] = "no OpenVPN output yet"
        printf "%s\t%s\t%s\n", code, x[code], evidence
    }'
}

# Codes after which waiting longer does not help (the setup script stops the service then).
# AUTH_FAILED counts only from the third attempt: right after another tunnel of the same account
# was stopped, the gateway may still hold its session for a short while.
msvpn_is_definitive() {   # msvpn_is_definitive <classify line>
    local code="${1%%$'\t'*}"
    case "$code" in
        auth_failed)
            local n
            n="$(sed -n 's/.*AUTH_FAILED, \([0-9]*\)x.*/\1/p' <<< "$1")"
            [[ "${n:-0}" -ge 3 ]] ;;
        other_tunnel|not_configured|auth_file|prompt|key_file|options_error|tls_verify|tun_device|permission|fatal) return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

msvpn_openvpn_has_systemd() {
    # Type=notify needs an OpenVPN built with systemd support (Debian/Ubuntu packages are).
    openvpn --version 2>&1 | grep -q 'enable_systemd=yes'
}

# Writes the service's OpenVPN config <dest> from a site's .ovpn <src> (GoodAccess download,
# NetworkManager export or hand-edited file). Notes go to stderr; returns 1 with a message on errors.
#   - Line ends are normalised to LF and the file ends with a newline. The old script appended
#     "auth-user-pass <file>" with tee -a: in a .ovpn without a final newline it was glued to the
#     closing </tls-auth> tag, where OpenVPN ignores it, so no credentials were sent (AUTH_FAILED).
#   - Every auth-user-pass line is replaced by "auth-user-pass <auth>". The old script kept a path that
#     was already there; our own CLI guide and the good_access README told sites to write
#     /etc/openvpn/credentials, which -n -s never creates, so OpenVPN exited at start.
#   - Key and certificate files the .ovpn references (ca, cert, key, tls-auth, ...) are copied to
#     <files_dir> (relative paths resolve against the .ovpn's folder). The service runs with
#     ProtectHome=true and cannot read a NetworkManager export's ~/.cert/nm-openvpn/ files.
#   - daemon, log/log-append, user/group are dropped: systemd supervises the process, OpenVPN's
#     output must reach the journal, and after dropping to "nobody" a reconnect cannot re-read the
#     root-only credentials file (auth-nocache).
#   - auth-nocache, persist-key, persist-tun, resolv-retry infinite, keepalive 10 60 are added when
#     missing (same as before).
msvpn_prepare_config() {   # msvpn_prepare_config <src.ovpn> <dest.conf> <auth file> <files dir>
    local src="$1" dest="$2" auth="$3" files_dir="$4"
    local src_dir tmp copylist present rc=0
    [[ -r "$src" ]] || { echo "cannot read $src" >&2; return 1; }
    src_dir="$(cd "$(dirname "$src")" && pwd)"
    tmp="$(mktemp "${dest}.XXXXXX")" || return 1
    copylist="$(mktemp)"; present="$(mktemp)"
    awk -v files_dir="$files_dir" -v src_dir="$src_dir" -v copylist="$copylist" -v present="$present" '
    BEGIN {
        split("ca cert key tls-auth tls-crypt tls-crypt-v2 pkcs12 extra-certs crl-verify secret", d, " ")
        for (i in d) filedir[d[i]] = 1
        inblock = ""
    }
    { sub(/\r$/, "") }
    inblock != "" {
        print
        t = $0; sub(/^[ \t]+/, "", t)
        if (index(t, "</" inblock ">") == 1) inblock = ""
        next
    }
    {
        t = $0; sub(/^[ \t]+/, "", t)
        if (t ~ /^<[A-Za-z0-9-]+>/) {
            tag = substr(t, 2); sub(/>.*/, "", tag)
            if (index(t, "</" tag ">") == 0) inblock = tag
            seen["<" tag ">"] = 1
            print; next
        }
        if (t == "" || t ~ /^[#;]/) { print; next }
        n = split(t, f, /[ \t]+/)
        dir = f[1]
        seen[dir] = 1
        if (dir == "auth-user-pass") {
            if (n > 1) printf "  replaced \"%s\" with the service credentials file\n", t > "/dev/stderr"
            next
        }
        if (dir == "daemon" || dir == "log" || dir == "log-append" || dir == "user" || dir == "group") {
            printf "  dropped \"%s\" (systemd runs the tunnel; output goes to the journal)\n", t > "/dev/stderr"
            next
        }
        if ((dir in filedir) && n > 1 && f[2] != "[inline]") {
            rest = substr(t, length(dir) + 1); sub(/^[ \t]+/, "", rest)
            if (substr(rest, 1, 1) == "\"") {
                path = substr(rest, 2); q = index(path, "\"")
                tail = (q > 0) ? substr(path, q + 1) : ""
                path = (q > 0) ? substr(path, 1, q - 1) : path
            } else {
                path = f[2]; tail = substr(rest, length(f[2]) + 1)
            }
            if (path !~ /^\//) path = src_dir "/" path
            base = path; sub(/.*\//, "", base); gsub(/[^A-Za-z0-9._-]/, "_", base)
            target = files_dir "/" dir "-" base
            printf "%s\t%s\n", path, target > copylist
            print dir " " target tail
            next
        }
        print
    }
    END {
        for (k in seen) print k > present
        if (inblock != "") { printf "the .ovpn ends inside <%s> (no </%s>)\n", inblock, inblock > "/dev/stderr"; exit 2 }
    }' "$src" > "$tmp" || rc=1
    if (( rc == 0 )) && ! grep -qx 'remote' "$present" && ! grep -qx '<connection>' "$present"; then
        echo "the .ovpn has no remote line: not a GoodAccess client config" >&2
        rc=1
    fi
    if (( rc == 0 )) && [[ -s "$copylist" ]]; then
        mkdir -p "$files_dir" && chmod 700 "$files_dir" || rc=1
        local from to
        while IFS=$'\t' read -r from to; do
            if [[ ! -r "$from" ]]; then
                echo "the .ovpn references $from, which does not exist or is not readable" >&2
                rc=1
                continue
            fi
            cp "$from" "$to" && chmod 600 "$to" || rc=1
            echo "  copied $from -> $to" >&2
        done < "$copylist"
    fi
    if (( rc == 0 )); then
        {
            echo ""
            echo "# --- added by setup_vpntunnel.sh -s ---"
            echo "auth-user-pass $auth"
            local item
            for item in "auth-nocache" "persist-key" "persist-tun" "resolv-retry infinite" "keepalive 10 60"; do
                grep -qx "${item%% *}" "$present" || echo "$item"
            done
        } >> "$tmp"
        chmod 600 "$tmp" && mv -f "$tmp" "$dest" || rc=1
    fi
    rm -f "$tmp" "$copylist" "$present"
    return "$rc"
}

# Findings for an installed config, one per line (empty = nothing wrong). Used by --diagnose and the
# start wrapper; never prints key material.
msvpn_lint_config() {   # msvpn_lint_config <conf>
    local conf="$1"
    [[ -r "$conf" ]] || { echo "config $conf is missing or unreadable"; return 0; }
    grep -q $'\r' "$conf" && echo "config has Windows line ends (CR)"
    [[ -z "$(tail -c 1 "$conf")" ]] || echo "config does not end with a newline"
    # Directive lines outside inline <tag>...</tag> blocks, CR stripped.
    local directives
    directives="$(awk '{ sub(/\r$/, ""); t = $0; sub(/^[ \t]+/, "", t) }
        b != "" { if (index(t, "</" b ">") == 1) b = ""; next }
        t ~ /^<[A-Za-z0-9-]+>/ { g = substr(t, 2); sub(/>.*/, "", g); if (index(t, "</" g ">") == 0) b = g; next }
        t != "" && t !~ /^[#;]/ { print t }' "$conf")"
    local auths
    auths="$(awk '$1 == "auth-user-pass" { print (NF > 1 ? $2 : "(no file)") }' <<< "$directives")"
    if [[ -z "$auths" ]]; then
        echo "no auth-user-pass line outside the inline blocks: OpenVPN sends no credentials (AUTH_FAILED)"
    else
        local a
        while read -r a; do
            [[ "$a" == "(no file)" ]] && { echo "auth-user-pass without a file: OpenVPN would prompt"; continue; }
            [[ -e "$a" ]] || echo "auth-user-pass file $a does not exist"
            [[ "$a" == /home/* || "$a" == /root/* ]] && echo "auth-user-pass file $a is invisible to the service (ProtectHome=true)"
        done <<< "$auths"
    fi
    local d p
    while read -r d p; do
        [[ -n "$d" ]] || continue
        [[ -e "$p" ]] || echo "$d file $p does not exist"
        [[ "$p" == /home/* || "$p" == /root/* ]] && echo "$d file $p is invisible to the service (ProtectHome=true)"
    done < <(awk '$1 ~ /^(ca|cert|key|tls-auth|tls-crypt|tls-crypt-v2|pkcs12|extra-certs|secret)$/ && NF > 1 && $2 != "[inline]" { print $1, $2 }' <<< "$directives")
    awk '$1 ~ /^(daemon|log|log-append|user|group)$/ { print "directive \"" $0 "\" does not belong in a systemd-run config" }' <<< "$directives"
    return 0
}

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

# Reads the GoodAccess account and password into MSVPN_ACCOUNT / MSVPN_PASSWORD. read -r keeps
# backslashes: the old prompt used plain read, which drops them, so such a password could never log
# in. Blanks around a pasted value are still trimmed.
msvpn_prompt_credentials() {
    MSVPN_ACCOUNT=""; MSVPN_PASSWORD=""
    read -r -p "VPN account: " MSVPN_ACCOUNT || true
    if [[ -t 0 ]]; then
        trap 'stty echo 2>/dev/null' EXIT
        stty -echo
    fi
    read -r -p "VPN password: " MSVPN_PASSWORD || true
    if [[ -t 0 ]]; then
        stty echo
        trap - EXIT
    fi
    echo ""
    [[ -n "$MSVPN_ACCOUNT" && -n "$MSVPN_PASSWORD" ]] || { echo "Account and password must not be empty." >&2; return 1; }
}

msvpn_write_auth_file() {   # msvpn_write_auth_file <path> <account> <password>
    local dir tmp
    dir="$(dirname "$1")"
    mkdir -p "$dir" || return 1
    tmp="$(umask 077 && mktemp "$1.XXXXXX")" || return 1
    printf '%s\n%s\n' "$2" "$3" > "$tmp" && chmod 600 "$tmp" && mv -f "$tmp" "$1"
}

# Problems with a credentials file, one per line (empty = usable). Never prints the content.
msvpn_check_auth_file() {
    local f="$1"
    [[ -e "$f" ]] || { echo "$f does not exist"; return 0; }
    [[ -r "$f" ]] || { echo "$f is not readable"; return 0; }
    [[ -n "$(sed -n 1p "$f" | tr -d '\r')" ]] || echo "$f: first line (account) is empty"
    [[ -n "$(sed -n 2p "$f" | tr -d '\r')" ]] || echo "$f: second line (password) is empty"
    return 0
}

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

msvpn_install_runtime() {   # msvpn_install_runtime <source dir>
    local f
    mkdir -p "$MSVPN_LIBEXEC" || return 1
    for f in "${MSVPN_RUNTIME_FILES[@]}"; do
        install -m 755 "$1/$f" "$MSVPN_LIBEXEC/$f" || return 1
    done
}

# Waits for the service's tunnel address. Returns 0 (address up; MSVPN_TUNNEL_ADDRS set), 4 (a failure
# that waiting does not fix) or 5 (timeout). MSVPN_VERDICT holds the classify line in both failure cases.
msvpn_wait_for_tunnel() {   # msvpn_wait_for_tunnel <start epoch> <timeout s>
    local t0="$1" deadline
    deadline=$(( $(date +%s) + $2 ))
    MSVPN_TUNNEL_ADDRS=""; MSVPN_VERDICT=""
    while :; do
        MSVPN_TUNNEL_ADDRS="$(msvpn_vpn_addrs)"
        [[ -n "$MSVPN_TUNNEL_ADDRS" ]] && return 0
        MSVPN_VERDICT="$(msvpn_journal_since "$t0" | msvpn_classify_log)"
        msvpn_is_definitive "$MSVPN_VERDICT" && return 4
        (( $(date +%s) >= deadline )) && return 5
        sleep 2
    done
}
