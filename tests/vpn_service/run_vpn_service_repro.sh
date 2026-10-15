#!/usr/bin/env bash
# =============================================================================
# Offline reproduction of the mediswarm-vpn service failures (#620)
# =============================================================================
# Builds a throwaway GoodAccess look-alike (OpenVPN 2.5/2.6 gateway with a test PKI, username/password
# on one shared client certificate, member address 172.24.4.69, and the push options the real gateway
# sends: redirect-gateway def1, DNS, explicit-exit-notify, ping 10 / ping-restart 120) and a "site host"
# container in which the site-side scripts run exactly as at a site:
#
#     sudo ./scripts/client_node_setup/setup_vpntunnel.sh -d TestSite -n -s
#
# with the account and password typed at the prompts. Each scenario runs with the scripts the sites
# have (BASELINE_REF, default f799667 = #418, unchanged until 1.9.0) and with this checkout's scripts.
#
# Runs unprivileged: both containers get only NET_ADMIN and /dev/net/tun, on an --internal docker network
# (no traffic leaves the host; the host's own tunnels and routes are untouched). The site container has no
# systemd: harness/fake_systemd.sh stands in for systemctl/journalctl and runs the installed unit files
# (Type, Restart, RestartSec, RestartPreventExitStatus, TimeoutStartSec, CapabilityBoundingSet, an
# emulation of ProtectHome, timers). Network access is needed once, to build the image.
#
# Requirements: docker, /dev/net/tun on the host. Not run in CI (needs docker with NET_ADMIN).
#
# Usage:
#   tests/vpn_service/run_vpn_service_repro.sh [options] [scenario ...]
#     --ubuntu 22.04|24.04   base image of both containers (default 22.04 = OpenVPN 2.5; 24.04 = 2.6)
#     --baseline-ref REF     git ref of the "before" scripts (default f799667)
#     --only-baseline        run only the before scripts
#     --only-fixed           run only the scripts of this checkout
#     --keep                 keep containers, network and logs for inspection
#   MSVPN_REPRO_LOGS=<dir>   copy the per-scenario logs there before cleaning up
#     -h                     help
#   Scenarios (default: all): happy docs_auth_path no_newline no_newline_timer backslash_pw wrong_pw
#                             nm_export user_group stale_ovpn second_kick second_reject slow_auth
#
# Exit status: 0 when every scenario behaves as expected with this checkout's scripts (the baseline
# results are informational: they show the failure a site sees).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

UBUNTU=22.04
BASELINE_REF=f799667
RUN_BASELINE=1
RUN_FIXED=1
KEEP=0
SUBNET="${MSVPN_REPRO_SUBNET:-10.231.97.0/24}"
SCENARIOS=()
ALL_SCENARIOS=(happy docs_auth_path no_newline no_newline_timer backslash_pw wrong_pw nm_export user_group
               stale_ovpn second_kick second_reject slow_auth)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ubuntu) UBUNTU="$2"; shift ;;
        --baseline-ref) BASELINE_REF="$2"; shift ;;
        --only-baseline) RUN_FIXED=0 ;;
        --only-fixed) RUN_BASELINE=0 ;;
        --keep) KEEP=1 ;;
        -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) SCENARIOS+=("$1") ;;
    esac
    shift
done
[[ ${#SCENARIOS[@]} -gt 0 ]] || SCENARIOS=("${ALL_SCENARIOS[@]}")

IMAGE="mediswarm-vpn-repro:${UBUNTU}"
TAG="mediswarm-vpn-repro-$$"
NET="$TAG-net"
GW="$TAG-gw"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/mediswarm-vpn-repro.XXXXXX")"
RESULTS="$WORK/results.tsv"
: > "$RESULTS"
FAILED=0

log() { printf '[repro %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

cleanup() {
    # MSVPN_REPRO_LOGS=<dir>: keep the per-scenario logs (setup output, service journal, gateway auth log).
    if [[ -n "${MSVPN_REPRO_LOGS:-}" ]]; then
        mkdir -p "$MSVPN_REPRO_LOGS" && cp "$WORK"/*.log "$WORK"/*.journal "$WORK"/results.tsv "$MSVPN_REPRO_LOGS"/ 2>/dev/null
    fi
    if [[ "$KEEP" == 1 ]]; then
        log "kept: containers $TAG-*, network $NET, logs in $WORK"
        return
    fi
    # Only the objects this run created (exact name prefix with this PID).
    local ids
    ids="$(docker ps -aq --filter "name=^${TAG}-")"
    [[ -n "$ids" ]] && docker rm -f $ids >/dev/null 2>&1
    docker network rm "$NET" >/dev/null 2>&1
    rm -rf "$WORK"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Image, scripts under test, PKI, gateway
# ---------------------------------------------------------------------------
command -v docker >/dev/null || { echo "docker not found"; exit 2; }
[[ -c /dev/net/tun ]] || { echo "/dev/net/tun missing on the host"; exit 2; }

log "building $IMAGE (ubuntu $UBUNTU)"
docker build -q -t "$IMAGE" --build-arg UBUNTU="$UBUNTU" -f "$HERE/Dockerfile" "$HERE" >/dev/null || exit 2

mkdir -p "$WORK/scripts/fixed" "$WORK/scripts/baseline" "$WORK/pki"
cp "$REPO"/scripts/client_node_setup/{setup_vpntunnel.sh,mediswarm-vpn.service,vpn_health_monitor.sh} "$WORK/scripts/fixed/"
for f in mediswarm_vpn_lib.sh vpn_service_start.sh; do
    [[ -f "$REPO/scripts/client_node_setup/$f" ]] && cp "$REPO/scripts/client_node_setup/$f" "$WORK/scripts/fixed/"
done
# The old monitor, for scenarios where the site installed its timer before (both variants).
git -C "$REPO" show "$BASELINE_REF:scripts/client_node_setup/vpn_health_monitor.sh" > "$WORK/scripts/baseline_monitor.sh" || exit 2
if [[ "$RUN_BASELINE" == 1 ]]; then
    for f in setup_vpntunnel.sh mediswarm-vpn.service vpn_health_monitor.sh; do
        git -C "$REPO" show "$BASELINE_REF:scripts/client_node_setup/$f" > "$WORK/scripts/baseline/$f" || {
            echo "cannot read $f at $BASELINE_REF"; exit 2; }
    done
fi
chmod +x "$WORK"/scripts/*.sh "$WORK"/scripts/*/*.sh 2>/dev/null

log "generating throwaway PKI and GoodAccess-shaped configs"
docker run --rm --network none -u "$(id -u):$(id -g)" -v "$HERE:/repro:ro" -v "$WORK/pki:/out" \
    --entrypoint /bin/bash "$IMAGE" /repro/harness/make_pki.sh /out >/dev/null || exit 2

docker network create --internal --subnet "$SUBNET" "$NET" >/dev/null || exit 2
log "starting test gateway (vpn-gw.test, 172.24.4.1/22)"
docker run -d --name "$GW" --network "$NET" --network-alias vpn-gw.test \
    --cap-add NET_ADMIN --device /dev/net/tun \
    -v "$HERE:/repro:ro" -v "$WORK/pki:/srv/pki:ro" \
    --entrypoint /bin/bash "$IMAGE" /repro/harness/server_entry.sh >/dev/null || exit 2

gw() { docker exec -i "$GW" "$@"; }
for _ in $(seq 1 30); do
    docker logs "$GW" 2>&1 | grep -q "Initialization Sequence Completed" && break
    sleep 1
done
gw sh -c 'mkdir -p /srv/vpn/ccd && echo "ifconfig-push 172.24.4.69 255.255.252.0" > /srv/vpn/ccd/cam'

gw_reset() {   # gw_reset <policy> <auth_delay>: no session of an earlier scenario may survive
    for _ in $(seq 1 15); do
        [[ -z "$(gw ls /srv/vpn/sessions 2>/dev/null)" ]] && break
        sleep 1
    done
    if [[ -n "$(gw ls /srv/vpn/sessions 2>/dev/null)" ]]; then
        log "gateway still holds a session; restarting it"
        docker restart -t 2 "$GW" >/dev/null
        gw sh -c 'rm -f /srv/vpn/sessions/*'
        for _ in $(seq 1 30); do
            docker logs --since 5s "$GW" 2>&1 | grep -q "Initialization Sequence Completed" && break
            sleep 1
        done
    fi
    # A login of the previous scenario may still be in its (slow_auth) password check.
    gw sh -c 'for i in $(seq 1 60); do pgrep -f "[c]heckpw.sh" >/dev/null || break; sleep 1; done'   # [c]: no self-match
    gw sh -c "echo $1 > /srv/vpn/session_policy; echo $2 > /srv/vpn/auth_delay; : > /srv/vpn/auth.log; : > /srv/vpn/sessions.log"
}

stop_site() {   # ends the site's tunnels gracefully (explicit-exit-notify) so the gateway frees the session
    docker exec "$1" sh -c 'systemctl stop mediswarm-vpn-health.timer mediswarm-vpn >/dev/null 2>&1;
        pkill -TERM -x openvpn; sleep 2' >/dev/null 2>&1
    docker rm -f "$1" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Site host helpers
# ---------------------------------------------------------------------------
start_site() {   # start_site <name>
    docker run -d --init --name "$1" --hostname site --network "$NET" \
        --cap-add NET_ADMIN --device /dev/net/tun \
        "$IMAGE" sleep infinity >/dev/null
}

vpn_addrs() {   # vpn_addrs <site>: "iface addr" for every 172.24.4.0/22 address
    docker exec "$1" ip -o -4 addr show 2>/dev/null \
        | awk '{split($4,a,"/"); if (a[1] ~ /^172\.24\.[4-7]\./) print $2, a[1]}'
}

service_prop() { docker exec "$1" systemctl show -p "$2" --value mediswarm-vpn 2>/dev/null; }

# An independent, hand-started tunnel with the same member account (what RUMC runs; CAM's
# NetworkManager profile is the same thing as far as the gateway can tell).
start_handstarted_tunnel() {   # start_handstarted_tunnel <site>
    docker exec -i "$1" sh -c 'mkdir -p /root/hand && cat > /root/hand/hand.ovpn &&
        printf "cam\nCamPass-2026\n" > /root/hand/hand.auth && chmod 600 /root/hand/hand.auth &&
        sed -i "s#^auth-user-pass\r\$#auth-user-pass /root/hand/hand.auth#" /root/hand/hand.ovpn &&
        openvpn --config /root/hand/hand.ovpn --daemon handstarted --log /root/hand/hand.log --writepid /root/hand/pid' \
        < "$WORK/pki/goodaccess.ovpn"
    for _ in $(seq 1 20); do
        [[ -n "$(vpn_addrs "$1")" ]] && return 0
        sleep 1
    done
    echo "hand-started tunnel did not come up"; return 1
}

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
# ovpn file | password typed | gateway session policy | gateway auth delay (s) | before setup: no /
# tunnel (hand-started tunnel of the same account is up) / timer (health timer of the old version installed) |
# health monitor answer | observation after setup (s) | VPN_WAIT_SECONDS for the fixed setup |
# expected outcome with the fixed scripts: up (rc 0 + address) / rc=N:<reason code>
scenario_params() {
    case "$1" in
        happy)          echo 'goodaccess.ovpn      CamPass-2026    kick   0  no     n  30  60 up' ;;
        docs_auth_path) echo 'docs_auth.ovpn       CamPass-2026    kick   0  no     n  30  60 up' ;;
        no_newline)     echo 'goodaccess_nonl.ovpn CamPass-2026    kick   0  no     n  70  60 up' ;;
        no_newline_timer) echo 'goodaccess_nonl.ovpn CamPass-2026  kick   0  timer  n  70  60 up' ;;
        backslash_pw)   echo 'goodaccess.ovpn      Cam\Pass-2026   kick   0  no     n  30  60 up' ;;
        wrong_pw)       echo 'goodaccess.ovpn      NotThePassword  kick   0  no     n  30  60 rc=4:auth_failed' ;;
        nm_export)      echo 'nm_export.ovpn       CamPass-2026    kick   0  no     n  30  60 up' ;;
        user_group)     echo 'user_group.ovpn      CamPass-2026    kick   0  no     n  30  60 up' ;;
        stale_ovpn)     echo 'stale.ovpn           CamPass-2026    kick   0  no     n  70  75 rc=5:tls_timeout' ;;
        second_kick)    echo 'goodaccess.ovpn      CamPass-2026    kick   0  tunnel n  45  60 rc=3:other' ;;
        second_reject)  echo 'goodaccess.ovpn      CamPass-2026    reject 0  tunnel Y  45  60 rc=3:other' ;;
        slow_auth)      echo 'goodaccess.ovpn      CamPass-2026    kick   40 timer  n  100 90 up' ;;
        *) return 1 ;;
    esac
}

classify() {   # the checkout's classifier, applied to both variants' service logs
    bash -c "source '$REPO/scripts/client_node_setup/mediswarm_vpn_lib.sh'; msvpn_classify_log" < "$1" | cut -f1
}

run_one() {   # run_one <baseline|fixed> <scenario>
    local variant="$1" sc="$2" site="$TAG-$1-$2"
    local ovpn typed policy delay pre health obs waits expect
    read -r ovpn typed policy delay pre health obs waits expect < <(scenario_params "$sc") || {
        echo "unknown scenario $sc"; return 1; }
    log "--- $sc / $variant"

    # The account's real password; for backslash_pw it contains the backslash, as in the email a site gets.
    local real_pw="CamPass-2026"
    [[ "$sc" == backslash_pw ]] && real_pw='Cam\Pass-2026'
    gw sh -c "printf '%s\n' 'cam:$real_pw' > /srv/vpn/users"
    gw_reset "$policy" "$delay"

    start_site "$site" || { printf '%s\t%s\tERROR site did not start\n' "$sc" "$variant" >> "$RESULTS"; return; }
    docker exec "$site" mkdir -p /opt/MediSwarm/scripts/client_node_setup /opt/MediSwarm/assets/openvpn_configs/good_access
    docker cp "$WORK/scripts/$variant/." "$site:/opt/MediSwarm/scripts/client_node_setup/" >/dev/null
    docker cp "$WORK/pki/$ovpn" "$site:/opt/MediSwarm/assets/openvpn_configs/good_access/TestSite.ovpn" >/dev/null
    if [[ "$sc" == nm_export ]]; then
        docker exec "$site" mkdir -p /home/siteuser/.cert
        docker cp "$WORK/pki/nm-openvpn" "$site:/home/siteuser/.cert/" >/dev/null
        docker exec "$site" sh -c 'chown -R siteuser:siteuser /home/siteuser/.cert && chmod -R go-rwx /home/siteuser/.cert'
    fi
    if [[ "$pre" == tunnel ]]; then
        start_handstarted_tunnel "$site" || true
        gw sh -c ': > /srv/vpn/auth.log; : > /srv/vpn/sessions.log'
    fi
    if [[ "$pre" == timer ]]; then
        # The site installed the health timer with the old version (as CAM did on 15 Sep).
        docker exec "$site" mkdir -p /opt/old
        docker cp "$WORK/scripts/baseline_monitor.sh" "$site:/opt/old/vpn_health_monitor.sh" >/dev/null
        docker exec "$site" /opt/old/vpn_health_monitor.sh --install-timer >/dev/null 2>&1
    fi
    local gw_since; gw_since="$(date +%s)"
    local pre_addrs; pre_addrs="$(vpn_addrs "$site" | tr '\n' ' ')"

    # The site types the account, the password, then answers the health-monitor prompt (harness/drive_setup.exp).
    local t0 out rc
    t0=$(date +%s)
    out="$(docker exec -e VPN_WAIT_SECONDS="$waits" -e REPRO_ACCOUNT=cam -e REPRO_PASSWORD="$typed" \
        -e REPRO_HEALTH="$health" "$site" expect -f /usr/local/lib/fake-systemd/drive_setup.exp 2>&1)"
    rc=$?
    printf '%s\n' "$out" | tr -d '\r' > "$WORK/$sc.$variant.setup.log"

    # Observe what the site would see afterwards.
    local pids="" newaddr=no first_up="" end=$(( $(date +%s) + obs )) p now_addrs
    while (( $(date +%s) < end )); do
        p="$(service_prop "$site" MainPID)"
        [[ -n "$p" && "$p" != 0 && " $pids " != *" $p "* ]] && pids="$pids $p"
        now_addrs="$(vpn_addrs "$site" | tr '\n' ' ')"
        if [[ -n "${now_addrs// /}" && "$now_addrs" != "$pre_addrs" ]]; then
            newaddr=yes; [[ -z "$first_up" ]] && first_up=$(( $(date +%s) - t0 ))
        fi
        sleep 2
    done
    now_addrs="$(vpn_addrs "$site" | tr '\n' ' ')"
    local active nrestarts
    active="$(service_prop "$site" ActiveState)"
    nrestarts="$(service_prop "$site" NRestarts)"
    docker exec "$site" journalctl -u mediswarm-vpn -u mediswarm-vpn-health --no-pager > "$WORK/$sc.$variant.journal" 2>&1
    docker exec "$site" journalctl -u mediswarm-vpn -o cat --no-pager > "$WORK/$sc.$variant.openvpn.log" 2>&1
    gw cat /srv/vpn/auth.log > "$WORK/$sc.$variant.gw_auth.log"
    gw cat /srv/vpn/sessions.log > "$WORK/$sc.$variant.gw_sessions.log"
    local accepted rejected reason gw_kicks
    gw_kicks="$(docker logs --since "$gw_since" "$GW" 2>&1 | grep -c 'previous active sessions')"
    accepted="$(grep -c verdict=accept "$WORK/$sc.$variant.gw_auth.log")"
    rejected="$(grep -c verdict=reject "$WORK/$sc.$variant.gw_auth.log")"
    reason="$(classify "$WORK/$sc.$variant.openvpn.log")"
    local health_restarts
    health_restarts="$(grep -c 'restarting mediswarm-vpn' "$WORK/$sc.$variant.journal")"

    # Verdict for the fixed scripts; their --diagnose report must work in every state.
    local verdict="-"
    if [[ "$variant" == fixed ]]; then
        verdict=PASS
        docker exec "$site" /opt/MediSwarm/scripts/client_node_setup/setup_vpntunnel.sh --diagnose \
            > "$WORK/$sc.$variant.diagnose.log" 2>&1 || verdict=FAIL
        case "$expect" in
            up) [[ "$rc" == 0 && "$newaddr" == yes ]] || verdict=FAIL ;;
            rc=3:other) [[ "$rc" == 3 && "$newaddr" == no && "$accepted" == 0 && "$gw_kicks" == 0 && "$now_addrs" == "$pre_addrs" ]] || verdict=FAIL ;;
            rc=*) [[ "$rc" == "${expect:3:1}" ]] && grep -q "$(cut -d: -f2 <<< "$expect")" <<< "$reason" || verdict=FAIL ;;
        esac
        [[ "$verdict" == PASS ]] || FAILED=1
    fi
    printf '%s\t%s\tsetup_rc=%s\tnew_vpn_addr=%s\tfirst_up=%ss\tservice=%s\topenvpn_pids=%s\tNRestarts=%s\thealth_restarts=%s\tgw_accept=%s\tgw_reject=%s\tgw_kicks=%s\tpre=[%s]\tnow=[%s]\treason=%s\texpect=%s\t%s\n' \
        "$sc" "$variant" "$rc" "$newaddr" "${first_up:--}" "${active:-?}" "$(wc -w <<< "$pids")" "${nrestarts:-?}" \
        "$health_restarts" "$accepted" "$rejected" "$gw_kicks" "${pre_addrs% }" "${now_addrs% }" "${reason:-none}" \
        "$( [[ $variant == fixed ]] && echo "$expect" || echo -)" "$verdict" >> "$RESULTS"
    tail -1 "$RESULTS"
    [[ "$variant" == fixed ]] && grep -E "did not come up|Another GoodAccess|tunnel up after|OpenVPN:" "$WORK/$sc.$variant.setup.log" | head -3 | sed 's/^/    setup: /'
    [[ "$KEEP" == 1 ]] || stop_site "$site"
}

for sc in "${SCENARIOS[@]}"; do
    [[ "$RUN_BASELINE" == 1 ]] && run_one baseline "$sc"
    [[ "$RUN_FIXED" == 1 ]] && run_one fixed "$sc"
done

echo
echo "================ results ($(docker exec "$GW" openvpn --version | head -1 | cut -d' ' -f1-2) on ubuntu $UBUNTU) ================"
column -t -s $'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
if [[ "$RUN_FIXED" == 1 ]]; then
    [[ "$FAILED" == 0 ]] && echo "fixed scripts: all scenarios as expected" || echo "fixed scripts: UNEXPECTED results above"
fi
exit "$FAILED"
