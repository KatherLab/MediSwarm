#!/bin/bash
# Minimal systemctl/journalctl stand-in for the offline VPN reproduction (installed as both names).
#
# Booting real systemd in a container needs CAP_SYS_ADMIN, and the reproduction runs unprivileged
# (NET_ADMIN and /dev/net/tun only). This stand-in implements what setup_vpntunnel.sh,
# vpn_service_start.sh and vpn_health_monitor.sh (old and new) use, following the unit files they
# install in /etc/systemd/system:
#   Type=simple|notify|oneshot (notify: "active" once OpenVPN logs "Initialization Sequence Completed"),
#   ExecStart, WorkingDirectory, Restart=always, RestartSec, RestartPreventExitStatus, TimeoutStartSec,
#   TimeoutStopSec, Environment, CapabilityBoundingSet (applied with setpriv --bounding-set),
#   ProtectHome=true (emulated: the process loses CAP_DAC_OVERRIDE/CAP_DAC_READ_SEARCH, so it cannot read
#   the mode-700 home of the site user; real systemd hides /home instead, same outcome for OpenVPN),
#   timers (OnUnitActiveSec; the host "booted long ago", so OnBootSec has elapsed and the first run is
#   immediate, as systemd does for an elapsed monotonic timer).
# Not emulated: ProtectSystem, LimitNPROC, sd_notify itself, journald rate limits.
set -u
SELF="$(readlink -f "$0")"
STATE=/run/fake-systemd
JOURNAL=/var/log/fake-journal.log
UNITDIR=/etc/systemd/system
mkdir -p "$STATE"; touch "$JOURNAL"

unit_name() { case "$1" in *.service|*.timer) echo "$1" ;; *) echo "$1.service" ;; esac; }
ufile() { echo "$UNITDIR/$1"; }
uget() {   # uget <unit> <Key>: last value
    local f; f="$(ufile "$1")"; [[ -f "$f" ]] || return 0
    sed -n "s/^$2=//p" "$f" | tail -1
}
sdir() { mkdir -p "$STATE/$1"; echo "$STATE/$1"; }
sget() { cat "$STATE/$1/$2" 2>/dev/null || echo "${3:-}"; }
sset() { mkdir -p "$STATE/$1"; printf '%s\n' "$3" > "$STATE/$1/$2"; }
jlog() { printf '%s\t%s\t%s\n' "$(date +%s)" "$1" "$2" >> "$JOURNAL"; }
secs() { local v="${1:-0}"; v="${v%s}"; [[ "$v" == infinity ]] && echo 999999 || echo "${v:-0}"; }
uptime_us() { awk '{ printf "%.0f", $1 * 1000000 }' /proc/uptime; }   # %d overflows in mawk

# ------------------------------------------------------------------ service supervisor
caps_for() {   # setpriv --bounding-set argument for the unit's CapabilityBoundingSet/ProtectHome
    local u="$1" want have out="" c
    want="$(uget "$u" CapabilityBoundingSet | tr 'A-Z' 'a-z' | sed 's/cap_//g')"
    have=" $(setpriv -d | sed -n 's/^Capability bounding set: //p' | tr ',' ' ') "
    if [[ -z "$want" ]]; then want="$have"; fi
    for c in $want; do
        [[ "$(uget "$u" ProtectHome)" == true && ( "$c" == dac_override || "$c" == dac_read_search ) ]] && continue
        [[ "$have" == *" $c "* ]] && out+=",+$c"
    done
    echo "-all$out"
}

supervise() {
    local u="$1" type wd restart rsec prevent tstart tstop caps status inv t0 result
    type="$(uget "$u" Type)"; type="${type:-simple}"
    wd="$(uget "$u" WorkingDirectory)"; wd="${wd:-/}"
    restart="$(uget "$u" Restart)"; restart="${restart:-no}"
    rsec="$(secs "$(uget "$u" RestartSec)")"; [[ "$rsec" == 0 ]] && rsec=1
    prevent=" $(uget "$u" RestartPreventExitStatus) "
    tstart="$(secs "$(uget "$u" TimeoutStartSec)")"; [[ "$tstart" == 0 ]] && tstart=90
    tstop="$(secs "$(uget "$u" TimeoutStopSec)")"; [[ "$tstop" == 0 ]] && tstop=90
    caps="$(caps_for "$u")"
    local -a argv
    read -r -a argv <<< "$(uget "$u" ExecStart)"
    local mainpid="" stopping=0
    trap 'stopping=1; [[ -n "$mainpid" ]] && kill -TERM "$mainpid" 2>/dev/null' TERM
    while :; do
        inv="$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
        sset "$u" InvocationID "$inv"; sset "$u" ActiveState activating; sset "$u" SubState start
        sset "$u" ExecMainStartTimestampMonotonic "$(uptime_us)"; sset "$u" ExecMainStartTimestamp "$(date)"
        jlog "$u" "Starting $u..."
        rm -f "$STATE/$u/fifo"; mkfifo "$STATE/$u/fifo"
        (   # journal reader; also the notify emulation
            while IFS= read -r line; do
                jlog "$u" "$line"
                if [[ "$type" == notify && "$line" == *"Initialization Sequence Completed"* ]] \
                        && [[ "$(sget "$u" ActiveState)" == activating ]]; then
                    sset "$u" ActiveState active; sset "$u" SubState running; jlog "$u" "Started $u."
                fi
            done < "$STATE/$u/fifo"
        ) &
        ( cd "$wd" && exec setpriv --bounding-set "$caps" -- "${argv[@]}" ) > "$STATE/$u/fifo" 2>&1 &
        mainpid=$!
        sset "$u" MainPID "$mainpid"
        if [[ "$type" != notify ]]; then sset "$u" ActiveState active; sset "$u" SubState running; fi
        t0="$(date +%s)"; result=success
        while kill -0 "$mainpid" 2>/dev/null; do
            if [[ "$stopping" == 0 && "$type" == notify && "$(sget "$u" ActiveState)" == activating ]] \
                    && (( $(date +%s) - t0 >= tstart )); then
                jlog "$u" "$u: start operation timed out. Terminating."
                result=timeout; kill -TERM "$mainpid" 2>/dev/null
            fi
            if [[ "$stopping" == 1 ]]; then
                local t1; t1="$(date +%s)"
                while kill -0 "$mainpid" 2>/dev/null && (( $(date +%s) - t1 < tstop )); do sleep 0.2; done
                kill -KILL "$mainpid" 2>/dev/null
            fi
            sleep 0.5
        done
        wait "$mainpid"; status=$?
        wait 2>/dev/null
        sset "$u" MainPID 0; sset "$u" ExecMainStatus "$status"
        if [[ "$stopping" == 1 ]]; then
            sset "$u" ActiveState inactive; sset "$u" SubState dead; jlog "$u" "Stopped $u."; exit 0
        fi
        [[ "$result" == success && "$status" != 0 ]] && result=exit-code
        sset "$u" Result "$result"
        jlog "$u" "$u: Main process exited, code=exited, status=$status"
        if [[ "$restart" != always || "$prevent" == *" $status "* ]]; then
            sset "$u" ActiveState failed; sset "$u" SubState failed; jlog "$u" "$u: Failed with result '$result'."
            exit 0
        fi
        sset "$u" ActiveState activating; sset "$u" SubState auto-restart
        sleep "$rsec" & wait $!
        [[ "$stopping" == 1 ]] && { sset "$u" ActiveState inactive; sset "$u" SubState dead; exit 0; }
        sset "$u" NRestarts $(( $(sget "$u" NRestarts 0) + 1 ))
        jlog "$u" "$u: Scheduled restart job, restart counter is at $(sget "$u" NRestarts)."
    done
}

run_oneshot() {
    local u="$1" line
    local -a env=()
    while IFS= read -r line; do env+=("$line"); done < <(sed -n 's/^Environment=//p' "$(ufile "$u")")
    sset "$u" ActiveState activating; sset "$u" ExecMainStartTimestampMonotonic "$(uptime_us)"
    local -a argv; read -r -a argv <<< "$(uget "$u" ExecStart)"
    env "${env[@]}" "${argv[@]}" 2>&1 | while IFS= read -r line; do jlog "$u" "$line"; done
    sset "$u" ActiveState inactive
}

timer_loop() {
    local t="$1" svc="${1%.timer}.service" every
    every="$(secs "$(uget "$t" OnUnitActiveSec)")"; [[ "$every" == 0 ]] && every=30
    while :; do run_oneshot "$svc"; sleep "$every"; done
}

start_unit() {   # start_unit <unit> <block 0|1>
    local u="$1"
    [[ -f "$(ufile "$u")" ]] || { echo "Failed to start $u: Unit $u not found." >&2; return 5; }
    case "$u" in
        *.timer)
            stop_unit "$u"
            setsid "$SELF" __timer "$u" </dev/null >/dev/null 2>&1 &
            sset "$u" pid $!; sset "$u" ActiveState active ;;
        *)
            if [[ "$(uget "$u" Type)" == oneshot ]]; then run_oneshot "$u"; return 0; fi
            stop_unit "$u"
            sset "$u" ActiveState activating
            setsid "$SELF" __supervise "$u" </dev/null >/dev/null 2>&1 &
            sset "$u" pid $!
            if [[ "$2" == 1 && "$(uget "$u" Type)" == notify ]]; then
                local t0; t0="$(date +%s)"
                sleep 0.5
                while [[ "$(sget "$u" ActiveState)" == activating && "$(sget "$u" SubState)" == start ]] \
                        && (( $(date +%s) - t0 < $(secs "$(uget "$u" TimeoutStartSec)") + 5 )); do sleep 0.5; done
                [[ "$(sget "$u" ActiveState)" == active ]] || { echo "Job for $u failed." >&2; return 1; }
            fi ;;
    esac
}

stop_unit() {
    local u="$1" pid
    pid="$(sget "$u" pid)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
        kill -KILL -- "-$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
    fi
    rm -f "$STATE/$u/pid"
    [[ -d "$STATE/$u" ]] && { sset "$u" ActiveState inactive; sset "$u" SubState dead; sset "$u" MainPID 0; }
    return 0
}

prop() {   # prop <unit> <Property>
    local u="$1"
    case "$2" in
        UnitFileState) [[ -e "$STATE/$u/enabled" ]] && echo enabled || echo disabled ;;
        ActiveState) sget "$u" ActiveState inactive ;;
        SubState) sget "$u" SubState dead ;;
        MainPID|NRestarts|ExecMainStatus|ExecMainStartTimestampMonotonic) sget "$u" "$2" 0 ;;
        ExecStart) echo "{ path=$(uget "$u" ExecStart | awk '{print $1}') ; argv[]=$(uget "$u" ExecStart) }" ;;
        *) sget "$u" "$2" "" ;;
    esac
}

systemctl_main() {
    local cmd="" noblock=0 now=0 value=0 a
    local -a units=() props=()
    while [[ $# -gt 0 ]]; do
        a="$1"; shift
        case "$a" in
            --no-block) noblock=1 ;;
            --now) now=1 ;;
            --value) value=1 ;;
            -p) props+=("$1"); shift ;;
            -p*) props+=("${a#-p}") ;;
            --property=*) props+=("${a#--property=}") ;;
            -q|--quiet|--no-pager|--no-legend|--full|-l) ;;
            -*) ;;
            *) if [[ -z "$cmd" ]]; then cmd="$a"; else units+=("$(unit_name "$a")"); fi ;;
        esac
    done
    local u rc=0
    case "$cmd" in
        daemon-reload|reset-failed|mask|unmask) return 0 ;;
        is-system-running) echo running ;;
        enable)  for u in "${units[@]}"; do sdir "$u" >/dev/null; touch "$STATE/$u/enabled"; [[ $now == 1 ]] && start_unit "$u" 0; done ;;
        disable) for u in "${units[@]}"; do rm -f "$STATE/$u/enabled"; [[ $now == 1 ]] && stop_unit "$u"; done; return 0 ;;
        start|restart|try-restart) for u in "${units[@]}"; do start_unit "$u" $(( 1 - noblock )) || rc=$?; done; return $rc ;;
        stop) for u in "${units[@]}"; do stop_unit "$u"; done ;;
        is-active) for u in "${units[@]}"; do a="$(prop "$u" ActiveState)"; echo "$a"; [[ "$a" == active ]] || rc=3; done; return $rc ;;
        is-enabled)
            for u in "${units[@]}"; do
                [[ -f "$(ufile "$u")" ]] || { rc=1; continue; }
                a="$(prop "$u" UnitFileState)"; echo "$a"; [[ "$a" == enabled ]] || rc=1
            done; return $rc ;;
        show)
            for u in "${units[@]}"; do for a in "${props[@]}"; do
                if [[ $value == 1 ]]; then prop "$u" "$a"; else echo "$a=$(prop "$u" "$a")"; fi
            done; done ;;
        status)
            for u in "${units[@]}"; do
                echo "* $u"; echo "   Active: $(prop "$u" ActiveState) ($(prop "$u" SubState)); Main PID: $(prop "$u" MainPID)"
                journalctl_main -u "$u" -n 10 -o cat
            done ;;
        *) echo "fake systemctl: unsupported command '$cmd'" >&2; return 1 ;;
    esac
    return $rc
}

journalctl_main() {
    local since=0 n=0 fmt=short a
    local -a units=()
    while [[ $# -gt 0 ]]; do
        a="$1"; shift
        case "$a" in
            -u) units+=("$(unit_name "$1")"); shift ;;
            --unit=*) units+=("$(unit_name "${a#--unit=}")") ;;
            -t) units+=("$1.service"); shift ;;
            --since) since="$1"; shift ;;
            --since=*) since="${a#--since=}" ;;
            -n) n="$1"; shift ;;
            -n*) n="${a#-n}" ;;
            -o) fmt="$1"; shift ;;
            -o*) fmt="${a#-o}" ;;
            *) ;;
        esac
    done
    if [[ "$since" == @* ]]; then since="${since#@}"; elif [[ "$since" != 0 ]]; then since="$(date -d "$since" +%s)"; fi
    awk -F'\t' -v since="$since" -v want=" ${units[*]} " -v fmt="$fmt" '
        $1 >= since && (want == "  " || index(want, " " $2 " ")) {
            if (fmt == "cat") print $3
            else { cmd = "date -d @" $1 " +%Y-%m-%dT%H:%M:%S"; cmd | getline ts; close(cmd); print ts " site " $2 ": " $3 }
        }' "$JOURNAL" | { if [[ "$n" != 0 ]]; then tail -n "$n"; else cat; fi; }
}

case "${1:-}" in
    __supervise) supervise "$2"; exit 0 ;;
    __timer) timer_loop "$2"; exit 0 ;;
esac
case "$(basename "$0")" in
    journalctl) journalctl_main "$@" ;;
    *) systemctl_main "$@" ;;
esac
