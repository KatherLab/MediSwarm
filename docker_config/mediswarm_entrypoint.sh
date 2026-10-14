#!/bin/bash
# ENTRYPOINT of the ODELIA image: exec the container command unchanged, after clearing a stale
# NVFlare lock in the startup kit when (and only when) the command launches the kit's daemon.
#
# Why the image does this: docker.sh starts the client/server container with
# --restart=unless-stopped and the command `bash -c "nohup ./start.sh ..."`. sub_start.sh refuses
# to start while <kit>/daemon_pid.fl names a live PID ("There seems to be one instance, pid=N").
# After a power loss the file survives in the bind-mounted kit, and in the restarted container
# that PID is often taken again (PIDs of a fresh namespace repeat, sub_start.sh can get its own
# old number), so the container shows "Up" while the node stays offline until someone deletes
# the file (F3; coordinator 30 Sep, USZ 3 Oct, VHIO 6 Oct 2026). Kits built from main clear the
# lock in the container command (#630); the sites keep their 1.8.1 kits, so the image does it.
#
# The locks <kit>/daemon_pid.fl and <kit>/pid.fl (what sub_start.sh uses) are removed only if ALL hold:
#   1. the command launches the kit daemon: ./start.sh or ./sub_start.sh, or an absolute path to
#      one, as the first word of a command (after nohup/exec), directly or inside `bash -c "..."`;
#   2. the script's directory is an NVFlare startup kit: it holds sub_start.sh and fed_server.json
#      (server), or fed_client.json with TRAINING_MODE=swarm (only docker.sh --start_client sets it);
#   3. the PID namespace is fresh: no process exists but this script and its ancestors (docker-init
#      under --init). No NVFlare daemon of this container can be alive, so any PID in a lock is stale.
# Every other command (preflight, local training, external validation, --run_script, fl_admin.sh,
# CI's `docker run ... python3 ...`) is exec'ed without touching anything. `docker run --entrypoint`
# and `docker exec` bypass this script. Escape hatch for a site:
#   MEDISWARM_DOCKER_OPTIONS="-e MEDISWARM_ENTRYPOINT_CLEAR_LOCKS=0" in startup/image.conf.
#
# Kept free of forks until the namespace check has run: a child would itself show up in /proc.

_PROC_ROOT=/proc
_LOG_PREFIX="[mediswarm-entrypoint]"

# Sets _KIT_SCRIPT to the start.sh/sub_start.sh path the command launches (as written), or "".
_find_kit_launcher () {
    _KIT_SCRIPT=""
    local script_re='^(\./|/([^[:space:];&|()<>]*/)?)(sub_)?start\.sh$'
    if [[ $# -ge 1 && "$1" =~ $script_re ]]; then
        _KIT_SCRIPT=$1
        return
    fi
    if [[ $# -ge 3 && "$2" == "-c" && ( "$1" == "/bin/bash" || "$1" == "bash" || "$1" == "/bin/sh" || "$1" == "sh"
          || "$1" == "/usr/bin/bash" ) ]]; then
        local cmd_re='(^|[;&|(])[[:space:]]*((nohup|exec)[[:space:]]+)*((\./|/([^[:space:];&|()<>]*/)?)(sub_)?start\.sh)([[:space:];&|)<>]|$)'
        if [[ "$3" =~ $cmd_re ]]; then
            _KIT_SCRIPT=${BASH_REMATCH[4]}
        fi
    fi
}

# True when the launcher's directory is an NVFlare startup kit this container may restart.
_is_kit_startup_dir () {
    local dir=$1
    [[ -f "$dir/sub_start.sh" ]] || return 1
    if [[ -f "$dir/fed_server.json" ]]; then
        return 0
    fi
    [[ -f "$dir/fed_client.json" && "${TRAINING_MODE:-}" == "swarm" ]]
}

# True when /proc shows no process except this shell and its ancestors, i.e. the container's PID
# namespace was created for this start. Fails closed: unreadable or unusual /proc -> not fresh.
_pid_namespace_is_fresh () {
    [[ -d "$_PROC_ROOT/$$" ]] || return 1
    local -A ours=()
    local pid=$$ ppid line entry
    while [[ "$pid" =~ ^[0-9]+$ && "$pid" != 0 && -z "${ours[$pid]:-}" ]]; do
        ours[$pid]=1
        ppid=""
        if [[ -r "$_PROC_ROOT/$pid/status" ]]; then
            while IFS= read -r line; do
                if [[ "$line" == PPid:* ]]; then
                    ppid=${line#PPid:}
                    ppid=${ppid//[[:space:]]/}
                    break
                fi
            done < "$_PROC_ROOT/$pid/status"
        fi
        pid=$ppid
    done
    for entry in "$_PROC_ROOT"/[0-9]*; do
        [[ -d "$entry" ]] || continue   # exited since the glob expanded
        [[ -n "${ours[${entry##*/}]:-}" ]] && continue
        return 1
    done
    return 0
}

_clear_stale_kit_locks () {
    _find_kit_launcher "$@"
    [[ -n "$_KIT_SCRIPT" ]] || return 0

    local startup_dir
    if [[ "$_KIT_SCRIPT" == /* ]]; then
        startup_dir=${_KIT_SCRIPT%/*}
        startup_dir=${startup_dir:-/}
    else
        startup_dir=$PWD
    fi
    _is_kit_startup_dir "$startup_dir" || return 0

    # Shown in the log only; the files are addressed as sub_start.sh addresses them ($DIR/..).
    local kit_root=${startup_dir%/*}
    kit_root=${kit_root:-/}

    if ! _pid_namespace_is_fresh; then
        printf '%s other processes are visible in this container (shared PID namespace?); leaving the locks in %s alone\n' \
            "$_LOG_PREFIX" "$kit_root" >&2
        return 0
    fi

    local name held
    for name in daemon_pid.fl pid.fl; do
        [[ -e "$startup_dir/../$name" ]] || continue
        held=""
        read -r held < "$startup_dir/../$name" 2>/dev/null
        if rm -f -- "$startup_dir/../$name"; then
            printf '%s removed stale %s/%s (pid %s): the container just started, so no NVFlare daemon of it is running\n' \
                "$_LOG_PREFIX" "$kit_root" "$name" "${held:-?}" >&2
        else
            printf '%s could not remove stale %s/%s; start.sh may refuse to start\n' "$_LOG_PREFIX" "$kit_root" "$name" >&2
        fi
    done
    return 0
}

# exec the command with exactly the environment Docker passed to this process. bash exports PWD and
# SHLVL to whatever it runs; replaying /proc/self/environ keeps a `python3 ...` command's environment
# identical to a run without this entrypoint. A first word containing '=' would be read by env as an
# assignment, so that (always failing) case keeps plain exec semantics.
_exec_unchanged () {
    local -a original_env=()
    local entry replay=1
    if [[ "$1" != *=* && -r "$_PROC_ROOT/$$/environ" && -x /usr/bin/env ]] \
        && mapfile -d '' -t original_env < "$_PROC_ROOT/$$/environ"; then
        for entry in "${original_env[@]}"; do
            [[ "$entry" == *=* ]] || replay=0   # env would take a bare word for the command
        done
        if [[ "$replay" == 1 ]]; then
            exec /usr/bin/env -i -- "${original_env[@]}" "$@"
        fi
    fi
    exec "$@"
}

main () {
    if [[ $# -eq 0 ]]; then
        set -- /bin/bash   # the image's CMD; only reachable if CMD is overridden with nothing
    fi
    if [[ "${MEDISWARM_ENTRYPOINT_CLEAR_LOCKS:-1}" != "0" ]]; then
        _clear_stale_kit_locks "$@"
    fi
    _exec_unchanged "$@"
}

# Run only when executed; the unit tests source this file to drive the pieces separately.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
