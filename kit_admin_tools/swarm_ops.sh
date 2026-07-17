#!/usr/bin/env bash
#
# swarm_ops.sh — operator CLI for the live ODELIA/MediSwarm NVFlare swarm.
#
# Wraps the admin console (./docker.sh --no_pull) and the container/host ops that
# were previously hand-driven via ad-hoc expect + ssh, and bakes in the failure
# modes learned the hard way (see issue #394 / docs/SWARM_FAILURE_MODES.md):
#
#   * A leftover `odelia_swarm_admin_*` container blocks the NEXT admin session by
#     name conflict (console silently returns nothing) -> we `docker rm -f` it
#     after every session.
#   * A stalled job does not abort cleanly; `restart-server` is the reliable clear.
#   * After `docker rm -f` of a client the stale `daemon_pid.fl` lock silently
#     prevents restart -> `restart-client` deletes it.
#   * `--restart=on-failure:5` does not survive a host reboot -> restart-* set
#     `unless-stopped` (also fixed at kit-build time in #393).
#
# Run this FROM the admin kit's startup dir (the one holding docker.sh and
# prepare_odelia_job.sh), or point at it with --admin-kit DIR.
#
# Credentials for restart-client come from ~/.mediswarm_secrets (never committed,
# never echoed): it must `export <SITE>_PASS=...`. Host/user come from a
# deploy_sites_*.conf (SITE_HOST / SITE_USER) or the --host/--user flags.
#
set -euo pipefail

# --------------------------------------------------------------------------- #
#  Locations / defaults
# --------------------------------------------------------------------------- #
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ADMIN_KIT="$DIR"
# The admin kit layout is <workspace>/<user-email>/startup/ ; derive the admin
# login name (the console's "User Name:") from the parent dir unless overridden.
ADMIN_USER="${SWARM_ADMIN_USER:-$(basename "$(dirname "$DIR")")}"
SERVER_LOG="${SWARM_SERVER_LOG:-}"          # server kit's nohup.out (for watch/progress)
SECRETS="${MEDISWARM_SECRETS:-$HOME/.mediswarm_secrets}"
ADMIN_CONTAINER_GLOB="odelia_swarm_admin"

err() { echo "swarm_ops: $*" >&2; }
die() { err "$*"; exit 1; }

strip_ansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }

# Remove any leftover admin-console container so it cannot block the next session.
clean_admin_container() {
  docker ps -a --format '{{.Names}}' 2>/dev/null \
    | grep -iE "$ADMIN_CONTAINER_GLOB" \
    | xargs -r docker rm -f >/dev/null 2>&1 || true
}

# Locate the server nohup.out if not provided (best effort).
resolve_server_log() {
  [ -n "$SERVER_LOG" ] && { echo "$SERVER_LOG"; return; }
  # admin kit is <ws>/<user>/startup; the server kit is a sibling <ws>/<serverhost>/startup
  local ws; ws="$(cd "$ADMIN_KIT/../.." && pwd -P)"
  find "$ws" -maxdepth 3 -path '*/startup/nohup.out' 2>/dev/null \
    | grep -viE "/${ADMIN_USER}/" | head -1
}

# --------------------------------------------------------------------------- #
#  Admin-console driver (expect). Usage: run_admin "cmd1" "cmd2" ...
#  Prints the raw (ANSI-stripped) console output; always cleans the container.
# --------------------------------------------------------------------------- #
run_admin() {
  command -v expect >/dev/null 2>&1 || die "expect not found (needed to drive the admin console)"
  [ -x "$ADMIN_KIT/docker.sh" ] || die "no docker.sh in $ADMIN_KIT (run from the admin kit startup dir or pass --admin-kit)"
  local timeout="${ADMIN_TIMEOUT:-90}"
  local exp; exp="$(mktemp)"
  {
    echo 'set timeout '"$timeout"
    echo 'spawn ./docker.sh --no_pull'
    echo 'expect "User Name: "'
    echo 'send "'"$ADMIN_USER"'\r"'
    local c
    for c in "$@"; do
      echo 'expect "> "'
      echo 'send "'"$c"'\r"'
    done
    echo 'expect "> "'
    echo 'send "bye\r"'
    echo 'expect eof'
  } > "$exp"
  ( cd "$ADMIN_KIT" && timeout "$((timeout + 15))" expect -f "$exp" 2>&1 ) | strip_ansi || true
  rm -f "$exp"
  clean_admin_container
}

# --------------------------------------------------------------------------- #
#  Subcommands
# --------------------------------------------------------------------------- #
cmd_status() {
  run_admin "check_status server" "check_status client" \
    | grep -aiE "Engine status|Registered clients|^\| [A-Z]" || true
}

cmd_submit() {
  local job="${1:-}"; [ -n "$job" ] || die "usage: swarm_ops.sh submit <container-job-path>"
  run_admin "submit_job $job" | grep -aiE "Submitted job|error|does not exist" || true
}

cmd_abort() {
  local jid="${1:-}"; [ -n "$jid" ] || die "usage: swarm_ops.sh abort <job-id>"
  err "aborting $jid (a stalled job may not abort cleanly; if it hangs, use restart-server)"
  ADMIN_TIMEOUT="${ADMIN_TIMEOUT:-60}" run_admin "abort_job $jid" | grep -aiE "abort|Done|Job" || true
}

cmd_restart_server() {
  local name
  name="$(docker ps -a --format '{{.Names}}' | grep -iE 'flserver' | head -1)"
  [ -n "$name" ] && { err "removing old server $name"; docker rm -f "$name" >/dev/null 2>&1 || true; }
  err "starting server via docker.sh --start_server"
  ( cd "$ADMIN_KIT" && nohup ./docker.sh --no_pull --start_server >/tmp/swarm_ops_srv.log 2>&1 & )
  sleep 25
  name="$(docker ps --format '{{.Names}}' | grep -iE 'flserver' | head -1)"
  [ -n "$name" ] || die "server did not come up; see /tmp/swarm_ops_srv.log"
  docker update --restart unless-stopped "$name" >/dev/null 2>&1 || true
  docker ps --format '{{.Names}}\t{{.Status}}' | grep -iE 'flserver'
}

cmd_restart_client() {
  local site="${1:-}"; [ -n "$site" ] || die "usage: swarm_ops.sh restart-client <SITE> [--host H --user U]"
  local host="${OPT_HOST:-}" user="${OPT_USER:-}"
  # Fall back to a deploy conf for host/user if not passed explicitly.
  if [ -z "$host" ] || [ -z "$user" ]; then
    local conf; conf="$(find "$ADMIN_KIT/../../.." -maxdepth 3 -name 'deploy_sites_*.conf' 2>/dev/null | head -1 || true)"
    if [ -n "${conf:-}" ]; then
      host="${host:-$(grep -aoE "^${site%_*}_HOST=[^ ]+" "$conf" 2>/dev/null | cut -d= -f2 | head -1)}"
      user="${user:-$(grep -aoE "^${site%_*}_USER=[^ ]+" "$conf" 2>/dev/null | cut -d= -f2 | head -1)}"
    fi
  fi
  [ -n "$host" ] && [ -n "$user" ] || die "no host/user for $site (pass --host/--user or add to a deploy_sites_*.conf)"
  [ -f "$SECRETS" ] || die "no secrets file at $SECRETS"
  # shellcheck disable=SC1090
  source "$SECRETS"
  local var="${site%_*}_PASS" pass
  pass="${!var:-}"
  [ -n "$pass" ] || die "no \$$var in $SECRETS"
  command -v sshpass >/dev/null 2>&1 || die "sshpass not found"
  err "restarting client at $user@$host (rm -f + clear daemon_pid.fl + start)"
  SSHPASS="$pass" sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$user@$host" '
    set -e
    cname=$(docker ps -a --format "{{.Names}}" | grep -iE "swarm_client|flclient" | head -1)
    [ -n "$cname" ] && docker rm -f "$cname" || true
    # clear the stale lock that otherwise silently blocks the restart
    find "$HOME" /opt /srv /mnt -maxdepth 6 -name daemon_pid.fl -delete 2>/dev/null || true
    kit=$(find "$HOME" /opt /srv /mnt -maxdepth 6 -name docker.sh -path "*startup*" 2>/dev/null | head -1)
    [ -n "$kit" ] && ( cd "$(dirname "$kit")" && nohup ./docker.sh --no_pull --start_client >/tmp/swarm_client.log 2>&1 & )
    echo "client restart issued (kit: ${kit:-NOT FOUND})"
  ' 2>&1 | grep -avE "Permanently|Warning: "
}

cmd_watch() {
  local jid="${1:-}" interval="${2:-900}"
  local log; log="$(resolve_server_log)"
  [ -n "$log" ] && [ -f "$log" ] || die "server log not found; set SWARM_SERVER_LOG=<path to server startup/nohup.out>"
  err "watching $log every ${interval}s (Ctrl-C to stop). job filter: ${jid:-<all>}"
  while true; do
    local hr fin
    hr="$(grep -aE "${jid}" "$log" 2>/dev/null | grep -aoE 'on round [0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)"
    fin="$(grep -aE "${jid}" "$log" 2>/dev/null | grep -aoE 'finished_learn_task' | wc -l | tr -d ' ')"
    echo "[$(date '+%H:%M:%S')] highest round=${hr:-none} finished_learn=${fin} log_mtime=$(date -d @"$(stat -c %Y "$log")" '+%H:%M:%S')"
    grep -aE "${jid}" "$log" 2>/dev/null | strip_ansi \
      | grep -aiE 'prune|active clients|ABORTED|Server runner finished|FATAL' | tail -2 || true
    sleep "$interval"
  done
}

usage() {
  cat <<EOF
swarm_ops.sh — operator CLI for the ODELIA/MediSwarm swarm (run from the admin kit startup dir)

  status                         registered clients + engine state
  submit <container-job-path>    submit_job (e.g. /fl_admin/local/mediswarm_jobs/<job>)
  abort  <job-id>                abort_job (falls back to restart-server if it hangs)
  restart-server                 recreate the FL server with --restart unless-stopped
  restart-client <SITE>          remote: rm -f client, clear daemon_pid.fl, restart
                                 [--host H --user U]  (else read from deploy_sites_*.conf)
  watch [job-id] [interval-s]    poll the server log for round progress / prunes / aborts
  clean-admin                    remove a stuck odelia_swarm_admin_* container

Env: SWARM_ADMIN_USER (default: derived from kit path = "$ADMIN_USER")
     SWARM_SERVER_LOG (server startup/nohup.out, for watch)
     MEDISWARM_SECRETS (default: ~/.mediswarm_secrets)
EOF
}

# --------------------------------------------------------------------------- #
#  Arg parsing
# --------------------------------------------------------------------------- #
OPT_HOST=""; OPT_USER=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --admin-kit) ADMIN_KIT="$(cd "$2" && pwd -P)"; shift 2 ;;
    --user)      OPT_USER="$2"; shift 2 ;;
    --host)      OPT_HOST="$2"; shift 2 ;;
    --server-log) SERVER_LOG="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *)           POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]:-}"

case "${1:-}" in
  status)          cmd_status ;;
  submit)          cmd_submit "${2:-}" ;;
  abort)           cmd_abort "${2:-}" ;;
  restart-server)  cmd_restart_server ;;
  restart-client)  cmd_restart_client "${2:-}" ;;
  watch)           cmd_watch "${2:-}" "${3:-900}" ;;
  clean-admin)     clean_admin_container; echo "admin container cleaned" ;;
  ""|-h|--help)    usage ;;
  *)               err "unknown subcommand: $1"; usage; exit 2 ;;
esac
