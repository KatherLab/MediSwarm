#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_CONF="$SCRIPT_DIR/sync.conf"

# Revision of this daemon, reported in every heartbeat (sync_version and live_sync_version).
# Every kit before 1.9.0 reported "2", so the monitor could not tell an old kit's daemon from
# a new one (#614). 3 = the 1.9.0 kits: kit upload key (#615), models and per-case predictions
# stay at the site by default, exit when the client container is gone (#614), sync status
# cleared after a clean cycle.
LIVE_SYNC_VERSION="3"

_ask_continue_without_sync() {
    # In non-interactive environments (no TTY), default to continuing without sync
    if [ ! -t 0 ]; then
        echo "[INFO] Non-interactive mode — continuing without live sync."
        return 0
    fi
    read -r -p "Continue training without live sync? [y/N] " ans
    case "$ans" in
        [yY]*) return 0 ;;
        *) echo "Aborted."; return 1 ;;
    esac
}

if [ ! -f "$SYNC_CONF" ]; then
    echo ""
    echo "[WARN] Live-sync: sync.conf not found — training artifacts will not be synced."
    echo ""
    _ask_continue_without_sync && exit 0 || exit 1
fi

# One SSH connection per kit instead of one per upload (#602).
#
# Every rsync/ssh call in this daemon is a separate SSH login: three to five per
# 30-second cycle per kit, about two logins per second across the consortium at
# the upload host. Each login is a logind session there; on 16 Sep 2026 the host
# hit logind's session cap and refused every SSH login, the operators' included.
# With connection multiplexing the first call opens a master connection and the
# later ones ride on it; the master lingers ControlPersist seconds after the last
# use, so a kit logs in once per ten minutes at most. Appended here rather than in
# sync.conf because sites carry their sync.conf across kit upgrades.
#
# A daemon that authenticates with the kit's own upload key gets a control socket of its
# own: sharing one with another kit's daemon would send this kit's uploads through a
# connection that the other kit's key opened.
add_ssh_multiplexing() {
  case " ${SSH_OPTS:-} " in *" -o ControlMaster="*) return 0 ;; esac
  local mux_dir="${TMPDIR:-/tmp}/mediswarm_ssh_mux" sock="%C"
  { mkdir -p "$mux_dir" && chmod 700 "$mux_dir"; } 2>/dev/null || return 0
  if [ "$UPLOAD_AUTH" = "kit-key" ]; then
    sock="k$(printf '%s' "$UPLOAD_KEY_USED" | sha256sum | cut -c1-8)-%C"
  fi
  SSH_OPTS="${SSH_OPTS:-} -o ControlMaster=auto -o ControlPath=${mux_dir}/${sock} -o ControlPersist=600"
}

# A site's choice in the environment wins over sync.conf, which sites carry across kits.
_env_sync_models="${MEDISWARM_SYNC_MODELS-}"
_env_sync_predictions="${MEDISWARM_SYNC_PREDICTIONS-}"
_env_container_gone="${MEDISWARM_SYNC_CONTAINER_GONE_MINUTES-}"
_env_container_grace="${MEDISWARM_SYNC_CONTAINER_START_GRACE_MINUTES-}"

# shellcheck source=/dev/null
source "$SYNC_CONF"
SSH_OPTS="${SSH_OPTS:-}"
# The SSH settings of sync.conf: used as they are by a kit without its own upload key, and as
# the fallback when the upload host does not accept the kit's key.
SSH_OPTS_CONF="$SSH_OPTS"

# ── What leaves the site (1.9.0, review finding DE-4) ───────────────
# Logs, heartbeats, TensorBoard event files, data_layout.json and metric summaries are uploaded
# as before. Model files (*.ckpt, *.pt, *.pth) only with MEDISWARM_SYNC_MODELS=1: a checkpoint
# is the site's trained, un-noised local model (also in DP jobs), and the CAM checkpoints alone
# are 3.2 GB. Per-case prediction CSVs (one row per validation/test case) only when the site
# opted in to per-case return (--env ODELIA_RETURN_PER_CASE=1 for its client, see
# per_case_predictions.py) or sets MEDISWARM_SYNC_PREDICTIONS=1; MEDISWARM_SYNC_PREDICTIONS=0
# never uploads them, and neither does a job that forces ODELIA_RETURN_PER_CASE=0 (DP jobs).
MEDISWARM_SYNC_MODELS="${_env_sync_models:-${MEDISWARM_SYNC_MODELS:-0}}"
MEDISWARM_SYNC_PREDICTIONS="${_env_sync_predictions:-${MEDISWARM_SYNC_PREDICTIONS:-}}"
MODEL_PATTERNS=('*.ckpt' '*.pt' '*.pth' '*.safetensors')
PER_CASE_PATTERNS=('*_gt_and_classprob_*.csv' '*prediction*.csv' '*per_case*.csv')

MODE=""
SITE_NAME=""
KIT_ROOT=""
STARTUP_DIR=""
SCRATCHDIR=""
CONTAINER_NAME_ARG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --site-name) SITE_NAME="${2:-}"; shift 2 ;;
    --kit-root) KIT_ROOT="${2:-}"; shift 2 ;;
    --startup-dir) STARTUP_DIR="${2:-}"; shift 2 ;;
    --scratch-dir) SCRATCHDIR="${2:-}"; shift 2 ;;
    --container-name) CONTAINER_NAME_ARG="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -n "$MODE" ] || { echo "MODE missing" >&2; exit 1; }
[ -n "$SITE_NAME" ] || { echo "SITE_NAME missing" >&2; exit 1; }
[ -n "$KIT_ROOT" ] || { echo "KIT_ROOT missing" >&2; exit 1; }
[ -n "$STARTUP_DIR" ] || { echo "STARTUP_DIR missing" >&2; exit 1; }

STATE_DIR="$STARTUP_DIR/.mediswarm_sync"

# Ensure a writable state dir. A client previously started via `sudo` can leave
# .mediswarm_sync (and its files) root-owned; under `set -euo pipefail` the
# mkdir/touch below would then EACCES and silently kill the whole daemon so the
# site stops reporting (issue #398). Reclaim ownership if we can (passwordless
# sudo), otherwise fall back to a user-writable dir so live-sync keeps running.
_state_dir_writable() { [ -d "$STATE_DIR" ] && ( : > "$STATE_DIR/.wtest" ) 2>/dev/null && rm -f "$STATE_DIR/.wtest" 2>/dev/null; }
_ensure_writable_state_dir() {
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  _state_dir_writable && return 0
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo -n chown -R "$(id -u):$(id -g)" "$STATE_DIR" 2>/dev/null || true
  fi
  _state_dir_writable && return 0
  local fallback="${TMPDIR:-/tmp}/mediswarm_sync_$(id -u)_$(printf '%s' "$STARTUP_DIR" | sha256sum | cut -c1-8)"
  mkdir -p "$fallback" 2>/dev/null || true
  echo "[WARN] $STATE_DIR not writable (likely root-owned from a sudo-launched client); using $fallback. Fix: sudo chown -R \$(id -un): '$STATE_DIR'" >&2
  STATE_DIR="$fallback"
}
_ensure_writable_state_dir

LIVE_SYNC_LOG="$STATE_DIR/live_sync_daemon.log"
touch "$LIVE_SYNC_LOG" 2>/dev/null || true
if [ -w "$LIVE_SYNC_LOG" ]; then
  exec > >(tee -a "$LIVE_SYNC_LOG") 2>&1
else
  echo "[WARN] cannot write $LIVE_SYNC_LOG; logging to stdout only" >&2
fi

SYNC_STATUS="ok"
SYNC_ERROR=""
CYCLE_FAILED=0

record_sync_error() {
  local msg="$1"
  SYNC_STATUS="error"
  SYNC_ERROR="$msg"
  CYCLE_FAILED=1
  echo "[WARN] $msg"
}

# A heartbeat used to report sync_status=error forever after a single failed upload (all six
# reporting sites showed "error" on 14 Oct 2026). The status now describes the last complete
# cycle: a cycle without a failed upload clears it, so the next heartbeat says ok again.
begin_cycle() { CYCLE_FAILED=0; }
end_cycle() {
  if [ "$CYCLE_FAILED" = "0" ]; then
    SYNC_STATUS="ok"
    SYNC_ERROR=""
  fi
}

# ── Upload authentication (#615) ────────────────────────────────────
# A 1.9.0 kit carries its own upload key (startup/upload_key, registered on the upload host for
# this site when the kit is built) and the upload host's SSH host key (startup/upload_known_hosts).
# With them the daemon authenticates with that key only and refuses an unknown host key, so
# uploads no longer depend on whichever ~/.ssh key or agent the person who started the client
# happened to have. A kit without upload_key keeps the settings of sync.conf, as before.
UPLOAD_KEY="$STARTUP_DIR/upload_key"
UPLOAD_KNOWN_HOSTS="$STARTUP_DIR/upload_known_hosts"
UPLOAD_AUTH="legacy"
UPLOAD_KEY_USED=""
SSH_OPTS_KIT=""

_ssh_opts_without() {
  # $1 = an SSH option string; prints it without -i and without the -o options named in $2
  # (lower case, space separated; ssh option names are case-insensitive).
  local -a words out=()
  local i=0 n w v key
  read -r -a words <<< "$1"
  n=${#words[@]}
  while [ "$i" -lt "$n" ]; do
    w="${words[$i]}"
    case "$w" in
      -o) v="${words[$((i + 1))]:-}"; i=$((i + 2)) ;;
      -o*) v="${w#-o}"; i=$((i + 1)) ;;
      -i) i=$((i + 2)); continue ;;
      -i*) i=$((i + 1)); continue ;;
      *) out+=("$w"); i=$((i + 1)); continue ;;
    esac
    key="${v%%=*}"
    case " $2 " in
      *" ${key,,} "*) ;;
      *) out+=("-o" "$v") ;;
    esac
  done
  printf '%s' "${out[*]:-}"
}

_private_copy() {
  # $1 = a file ssh must read -> path of a copy that ssh accepts: mode 600, in a directory that
  # only this user owns, on a path without whitespace (SSH_OPTS is split on spaces).
  local dir
  dir="${TMPDIR:-/tmp}/mediswarm_upload_$(id -u)_$(printf '%s' "$1" | sha256sum | cut -c1-8)"
  ( umask 077 && mkdir -p "$dir" ) 2>/dev/null || return 1
  [ -O "$dir" ] && [ "$(stat -c %a "$dir" 2>/dev/null)" = "700" ] || return 1
  [[ "$dir" != *[[:space:]]* ]] || return 1
  ( umask 077 && cp -f "$1" "$dir/$(basename "$1")" ) 2>/dev/null || return 1
  chmod 600 "$dir/$(basename "$1")" 2>/dev/null || return 1
  printf '%s' "$dir/$(basename "$1")"
}

_usable_key() {
  # ssh ignores a private key that group or others can read ("UNPROTECTED PRIVATE KEY FILE"),
  # and a kit unzipped by some tools loses its modes (UMCU_1, 25 Sep 2026).
  local key="$1" mode
  [ -r "$key" ] || return 1
  mode="$(stat -c %a "$key" 2>/dev/null || echo 777)"
  if [ $((8#$mode & 8#077)) -ne 0 ]; then
    chmod 600 "$key" 2>/dev/null || true
    mode="$(stat -c %a "$key" 2>/dev/null || echo 777)"
  fi
  if [ $((8#$mode & 8#077)) -eq 0 ] && [ -O "$key" ] && [[ "$key" != *[[:space:]]* ]]; then
    printf '%s' "$key"
    return 0
  fi
  _private_copy "$key"
}

_known_host_listed() {
  # Does known_hosts file $2 hold a key for host $1? ssh-keygen also reads hashed entries; the
  # awk fallback (hosts without ssh-keygen) reads plain ones, as the kit build writes them.
  if command -v ssh-keygen >/dev/null 2>&1; then
    ssh-keygen -F "$1" -f "$2" >/dev/null 2>&1
    return
  fi
  awk -v h="$1" '!/^[[:space:]]*#/ && NF >= 3 {
      n = split($1, names, ",")
      for (i = 1; i <= n; i++) if (names[i] == h || names[i] == "[" h "]:22") found = 1
    } END { exit !found }' "$2" 2>/dev/null
}

configure_upload_ssh() {
  if [ ! -f "$UPLOAD_KEY" ]; then
    echo "[INFO] Upload authentication: the SSH settings of sync.conf (this kit has no startup/upload_key)."
    return 0
  fi
  local key known=""
  if ! key="$(_usable_key "$UPLOAD_KEY")" || [ -z "$key" ]; then
    echo "[WARN] Cannot use $UPLOAD_KEY (unreadable, or its mode cannot be made 600): using the SSH settings of sync.conf."
    return 0
  fi
  if [ -f "$UPLOAD_KNOWN_HOSTS" ] && _known_host_listed "$REMOTE_HOST" "$UPLOAD_KNOWN_HOSTS"; then
    known="$UPLOAD_KNOWN_HOSTS"
    [[ "$known" != *[[:space:]]* ]] || known="$(_private_copy "$UPLOAD_KNOWN_HOSTS")" || known=""
  fi
  UPLOAD_AUTH="kit-key"
  UPLOAD_KEY_USED="$key"
  if [ -n "$known" ]; then
    SSH_OPTS_KIT="-i $key -o IdentitiesOnly=yes -o UserKnownHostsFile=$known -o StrictHostKeyChecking=yes -o CheckHostIP=no $(_ssh_opts_without "$SSH_OPTS_CONF" "stricthostkeychecking userknownhostsfile globalknownhostsfile identitiesonly identityfile checkhostip")"
    echo "[INFO] Upload authentication: the kit's upload key; host key pinned by startup/upload_known_hosts."
  else
    # The kit only knows the host key of the names it was built for. Another REMOTE_HOST in
    # sync.conf (sites carry it across kits) keeps that file's host key settings.
    SSH_OPTS_KIT="-i $key -o IdentitiesOnly=yes $(_ssh_opts_without "$SSH_OPTS_CONF" "identitiesonly identityfile")"
    echo "[WARN] Upload authentication: the kit's upload key, but $UPLOAD_KNOWN_HOSTS does not list REMOTE_HOST=${REMOTE_HOST}: the host key settings of sync.conf apply."
  fi
  SSH_OPTS="$SSH_OPTS_KIT"
}

use_ssh_opts() {
  # $1 = kit-key | legacy | legacy-fallback: SSH_OPTS for that authentication, multiplexed.
  UPLOAD_AUTH="$1"
  if [ "$1" = "kit-key" ]; then SSH_OPTS="$SSH_OPTS_KIT"; else SSH_OPTS="$SSH_OPTS_CONF"; fi
  add_ssh_multiplexing
}

configure_upload_ssh
use_ssh_opts "$UPLOAD_AUTH"

# ── Client container watchdog (#614) ────────────────────────────────
# The swarm daemon is a host process started by `docker.sh --start_client`, and nothing stopped
# it when its client went away: the 1.6.0 daemon at CAM kept uploading for weeks after the site
# had moved to a newer kit, and clobbered that kit's uploads. The daemon now exits once the kit's
# client container has not been running for MEDISWARM_SYNC_CONTAINER_GONE_MINUTES (default 10;
# 0 = never). Until it has seen the container running once it waits
# MEDISWARM_SYNC_CONTAINER_START_GRACE_MINUTES (default 60) instead, for a `docker run` that
# first has to pull the image. When Docker cannot be asked, nothing is decided.
_minutes_to_seconds() {
  # $1 = minutes (may be fractional), rounded up to whole seconds; $2 = default for a bad value
  awk -v m="$1" -v d="$2" 'BEGIN { if (m !~ /^[0-9]+(\.[0-9]+)?$/) m = d; s = m * 60
                                   printf "%d", (s == int(s)) ? s : int(s) + 1 }'
}
CONTAINER_GONE_SECONDS="$(_minutes_to_seconds "${_env_container_gone:-${MEDISWARM_SYNC_CONTAINER_GONE_MINUTES:-10}}" 10)"
CONTAINER_START_GRACE_SECONDS="$(_minutes_to_seconds "${_env_container_grace:-${MEDISWARM_SYNC_CONTAINER_START_GRACE_MINUTES:-60}}" 60)"
[ "$CONTAINER_START_GRACE_SECONDS" -ge "$CONTAINER_GONE_SECONDS" ] || CONTAINER_START_GRACE_SECONDS="$CONTAINER_GONE_SECONDS"
DAEMON_START_TS="$(date +%s)"
CLIENT_LAST_SEEN=""
CLIENT_CONTAINER_SEEN=""
CLIENT_STATE=""
FINAL_STATUS="finished"
KIT_ROOT_PHYSICAL="$(cd "$KIT_ROOT" 2>/dev/null && pwd -P || true)"

_mounts_this_kit() {
  # Does container $1 mount this kit at /startupkit (as every ODELIA and DECADE client does)?
  local dst src
  while IFS='|' read -r dst src; do
    case "$dst" in /startupkit|/startupkit/) ;; *) continue ;; esac
    [ "$src" = "$KIT_ROOT" ] && return 0
    if [ -n "$KIT_ROOT_PHYSICAL" ] && [ "$(cd "$src" 2>/dev/null && pwd -P || true)" = "$KIT_ROOT_PHYSICAL" ]; then
      return 0
    fi
  done <<< "$(docker inspect --format '{{range .Mounts}}{{.Destination}}|{{.Source}}{{println}}{{end}}' "$1" 2>/dev/null || true)"
  return 1
}

refresh_client_state() {
  # CLIENT_STATE = running | stopped | unknown (Docker cannot be asked). The kit's client is the
  # container docker.sh named (--container-name), else any running container that mounts this
  # kit: a DECADE docker.sh does not pass the name, and a client re-created under another
  # name still counts. Called in the main shell, so CLIENT_CONTAINER_SEEN survives.
  CLIENT_STATE=unknown
  command -v docker >/dev/null 2>&1 || return 0
  local names n
  names="$(docker ps --format '{{.Names}}' 2>/dev/null)" || return 0
  CLIENT_STATE=stopped
  if [ -n "$CONTAINER_NAME_ARG" ] && grep -qxF -- "$CONTAINER_NAME_ARG" <<< "$names"; then
    CLIENT_STATE=running
    CLIENT_CONTAINER_SEEN="$CONTAINER_NAME_ARG"
    return 0
  fi
  while read -r n; do
    [ -n "$n" ] || continue
    if _mounts_this_kit "$n"; then
      CLIENT_STATE=running
      CLIENT_CONTAINER_SEEN="$n"
      return 0
    fi
  done <<< "$names"
  return 0
}

client_watchdog() {
  [ "$MODE" = "swarm" ] || return 0
  refresh_client_state
  [ "$CONTAINER_GONE_SECONDS" -gt 0 ] || return 0
  local now since limit
  now="$(date +%s)"
  case "$CLIENT_STATE" in
    running) CLIENT_LAST_SEEN="$now"; return 0 ;;
    stopped) ;;
    *) return 0 ;;
  esac
  if [ -n "$CLIENT_LAST_SEEN" ]; then
    since="$CLIENT_LAST_SEEN"; limit="$CONTAINER_GONE_SECONDS"
  else
    since="$DAEMON_START_TS"; limit="$CONTAINER_START_GRACE_SECONDS"
  fi
  if [ $((now - since)) -ge "$limit" ]; then
    echo "[INFO] The client container of $KIT_ROOT (${CONTAINER_NAME_ARG:-the one mounting this kit}) has not been running for $(( (now - since) / 60 )) min: live sync stops (#614)."
    FINAL_STATUS="stopped"
    exit 0
  fi
  return 0
}

export_sync_env() {
  export MEDISWARM_SYNC_STATUS="$SYNC_STATUS"
  export MEDISWARM_SYNC_ERROR="$SYNC_ERROR"
  export MEDISWARM_LIVE_SYNC_VERSION="$LIVE_SYNC_VERSION"
  export MEDISWARM_UPLOAD_AUTH="$UPLOAD_AUTH"
  export MEDISWARM_SYNC_CLIENT_CONTAINER="${CLIENT_CONTAINER_SEEN:-$CONTAINER_NAME_ARG}"
  export MEDISWARM_SYNC_CLIENT_STATE="$CLIENT_STATE"
}

# ── Upload policy for models and per-case predictions ───────────────
SYNC_MODELS_NOW=0
SYNC_PREDICTIONS_NOW=0
LAST_POLICY_NOTE=""

_options_opt_in() {
  # Do docker options $1 (or an image.conf line that sets them) pass ODELIA_RETURN_PER_CASE=1?
  local re="(^|[[:space:]\"'=])(--env[=[:space:]]+|-e[[:space:]]+)ODELIA_RETURN_PER_CASE=1([^0-9]|\$)"
  [[ "$1" =~ $re ]]
}

per_case_opted_in() {
  # The opt-in reaches the client only through docker options, so the client container's own
  # environment decides. Without a container to ask: the options docker.sh passes (exported to
  # this daemon as MEDISWARM_SYNC_DOCKER_OPTIONS by a 1.9.0 docker.sh), else image.conf.
  local c env_list line
  if command -v docker >/dev/null 2>&1; then
    for c in "$CLIENT_CONTAINER_SEEN" "$CONTAINER_NAME_ARG"; do
      [ -n "$c" ] || continue
      if env_list="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$c" 2>/dev/null)"; then
        grep -qxF 'ODELIA_RETURN_PER_CASE=1' <<< "$env_list" && return 0 || return 1
      fi
    done
  fi
  if [ -n "${MEDISWARM_SYNC_DOCKER_OPTIONS+x}" ]; then
    _options_opt_in "$MEDISWARM_SYNC_DOCKER_OPTIONS" && return 0 || return 1
  fi
  [ -f "$STARTUP_DIR/image.conf" ] || return 1
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"   # leading blanks
    line="${line%%[[:space:]]#*}"              # a trailing comment
    [ "${line:0:1}" != "#" ] || continue
    _options_opt_in "$line" && return 0
  done < "$STARTUP_DIR/image.conf"
  return 1
}

job_forces_per_case_off() {
  # The job's launcher line forces ODELIA_RETURN_PER_CASE=0 (the job builder does for DP jobs):
  # its per-case rows must not leave the site by this route either.
  local job_id="${1:-}" f
  [ -n "$job_id" ] || return 1
  for f in "$KIT_ROOT/$job_id"/app_*/config/config_fed_client.conf "$KIT_ROOT/$job_id"/app_*/config/config_fed_client.json; do
    [ -f "$f" ] || continue
    grep -qE '(^|[^A-Za-z0-9_])ODELIA_RETURN_PER_CASE=0([^0-9]|$)' "$f" && return 0
  done
  return 1
}

refresh_upload_policy() {
  # $1 = swarm job id ("" in local mode and while waiting for a job). Decides once per upload
  # what may leave the site, for the uploads and for the heartbeat (sync_models/sync_predictions).
  local job_id="${1:-}" why note
  SYNC_MODELS_NOW=0
  [ "$MEDISWARM_SYNC_MODELS" != "1" ] || SYNC_MODELS_NOW=1
  if [ "$MEDISWARM_SYNC_PREDICTIONS" = "0" ]; then
    SYNC_PREDICTIONS_NOW=0; why="MEDISWARM_SYNC_PREDICTIONS=0"
  elif job_forces_per_case_off "$job_id"; then
    SYNC_PREDICTIONS_NOW=0; why="job $job_id sets ODELIA_RETURN_PER_CASE=0"
  elif [ "$MEDISWARM_SYNC_PREDICTIONS" = "1" ]; then
    SYNC_PREDICTIONS_NOW=1; why="MEDISWARM_SYNC_PREDICTIONS=1"
  elif per_case_opted_in; then
    SYNC_PREDICTIONS_NOW=1; why="the client runs with ODELIA_RETURN_PER_CASE=1"
  else
    SYNC_PREDICTIONS_NOW=0; why="no per-case opt-in"
  fi
  export MEDISWARM_SYNC_MODELS_EFFECTIVE="$SYNC_MODELS_NOW"
  export MEDISWARM_SYNC_PREDICTIONS_EFFECTIVE="$SYNC_PREDICTIONS_NOW"
  note="models: $([ "$SYNC_MODELS_NOW" = 1 ] && echo "uploaded (MEDISWARM_SYNC_MODELS=1)" || echo "stay at the site")"
  note="$note; per-case predictions: $([ "$SYNC_PREDICTIONS_NOW" = 1 ] && echo uploaded || echo "stay at the site") ($why)"
  if [ "$note" != "$LAST_POLICY_NOTE" ]; then
    echo "[INFO] Upload policy: $note"
    LAST_POLICY_NOTE="$note"
  fi
  return 0
}

RUN_DIR_EXCLUDES=()
set_run_dir_excludes() {
  # rsync filters for a whole-run-dir upload: leave out what the policy keeps at the site.
  local p
  RUN_DIR_EXCLUDES=()
  if [ "$SYNC_MODELS_NOW" != "1" ]; then
    for p in "${MODEL_PATTERNS[@]}"; do RUN_DIR_EXCLUDES+=("--exclude=$p"); done
  fi
  if [ "$SYNC_PREDICTIONS_NOW" != "1" ]; then
    for p in "${PER_CASE_PATTERNS[@]}"; do RUN_DIR_EXCLUDES+=("--exclude=$p"); done
  fi
  return 0
}

RUN_DIR_PERIODIC_FILTERS=()
set_periodic_filters() {
  # The every-CKPT_SYNC_INTERVAL upload of a running run dir: checkpoints and per-case CSVs where
  # the policy allows them, and DECADE's STAMP metrics summary and predictions (#492; that
  # consortium has no per-case opt-in and evaluates centrally from them) as before.
  RUN_DIR_PERIODIC_FILTERS=(--include='*/')
  if [ "$SYNC_MODELS_NOW" = "1" ]; then
    RUN_DIR_PERIODIC_FILTERS+=(--include='last.ckpt' --include='epoch=*.ckpt')
  fi
  if [ "$SYNC_PREDICTIONS_NOW" = "1" ]; then
    RUN_DIR_PERIODIC_FILTERS+=(--include='*_model_gt_and_classprob_*.csv')
  fi
  RUN_DIR_PERIODIC_FILTERS+=(--include='stamp_metrics_summary.csv' --include='stamp_gt_predprob_*.csv' --exclude='*')
  return 0
}

# ── SSH connectivity check ──────────────────────────────────────────
# Verify we can reach the monitoring server before entering the sync
# loop.  With BatchMode=yes the connection will fail immediately if
# key-based auth is not set up instead of prompting for a password.
upload_host_reachable() {
  ssh -n ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" 'echo ok' >/dev/null 2>&1
}

connect_upload_host() {
  # The kit's key first. When the upload host does not accept it (not registered there yet) but
  # does accept this host's own settings (the key a site registered by email before 1.9.0),
  # uploads go on with those instead of stopping.
  if [ "$UPLOAD_AUTH" = "legacy-fallback" ]; then use_ssh_opts kit-key; fi
  upload_host_reachable && return 0
  [ "$UPLOAD_AUTH" = "kit-key" ] || return 1
  use_ssh_opts legacy-fallback
  if upload_host_reachable; then
    echo "[WARN] The upload host does not accept this kit's upload key (startup/upload_key) but accepts the SSH settings of sync.conf: uploading with those. Ask the swarm operator to register the kit key for ${SITE_NAME}."
    return 0
  fi
  use_ssh_opts kit-key
  return 1
}

explain_no_connection() {
  echo ""
  echo "============================================================"
  echo "[WARN] Live-sync: cannot connect to ${REMOTE_USER}@${REMOTE_HOST}"
  echo ""
  if [ "$UPLOAD_AUTH" = "kit-key" ]; then
    echo "  This kit authenticates with its own upload key (startup/upload_key),"
    echo "  which the swarm operator registers on the upload host for ${SITE_NAME}."
    echo "  Check the VPN to the upload host; if this persists, tell the operator."
  else
    echo "  To enable live sync, share your SSH public key with your"
    echo "  swarm operator.  Generate one (if you haven't already) with:"
    echo ""
    echo "    ssh-keygen -t ed25519 -C \"\$(hostname)@mediswarm\""
    echo "    cat ~/.ssh/id_ed25519.pub  # send this to the swarm operator"
  fi
  echo ""
  echo "  Test connectivity exactly as this daemon does with:"
  echo "    ssh ${SSH_OPTS} ${REMOTE_USER}@${REMOTE_HOST} 'echo ok'"
  echo "============================================================"
  echo ""
}

echo "Testing SSH connectivity to ${REMOTE_USER}@${REMOTE_HOST}..."
if connect_upload_host; then
    echo "SSH OK — live sync enabled."
else
    record_sync_error "Live-sync cannot connect to ${REMOTE_USER}@${REMOTE_HOST}"
    explain_no_connection
    if [ "$MODE" = "swarm" ] && [ ! -t 0 ]; then
        # The swarm daemon keeps trying instead of giving up for the whole run: the VPN may come
        # up, or the operator register the kit's key, after the client was started. It still
        # ends with its client container. The tries are minutes apart, so that failed logins do
        # not get the site banned by the upload host's SSH protection.
        retry_seconds="${MEDISWARM_SYNC_CONNECT_RETRY_SECONDS:-300}"
        until connect_upload_host; do
            client_watchdog
            echo "[WARN] still cannot connect to ${REMOTE_USER}@${REMOTE_HOST}; next try in ${retry_seconds}s"
            sleep "$retry_seconds"
        done
        echo "SSH OK — live sync enabled."
    else
        _ask_continue_without_sync && exit 0 || exit 1
    fi
fi

LAST_CKPT_SYNC_FILE="$STATE_DIR/${MODE}_last_ckpt_sync_ts"
touch "$LAST_CKPT_SYNC_FILE" 2>/dev/null || true

# The remote swarm/_active folder this daemon created, if any (see sync_swarm). It is shared by
# every kit of the site, so a daemon removes it only when it wrote it itself.
ACTIVE_OWNER_FILE="$STATE_DIR/swarm_active_owner"

# Track the run we are currently syncing so we can finalize it if a new run appears
CURRENT_LOCAL_RUN=""

ssh_cmd() {
  # -n: never read the daemon's stdin. With multiplexing the first call forks a
  # background master that would otherwise inherit and hold that descriptor.
  ssh -n ${SSH_OPTS} "$@" || {
    record_sync_error "ssh failed: $*"
    return 1
  }
}

rsync_cmd() {
  local rc=0
  rsync -az --partial --mkpath -e "ssh ${SSH_OPTS}" "$@" || rc=$?
  # 24: source files vanished during the transfer (a checkpoint rotated, a CSV rewritten),
  # normal for a run that is still writing; not an upload failure.
  if [ "$rc" = "0" ] || [ "$rc" = "24" ]; then
    return 0
  fi
  record_sync_error "rsync failed (exit $rc): $*"
  return 1
}

ensure_remote_dir() {
  local remote_dir="$1"
  ssh_cmd "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${remote_dir}'"
}

sync_daemon_log() {
  local remote_dir="$1"
  [ -f "$LIVE_SYNC_LOG" ] || return 0
  rsync_cmd "$LIVE_SYNC_LOG" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/live_sync_daemon.log" || true
}

build_heartbeat() {
  export_sync_env
  # Through bash, not exec: a kit unzipped without the exec bit (UMCU_1, 25 Sep 2026)
  # otherwise fails with "Permission denied" every cycle and never sends a heartbeat.
  bash "$SCRIPT_DIR/build_heartbeat.sh" "$@"
}

find_latest_job_id() {
  # Newest job directory by modification time. Job ids are random UUIDs, so sorting
  # them by name picks an arbitrary job: on 30 Sep 2026 every site kept uploading the
  # new job's logs under the previous job's id (79dfd8ce sorts after 644a5ee5).
  find "$KIT_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name startup ! -name local ! -name transfer \
      -printf '%T@ %f\n' 2>/dev/null | while read -r ts b; do
    case "$b" in
      *-*-*-*-*) printf '%s %s\n' "$ts" "$b" ;;
    esac
  done | sort -n | tail -n 1 | cut -d' ' -f2 || true
}

extract_run_name_from_nohup() {
  local nohup_file="$STARTUP_DIR/nohup.out"
  [ -f "$nohup_file" ] || return 0

  grep -E 'Run name: |Run directory:' "$nohup_file" 2>/dev/null | tail -n 1 | \
    sed -E 's#.*Run name: ##; s#.*Run directory: .*/##' || true
}

find_latest_local_run_name() {
  local base result=""
  # Check scratch dir first (primary location since all jobs now use SCRATCH_DIR)
  if [ -n "$SCRATCHDIR" ]; then
    base="$SCRATCHDIR/runs/$SITE_NAME"
    if [ -d "$base" ]; then
      result="$(find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null | \
        awk -v min_ts="${MEDISWARM_LOCAL_SYNC_START_TS:-0}" '$1 >= min_ts {print $2}' | \
        sort | tail -n 1 || true)"
    fi
  fi
  # Fall back to startup dir (backward compatibility with older builds)
  if [ -z "$result" ]; then
    base="$STARTUP_DIR/runs/$SITE_NAME"
    if [ -d "$base" ]; then
      result="$(find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null | \
        awk -v min_ts="${MEDISWARM_LOCAL_SYNC_START_TS:-0}" '$1 >= min_ts {print $2}' | \
        sort | tail -n 1 || true)"
    fi
  fi
  printf '%s' "$result"
}

find_latest_swarm_run_name() {
  # Swarm mode writes its STAMP run under the same runs/$SITE_NAME/ tree as local
  # mode (stamp_training.load_stamp_environment). Locate it by directory listing —
  # the robust approach local mode already uses — instead of relying only on
  # nohup parsing, which failed because training logged "Output directory:" rather
  # than the phrase extract_run_name_from_nohup matched, so the run dir (and its
  # metrics/predprob CSVs) never synced (#492). Names are timestamped, so a
  # lexicographic sort is chronological.
  local base result=""
  for base in "${SCRATCHDIR:+$SCRATCHDIR/runs/$SITE_NAME}" "$STARTUP_DIR/runs/$SITE_NAME"; do
    [ -n "$base" ] && [ -d "$base" ] || continue
    result="$(find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -n 1 || true)"
    [ -n "$result" ] && break
  done
  printf '%s' "$result"
}

build_remote_dir() {
  local run_id="$1"
  printf '%s/%s/%s/%s' "$REMOTE_BASE" "$SITE_NAME" "$MODE" "$run_id"
}

upload_global_models() {
  # FL_global_model.pt / best_FL_global_model.pt of a swarm job: model files, so only with
  # MEDISWARM_SYNC_MODELS=1.
  local job_id="$1" remote_dir="$2" f
  [ "$SYNC_MODELS_NOW" = "1" ] || return 0
  for f in FL_global_model.pt best_FL_global_model.pt; do
    [ -f "$KIT_ROOT/$job_id/app_${SITE_NAME}/$f" ] || continue
    rsync_cmd "$KIT_ROOT/$job_id/app_${SITE_NAME}/$f" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/$f" || true
  done
}

release_active_dir() {
  # Remove the remote swarm/_active folder once this kit has a job, but only the path this
  # daemon wrote (#614: the 1.6.0 daemon at CAM, whose kit had an old job, deleted the waiting
  # 1.8.1 client's _active folder every 35 s).
  local own
  [ -f "$ACTIVE_OWNER_FILE" ] || return 0
  own="$(head -n 1 "$ACTIVE_OWNER_FILE" 2>/dev/null || true)"
  case "$own" in
    */"$SITE_NAME"/swarm/_active) ;;
    *) rm -f "$ACTIVE_OWNER_FILE"; return 0 ;;
  esac
  if [[ "$own" != *"'"* ]] && ssh_cmd "${REMOTE_USER}@${REMOTE_HOST}" "rm -rf '$own'"; then
    rm -f "$ACTIVE_OWNER_FILE"
  fi
  return 0
}

_finalize_local_run() {
  # Write a heartbeat_final.json for a given run_name and mark it "finished"
  local old_run="$1"
  [ -n "$old_run" ] || return 0

  local old_remote_dir old_hb_final old_run_dir
  old_remote_dir="$(build_remote_dir "$old_run")"
  old_hb_final="$STATE_DIR/local_heartbeat_final_${old_run}.json"
  export SCRATCHDIR
  refresh_upload_policy ""
  build_heartbeat "$SITE_NAME" "local" "$KIT_ROOT" "" "$old_run" "finished" "$old_hb_final" >/dev/null
  rsync_cmd "$old_hb_final" "${REMOTE_USER}@${REMOTE_HOST}:${old_remote_dir}/heartbeat_final.json" || true
  sync_daemon_log "$old_remote_dir"
  # Final sync of the old run directory
  old_run_dir=""
  if [ -n "$SCRATCHDIR" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$old_run" ]; then
    old_run_dir="$SCRATCHDIR/runs/$SITE_NAME/$old_run"
  elif [ -d "$STARTUP_DIR/runs/$SITE_NAME/$old_run" ]; then
    old_run_dir="$STARTUP_DIR/runs/$SITE_NAME/$old_run"
  fi
  if [ -n "$old_run_dir" ]; then
    set_run_dir_excludes
    rsync_cmd ${RUN_DIR_EXCLUDES[@]+"${RUN_DIR_EXCLUDES[@]}"} \
      "$old_run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${old_remote_dir}/run_dir/" || true
  fi
}

sync_local() {
  local run_name run_dir remote_dir hb_file now last

  run_name="$(find_latest_local_run_name || true)"
  [ -n "$run_name" ] || return 0

  # If the run changed (new training started), finalize the old one
  if [ -n "$CURRENT_LOCAL_RUN" ] && [ "$CURRENT_LOCAL_RUN" != "$run_name" ]; then
    _finalize_local_run "$CURRENT_LOCAL_RUN" || true
  fi
  CURRENT_LOCAL_RUN="$run_name"

  # Determine run_dir: check scratch dir first, fall back to startup dir
  run_dir=""
  if [ -n "$SCRATCHDIR" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$run_name" ]; then
    run_dir="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
  elif [ -d "$STARTUP_DIR/runs/$SITE_NAME/$run_name" ]; then
    run_dir="$STARTUP_DIR/runs/$SITE_NAME/$run_name"
  fi
  [ -n "$run_dir" ] || return 0

  remote_dir="$(build_remote_dir "$run_name")"
  ensure_remote_dir "$remote_dir"

  hb_file="$STATE_DIR/local_heartbeat.json"
  export SCRATCHDIR
  refresh_upload_policy ""
  build_heartbeat "$SITE_NAME" "local" "$KIT_ROOT" "" "$run_name" "running" "$hb_file" >/dev/null

  rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json" || true
  sync_daemon_log "$remote_dir"

  [ -f "$STARTUP_DIR/local_training_console_output.txt" ] && \
    rsync_cmd "$STARTUP_DIR/local_training_console_output.txt" \
      "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/local_training_console_output.txt" || true

  now="$(date +%s)"
  last="$(cat "$LAST_CKPT_SYNC_FILE" 2>/dev/null || echo 0)"

  rsync_cmd \
    --include='*/' \
    --include='events.out.tfevents*' \
    --exclude='*' \
    "$run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true

  if [ $((now - ${last:-0})) -ge "$CKPT_SYNC_INTERVAL" ]; then
    set_periodic_filters
    rsync_cmd "${RUN_DIR_PERIODIC_FILTERS[@]}" \
      "$run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
    echo "$now" > "$LAST_CKPT_SYNC_FILE" 2>/dev/null || true
  fi
}

sync_swarm() {
  local job_id run_name remote_dir hb_file now last

  job_id="$(find_latest_job_id || true)"
  if [ -z "$job_id" ]; then
    remote_dir="$(build_remote_dir "_active")"
    ensure_remote_dir "$remote_dir" || return 0
    printf '%s\n' "$remote_dir" > "$ACTIVE_OWNER_FILE" 2>/dev/null || true
    hb_file="$STATE_DIR/swarm_active_heartbeat.json"
    export SCRATCHDIR
    refresh_upload_policy ""
    build_heartbeat "$SITE_NAME" "swarm" "$KIT_ROOT" "" "" "waiting" "$hb_file" >/dev/null
    rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json" || true
    [ -f "$STARTUP_DIR/nohup.out" ] && \
      rsync_cmd "$STARTUP_DIR/nohup.out" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/nohup.out" || true
    [ -f "$KIT_ROOT/log.txt" ] && \
      rsync_cmd "$KIT_ROOT/log.txt" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/log.txt" || true
    sync_daemon_log "$remote_dir"
    return 0
  fi

  release_active_dir

  run_name="$(extract_run_name_from_nohup || true)"
  [ -n "$run_name" ] || run_name="$(find_latest_swarm_run_name || true)"
  remote_dir="$(build_remote_dir "$job_id")"
  ensure_remote_dir "$remote_dir"

  export SCRATCHDIR
  refresh_upload_policy "$job_id"
  hb_file="$STATE_DIR/swarm_heartbeat.json"
  build_heartbeat "$SITE_NAME" "swarm" "$KIT_ROOT" "$job_id" "$run_name" "running" "$hb_file" >/dev/null

  rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json" || true
  sync_daemon_log "$remote_dir"

  [ -f "$STARTUP_DIR/nohup.out" ] && \
    rsync_cmd "$STARTUP_DIR/nohup.out" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/nohup.out" || true

  [ -f "$KIT_ROOT/log.txt" ] && \
    rsync_cmd "$KIT_ROOT/log.txt" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/log.txt" || true

  upload_global_models "$job_id" "$remote_dir"

  # Determine run dir: check scratch dir first, fall back to startup dir
  local run_dir_for_sync=""
  if [ -n "$SCRATCHDIR" ] && [ -n "$run_name" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$run_name" ]; then
    run_dir_for_sync="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
  elif [ -n "$run_name" ] && [ -d "$STARTUP_DIR/runs/$SITE_NAME/$run_name" ]; then
    run_dir_for_sync="$STARTUP_DIR/runs/$SITE_NAME/$run_name"
  fi

  if [ -n "$run_dir_for_sync" ]; then
    rsync_cmd \
      --include='*/' \
      --include='events.out.tfevents*' \
      --exclude='*' \
      "$run_dir_for_sync/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true

    now="$(date +%s)"
    last="$(cat "$LAST_CKPT_SYNC_FILE" 2>/dev/null || echo 0)"
    if [ $((now - ${last:-0})) -ge "$CKPT_SYNC_INTERVAL" ]; then
      set_periodic_filters
      rsync_cmd "${RUN_DIR_PERIODIC_FILTERS[@]}" \
        "$run_dir_for_sync/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
      echo "$now" > "$LAST_CKPT_SYNC_FILE" 2>/dev/null || true
    fi
  fi
}

final_sync() {
  if [ "$MODE" = "local" ]; then
    sync_local || true
    local run_name run_dir
    run_name="$(find_latest_local_run_name || true)"
    if [ -n "$run_name" ]; then
      # Determine run_dir: check scratch dir first, fall back to startup dir
      run_dir=""
      if [ -n "$SCRATCHDIR" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$run_name" ]; then
        run_dir="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
      elif [ -d "$STARTUP_DIR/runs/$SITE_NAME/$run_name" ]; then
        run_dir="$STARTUP_DIR/runs/$SITE_NAME/$run_name"
      fi

      remote_dir="$(build_remote_dir "$run_name")"
      hb_file="$STATE_DIR/local_heartbeat_final.json"
      export SCRATCHDIR
      refresh_upload_policy ""
      build_heartbeat "$SITE_NAME" "local" "$KIT_ROOT" "" "$run_name" "finished" "$hb_file" >/dev/null
      rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat_final.json" || true
      sync_daemon_log "$remote_dir"
      if [ -n "$run_dir" ]; then
        set_run_dir_excludes
        rsync_cmd ${RUN_DIR_EXCLUDES[@]+"${RUN_DIR_EXCLUDES[@]}"} \
          "$run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
      fi
    fi
  else
    sync_swarm || true
    local job_id run_name remote_dir hb_file hb_name="heartbeat_final.json"
    job_id="$(find_latest_job_id || true)"
    if [ -z "$job_id" ]; then
      # Waiting for a job when the client went away: say so instead of a "waiting" that goes stale.
      if [ "$FINAL_STATUS" = "stopped" ] && [ -f "$ACTIVE_OWNER_FILE" ]; then
        remote_dir="$(build_remote_dir "_active")"
        hb_file="$STATE_DIR/swarm_active_heartbeat.json"
        build_heartbeat "$SITE_NAME" "swarm" "$KIT_ROOT" "" "" "stopped" "$hb_file" >/dev/null
        rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/heartbeat.json" || true
        sync_daemon_log "$remote_dir"
      fi
      return 0
    fi
    run_name="$(extract_run_name_from_nohup || true)"
    [ -n "$run_name" ] || run_name="$(find_latest_swarm_run_name || true)"
    remote_dir="$(build_remote_dir "$job_id")"
    hb_file="$STATE_DIR/swarm_heartbeat_final.json"
    export SCRATCHDIR
    refresh_upload_policy "$job_id"
    # Stopped because the client container is gone (#614): the newest job may have finished
    # long ago or been cut off, so no heartbeat_final.json (it would overrule what the monitor
    # reads from the job's logs); the heartbeat says "stopped" instead of "running".
    if [ "$FINAL_STATUS" = "stopped" ]; then
      hb_name="heartbeat.json"
      hb_file="$STATE_DIR/swarm_heartbeat.json"   # the file docker.sh --status reads
    fi
    build_heartbeat "$SITE_NAME" "swarm" "$KIT_ROOT" "$job_id" "$run_name" "$FINAL_STATUS" "$hb_file" >/dev/null
    rsync_cmd "$hb_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/${hb_name}" || true
    sync_daemon_log "$remote_dir"

    [ -f "$STARTUP_DIR/nohup.out" ] && \
      rsync_cmd "$STARTUP_DIR/nohup.out" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/nohup.out" || true

    [ -f "$KIT_ROOT/log.txt" ] && \
      rsync_cmd "$KIT_ROOT/log.txt" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/log.txt" || true

    upload_global_models "$job_id" "$remote_dir"

    # Sync run dir: check scratch dir first, fall back to startup dir
    local final_run_dir=""
    if [ -n "$SCRATCHDIR" ] && [ -n "$run_name" ] && [ -d "$SCRATCHDIR/runs/$SITE_NAME/$run_name" ]; then
      final_run_dir="$SCRATCHDIR/runs/$SITE_NAME/$run_name"
    elif [ -n "$run_name" ] && [ -d "$STARTUP_DIR/runs/$SITE_NAME/$run_name" ]; then
      final_run_dir="$STARTUP_DIR/runs/$SITE_NAME/$run_name"
    fi
    if [ -n "$final_run_dir" ]; then
      set_run_dir_excludes
      rsync_cmd ${RUN_DIR_EXCLUDES[@]+"${RUN_DIR_EXCLUDES[@]}"} \
        "$final_run_dir/" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/run_dir/" || true
    fi
  fi
}

trap final_sync EXIT
# docker.sh and --migrate_from stop the daemon with SIGTERM. Without a handler bash dies on the
# signal and skips the EXIT trap, so the last log lines of a run would never be uploaded.
trap 'exit 0' TERM INT HUP

while true; do
  begin_cycle
  client_watchdog
  if [ "$MODE" = "local" ]; then
    sync_local || true
  else
    sync_swarm || true
  fi
  end_cycle
  sleep "$LOG_SYNC_INTERVAL"
done
