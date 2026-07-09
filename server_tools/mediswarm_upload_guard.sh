#!/bin/bash
# Forced-command guard for the `mediswarm-upload` account on the live-monitor host.
#
# Purpose: the upload account only needs to receive live_sync uploads. Without a
# forced command, an authorized key grants a full interactive shell on the host
# that also runs the FL server (see #419). This script restricts every key to the
# three operations kit_live_sync/live_sync.sh performs — and nothing else.
#
# Install (as root, on the monitor host):
#   install -o root -g root -m 755 server_tools/mediswarm_upload_guard.sh \
#           /etc/ssh/mediswarm_upload_guard.sh
# Then prefix EVERY key line in ~mediswarm-upload/.ssh/authorized_keys with:
#   restrict,command="/etc/ssh/mediswarm_upload_guard.sh" ssh-ed25519 AAAA... comment
# (`restrict` disables pty + all forwarding; `command=` forces this script
#  regardless of what the client asks to run.)
#
# Permits ONLY:
#   - `echo ok`                                     (connectivity check)
#   - `mkdir -p '<path under /srv/mediswarm/live>'`  (ensure_remote_dir)
#   - `rsync --server ...`  -> rrsync -wo /          (upload; write-only, no exfil)
# Everything else is rejected and logged to syslog (tag: mediswarm-upload-guard).
set -u

UPLOAD_ROOT="/srv/mediswarm/live"
cmd="${SSH_ORIGINAL_COMMAND:-}"
from="${SSH_CLIENT%% *}"
log() { logger -t mediswarm-upload-guard -- "from=$from action=$1 cmd=[$cmd]"; }

# 1) connectivity check
if [ "$cmd" = "echo ok" ]; then echo ok; exit 0; fi

# 2) mkdir -p '<path>' strictly under the upload root, no traversal
if [[ "$cmd" =~ ^mkdir\ -p\ \'(/srv/mediswarm/live/[A-Za-z0-9._@=+/-]+)\'$ ]]; then
    d="${BASH_REMATCH[1]}"
    if [[ "$d" == *..* ]]; then log reject-traversal; exit 1; fi
    log allow-mkdir
    exec mkdir -p -- "$d"
fi

# 3) rsync upload only: write-only (blocks any download/exfil). OS permissions
#    confine writes to what the account owns (the upload root). rrsync also
#    filters dangerous rsync options (e.g. --rsync-path).
if [[ "$cmd" == "rsync --server "* ]]; then
    log allow-rsync
    exec /usr/bin/rrsync -wo /
fi

log REJECT
echo "This account only accepts MediSwarm live-sync uploads." >&2
exit 1
