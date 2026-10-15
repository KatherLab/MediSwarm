#!/bin/bash
# Forced-command guard for the `mediswarm-upload` account on the live-monitor host.
#
# Purpose: the upload account only needs to receive live_sync uploads. Without a
# forced command, an authorized key grants a full interactive shell on the host
# that also runs the FL server (see #419). This script restricts every key to the
# three operations kit_live_sync/live_sync.sh performs, and nothing else.
#
# Install (as root, on the monitor host):
#   install -o root -g root -m 755 server_tools/mediswarm_upload_guard.sh \
#           /etc/ssh/mediswarm_upload_guard.sh
# Every key line in ~mediswarm-upload/.ssh/authorized_keys forces this script, with the
# site's name as the only argument for the keys the 1.9.0 kit build generates:
#   restrict,command="/etc/ssh/mediswarm_upload_guard.sh UKA_1" ssh-ed25519 AAAA... UKA_1@kit-1.9.0
# (scripts/admin/register_upload_keys.sh installs these lines), and without one for the
# keys sites sent by hand before 1.9.0:
#   restrict,command="/etc/ssh/mediswarm_upload_guard.sh" ssh-ed25519 AAAA... comment
# (`restrict` disables pty + all forwarding; `command=` forces this script
#  regardless of what the client asks to run.)
#
# The key's area is /srv/mediswarm/live/<SITE> for a key with a site, else /srv/mediswarm/live.
# Permits ONLY:
#   - `echo ok`                           (connectivity check)
#   - `mkdir -p '<dir in the area>'`      (ensure_remote_dir)
#   - `rm -rf '<path below the area>'`    (clears the per-run _active marker)
#   - `rsync --server ... . <dest>`       (upload of a dest in the area: write-only, no exfil,
#                                          files above 1500M skipped)
# A path must be absolute, made of plain characters, free of `..` and empty or `.`
# components, and must still lie in the area once symlinks are resolved.
# Everything else is rejected and logged to syslog (tag: mediswarm-upload-guard).
#
# Why the destination is checked here (PS-1): the account's authorized_keys used to live in
# a home it owned, and rrsync was confined to "/", where it does no path checking at all. A
# site key could therefore rsync its own authorized_keys over the forced-command one and log
# in with a shell. The home is now root-owned (server_tools/README.md); this check is the
# second line: no key writes outside the upload tree, and a site key nowhere but its folder.
set -u

UPLOAD_ROOT="/srv/mediswarm/live"
SITE="${1:-}"
cmd="${SSH_ORIGINAL_COMMAND:-}"
from="${SSH_CLIENT:-}"
from="${from%% *}"
log() { logger -t mediswarm-upload-guard -- "from=$from site=${SITE:--} action=$1 cmd=[$cmd]"; }
deny() {
    log "$1"
    echo "This account only accepts MediSwarm live-sync uploads${SITE:+ for $SITE}." >&2
    exit 1
}

# A malformed key line must not fall back to the wider legacy area.
if [ "$#" -gt 1 ]; then deny reject-guard-arguments; fi
if [ -n "$SITE" ] && [[ ! "$SITE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then deny reject-bad-site; fi

AREA="$UPLOAD_ROOT${SITE:+/$SITE}"
REAL_ROOT="$(realpath -m -- "$UPLOAD_ROOT")" || deny reject-no-upload-root
# Built from the resolved root rather than resolved itself: a symlink planted in place of
# a site folder must not move that site's area somewhere else.
REAL_AREA="$REAL_ROOT${SITE:+/$SITE}"

# Prints $1 relative to the area ("." for the area itself, a trailing slash kept) if it is
# an absolute path inside it; fails otherwise.
area_relative() {
    local p="$1" trail="" rel real
    [[ "$p" =~ ^/[A-Za-z0-9._@=+/-]*$ ]] || return 1
    [[ "$p" == *..* ]] && return 1
    if [ "$p" != "/" ] && [ "${p%/}" != "$p" ]; then trail="/"; p="${p%/}"; fi
    case "$p/" in *//* | */./*) return 1 ;; esac
    case "$p" in
        "$AREA") rel="." ;;
        "$AREA"/*) rel="${p#"$AREA"/}" ;;
        *) return 1 ;;
    esac
    # A symlink inside the area (an uploaded one included) must not lead out of it.
    real="$(realpath -m -- "$p")" || return 1
    case "$real" in
        "$REAL_AREA" | "$REAL_AREA"/*) ;;
        *) return 1 ;;
    esac
    if [ "$rel" = "." ]; then echo "."; else echo "$rel$trail"; fi
}

# 1) connectivity check
if [ "$cmd" = "echo ok" ]; then echo ok; exit 0; fi

# 2) mkdir -p '<dir>' inside the area
if [[ "$cmd" =~ ^mkdir\ -p\ \'([^\']*)\'$ ]]; then
    d="${BASH_REMATCH[1]}"
    area_relative "$d" > /dev/null || deny reject-outside-area
    log allow-mkdir
    exec mkdir -p -- "$d"
fi

# 3) rm -rf '<path>' strictly below the area. live_sync.sh clears its per-run `_active`
#    marker this way. A key can already delete within its area via `rsync --delete`, so
#    this grants no new power; the area itself (a whole site, or the whole upload tree)
#    is never removed.
if [[ "$cmd" =~ ^rm\ -rf\ \'([^\']*)\'$ ]]; then
    d="${BASH_REMATCH[1]}"
    rel="$(area_relative "$d")" || deny reject-outside-area
    if [ "$rel" = "." ]; then deny reject-remove-area; fi
    log allow-rm
    exec rm -rf -- "$d"
fi

# 4) rsync upload only. The client runs `rsync --server <options> . <dest>` for a push;
#    `--sender` would be a download. rrsync then enforces write-only and filters dangerous
#    rsync options (e.g. --rsync-path, -s), but only checks paths when its restricted
#    directory is not "/", so the destination is checked here and rrsync gets the area as
#    its root, with the destination made relative to it.
if [[ "$cmd" == "rsync --server "* ]]; then
    read -r -a words <<< "$cmd"
    n="${#words[@]}"
    if [ "$n" -lt 4 ] || [ "${words[n-2]}" != "." ]; then deny reject-rsync-arguments; fi
    opts=("${words[@]:2:n-4}")
    for o in "${opts[@]}"; do
        case "$o" in
            --sender | --server) deny reject-rsync-download ;;
            -*) ;;
            *) deny reject-rsync-arguments ;;
        esac
        # Options that take a path (--temp-dir, --backup-dir, --partial-dir, ...) must not
        # point anywhere else either; live_sync sends none of them.
        case "$o" in *..* | *=/* | *=~*) deny reject-rsync-path-option ;; esac
    done
    rel="$(area_relative "${words[n-1]}")" || deny reject-outside-area
    # rrsync refuses a restricted directory that does not exist: create a new site's folder.
    if [ ! -d "$REAL_AREA" ]; then mkdir -p -- "$REAL_AREA" || deny reject-no-area; fi
    log allow-rsync
    # Cap the size of any single uploaded file. --max-size is a receiver-side option, so
    # adding it here takes effect without any change at the sites, and rrsync accepts it.
    # Oversized files are skipped silently (rsync exits 0, so live_sync records no error).
    # Why: the CAM model's Lightning checkpoints are 3.2 GB and live_sync re-sends them every
    # epoch, competing with the swarm's own 1 GB model transfers on the sites' VPN uplinks.
    # The default lets MST (~0.3 GB) and 1DivideAndConquer (1.1 GB) checkpoints through.
    SSH_ORIGINAL_COMMAND="rsync --server --max-size=${MEDISWARM_UPLOAD_MAX_SIZE:-1500M}${opts[*]:+ ${opts[*]}} . $rel"
    export SSH_ORIGINAL_COMMAND
    # -no-lock: rrsync otherwise takes an exclusive flock() on its restricted directory,
    # which serialises the uploads of every key with that root. With several sites
    # live-syncing every 30s, the first site's daemon holds the lock and every other site
    # is refused with
    #     rrsync error: Another instance of rrsync is already accessing this directory.
    # Observed in DECADE: one site's daemon blocked uploads for all the others, so we
    # were blind during the run. Each site writes into its own subdirectory, so the
    # single-run lock buys us nothing.
    exec /usr/bin/rrsync -wo -no-lock "$REAL_AREA"
fi

deny REJECT
