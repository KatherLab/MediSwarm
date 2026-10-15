#!/usr/bin/env bash
#
# Install the live-sync upload keys of a kit build into the upload account's authorized_keys.
#
#   sudo scripts/admin/register_upload_keys.sh <fragment> [--authorized-keys PATH] [--dry-run]
#                                              [--retire-comment-prefix P]... [--owner USER:GROUP]
#
# <fragment> is the upload_authorized_keys.txt the kit build writes next to the prod_NN zips
# (scripts/build/_injectLiveSyncIntoStartupKits.sh): one line per client kit,
#   restrict,command="/etc/ssh/mediswarm_upload_guard.sh <SITE>" ssh-ed25519 AAAA... <SITE>@kit-<version>
#
# Why a script: the upload account's home is root-owned so that an upload can never rewrite
# its own authorized_keys (PS-1). Editing the file by hand as root easily leaves it owned by
# root:root 600, which sshd cannot read as the account, so every site's live sync stops.
#
# What it does:
#   - checks every fragment line BEFORE touching anything: only lines that force the upload
#     guard (restrict + command="/etc/ssh/mediswarm_upload_guard.sh [SITE]") are accepted, so
#     a mistake can never install a key that opens a shell;
#   - is idempotent: a key already in the file with the same options is left as it is; the
#     same key with OTHER options aborts the run (sshd uses the first matching line, so which
#     one wins must be decided by hand);
#   - never removes a line unless asked: --retire-comment-prefix P (repeatable) drops the
#     existing keys whose comment starts with P, except the keys of this fragment. Preview with
#     --dry-run, e.g. --retire-comment-prefix UKA_1@kit- once UKA_1 runs the new kit;
#   - backs the file up first (authorized_keys.bak-<UTC time>, same owner and mode), writes
#     the new version next to it and renames it into place, owner root:mediswarm-upload, 640.
#
# --authorized-keys defaults to ~mediswarm-upload/.ssh/authorized_keys. --owner is for tests
# and other hosts; with the default owner the script must run as root.

set -euo pipefail

GUARD="/etc/ssh/mediswarm_upload_guard.sh"
ACCOUNT="mediswarm-upload"
MODE=640
SITE_RE='^[A-Za-z0-9][A-Za-z0-9._-]*$'
KEY_TYPE_RE='^(ssh-ed25519|ecdsa-sha2-nistp(256|384|521)|ssh-rsa|sk-ssh-ed25519@openssh\.com|sk-ecdsa-sha2-nistp256@openssh\.com)$'
BLOB_RE='^[A-Za-z0-9+/]+={0,3}$'

usage() {
    sed -n '3,7p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}
die() {
    echo "ERROR: $*" >&2
    exit 1
}

FRAGMENT=""
AUTHORIZED_KEYS=""
DRY_RUN=0
OWNER=""
RETIRE_PREFIXES=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --authorized-keys) AUTHORIZED_KEYS="${2:?--authorized-keys needs a path}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --retire-comment-prefix)
            [ -n "${2:-}" ] || die "--retire-comment-prefix needs a non-empty prefix"
            RETIRE_PREFIXES+=("$2"); shift 2 ;;
        --owner) OWNER="${2:?--owner needs USER:GROUP}"; shift 2 ;;
        -h | --help) usage ;;
        -*) echo "Unknown option: $1" >&2; usage ;;
        *) [ -z "$FRAGMENT" ] || usage; FRAGMENT="$1"; shift ;;
    esac
done
[ -n "$FRAGMENT" ] || usage
[ -f "$FRAGMENT" ] || die "no such fragment: $FRAGMENT"

if [ -z "$AUTHORIZED_KEYS" ]; then
    home="$(getent passwd "$ACCOUNT" | cut -d: -f6)"
    [ -n "$home" ] || die "no account $ACCOUNT on this host; pass --authorized-keys"
    AUTHORIZED_KEYS="$home/.ssh/authorized_keys"
fi
if [ -z "$OWNER" ]; then
    OWNER="root:$ACCOUNT"
    if [ "$DRY_RUN" = 0 ] && [ "$(id -u)" -ne 0 ]; then
        die "run as root (the file stays owned by $OWNER): sudo $0 ${*:-$FRAGMENT}"
    fi
fi

# Splits an authorized_keys line into KEY_OPTS, KEY_TYPE, KEY_BLOB and KEY_COMMENT. Options
# come first when present and end at the first blank outside double quotes, as for sshd.
split_key_line() {
    local line="$1" i=0 n c quoted=0
    KEY_OPTS="" KEY_TYPE="" KEY_BLOB="" KEY_COMMENT=""
    line="${line#"${line%%[![:space:]]*}"}"
    if [[ ! "$line" =~ ^(ssh-|ecdsa-|sk-) ]]; then
        n="${#line}"
        while [ "$i" -lt "$n" ]; do
            c="${line:i:1}"
            if [ "$quoted" = 1 ] && [ "$c" = '\' ]; then
                i=$((i + 2))
                continue
            fi
            if [ "$c" = '"' ]; then
                quoted=$((1 - quoted))
            elif [ "$quoted" = 0 ] && [[ "$c" == [[:space:]] ]]; then
                break
            fi
            i=$((i + 1))
        done
        KEY_OPTS="${line:0:i}"
        line="${line:i}"
    fi
    read -r KEY_TYPE KEY_BLOB KEY_COMMENT <<< "$line" || true
    [[ "$KEY_TYPE" =~ $KEY_TYPE_RE ]] && [[ "$KEY_BLOB" =~ $BLOB_RE ]]
}

is_entry() { [[ ! "$1" =~ ^[[:space:]]*(#.*)?$ ]]; }

# ── 1. The fragment: every line must be a guarded upload key ────────────────────────────
declare -A NEW_OPTS=() NEW_SITE=()
NEW_LINES=() NEW_BLOBS=()
lineno=0
while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))
    is_entry "$line" || continue
    split_key_line "$line" || die "$FRAGMENT:$lineno is not an SSH public key line"
    site=""
    if [ "$KEY_OPTS" = "restrict,command=\"$GUARD\"" ]; then
        site="(any site: legacy key)"
    elif [[ "$KEY_OPTS" =~ ^restrict,command=\"$GUARD\ ([^\"]*)\"$ ]] && site="${BASH_REMATCH[1]}" \
            && [[ "$site" =~ $SITE_RE ]]; then
        :   # site taken before the second =~ overwrites BASH_REMATCH
    else
        die "$FRAGMENT:$lineno does not force the upload guard (expected restrict,command=\"$GUARD <SITE>\"); refusing to install it"
    fi
    [ -z "${NEW_OPTS[$KEY_BLOB]+x}" ] || die "$FRAGMENT:$lineno repeats a key of an earlier line"
    NEW_OPTS[$KEY_BLOB]="$KEY_OPTS"
    NEW_SITE[$KEY_BLOB]="$site"
    NEW_LINES+=("$KEY_OPTS $KEY_TYPE $KEY_BLOB${KEY_COMMENT:+ $KEY_COMMENT}")
    NEW_BLOBS+=("$KEY_BLOB")
done < "$FRAGMENT"
[ "${#NEW_LINES[@]}" -gt 0 ] || die "$FRAGMENT holds no key lines"

# ── 2. Plan against the current file ─────────────────────────────────────────────────────
[ -d "$(dirname "$AUTHORIZED_KEYS")" ] \
    || die "$(dirname "$AUTHORIZED_KEYS") does not exist; create and harden it first (server_tools/README.md)"
declare -A PRESENT=()
KEEP=() RETIRED=()
if [ -f "$AUTHORIZED_KEYS" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        if is_entry "$line" && split_key_line "$line"; then
            if [ -n "${NEW_OPTS[$KEY_BLOB]+x}" ]; then
                if [ "$KEY_OPTS" != "${NEW_OPTS[$KEY_BLOB]}" ]; then
                    die "the key of ${NEW_SITE[$KEY_BLOB]} is already in $AUTHORIZED_KEYS with other options:
  ${line}
sshd uses the first matching line: remove or fix that line by hand, then run this again. Nothing was changed."
                fi
                PRESENT[$KEY_BLOB]=1
            else
                retire=0
                for p in "${RETIRE_PREFIXES[@]}"; do
                    [[ "$KEY_COMMENT" == "$p"* ]] && retire=1
                done
                if [ "$retire" = 1 ]; then
                    RETIRED+=("$KEY_TYPE ${KEY_BLOB:0:16}... ${KEY_COMMENT}")
                    continue
                fi
            fi
        fi
        KEEP+=("$line")
    done < "$AUTHORIZED_KEYS"
fi

ADD=()
for i in "${!NEW_BLOBS[@]}"; do
    blob="${NEW_BLOBS[$i]}"
    if [ -n "${PRESENT[$blob]+x}" ]; then
        echo "already registered: ${NEW_SITE[$blob]}  (${NEW_LINES[$i]##* })"
    else
        ADD+=("${NEW_LINES[$i]}")
        echo "add:     ${NEW_SITE[$blob]}  (${NEW_LINES[$i]##* })"
    fi
done
for r in "${RETIRED[@]}"; do echo "retire:  $r"; done

if [ "${#ADD[@]}" -eq 0 ] && [ "${#RETIRED[@]}" -eq 0 ]; then
    echo "Nothing to change in $AUTHORIZED_KEYS."
    exit 0
fi
if [ "$DRY_RUN" = 1 ]; then
    echo "Dry run: $AUTHORIZED_KEYS was not changed (${#ADD[@]} to add, ${#RETIRED[@]} to retire)."
    exit 0
fi

# ── 3. Write: backup, new file next to the old one, owner and mode, rename ─────────────────
if [ ! -f "$GUARD" ]; then
    echo "WARNING: $GUARD is not installed here; these keys cannot log in until it is (server_tools/README.md)." >&2
fi
dir="$(dirname "$AUTHORIZED_KEYS")"
tmp="$(mktemp "$dir/.authorized_keys.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
{
    if [ "${#KEEP[@]}" -gt 0 ]; then printf '%s\n' "${KEEP[@]}"; fi
    printf '%s\n' "${ADD[@]+"${ADD[@]}"}"
} | sed '/^$/{$d}' > "$tmp"
chown "$OWNER" "$tmp"
chmod "$MODE" "$tmp"
if [ -f "$AUTHORIZED_KEYS" ]; then
    backup="$AUTHORIZED_KEYS.bak-$(date -u +%Y%m%dT%H%M%SZ)"
    cp -p "$AUTHORIZED_KEYS" "$backup"
    echo "backup:  $backup"
fi
mv -f "$tmp" "$AUTHORIZED_KEYS"
trap - EXIT
echo "Updated $AUTHORIZED_KEYS: ${#ADD[@]} added, ${#RETIRED[@]} retired (owner $OWNER, mode $MODE)."
