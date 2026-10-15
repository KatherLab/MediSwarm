#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: _injectLiveSyncIntoStartupKits.sh SWARM_PROJECT.yml"
  exit 1
fi

PROJECT_YML="$1"
OUTPUT_FOLDER="workspace/$(grep '^name: ' "$PROJECT_YML" | sed 's/name: //')"
TARGET_FOLDER="$(ls -d "$OUTPUT_FOLDER"/prod_* | tail -n 1)"

# Image repository this project's kits follow, taken from the project YAML rather than
# hardcoded: this script provisions BOTH consortia (ODELIA -> jefftud/odelia, DECADE ->
# jefftud/decade). A hardcoded repo here writes the wrong channel into the other
# consortium's image.conf, pointing every one of its sites at a foreign image.
_docker_image_value="$(grep -m1 -E '^[[:space:]]*docker_image:' "$PROJECT_YML" \
  | sed -E 's/^[[:space:]]*docker_image:[[:space:]]*//; s/[[:space:]]*$//' || true)"
if [ -z "$_docker_image_value" ]; then
  echo "Could not read docker_image from $PROJECT_YML -- cannot pick a release channel."
  exit 1
fi
# Strip the tag to get the bare repository. Only treat the trailing ':...' as a tag when
# it is not a registry port (a port segment still contains a '/', e.g. host:5000/repo).
case "${_docker_image_value##*:}" in
  */*|"") IMAGE_REPO="$_docker_image_value" ;;
  *)      IMAGE_REPO="${_docker_image_value%:*}" ;;
esac

# Only used to make the "pin a specific version" example in image.conf concrete.
PINNED_EXAMPLE_TAG="$(grep -m1 -E '^[0-9]' odelia_image.version 2>/dev/null | tr -d '[:space:]' || true)"
PINNED_EXAMPLE_TAG="${PINNED_EXAMPLE_TAG:-1.6.0}"

echo "Release channel for these kits: ${IMAGE_REPO}:current"

HELPER_SOURCE_DIR="kit_live_sync"
SYNC_CONF_SOURCE="$HELPER_SOURCE_DIR/sync.conf"
SYNC_CONF_EXAMPLE_SOURCE="$HELPER_SOURCE_DIR/sync.conf.example"
ADMIN_HELPER_SOURCE="kit_admin_tools/prepare_odelia_job.sh"
ADMIN_PATCHER_SOURCE="scripts/admin/patch_warm_start_job.py"

if [ ! -d "$HELPER_SOURCE_DIR" ]; then
  echo "Missing helper directory: $HELPER_SOURCE_DIR"
  exit 1
fi

if [ ! -f "$SYNC_CONF_SOURCE" ] && [ ! -f "$SYNC_CONF_EXAMPLE_SOURCE" ]; then
  echo "Missing live sync config files in $HELPER_SOURCE_DIR"
  exit 1
fi

if [ ! -f "$ADMIN_HELPER_SOURCE" ]; then
  echo "Missing admin helper: $ADMIN_HELPER_SOURCE"
  exit 1
fi

if [ ! -f "$ADMIN_PATCHER_SOURCE" ]; then
  echo "Missing admin job patcher: $ADMIN_PATCHER_SOURCE"
  exit 1
fi

echo "Injecting live sync helpers into startup kits under $TARGET_FOLDER"

if [ ! -f "$SYNC_CONF_SOURCE" ]; then
  echo "No local live sync config found; injecting sync.conf.example only."
fi

# ── Per-site upload keys (1.9.0) ─────────────────────────────────────────────────────────
# Until 1.8.1 every site generated an SSH key and emailed it, and we added it by hand (#615).
# Now each client kit gets its own key pair (startup/upload_key + .pub) and
# startup/upload_known_hosts with the upload host's key, which the sync.conf block between
# the markers below makes live sync use. The public keys go into ONE file next to the
# prod_NN zips, never into a kit: upload_authorized_keys.txt, the authorized_keys lines
# that confine each key to its own folder (server_tools/mediswarm_upload_guard.sh <SITE>).
# Register it on the upload host before handing out the kits:
#   sudo scripts/admin/register_upload_keys.sh workspace/<project>/prod_NN/upload_authorized_keys.txt
#
#   MEDISWARM_UPLOAD_KEYS          auto (default): generate when the upload host's key is
#                                  available, else warn and build kits without keys;
#                                  1: same, but fail instead of warning; 0: never.
#   MEDISWARM_UPLOAD_HOST_KEY_PUB  the upload host's public host key (default
#                                  /etc/ssh/ssh_host_ed25519_key.pub: kits are built on the
#                                  upload host)
#   MEDISWARM_UPLOAD_HOST_KEY      that key's line itself; wins over the path.
#                                  _buildStartupKits.sh reads the file on the build host and
#                                  passes the line, since this script runs in the image.
#   MEDISWARM_UPLOAD_HOST_NAMES    names the sites reach the upload host by, recorded in
#                                  upload_known_hosts (default dl3.tud.de,172.24.4.65; the
#                                  REMOTE_HOST of the shipped sync.conf is added). live sync
#                                  checks the key under a fixed alias, so a site whose
#                                  REMOTE_HOST is another name for the same host still works.
UPLOAD_KEYS_MODE="${MEDISWARM_UPLOAD_KEYS:-auto}"
UPLOAD_HOST_KEY_PUB="${MEDISWARM_UPLOAD_HOST_KEY_PUB:-/etc/ssh/ssh_host_ed25519_key.pub}"
UPLOAD_HOST_NAMES="${MEDISWARM_UPLOAD_HOST_NAMES:-dl3.tud.de,172.24.4.65}"
UPLOAD_HOST_ALIAS="mediswarm-upload-host"   # must match HostKeyAlias in the sync.conf block
UPLOAD_GUARD="/etc/ssh/mediswarm_upload_guard.sh"
UPLOAD_FRAGMENT="$TARGET_FOLDER/upload_authorized_keys.txt"
UPLOAD_KEY_BLOCK_BEGIN="# >>> kit upload key >>>"
UPLOAD_KEY_BLOCK_END="# <<< kit upload key <<<"

# The sync.conf block that uses the key, taken from sync.conf.example so there is one copy.
# A local sync.conf (gitignored, carries the real REMOTE_HOST) predates it and gets it appended.
UPLOAD_KEY_BLOCK=""
if [ -f "$SYNC_CONF_EXAMPLE_SOURCE" ]; then
  UPLOAD_KEY_BLOCK="$(awk -v b="$UPLOAD_KEY_BLOCK_BEGIN" -v e="$UPLOAD_KEY_BLOCK_END" \
    '$0 == b {on = 1} on {print} $0 == e {on = 0}' "$SYNC_CONF_EXAMPLE_SOURCE")"
fi

# Kit version for the key comments (<SITE>@kit-<version>): _buildStartupKits.sh exports
# VERSION; otherwise the tag of the project's docker_image, unless it is the placeholder.
UPLOAD_KIT_VERSION="${VERSION:-}"
if [ -z "$UPLOAD_KIT_VERSION" ]; then
  case "${_docker_image_value##*:}" in
    */* | __*) ;;
    *) UPLOAD_KIT_VERSION="${_docker_image_value##*:}" ;;
  esac
fi
UPLOAD_KIT_VERSION="${UPLOAD_KIT_VERSION:-unknown}"

_upload_keys_unavailable() {
  if [ "$UPLOAD_KEYS_MODE" = 1 ]; then
    echo "ERROR: MEDISWARM_UPLOAD_KEYS=1 but $1" >&2
    exit 1
  fi
  echo "WARNING: $1"
  echo "WARNING: building the kits WITHOUT upload keys; their live sync logs in with the site's own key as in 1.8.1."
  UPLOAD_HOST_KEY=""
}

UPLOAD_HOST_KEY=""
case "$UPLOAD_KEYS_MODE" in
  0) echo "MEDISWARM_UPLOAD_KEYS=0: no upload keys in these kits." ;;
  auto | 1)
    _host_key_line="${MEDISWARM_UPLOAD_HOST_KEY:-}"
    _host_key_source="MEDISWARM_UPLOAD_HOST_KEY"
    if [ -z "$_host_key_line" ]; then
      _host_key_source="$UPLOAD_HOST_KEY_PUB"
      _host_key_line="$(head -n 1 "$UPLOAD_HOST_KEY_PUB" 2>/dev/null || true)"
    fi
    read -r _host_key_type _host_key_blob _ <<< "$_host_key_line" || true
    if [ -z "$_host_key_line" ]; then
      _upload_keys_unavailable "no upload host key: $UPLOAD_HOST_KEY_PUB is not readable here and MEDISWARM_UPLOAD_HOST_KEY is empty."
    elif ! [[ "${_host_key_type:-}" =~ ^(ssh-ed25519|ecdsa-sha2-nistp(256|384|521)|ssh-rsa)$ ]] \
        || ! [[ "${_host_key_blob:-}" =~ ^[A-Za-z0-9+/]+={0,3}$ ]]; then
      _upload_keys_unavailable "the upload host key from $_host_key_source is not an SSH public key line."
    elif ! python3 -c 'from cryptography.hazmat.primitives.asymmetric import ed25519' > /dev/null 2>&1; then
      _upload_keys_unavailable "python3 with the cryptography package is needed to generate the upload keys."
    else
      UPLOAD_HOST_KEY="$_host_key_type $_host_key_blob"
      # The known_hosts patterns: the alias live sync checks against, then every name of the
      # host. A name is a hostname or an IP address; anything else is left out.
      _shipped_conf="$SYNC_CONF_EXAMPLE_SOURCE"
      [ ! -f "$SYNC_CONF_SOURCE" ] || _shipped_conf="$SYNC_CONF_SOURCE"
      _shipped_remote_host="$(sed -n 's/^REMOTE_HOST=["'\'']\{0,1\}\([^"'\'']*\)["'\'']\{0,1\}[[:space:]]*$/\1/p' \
        "$_shipped_conf" | tail -n 1)"
      IFS=', ' read -r -a _names <<< "$UPLOAD_HOST_NAMES $_shipped_remote_host" || true
      UPLOAD_KNOWN_HOSTS_PATTERNS="$UPLOAD_HOST_ALIAS"
      for _name in "${_names[@]}"; do
        [[ "$_name" =~ ^[A-Za-z0-9][A-Za-z0-9.:_-]*$ ]] || { echo "WARNING: ignoring upload host name '$_name'"; continue; }
        case ",$UPLOAD_KNOWN_HOSTS_PATTERNS," in *",$_name,"*) ;; *) UPLOAD_KNOWN_HOSTS_PATTERNS+=",$_name" ;; esac
      done
      UPLOAD_HOST_FINGERPRINT="$(python3 -c 'import base64, hashlib, sys
print("SHA256:" + base64.b64encode(hashlib.sha256(base64.b64decode(sys.argv[1])).digest()).decode().rstrip("="))' \
        "$_host_key_blob")"
      echo "Upload keys: one per client kit; upload host key $UPLOAD_HOST_FINGERPRINT ($_host_key_type, from $_host_key_source)"
      echo "             pinned for $UPLOAD_KNOWN_HOSTS_PATTERNS"
      # Collected by the kit loop below (a subshell), installed after it.
      UPLOAD_FRAGMENT_TMP="$(mktemp "$TARGET_FOLDER/.upload_authorized_keys.XXXXXX")"
      trap 'rm -f "$UPLOAD_FRAGMENT_TMP"' EXIT
    fi
    ;;
  *)
    echo "ERROR: MEDISWARM_UPLOAD_KEYS must be auto, 1 or 0 (got '$UPLOAD_KEYS_MODE')" >&2
    exit 1
    ;;
esac

# Writes a new ed25519 key pair in OpenSSH format: $1 (mode 600) and $1.pub with comment $2.
# Python's cryptography, as NVFlare's provisioning uses it: the image has no ssh-keygen.
_generate_upload_key() {
  python3 - "$1" "$2" << 'PY'
import os
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

path, comment = sys.argv[1], sys.argv[2]
key = Ed25519PrivateKey.generate()
private = key.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption()
)
public = key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(private)
with open(path + ".pub", "w") as handle:
    handle.write(f"{public.decode()} {comment}\n")
PY
}

# Gives the client kit in $1 (startup dir) for site $2 its upload key and the pinned host
# key, and records the authorized_keys line for it.
_add_upload_key() {
  local startup="$1" site="$2" comment pub_type pub_blob
  if ! [[ "$site" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: client kit '$site': the upload guard only accepts site names of letters, digits, . _ -" >&2
    exit 1
  fi
  comment="${site}@kit-${UPLOAD_KIT_VERSION}"
  # A rerun on the same prod folder keeps the keys it made; a half-written pair is redone.
  if [ ! -s "$startup/upload_key" ] || [ ! -s "$startup/upload_key.pub" ]; then
    rm -f "$startup/upload_key" "$startup/upload_key.pub"
    _generate_upload_key "$startup/upload_key" "$comment"
  fi
  chmod 600 "$startup/upload_key"
  chmod 644 "$startup/upload_key.pub"
  printf '%s %s\n' "$UPLOAD_KNOWN_HOSTS_PATTERNS" "$UPLOAD_HOST_KEY" > "$startup/upload_known_hosts"
  chmod 644 "$startup/upload_known_hosts"
  read -r pub_type pub_blob _ < "$startup/upload_key.pub"
  printf 'restrict,command="%s %s" %s %s %s\n' "$UPLOAD_GUARD" "$site" "$pub_type" "$pub_blob" "$comment" \
    >> "$UPLOAD_FRAGMENT_TMP"
  echo "Upload key for $site: $startup/upload_key ($comment)"
}

find "$TARGET_FOLDER" -mindepth 1 -maxdepth 1 -type d | sort | while read -r KIT_DIR; do
  STARTUP_DIR="$KIT_DIR/startup"
  ORIGINAL_DOCKER_SH="$STARTUP_DIR/docker.sh"

  if [ ! -d "$STARTUP_DIR" ]; then
    continue
  fi

  if [ ! -f "$ORIGINAL_DOCKER_SH" ]; then
    continue
  fi

  if [ -f "$SYNC_CONF_SOURCE" ]; then
    cp "$SYNC_CONF_SOURCE" "$STARTUP_DIR/sync.conf"
  else
    cp "$SYNC_CONF_EXAMPLE_SOURCE" "$STARTUP_DIR/sync.conf"
  fi
  if [ -f "$SYNC_CONF_EXAMPLE_SOURCE" ]; then
    cp "$SYNC_CONF_EXAMPLE_SOURCE" "$STARTUP_DIR/sync.conf.example"
  fi
  # Without key files the block leaves SSH_OPTS alone, so every kit gets it.
  if [ -n "$UPLOAD_KEY_BLOCK" ] && ! grep -qxF "$UPLOAD_KEY_BLOCK_BEGIN" "$STARTUP_DIR/sync.conf"; then
    printf '\n%s\n' "$UPLOAD_KEY_BLOCK" >> "$STARTUP_DIR/sync.conf"
  fi
  # Client kits only (the archiver's rule): the server and admin kits upload nothing.
  if [ -n "$UPLOAD_HOST_KEY" ] && [ -f "$STARTUP_DIR/fed_client.json" ] \
      && [ ! -f "$STARTUP_DIR/fed_server.json" ] && [ ! -f "$STARTUP_DIR/fl_admin.sh" ] \
      && [ ! -f "$STARTUP_DIR/fed_admin.json" ]; then
    _add_upload_key "$STARTUP_DIR" "$(basename "$KIT_DIR")"
  fi
  cp "$HELPER_SOURCE_DIR/build_heartbeat.sh" "$STARTUP_DIR/build_heartbeat.sh"
  cp "$HELPER_SOURCE_DIR/live_sync.sh" "$STARTUP_DIR/live_sync.sh"

  chmod +x "$STARTUP_DIR/build_heartbeat.sh" "$STARTUP_DIR/live_sync.sh"

  # Ship image.conf so a site follows the release channel out of the box. docker.sh
  # sources this on every run, so re-tagging :current centrally updates every node on
  # its next start -- no action at the site. (A site still chooses WHEN to run; we
  # never pull or restart under them.) Don't clobber a site's own edit on re-issue.
  #
  # The channel MUST follow this project's own image repository. We run two consortia
  # off this one build script (ODELIA -> jefftud/odelia, DECADE -> jefftud/decade), so
  # a hardcoded repo here silently hands one consortium the other's image. $IMAGE_REPO
  # is derived from the project YAML's docker_image line for exactly that reason.
  if [ ! -f "$STARTUP_DIR/image.conf" ]; then
    cat > "$STARTUP_DIR/image.conf" <<IMAGECONF
# Which image this node runs. Read by docker.sh on every run.
#
# Default: the release channel. When a new release ships, we re-tag
# ${IMAGE_REPO}:current and this node picks it up the next time you start it --
# nothing to do here. The Run Board's "Run schedule" tab always names the exact
# tag of a run.
#
# To pin a specific version instead, replace the line below, e.g.
#   MEDISWARM_IMAGE=${IMAGE_REPO}:${PINNED_EXAMPLE_TAG}
# For a one-off without editing this file:  ./docker.sh --image <ref> ...
MEDISWARM_IMAGE=${IMAGE_REPO}:current
IMAGECONF
  fi

  if [ -f "$STARTUP_DIR/fl_admin.sh" ]; then
    cp "$ADMIN_HELPER_SOURCE" "$STARTUP_DIR/prepare_odelia_job.sh"
    cp "$ADMIN_PATCHER_SOURCE" "$STARTUP_DIR/patch_warm_start_job.py"
    chmod +x "$STARTUP_DIR/prepare_odelia_job.sh"
    echo "Injected admin warm-start job helper and matching patcher into $STARTUP_DIR"
  fi

  # Clean up legacy docker_original.sh wrapper if present from a previous build
  if [ -f "$STARTUP_DIR/docker_original.sh" ]; then
    rm -f "$STARTUP_DIR/docker_original.sh"
  fi

  echo "Injected live sync helpers into $STARTUP_DIR"
done

if [ -n "$UPLOAD_HOST_KEY" ]; then
  if [ -s "$UPLOAD_FRAGMENT_TMP" ]; then
    {
      echo "# MediSwarm live-sync upload keys of the client kits in $TARGET_FOLDER"
      echo "# kit version $UPLOAD_KIT_VERSION, built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "# The kits pin the upload host key $UPLOAD_HOST_FINGERPRINT; compare it on the upload host with"
      echo "#   ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub"
      echo "# Install these lines BEFORE the kits go out (as root, keeps root:mediswarm-upload 640):"
      echo "#   sudo scripts/admin/register_upload_keys.sh $UPLOAD_FRAGMENT"
      cat "$UPLOAD_FRAGMENT_TMP"
    } > "$UPLOAD_FRAGMENT"
    chmod 644 "$UPLOAD_FRAGMENT"
    echo "Upload key lines for authorized_keys: $UPLOAD_FRAGMENT (public keys only; not part of any kit)"
    echo "Register them on the upload host before the kits go out:"
    echo "  sudo scripts/admin/register_upload_keys.sh $UPLOAD_FRAGMENT"
  else
    echo "No client kit in $TARGET_FOLDER: no upload keys made."
  fi
fi

echo "Live sync injection finished"
