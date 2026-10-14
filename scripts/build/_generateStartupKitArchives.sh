#!/usr/bin/env bash
#
# Package each participant's startup kit of the newest prod_NN folder.
#
# A kit contains that site's PRIVATE KEY (startup/client.key). The ODELIA
# consortium is a closed, mutually-trusting group and hands out kits through a
# members-only shared folder: every site can see the folder and simply picks its
# own kit. Kits are therefore packaged as plain zips.
#
# This deliberately reverses the per-site AES encryption introduced in #449 -- a
# consortium decision (2026-07-16) that per-site passwords were unnecessary
# friction for a trusted group sharing one folder. The private key is still real,
# so the shared folder must stay members-only, and a folder leak means rotating
# certs (re-issuing kits).
#
# Outputs, alongside the kits:
#   <KIT>/kit_manifest.txt     inside every kit: site, kind, kit version, expected image,
#                              root CA, and the sha256 of every file as issued
#   <SITE>_<VERSION>.zip       the kit                                   -> members-only shared folder
#   SHA256SUMS.txt             sha256sum lines of the SITE (client) kits  -> upload next to the zips
#   kit_manifest.csv           one row per kit, server and admin included -> feeds the kit registry
#
# A site checks and installs with:
#   sha256sum -c --ignore-missing SHA256SUMS.txt
#   unzip <SITE>_<VERSION>.zip -d <new empty folder>
#
# Root CA check: all kits of one build must carry the same rootCA.pem, and it must be the
# CA of the previous build of the same project. A different CA means the state folder that
# holds it was lost or replaced; such kits cannot connect to a server provisioned earlier
# (F9 in docs/SWARM_FAILURE_MODES.md), so nothing is packaged unless
# MEDISWARM_ALLOW_NEW_ROOT_CA=1 says the new CA is intended.

set -euo pipefail

PROJECT_YML=$1
LONG_VERSION=$2
PROJECT_NAME="$(grep '^name: ' "$PROJECT_YML" | sed 's/name: //')"
OUTPUT_FOLDER="workspace/$PROJECT_NAME"
TARGET_FOLDER="$(ls -d "$OUTPUT_FOLDER"/prod_* | tail -n 1)"

MANIFEST_FILE="kit_manifest.csv"
SUMS_FILE="SHA256SUMS.txt"
KIT_MANIFEST="kit_manifest.txt"

# The image the project provisions into docker.sh, used when a kit has no docker.sh line.
PROJECT_IMAGE="$(grep -m1 -E '^[[:space:]]*docker_image:' "$PROJECT_YML" \
  | sed -E 's/^[[:space:]]*docker_image:[[:space:]]*//; s/[[:space:]]*$//' || true)"
PROJECT_IMAGE="${PROJECT_IMAGE//__REPLACED_BY_CURRENT_VERSION_NUMBER_WHEN_BUILDING_STARTUP_KITS__/$LONG_VERSION}"

# sha256 of every distinct rootCA.pem in the kits of one prod_NN folder, one per line
_root_cas() {
  find "$1" -mindepth 3 -maxdepth 3 -path '*/startup/rootCA.pem' -type f -exec sha256sum {} + 2>/dev/null \
    | awk '{print $1}' | sort -u
}

_kit_kind() {
  if [ -f "$1/startup/fed_server.json" ]; then
    echo server
  elif [ -f "$1/startup/fl_admin.sh" ] || [ -f "$1/startup/fed_admin.json" ]; then
    echo admin
  elif [ -f "$1/startup/fed_client.json" ]; then
    echo client
  else
    echo unknown
  fi
}

ROOT_CA="$(_root_cas "$TARGET_FOLDER")"
if [ "$(printf '%s\n' "$ROOT_CA" | grep -c .)" -gt 1 ]; then
  echo "ERROR: the kits in $TARGET_FOLDER carry different root CAs:" >&2
  printf '  %s\n' $ROOT_CA >&2
  exit 1
fi

# The reference is the newest earlier build whose kits were packaged (it holds zips). A build
# refused below keeps its kits unpackaged, and NVFlare has meanwhile stored the new CA in
# state/, so comparing with that build would let a plain rerun pass.
PREVIOUS_FOLDER=""
PREVIOUS_CA=""
for folder in $(ls -d "$OUTPUT_FOLDER"/prod_* | sort -r); do
  [ "$folder" = "$TARGET_FOLDER" ] && continue
  ls "$folder"/*.zip > /dev/null 2>&1 || continue
  PREVIOUS_CA="$(_root_cas "$folder")"
  if [ -n "$PREVIOUS_CA" ]; then
    PREVIOUS_FOLDER="$folder"
    break
  fi
done
if [ -n "$PREVIOUS_CA" ] && [ -n "$ROOT_CA" ] && ! printf '%s\n' "$PREVIOUS_CA" | grep -qx "$ROOT_CA"; then
  if [ "${MEDISWARM_ALLOW_NEW_ROOT_CA:-0}" = "1" ]; then
    echo "WARNING: $TARGET_FOLDER has a NEW root CA ($ROOT_CA); $PREVIOUS_FOLDER used another one."
    echo "         MEDISWARM_ALLOW_NEW_ROOT_CA=1: every site and the server need these new kits."
  else
    cat >&2 <<EOF
ERROR: the kits in $TARGET_FOLDER were signed by a different root CA than those in
$PREVIOUS_FOLDER ($ROOT_CA instead of $PREVIOUS_CA).
They cannot connect to a server provisioned before (F9 in docs/SWARM_FAILURE_MODES.md).
This happens when $OUTPUT_FOLDER/state/cert.json was deleted or replaced. Nothing was
packaged; do not hand out $TARGET_FOLDER. To keep the old CA, restore state/cert.json
from a backup and rebuild. If a new CA is intended, rerun with MEDISWARM_ALLOW_NEW_ROOT_CA=1.
EOF
    exit 1
  fi
fi

# Written into the kit folder before zipping, so the unpacked kit carries it. The checksum
# lines are plain sha256sum output; the '#' header keeps them usable with sha256sum -c.
_write_kit_manifest() {
  local kit="$1" kind="$2" image channel template
  # A kit without docker.sh (no docker_image in the project) or without image.conf is valid.
  image="$(sed -n 's/^DOCKER_IMAGE=//p' "$kit/startup/docker.sh" 2>/dev/null | head -n 1 || true)"
  channel="$(sed -n 's/^MEDISWARM_IMAGE=//p' "$kit/startup/image.conf" 2>/dev/null | tail -n 1 || true)"
  template="$(grep -o 'MEDISWARM_VERSION=[^ ]*' "$kit/startup/docker.sh" 2>/dev/null | head -n 1 | cut -d= -f2 || true)"
  if [ "$kind" = client ] && [ -n "$template" ] && [ "$template" != "$LONG_VERSION" ]; then
    # A kit-only build inside an image of another version: the kit is labelled
    # $LONG_VERSION but its docker.sh is that image's template.
    echo "WARNING: $kit: docker.sh comes from the $template template, the kit is labelled $LONG_VERSION"
  fi
  {
    echo "# MediSwarm startup kit manifest"
    echo "# site: $kit"
    echo "# kind: $kind"
    echo "# kit_version: $LONG_VERSION"
    echo "# image: ${image:-$PROJECT_IMAGE}"
    echo "# image_channel: ${channel:-none}"
    if [ -n "$template" ]; then
      echo "# template_version: $template"
    fi
    echo "# project: $PROJECT_NAME"
    echo "# root_ca_sha256: ${ROOT_CA:-none}"
    echo "# created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# Every file of this kit as issued. To check them, run in this folder:"
    echo "#   grep -v '^#' $KIT_MANIFEST | sha256sum -c --quiet"
    (cd "$kit" && find . -type f ! -name "$KIT_MANIFEST" -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum \
      | sed 's#  \./#  #')
  } > "$kit/$KIT_MANIFEST"
}

cd "$TARGET_FOLDER"

# the archives carry a private key; keep the local build artifacts owner-only
umask 077

echo "site,version,file,sha256,kind,image,root_ca_sha256" > "$MANIFEST_FILE"
: > "$SUMS_FILE"

for startupkit in *; do
    # skip whatever this script itself produces
    case "$startupkit" in
        "$MANIFEST_FILE"|"$SUMS_FILE"|*.zip) continue ;;
    esac
    [ -d "$startupkit" ] || continue

    kind="$(_kit_kind "$startupkit")"
    _write_kit_manifest "$startupkit" "$kind"
    image="$(sed -n 's/^# image: //p' "$startupkit/$KIT_MANIFEST")"

    archive="${startupkit}_${LONG_VERSION}.zip"

    # zip -r appends to an existing archive; start clean so a rebuild is reproducible
    rm -f "$archive"
    zip -rq "$archive" "$startupkit"
    sha="$(sha256sum "$archive" | cut -d' ' -f1)"

    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$startupkit" "$LONG_VERSION" "$archive" "$sha" "$kind" "$image" "${ROOT_CA:-none}" >> "$MANIFEST_FILE"
    # Only site kits go to the shared folder; server and admin kits stay at TUD.
    if [ "$kind" = client ]; then
        printf '%s  %s\n' "$sha" "$archive" >> "$SUMS_FILE"
    fi

    echo "Generated startup kit $TARGET_FOLDER/$archive"
done

echo "Manifest:  $TARGET_FOLDER/$MANIFEST_FILE  (sha256 per kit -- publish these in the kit registry)"
echo "Checksums: $TARGET_FOLDER/$SUMS_FILE  (site kits; upload next to the zips)"
echo "Root CA:   ${ROOT_CA:-none}"
