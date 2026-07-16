#!/usr/bin/env bash
#
# Package each site's startup kit.
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
#   <SITE>_<VERSION>.zip       the kit                    -> members-only shared folder
#   kit_manifest.csv           site,version,file,sha256   -> feeds the kit registry
#
# A site installs with:
#   unzip <SITE>_<VERSION>.zip

set -euo pipefail

OUTPUT_FOLDER="workspace/$(grep '^name: ' "$1" | sed 's/name: //')"
TARGET_FOLDER="$(ls -d "$OUTPUT_FOLDER"/prod_* | tail -n 1)"
LONG_VERSION=$2

MANIFEST_FILE="kit_manifest.csv"

cd "$TARGET_FOLDER"

# the archives carry a private key; keep the local build artifacts owner-only
umask 077

echo "site,version,file,sha256" > "$MANIFEST_FILE"

for startupkit in *; do
    # skip whatever this script itself produces
    case "$startupkit" in
        "$MANIFEST_FILE"|*.zip) continue ;;
    esac
    [ -d "$startupkit" ] || continue

    archive="${startupkit}_${LONG_VERSION}.zip"

    # zip -r appends to an existing archive; start clean so a rebuild is reproducible
    rm -f "$archive"
    zip -rq "$archive" "$startupkit"

    printf '%s,%s,%s,%s\n' \
        "$startupkit" "$LONG_VERSION" "$archive" \
        "$(sha256sum "$archive" | cut -d' ' -f1)" >> "$MANIFEST_FILE"

    echo "Generated startup kit $TARGET_FOLDER/$archive"
done

echo "Manifest:  $TARGET_FOLDER/$MANIFEST_FILE  (sha256 per kit -- publish these in the kit registry)"
