#!/usr/bin/env bash
#
# Package each site's startup kit.
#
# A kit contains that site's PRIVATE KEY (startup/client.key). Kits used to be
# packaged as plain, unencrypted zips, so they could not safely be put anywhere
# shared -- any site could have read any other site's credentials.
#
# Each kit is therefore encrypted with AES-256 (PBKDF2, 600k iterations, salted)
# under its own random password. The whole archive is encrypted, so the file names
# inside leak nothing either.
#
# openssl is used rather than a password-protected zip on purpose: standard zip
# encryption is ZipCrypto, which is broken by a known-plaintext attack and would not
# actually protect a private key sitting on a shared link. openssl is already present
# in the image and on every partner host, so this needs no new dependency.
#
# Outputs, alongside the kits:
#   <SITE>_<VERSION>.zip.enc   encrypted kit             -> safe to put on a shared link
#   kit_passwords.txt          one password per site     -> NEVER share; deliver per site
#   kit_manifest.csv           site,version,file,sha256  -> feeds the kit registry
#
# A site decrypts with:
#   openssl enc -d -aes-256-cbc -pbkdf2 -iter 600000 \
#           -in <SITE>_<VERSION>.zip.enc -out <SITE>.zip -pass pass:'<password>'
#   unzip <SITE>.zip

set -euo pipefail

OUTPUT_FOLDER="workspace/$(grep '^name: ' "$1" | sed 's/name: //')"
TARGET_FOLDER="$(ls -d "$OUTPUT_FOLDER"/prod_* | tail -n 1)"
LONG_VERSION=$2

PBKDF2_ITER=600000
PASSWORD_FILE="kit_passwords.txt"
MANIFEST_FILE="kit_manifest.csv"

cd "$TARGET_FOLDER"

umask 077
: > "$PASSWORD_FILE"
{
  echo "# MediSwarm startup-kit passwords -- version $LONG_VERSION"
  echo "# Deliver each password to ITS OWN SITE ONLY, out of band. Never share this file."
} >> "$PASSWORD_FILE"

echo "site,version,file,sha256" > "$MANIFEST_FILE"

for startupkit in *; do
    # skip whatever this script itself produces
    case "$startupkit" in
        "$PASSWORD_FILE"|"$MANIFEST_FILE"|*.zip|*.zip.enc) continue ;;
    esac
    [ -d "$startupkit" ] || continue

    archive="${startupkit}_${LONG_VERSION}.zip"
    encrypted="${archive}.enc"

    zip -rq "$archive" "$startupkit"

    password="$(openssl rand -base64 24)"
    openssl enc -aes-256-cbc -pbkdf2 -iter "$PBKDF2_ITER" -salt \
        -in "$archive" -out "$encrypted" -pass pass:"$password"

    # the plaintext archive contains the private key -- it must not survive
    rm -f "$archive"

    printf '%s\t%s\n' "$startupkit" "$password" >> "$PASSWORD_FILE"
    printf '%s,%s,%s,%s\n' \
        "$startupkit" "$LONG_VERSION" "$encrypted" \
        "$(sha256sum "$encrypted" | cut -d' ' -f1)" >> "$MANIFEST_FILE"

    echo "Generated encrypted startup kit $TARGET_FOLDER/$encrypted"
done

echo "Passwords: $TARGET_FOLDER/$PASSWORD_FILE  (deliver per site, out of band -- never share the file)"
echo "Manifest:  $TARGET_FOLDER/$MANIFEST_FILE  (sha256 per kit -- publish these in the kit registry)"
