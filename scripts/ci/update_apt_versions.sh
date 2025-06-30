#!/usr/bin/env bash
set -e

DOCKERFILE_PATH="docker_config/Dockerfile_ODELIA"
APT_LOG="out.txt"

echo "[INFO] Removing APT version pins from Dockerfile..."
scripts/dev_utils/dockerfile_update_removeVersionApt.py "$DOCKERFILE_PATH"

echo "[INFO] Committing temporarily without version constraints..."
git config user.email "ci@github.com"
git config user.name "GitHub CI"
git commit "$DOCKERFILE_PATH" -m "WIP: remove apt versions for rebuild" || echo "[INFO] No version pin removal change to commit."

echo "[INFO] Rebuilding Docker image and capturing logs..."
./buildDockerImageAndStartupKits.sh -p tests/provision/dummy_project_for_testing.yml 2>&1 | tee "$APT_LOG"

echo "[INFO] Re-adding updated APT version pins to Dockerfile..."
scripts/dev_utils/dockerfile_update_addAptVersionNumbers.py "$DOCKERFILE_PATH" "$APT_LOG"
rm "$APT_LOG"

# === Validate pinned versions, remove invalid ===
echo "[INFO] Validating all pinned versions, removing invalid ones..."
grep -oP '[a-z0-9.\-]+=[a-zA-Z0-9:~.+-]+' "$DOCKERFILE_PATH" | while read -r pkgver; do
    pkg="${pkgver%%=*}"
    ver="${pkgver#*=}"
    echo -n "Checking $pkg=$ver... "
    if ! apt-cache madison "$pkg" | grep -q "$ver"; then
        echo "NOT FOUND – removing pin"
        sed -i "s|$pkg=$ver|$pkg|g" "$DOCKERFILE_PATH"
    else
        echo "OK"
    fi
done

# === Final diff check ===
echo "[DEBUG] Final git diff:"
git diff || true

if git diff --quiet; then
  echo "[INFO] No changes to apt versions found. Skipping commit."
else
  echo "[INFO] Committing updated apt versions..."
  git commit "$DOCKERFILE_PATH" --amend -m "chore: update apt versions based on rebuild" || echo "[INFO] Nothing to amend. Skipping."
fi
