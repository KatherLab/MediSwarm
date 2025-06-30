#!/usr/bin/env bash

set -e

DOCKERFILE_PATH="docker_config/Dockerfile_ODELIA"
LOG_PATH="out.txt"
PROJECT_YML="tests/provision/dummy_project_for_testing.yml"

echo "[INFO] Removing APT version pins from Dockerfile..."
scripts/dev_utils/dockerfile_update_removeVersionApt.py "$DOCKERFILE_PATH"

echo "[INFO] Committing temporarily without version constraints..."
git config user.email "ci@github.com"
git config user.name "GitHub CI"
git commit "$DOCKERFILE_PATH" -m "WIP: remove apt versions for rebuild" || echo "[INFO] No version pin removal change to commit."

echo "[INFO] Rebuilding Docker image and capturing logs..."
if ! ./buildDockerImageAndStartupKits.sh -p "$PROJECT_YML" 2>&1 | tee "$LOG_PATH"; then
    echo "[ERROR] Docker build failed. Proceeding to clean invalid versions..."
fi

echo "[INFO] Re-adding updated APT version pins to Dockerfile..."
scripts/dev_utils/dockerfile_update_addAptVersionNumbers.py "$DOCKERFILE_PATH" "$LOG_PATH"

echo "[INFO] Validating all pinned versions, removing invalid ones..."
has_invalid_versions=0
while IFS= read -r match; do
    pkg="$(echo "$match" | cut -d= -f1)"
    ver="$(echo "$match" | cut -d= -f2)"
    echo -n "Checking $pkg=$ver... "
    if ! apt-cache madison "$pkg" | grep -q "$ver"; then
        echo "NOT FOUND – removing pin"
        sed -i "s|\b$pkg=$ver\b|$pkg|" "$DOCKERFILE_PATH"
        has_invalid_versions=1
    else
        echo "OK"
    fi
done < <(grep -oP '\b[a-z0-9\.\-]+=[a-zA-Z0-9:~.+-]+\b' "$DOCKERFILE_PATH")

rm "$LOG_PATH"

if git diff --quiet; then
  if [[ "$has_invalid_versions" -eq 1 ]]; then
    echo "[INFO] Pinned versions were removed. Committing fallback without version locks..."
    git commit "$DOCKERFILE_PATH" --amend -m "fix: remove obsolete apt version pins"
  else
    echo "[INFO] No changes to apt versions found. Skipping commit."
  fi
else
  echo "[INFO] Committing updated apt versions..."
  git commit "$DOCKERFILE_PATH" --amend -m "chore: update apt versions based on rebuild"
fi
