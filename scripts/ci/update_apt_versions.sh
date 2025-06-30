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
    echo "[ERROR] Docker build failed. Aborting update."
    exit 1
fi

echo "[INFO] Re-adding updated APT version pins to Dockerfile..."
scripts/dev_utils/dockerfile_update_addAptVersionNumbers.py "$DOCKERFILE_PATH" "$LOG_PATH"
rm "$LOG_PATH"

# Optional: validate if the versions exist in apt repository
echo "[INFO] Validating all pinned versions..."
while IFS= read -r line; do
    if [[ $line =~ ([a-z0-9\-]+)=([a-zA-Z0-9:~.+-]+) ]]; then
        pkg="${BASH_REMATCH[1]}"
        ver="${BASH_REMATCH[2]}"
        echo -n "Checking $pkg=$ver... "
        if ! apt-cache madison "$pkg" | grep -q "$ver"; then
            echo "[MISSING]"
            sed -i "s|$pkg=$ver|$pkg|" "$DOCKERFILE_PATH"
        else
            echo "OK"
        fi
    fi
done < <(grep -oP '\b[a-z0-9\.\-]+=[a-zA-Z0-9:~.+-]+\b' "$DOCKERFILE_PATH")

# Check if there are changes to commit
if git diff --quiet; then
  echo "[INFO] No changes to apt versions found. Skipping commit."
else
  echo "[INFO] Committing updated apt versions..."
  git commit "$DOCKERFILE_PATH" --amend -m "chore: update apt versions based on rebuild" || echo "[INFO] Nothing to amend. Skipping."
fi
