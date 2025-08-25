#!/usr/bin/env bash
set -e

DOCKERFILE_PATH="docker_config/Dockerfile_ODELIA"
LOG_PATH=$(mktemp)
PROJECT_YML="tests/provision/dummy_project_for_testing.yml"

echo "[INFO] Removing APT version pins from Dockerfile..."
scripts/dev_utils/dockerfile_update_removeVersionApt.py "$DOCKERFILE_PATH"

echo "[INFO] Committing temporarily without version constraints..."
git config user.email "ci@github.com"
git config user.name "GitHub CI"
git commit "$DOCKERFILE_PATH" -m "WIP: remove apt versions for rebuild" || echo "[INFO] No version pin removal change to commit."

echo "[INFO] Rebuilding Docker image and capturing logs..."
if ! ./buildDockerImageAndStartupKits.sh -p "$PROJECT_YML" > "$LOG_PATH" 2>&1; then
  echo "Build failed. Output:"
  cat "$LOG_PATH"
  exit 1
fi

echo "[DEBUG] First 20 lines of build log:"
head -n 20 "$LOG_PATH"

echo "[DEBUG] Checking for apt install commands:"
grep "apt install" "$LOG_PATH" || echo "[WARN] No apt install command found in log!"

echo "[INFO] Re-adding updated APT version pins to Dockerfile..."
scripts/dev_utils/dockerfile_update_addAptVersionNumbers.py "$DOCKERFILE_PATH" "$LOG_PATH"
rm "$LOG_PATH"

git fetch origin main
if git diff --quiet origin/main..HEAD; then
  echo "NO_CHANGES=true" >> "$GITHUB_ENV"
else
  echo "NO_CHANGES=false" >> "$GITHUB_ENV"
fi
