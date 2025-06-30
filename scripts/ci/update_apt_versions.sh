#!/usr/bin/env bash

set -e

echo "[INFO] Removing APT version pins from Dockerfile..."
scripts/dev_utils/dockerfile_update_removeVersionApt.py docker_config/Dockerfile_ODELIA

echo "[INFO] Committing temporarily without version constraints..."
git config user.email "ci@github.com"
git config user.name "GitHub CI"
git commit docker_config/Dockerfile_ODELIA -m "WIP: remove apt versions for rebuild" || echo "[INFO] No version pin removal change to commit."

echo "[INFO] Rebuilding Docker image and capturing logs..."
./buildDockerImageAndStartupKits.sh -p tests/provision/dummy_project_for_testing.yml 2>&1 | tee out.txt

echo "[INFO] Re-adding updated APT version pins to Dockerfile..."
scripts/dev_utils/dockerfile_update_addAptVersionNumbers.py docker_config/Dockerfile_ODELIA out.txt
rm out.txt

# Check if there are changes to commit
if git diff --quiet; then
  echo "[INFO] No changes to apt versions found. Skipping commit."
else
  echo "[INFO] Committing updated apt versions..."
  # Allow empty amend to avoid error
  git commit docker_config/Dockerfile_ODELIA --amend -m "chore: update apt versions based on rebuild" || echo "[INFO] Nothing to amend. Skipping."
fi
