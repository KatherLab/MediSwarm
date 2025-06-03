#!/usr/bin/env bash

set -e

echo "[INFO] Removing apt version pins from Dockerfile..."
scripts/dev_utils/dockerfile_update_removeVersionApt.py docker_config/Dockerfile_ODELIA

echo "[INFO] Committing temporarily without version constraints..."
git config user.email "ci@github.com"
git config user.name "GitHub CI"
git commit docker_config/Dockerfile_ODELIA -m "WIP: remove apt versions for rebuild"

echo "[INFO] Rebuilding Docker image and capturing logs..."
./buildDockerImageAndStartupKits.sh -p tests/provision/dummy_project_for_testing.yml 2>&1 | tee out.txt

echo "[INFO] Re-adding updated apt version pins..."
scripts/dev_utils/dockerfile_update_addAptVersionNumbers.py docker_config/Dockerfile_ODELIA out.txt
rm out.txt

if git diff --quiet; then
  echo "[INFO] No changes to apt versions found. Cleaning up..."
  git reset --hard HEAD~1
else
  git commit docker_config/Dockerfile_ODELIA --amend -m "chore: update apt versions based on rebuild"
  echo "[INFO] Updated apt versions committed."
fi
