#!/usr/bin/env bash

set -e

echo "[INFO] Removing APT version pins from Dockerfile..."
scripts/dev_utils/dockerfile_update_removeVersionApt.py docker_config/Dockerfile_ODELIA

echo "[INFO] Rebuilding Docker image and capturing logs..."
./buildDockerImageAndStartupKits.sh -p tests/provision/dummy_project_for_testing.yml 2>&1 | tee out.txt

echo "[INFO] Re-adding updated APT version pins to Dockerfile..."
scripts/dev_utils/dockerfile_update_addAptVersionNumbers.py docker_config/Dockerfile_ODELIA out.txt
rm out.txt

echo "[INFO] Update process completed. Check Dockerfile for changes."
