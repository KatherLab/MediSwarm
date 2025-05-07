#!/usr/bin/env bash

# This script should be called only at the beginning of a swarm training.

DISTRIBUTED_CODE_FOLDER=`pwd`
EXPECTED_CODE_FOLDER=/MediSwarm/application/jobs/3dcnn_ptl/app

cd $EXPECTED_CODE_FOLDER
find . \( -name "*.py" -o -name "*.conf" \) -exec shasum {} \; | sort > /tmp/expected_sha_sums.txt

cd $DISTRIBUTED_CODE_FOLDER
COMPARISON_RESULT=$(shasum -c /tmp/expected_sha_sums.txt)

if [ $? = 0 ]; then
  echo "MediSwarm code verification succeeded: " $COMPARISON_RESULT
else
  echo "MediSwarm code verification FAILED: " $COMPARISON_RESULT
  exit 1
fi
