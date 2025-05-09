#!/usr/bin/env bash

# This script should be called only at the beginning of a swarm training.

if [ ! -f custom/MediSwarmAPCFolderName.txt ]; then
  echo "MediSwarm APC folder name not found"
  exit 1
fi
APC_FOLDER_NAME_=`cat custom/MediSwarmAPCFolderName.txt`
APC_FOLDER_NAME=`basename $APC_FOLDER_NAME_`

/MediSwarm/_validateMediSwarmAPCFolderName.py $APC_FOLDER_NAME
if [ $? -ne 0 ]; then
  echo "Invalid application folder name" $APC_FOLDER_NAME
  exit 1
fi

DISTRIBUTED_CODE_FOLDER=`pwd`
EXPECTED_CODE_FOLDER=/MediSwarm/application/jobs/$APC_FOLDER_NAME/app

cd $EXPECTED_CODE_FOLDER
find . \( -name "*.py" -o -name "*.conf" \) -exec shasum {} \; | sort > /tmp/expected_sha_sums.txt

cd $DISTRIBUTED_CODE_FOLDER
COMPARISON_RESULT=$(shasum -c /tmp/expected_sha_sums.txt)

if [ $? -eq 0 ]; then
  echo "MediSwarm code verification succeeded: " $COMPARISON_RESULT
else
  echo "MediSwarm code verification FAILED: " $COMPARISON_RESULT
  exit 1
fi
