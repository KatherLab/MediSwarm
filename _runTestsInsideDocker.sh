#!/usr/bin/env bash

# run NVFlare's unit tests (takes about 2 minutes), does not include NVFlare's integration tests yet
# cd nvflare
# ./runtest.sh
# cd ..

export MPLCONFIGDIR=/tmp

cd tests/unit_tests/controller

python3 -m coverage run --source=/workspace/controller/controller -m unittest discover
coverage report -m
rm .coverage
