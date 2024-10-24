#!/usr/bin/env bash

# run NVFlare's unit tests (takes about 2 minutes), does not include NVFlare's integration tests yet
# cd nvflare
# ./runtest.sh -c -r
# coverage report -m
# cd ..

# run unit tests of ODELIA swarm learning and report coverage
export MPLCONFIGDIR=/tmp
cd tests/unit_tests/controller
python3 -m coverage run --source=/workspace/controller/controller -m unittest discover
coverage report -m
rm .coverage

# run simulation mode for minimal example
cd /workspace
nvflare simulator -w /tmp/minimal_training_test -n 2 -t 2 application/jobs/minimal_training_test -c simulated_node_0,simulated_node_1

# run proof-of-concept mode for minimal example
cd /workspace
nvflare poc prepare -c poc_client_0 poc_client_1
nvflare poc prepare-jobs-dir -j application/jobs/
nvflare poc start -ex admin@nvidia.com
sleep 15
echo "Will submit job now after sleeping 15 seconds to allow the background process to complete"
nvflare job submit -j application/jobs/minimal_training_test
sleep 60
echo "Will shut down now after sleeping 60 seconds to allow the background process to complete"
sleep 2
nvflare poc stop
