#!/usr/bin/env bash

# run NVFlare's unit tests (takes about 2 minutes), does not include NVFlare's integration tests yet
# cd nvflare
# ./runtest.sh
# cd ..

# run unit tests of ODELIA swarm learning and report coverage
export MPLCONFIGDIR=/tmp
cd controller/controller
python3 -m coverage run --source=. -m unittest discover
coverage report -m
rm .coverage

# run simulation mode for (not yet) minimal example
cd /workspace
nvflare simulator -w /tmp/minimal_training -n 2 -t 2 application/test_jobs/minimal_training -c simulated_node_0,simulated_node_1
