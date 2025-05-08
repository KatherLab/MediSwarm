#!/usr/bin/env bash

set -e

# run unit tests of ODELIA swarm learning and report coverage
export MPLCONFIGDIR=/tmp
cd /MediSwarm/tests/unit_tests/controller
PYTHONPATH=/MediSwarm/controller/controller python3 -m coverage run --source=/MediSwarm/controller/controller -m unittest discover
coverage report -m
rm .coverage

# uncomment to run NVFlare's unit tests (takes about 2 minutes and will install python packages in the container)
# cd /MediSwarm/docker_config/NVFlare
# ./runtest.sh -c -r
# coverage report -m
# cd ..

# run standalone version of minimal example
cd /MediSwarm/application/jobs/minimal_training_pytorch_cnn/app/custom/
export TRAINING_MODE="local_training"
./main.py

# run simulation mode for minimal example
cd /MediSwarm
export TRAINING_MODE="swarm"
nvflare simulator -w /tmp/minimal_training_pytorch_cnn -n 2 -t 2 application/jobs/minimal_training_pytorch_cnn -c simulated_node_0,simulated_node_1 | tee /scratch/minimal_training_pytorch_cnn_sim.log

if [[ ! -z $(grep "MediSwarm code verification succeeded" /scratch/minimal_training_pytorch_cnn_sim.log) ]] && \
   [[ ! -z $(grep "Round 4 started" /scratch/minimal_training_pytorch_cnn_sim.log) ]]; then
    echo "✅ Simulation mode succeeded."
else
    echo "❌ Simulation mode failed."
    exit 1
fi
rm /scratch/minimal_training_pytorch_cnn_sim.log

# run proof-of-concept mode for minimal example
cd /MediSwarm
export TRAINING_MODE="swarm"
nvflare poc prepare -c poc_client_0 poc_client_1
nvflare poc prepare-jobs-dir -j application/jobs/
nvflare poc start -ex admin@nvidia.com | tee /scratch/minimal_training_pytorch_cnn_poc.log &
sleep 15
echo "Will submit job now after sleeping 15 seconds to allow the background process to complete"
nvflare job submit -j application/jobs/minimal_training_pytorch_cnn
sleep 60
echo "Will shut down now after sleeping 60 seconds to allow the background process to complete"
sleep 2
nvflare poc stop

if [[ ! -z $(grep "MediSwarm code verification succeeded" /scratch/minimal_training_pytorch_cnn_poc.log) ]] && \
   [[ ! -z $(grep "Round 4 started" /scratch/minimal_training_pytorch_cnn_poc.log) ]]; then
    echo "✅ Proof-of-concept mode succeeded."
else
    echo "❌ Proof-of-concept mode failed."
    exit 1
fi
rm /scratch/minimal_training_pytorch_cnn_poc.log

# check, via simulation mode, that differences between intended and distributed code are detected
# note that we cannot directly modify only the code to be distributed compared to the code in the image here
cd /MediSwarm
export TRAINING_MODE="swarm"
cp -r application/jobs/minimal_training_pytorch_cnn application/jobs/minimal_training_pytorch_cnn_modified
echo "# modification" >> application/jobs/minimal_training_pytorch_cnn_modified/app/custom/main.py
# but leave the pointer to the original code as is so that we have a discrepancy

nvflare simulator -w /tmp/minimal_training_pytorch_cnn_modified -n 2 -t 2 application/jobs/minimal_training_pytorch_cnn_modified -c simulated_node_0,simulated_node_1 | tee /scratch/minimal_training_pytorch_cnn_modified_sim.log

if [[ ! -z $(grep "MediSwarm code verification FAILED" /scratch/minimal_training_pytorch_cnn_modified_sim.log) ]] && \
   [[ -z $(grep "Round 1 started" /scratch/minimal_training_pytorch_cnn_modified_sim.log) ]]; then
    echo "✅ Code modification detected successfully. (Execution exceptions in the output above are expected)"
else
    echo "❌ Code modification not detected."
    exit 1
fi
rm /scratch/minimal_training_pytorch_cnn_modified_sim.log

nvflare poc start -ex admin@nvidia.com | tee /scratch/minimal_training_pytorch_cnn_modified_poc.log &
sleep 15
echo "Will submit job now after sleeping 15 seconds to allow the background process to complete"
nvflare job submit -j application/jobs/minimal_training_pytorch_cnn_modified
sleep 30
echo "Will shut down now after sleeping 30 seconds to allow the background process to complete"
sleep 2
nvflare poc stop

if [[ ! -z $(grep "MediSwarm code verification FAILED" /scratch/minimal_training_pytorch_cnn_modified_poc.log) ]] && \
   [[ -z $(grep "Round 1 started" /scratch/minimal_training_pytorch_cnn_modified_poc.log) ]]; then
    echo "✅ Code modification detected successfully. (Execution exceptions in the output above are expected)"
else
    echo "❌ Code modification not detected."
    exit 1
fi
rm /scratch/minimal_training_pytorch_cnn_modified_poc.log

rm -rf application/jobs/minimal_training_pytorch_cnn_modified
