#! /bin/bash

VERSION=$(../../getVersionNumber.sh)
CONTAINER_VERSION_SUFFIX=$(git rev-parse --short HEAD)
CWD=$(pwd)

run_dir=odelia_1.0.2${VERSION}_MEVIS_test
base_dir=/home/steffen/projects/ODELIA/MediSwarm/workspace/${run_dir}/prod_00/

# submit_job MediSwarm/application/jobs/minimal_training_pytorch_cnn

cd $base_dir/localhost/startup
./docker.sh --no_pull --start_server

cd $base_dir/local1/startup
./docker.sh --no_pull --data_dir . --scratch_dir . --GPU 0 --start_client

cd $base_dir/local2/startup
./docker.sh --no_pull --data_dir . --scratch_dir . --GPU 0 --start_client

cd $base_dir/admin@mevis.odelia/startup
# ./docker.sh --no_pull << "admin@mevis.odelia\nsubmit_job MediSwarm/application/jobs/minimal_training_pytorch_cnn\nlist_jobs\n"
# expect -c "spawn ./docker.sh --no_pull; send \"admin@mevis.odelia; send submit_job MediSwarm/application/jobs/minimal_training_pytorch_cnn\"; interact"
expect -f ${CWD}/_submitDummyTraining.exp
docker kill odelia_swarm_admin_$CONTAINER_VERSION_SUFFIX
sleep 120
cd "$CWD"
