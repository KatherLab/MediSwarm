Each folder in this challenge folder contains code related to the according challenge team.

The following steps need to be performed to make their code swarm ready and compatible with MediSwarm:
1. Include your code in the according directory under ./application/jobs/ODELIA_Ternary_classification/app/custom/models/challenge/<your_teams_folder>
2. Your model needs to inherit from BasicClassifier in ./application/jobs/ODELIA_Ternary_classification/app/custom/models/base_model.py (Recommended). Alternatively, you can use the ModelWrapper in base_model.py and make your model inherit from torch.nn.module.


In order to test your training run the following:

export SITE_NAME=<site>                                 # Chose the according name from [CAM, MHA, RSH, RUMC, UKA, UMCU, USZ, VHIO]
export DATADIR=<sites_data_folder>                      # the data directory to be used
export TRAINING_MODE=local_training                     # one out of local_training, preflight_check, swarm
export MODEL_VARIANT=<model_team>                       # can be either the model name: mst, resnet or challenge_<team_suffix>                                 # like challenge_3agaldran with suffixes out of
                                                        # [1DivideAndConquer, 2BCN_AIM, 3agaldran, 4LME_ABMIL, 5Pimed]
export SCRATCHDIR=<scratch_dir> 

### Creating startup kits: if there is an issue with the versions:
https://github.com/deboraJ1/MediSwarm/blob/include_abmil/scripts/dev_utils/README_dockerfile_update.md


Example:
export SITE_NAME=CAM                                 
export DATADIR=/mnt/d/Projects/ODELIA/Challenge/CAM/    
export TRAINING_MODE=local_training                     
export MODEL_VARIANT=challenge_3agaldran
export SCRATCH_DIR=/mnt/d/Projects/ODELIA/Challenge/test_scratch  # hpcwork/it336446/test_scratch


cd <startupkit_directory/startup>                       #  workspace/odelia_1.0.2-dev.260116.a701bc2_dummy_project_for_testing/prod_01/localhost/startup/ /hpcwork/it336446/startup_UKA/startup
chmod +x docker.sh
sudo ./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training 2>&1 | tee dummy_training_console_output.txt
sudo ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --preflight_check 2>&1 | tee preflight_check_console_output.txt
sudo ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --local_training 2>&1 | tee local_training_console_output.txt

chmod +x start.sh
chmod +x sub_start.sh
sudo ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
sudo chmod a+r nohup.out

// check training logs:
tail -f nohup.out
nvidia-smi

//check connection to server
ping dl3.tud.de

