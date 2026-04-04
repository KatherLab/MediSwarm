Each folder in this challenge folder contains code related to the according challenge team.

The following steps need to be performed to make their code swarm ready and compatible with MediSwarm:
1. Include your code in the according directory under ./application/jobs/ODELIA_Ternary_classification/app/custom/models/challenge/<your_teams_folder>
2. Your model needs to inherit from BasicClassifier in ./application/jobs/ODELIA_Ternary_classification/app/custom/models/base_model.py (Recommended). Alternatively, you can use the ModelWrapper in base_model.py and make your model inherit from torch.nn.module.


In order to test your training run the following:

Note: When using `sudo`, environment variables (export ...) are NOT forwarded.
Use the CLI flags (--model_name, --num_epochs, --config) instead to pass values through sudo.

Set up your paths (these are expanded by your shell before sudo runs, so they work fine):
export DATADIR=<sites_data_folder>                      # the data directory to be used
export SCRATCHDIR=<scratch_dir>

### Creating startup kits: if there is an issue with the versions:
https://github.com/deboraJ1/MediSwarm/blob/include_abmil/scripts/dev_utils/README_dockerfile_update.md


Example:
DATADIR=/mnt/d/Projects/ODELIA/Challenge/CAM/
SCRATCHDIR=/mnt/d/Projects/ODELIA/Challenge/test_scratch  # hpcwork/it336446/test_scratch

cd <startupkit_directory/startup>                       #  workspace/odelia_1.0.2-dev.260116.a701bc2_dummy_project_for_testing/prod_01/localhost/startup/ /hpcwork/it336446/startup_UKA/startup
chmod +x docker.sh
sudo ./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training 2>&1 | tee dummy_training_console_output.txt
sudo ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --preflight_check 2>&1 | tee preflight_check_console_output.txt
sudo ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --model_name challenge_3agaldran --local_training 2>&1 | tee local_training_console_output.txt

chmod +x start.sh
chmod +x sub_start.sh
sudo ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
sudo chmod a+r nohup.out

// check training logs:
tail -f nohup.out
nvidia-smi

//check connection to server
ping dl3.tud.de

