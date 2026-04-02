# Evaluation of Local and Swarm Trainings

## From Class Predictions

* During training, class predictions and ground truth values are logged for later evaluation.
* This script reads these values and computes ROC_AUCs (by default, one-vs-one, which does not hide low performance on rare classes).
* The script expects a folder structure
  ```bash
  local
  ├── CAM
  │   ├── site_model_gt_and_classprob_train.csv
  │   └── site_model_gt_and_classprob_validation.csv
  └── … (further sites)
  swarm
  ├── CAM
  │   ├── site_model_gt_and_classprob_train.csv
  │   ├── site_model_gt_and_classprob_validation.csv
  │   ├── aggregated_model_gt_and_classprob_train.csv
  │   └── aggregated_model_gt_and_classprob_validation.csv
  └── … (further sites)
  ```

## From Training Logs (outdated)

* During training, ROC_AUC values are computed and written in the log. These are one-vs-rest ROC_AUCs, which may hide poor performance on rare classes.
* These values can be parsed from local training logs and swarm training captured output using the script
  ```bash
  ./parse_logs_and_plot.py
  ```
* The script expects a folder structure
  ```bash
  CAM
  ├── local_training_console_output.txt
  └── nohup.out
  MHA
  ├── local_training_console_output.txt
  └── nohup.out
  …
  ```
* This script is kept for comparison only, use the evaluation bsaed on class predicitons instead.
