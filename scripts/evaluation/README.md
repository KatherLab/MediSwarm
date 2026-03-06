# Evaluation of Local and Swarm Trainings

## From Training Logs

TODO when merging, mention that this script is just for comparison, the other one should be used.

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