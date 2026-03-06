# Evaluation of Local and Swarm Trainings

## From Training Logs

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

## From Class Predictions

* TODO (separate branch)
* uses one-vs-one ROC_AUC