import os
from datetime import datetime
from pathlib import Path
import argparse


def load_environment_variables():
    return {
        'task_data_name': os.getenv('DATA_FOLDER', 'Odelia'),
        'scratch_dir': os.getenv('SCRATCH_DIR', '/scratch/'),
        'data_dir': os.getenv('DATA_DIR', '/data/'),
        'max_epochs': int(os.getenv('MAX_EPOCHS', 100)),
        'min_peers': int(os.getenv('MIN_PEERS', 2)),
        'max_peers': int(os.getenv('MAX_PEERS', 10)),
        'local_compare_flag': os.getenv('LOCAL_COMPARE_FLAG', 'False').lower() == 'true',
        'use_adaptive_sync': os.getenv('USE_ADAPTIVE_SYNC', 'False').lower() == 'true',
        'sync_frequency': int(os.getenv('SYNC_FREQUENCY', 1024)),
        'model_name': os.getenv('MODEL_NAME', 'ResNet101'),
        'prediction_flag': os.getenv('PREDICT_FLAG', 'ext')
    }


def load_prediction_modules(prediction_flag):
    from predict import predict
    return predict, prediction_flag


def prepare_dataset(task_data_name, data_dir, site_name, split="train"):
    try:
        available_dirs = next(os.walk(data_dir))[1]
    except StopIteration:
        raise ValueError(f"No directories found under data_dir: {data_dir}")
    if 'DUKE_ext' in available_dirs:
        task_data_name = "DUKE_ext"

    dataset_class = None
    if task_data_name == "multi_ext":
        from data.datasets import DUKE_Dataset3D_collab as dataset_class
    elif task_data_name == "DUKE_ext":
        from data.datasets import DUKE_Dataset3D_external as dataset_class
    elif task_data_name == "DUKE":
        from data.datasets import DUKE_Dataset3D as dataset_class
    else:
        raise ValueError("Invalid task data name specified")

    return dataset_class(flip=True, path_root=os.path.join(data_dir, site_name)), task_data_name


def prepare_odelia_dataset(task_data_name, data_dir, site_name, split="train"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--institution', default='ODELIA', type=str)
    parser.add_argument('--model', type=str, default='MST', choices=['ResNet', 'MST'])
    parser.add_argument('--task', type=str, default="binary", choices=['binary', 'ordinal'])
    parser.add_argument('--config', type=str, default="unilateral", choices=['original', 'unilateral'])
    args = parser.parse_args()
    binary = args.task == 'binary'

    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_name = f'{args.model}_{args.task}_{args.config}_{current_time}'
    path_run_dir = Path.cwd() / 'runs' / args.institution / run_name
    path_run_dir.mkdir(parents=True, exist_ok=True)

    from data.datasets import ODELIA_Dataset3D
    ds_train = ODELIA_Dataset3D(institutions=args.institution, split='train', binary=binary, config=args.config,
                                random_flip=True, random_rotate=True, random_inverse=False, noise=True)
    ds_val = ODELIA_Dataset3D(institutions=args.institution, split='val', binary=binary, config=args.config)

    return ds_train, ds_val, path_run_dir, run_name, binary


def generate_run_directory(scratch_dir, task_data_name, model_name, local_compare_flag):
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    mode = 'local_compare' if local_compare_flag else 'swarm_learning'
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    return os.path.join(scratch_dir, f"{current_time}_{task_data_name}_{model_name}_{mode}")


def cal_weightage(train_size):
    estimated_full_dataset_size = 808
    weightage = int(100 * train_size / estimated_full_dataset_size)
    return min(weightage, 100)


def cal_max_epochs(preset_max_epochs, weightage):
    return int(preset_max_epochs / (weightage / 100))
