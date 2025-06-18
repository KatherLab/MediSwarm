import os
from datetime import datetime
from pathlib import Path
import argparse


def load_environment_variables():
    return {
        'site_name': os.environ['SITE_NAME'],
        'task_data_name': os.environ.get('DATA_FOLDER', 'Odelia'),
        'scratch_dir': os.environ['SCRATCH_DIR'],
        'data_dir': os.environ['DATA_DIR'],
        'max_epochs': int(os.environ.get('MAX_EPOCHS', 100)),
        'min_peers': int(os.environ.get('MIN_PEERS', 2)),
        'max_peers': int(os.environ.get('MAX_PEERS', 10)),
        'local_compare_flag': os.environ.get('LOCAL_COMPARE_FLAG', 'False').lower() == 'true',
        'use_adaptive_sync': os.environ.get('USE_ADAPTIVE_SYNC', 'False').lower() == 'true',
        'sync_frequency': int(os.environ.get('SYNC_FREQUENCY', 1024)),
        'model_name': os.environ.get('MODEL_NAME', 'ResNet101'),
        'prediction_flag': os.environ.get('PREDICT_FLAG', 'ext')
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
    # parser removed, now read from environment
    institution = os.environ.get('INSTITUTION', 'ODELIA')
    model = os.environ.get('MODEL_NAME', 'MST')
    task = os.environ.get('TASK', 'binary')
    config = os.environ.get('CONFIG', 'unilateral')
    task_type = os.environ.get('TASK', 'binary')  # 'binary' or 'ordinal'
    is_binary_task = task_type == 'binary'

    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_name = f'{model}_{task}_{config}_{current_time}'
    path_run_dir = Path.cwd() / 'runs' / institution / run_name
    path_run_dir.mkdir(parents=True, exist_ok=True)

    from data.datasets import ODELIA_Dataset3D
    ds_train = ODELIA_Dataset3D(institutions=institution, split='train', binary=is_binary_task, config=config,
                                random_flip=True, random_rotate=True, random_inverse=False, noise=True)
    ds_val = ODELIA_Dataset3D(institutions=institution, split='val', binary=is_binary_task, config=config)

    print(f"Total samples loaded: {len(ds_train)} (train) + {len(ds_val)} (val)")
    print(f"Train set: {len(ds_train)}, Val set: {len(ds_val)}")
    #print(f"Labels in val: {[sample['label'] for sample in ds_val]}")

    return ds_train, ds_val, path_run_dir, run_name, is_binary_task



def generate_run_directory(scratch_dir, task_data_name, model_name, local_compare_flag):
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    mode = 'local_compare' if local_compare_flag else 'swarm_learning'
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    return os.path.join(scratch_dir, f"{current_time}_{task_data_name}_{model_name}_{mode}")

# TODO: Implement dynamic weightage calculation based on actual dataset size
def cal_weightage(train_size):
    """
    Placeholder function for calculating training weightage.
    Currently unused.
    """
    pass  # To be implemented


# TODO: Implement max epochs adjustment logic based on weightage
def cal_max_epochs(preset_max_epochs, weightage):
    """
    Placeholder function for dynamically adjusting max epochs.
    Currently unused.
    """
    pass  # To be implemented
