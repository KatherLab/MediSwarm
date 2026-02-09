import os
from datetime import datetime
from pathlib import Path
from data.datasets import ODELIA_Dataset3D

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
        'prediction_flag': os.environ.get('PREDICT_FLAG', 'ext'),
        'mediswarm_version': os.environ.get('MEDISWARM_VERSION', 'unset'),
    }


def load_prediction_modules(prediction_flag):
    from predict import predict
    return predict, prediction_flag


def prepare_odelia_dataset():
    # parser removed, now read from environment
    institution = os.environ.get('INSTITUTION', os.environ['SITE_NAME'])  # TODO think about how this should be handled
    model = os.environ.get('MODEL_NAME', 'MST')
    config = os.environ.get('CONFIG', 'unilateral')

    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_name = f'{model}_{config}_{current_time}'
    path_run_dir = Path.cwd() / 'runs' / institution / run_name
    path_run_dir.mkdir(parents=True, exist_ok=True)

    ds_train = ODELIA_Dataset3D(institutions=institution, split='train', config=config,
                                random_flip=True, random_rotate=True, random_inverse=False, noise=True)
    ds_val = ODELIA_Dataset3D(institutions=institution, split='val', config=config)

    print(f"Total samples loaded: {len(ds_train)} (train) + {len(ds_val)} (val)")
    print(f"Train set: {len(ds_train)}, Val set: {len(ds_val)}")
    # print(f"Labels in val: {[sample['label'] for sample in ds_val]}")

    return ds_train, ds_val, path_run_dir, run_name


def prepare_odelia_dataset_without_augmentation():
    institution = os.environ.get('INSTITUTION', os.environ['SITE_NAME'])
    config = os.environ.get('CONFIG', 'unilateral')

    ds_train = ODELIA_Dataset3D(institutions=institution, split='train', config=config, transform='USE_UNPROCESSED_IMAGES')
    ds_val = ODELIA_Dataset3D(institutions=institution, split='val', config=config, transform='USE_UNPROCESSED_IMAGES')

    return ds_train, ds_val


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
