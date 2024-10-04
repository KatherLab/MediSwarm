import os
from datetime import datetime


def load_environment_variables():
    """Load environment variables and return them as a dictionary."""
    return {
        'clini_table': os.getenv('CLINI_TABLE', '/mnt/swarm_beta/stamp_test_data/data/1/clinical_table.csv'),
        'slide_table': os.getenv('SLIDE_TABLE', '/mnt/swarm_beta/stamp_test_data/data/1/slide_table.csv'),
        'feature_dir': os.getenv('FEATURE_DIR', '/mnt/swarm_beta/stamp_test_data/data/1/All_merged_features_CTRANSPATH'),
        'output_path': os.getenv('OUTPUT_PATH', '/mnt/swarm_beta/stamp_test_data/data/1/out'),
        'target_label': os.getenv('TARGET_LABEL', 'Disease'),
        'cat_labels': os.getenv('CAT_LABELS', '').split(','),
        'cont_labels': os.getenv('CONT_LABELS', '').split(','),
        'categories': os.getenv('CATEGORIES', '').split(','),

        'task_data_name': os.getenv('DATA_FOLDER', 'stamp_experiment'),
        'scratch_dir': os.getenv('SCRATCH_DIR', '/scratch/'),
        'data_dir': os.getenv('DATA_DIR', '/data/'),
        'max_epochs': int(os.getenv('MAX_EPOCHS', 60)),
        'min_peers': int(os.getenv('MIN_PEERS', 2)),
        'max_peers': int(os.getenv('MAX_PEERS', 7)),
        'local_compare_flag': os.getenv('LOCAL_COMPARE_FLAG', 'False').lower() == 'true',
        'use_adaptive_sync': os.getenv('USE_ADAPTIVE_SYNC', 'False').lower() == 'true',
        'sync_frequency': int(os.getenv('SYNC_FREQUENCY', 5)),
    }

def generate_run_directory(scratch_dir, task_data_name, model_name, local_compare_flag):
    """Generate the directory path for the current run."""
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    mode = 'local_compare' if local_compare_flag else 'swarm_learning'
    # make dir if not exist
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    return os.path.join(scratch_dir, f"{current_time}_{task_data_name}_{model_name}_{mode}")

def cal_weightage(train_size):
    estimated_full_dataset_size = 808 # exact training size of Duke 80% dataset, which is the largest across multiple nodes
    weightage = int(100 * train_size / estimated_full_dataset_size)
    if weightage > 100:
        weightage = 100
    return weightage

def cal_max_epochs(preset_max_epochs, weightage):
    return int(preset_max_epochs / (weightage / 100))