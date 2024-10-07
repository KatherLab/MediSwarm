import os
from datetime import datetime


def load_environment_variables():
    """Load environment variables and return them as a dictionary."""
    return {
        'scratch_dir': os.getenv('SCRATCH_DIR', '/scratch/'),
        'max_epochs': int(os.getenv('MAX_EPOCHS', 100)),
        'min_peers': int(os.getenv('MIN_PEERS', 2)),
        'max_peers': int(os.getenv('MAX_PEERS', 7)),
        'use_adaptive_sync': os.getenv('USE_ADAPTIVE_SYNC', 'False').lower() == 'true',
        'sync_frequency': int(os.getenv('SYNC_FREQUENCY', 1024)),
        'prediction_flag': os.getenv('PREDICT_FLAG', 'ext')
    }

def create_run_directory(scratch_dir):
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # make dir if not exist
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    return os.path.join(scratch_dir, f"{current_time}_minimal_training")
