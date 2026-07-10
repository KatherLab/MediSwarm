import os
import torch.multiprocessing as mp
from datetime import datetime
from pathlib import Path

from data.datasets import ODELIA_Dataset3D


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


NUM_FOLDS = 5  # split.csv is generated with 5-fold nested StratifiedGroupKFold


def _fold_from_env():
    """Cross-validation fold to train on (#411).

    Run-level, not per-site: NVFlare distributes one job to every client, so all
    sites train the same fold. K-fold CV = K sequential swarm runs. Defaults to 0,
    which is the fold the code trained before this was configurable.
    """
    raw = os.environ.get('FOLD', '0').strip() or '0'
    try:
        fold = int(raw)
    except ValueError:
        raise ValueError(f"FOLD must be an integer, got {raw!r}")
    if not 0 <= fold < NUM_FOLDS:
        raise ValueError(f"FOLD must be in [0, {NUM_FOLDS - 1}], got {fold}")
    return fold


def load_environment_variables():
    scratch_dir = os.environ['SCRATCH_DIR']
    # Default DataLoader worker count. PR301 capped this at 8, which regressed
    # loader concurrency on the ~16-CPU deploy nodes (the older path used all
    # CPUs). Raise the cap to 16 so those nodes are not throttled by default,
    # while still bounding very-large hosts. Tune per node via ODELIA_NUM_WORKERS
    # (lower it on RAM-constrained nodes). See #315.
    default_loader_workers = min(mp.cpu_count(), 16)

    return {
        'site_name': os.environ['SITE_NAME'],
        'task_data_name': os.environ.get('DATA_FOLDER', 'Odelia'),
        'scratch_dir': scratch_dir,
        'data_dir': os.environ['DATA_DIR'],
        'max_epochs': int(os.environ.get('MAX_EPOCHS', 100)),
        'min_peers': int(os.environ.get('MIN_PEERS', 2)),
        'max_peers': int(os.environ.get('MAX_PEERS', 10)),
        'local_compare_flag': _env_flag('LOCAL_COMPARE_FLAG', default=False),
        'use_adaptive_sync': _env_flag('USE_ADAPTIVE_SYNC', default=False),
        'sync_frequency': int(os.environ.get('SYNC_FREQUENCY', 1024)),
        'model_name': os.environ.get('MODEL_NAME', 'ResNet101'),
        'fold': _fold_from_env(),
        'prediction_flag': os.environ.get('PREDICT_FLAG', 'ext'),
        'mediswarm_version': os.environ.get('MEDISWARM_VERSION', 'unset'),
        'odelia_num_workers': int(os.environ.get('ODELIA_NUM_WORKERS', default_loader_workers)),
        'odelia_hash_num_workers': int(os.environ.get('ODELIA_HASH_NUM_WORKERS', 0)),
        'odelia_threads_per_worker': int(os.environ.get('ODELIA_THREADS_PER_WORKER', 1)),
        'odelia_interop_threads': int(os.environ.get('ODELIA_INTEROP_THREADS', 1)),
        'odelia_hash_cache_dir': os.environ.get(
            'ODELIA_HASH_CACHE_DIR',
            str(Path(scratch_dir) / 'odelia_hash_cache')
        ),
        'odelia_enable_preprocess_cache': _env_flag('ODELIA_ENABLE_PREPROCESS_CACHE', default=False),
        'odelia_preprocess_cache_dir': os.environ.get(
            'ODELIA_PREPROCESS_CACHE_DIR',
            str(Path(scratch_dir) / 'odelia_preprocess_cache')
        ),
        'odelia_preprocess_cache_version': os.environ.get('ODELIA_PREPROCESS_CACHE_VERSION', 'v1'),
    }


def load_prediction_modules(prediction_flag):
    from predict import predict
    return predict, prediction_flag


def resolve_odelia_runtime_settings(env_vars=None):
    if env_vars is None:
        env_vars = load_environment_variables()

    return {
        'institution': os.environ.get('INSTITUTION', env_vars['site_name']),
        'model_name': os.environ.get('MODEL_NAME', env_vars['model_name']),
        'config': os.environ.get('CONFIG', 'unilateral'),
        'fold': env_vars['fold'],
        'data_dir': env_vars['data_dir'],
        'scratch_dir': env_vars['scratch_dir'],
    }


def build_odelia_manifests(env_vars=None):
    settings = resolve_odelia_runtime_settings(env_vars)
    return ODELIA_Dataset3D.build_manifests(
        path_root=settings['data_dir'],
        institutions=settings['institution'],
        config=settings['config'],
    )


def prepare_odelia_dataset(logger, log_dataset_details: bool = False, manifests=None, env_vars=None):
    if env_vars is None:
        env_vars = load_environment_variables()
    settings = resolve_odelia_runtime_settings(env_vars)
    institution = settings['institution']
    model = settings['model_name']
    config = settings['config']

    if manifests is None:
        manifests = build_odelia_manifests(env_vars)

    fold = settings['fold']
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_name = f'{model}_{config}_fold{fold}_{current_time}'
    path_run_dir = Path(settings['scratch_dir']) / 'runs' / institution / run_name
    path_run_dir.mkdir(parents=True, exist_ok=True)

    ds_train = ODELIA_Dataset3D(
        path_root=settings['data_dir'],
        institutions=institution,
        split='train',
        config=config,
        fold=fold,
        random_flip=True,
        random_rotate=True,
        random_inverse=False,
        noise=True,
        manifests=manifests,
        enable_preprocess_cache=env_vars['odelia_enable_preprocess_cache'] if env_vars else False,
        preprocess_cache_dir=env_vars['odelia_preprocess_cache_dir'] if env_vars else None,
        preprocess_cache_version=env_vars['odelia_preprocess_cache_version'] if env_vars else 'v1',
    )
    ds_val = ODELIA_Dataset3D(
        path_root=settings['data_dir'],
        institutions=institution,
        split='val',
        config=config,
        fold=fold,
        manifests=manifests,
        enable_preprocess_cache=env_vars['odelia_enable_preprocess_cache'] if env_vars else False,
        preprocess_cache_dir=env_vars['odelia_preprocess_cache_dir'] if env_vars else None,
        preprocess_cache_version=env_vars['odelia_preprocess_cache_version'] if env_vars else 'v1',
    )
    ds_test = ODELIA_Dataset3D(
        path_root=settings['data_dir'],
        institutions=institution,
        split='test',
        config=config,
        fold=fold,
        manifests=manifests,
        enable_preprocess_cache=env_vars['odelia_enable_preprocess_cache'] if env_vars else False,
        preprocess_cache_dir=env_vars['odelia_preprocess_cache_dir'] if env_vars else None,
        preprocess_cache_version=env_vars['odelia_preprocess_cache_version'] if env_vars else 'v1',
    )

    print(f"Total samples loaded: {len(ds_train)} (train) + {len(ds_val)} (val)")
    print(f"Train set: {len(ds_train)}, Val set: {len(ds_val)}")

    return ds_train, ds_val, ds_test, path_run_dir, run_name


def prepare_odelia_dataset_without_augmentation(manifests=None, env_vars=None):
    if env_vars is None:
        env_vars = load_environment_variables()
    settings = resolve_odelia_runtime_settings(env_vars)
    institution = settings['institution']
    config = settings['config']

    if manifests is None:
        manifests = build_odelia_manifests(env_vars)

    ds_train = ODELIA_Dataset3D(
        path_root=settings['data_dir'],
        institutions=institution,
        split='train',
        config=config,
        fold=settings['fold'],
        transform='USE_UNPROCESSED_IMAGES',
        manifests=manifests,
    )
    ds_val = ODELIA_Dataset3D(
        path_root=settings['data_dir'],
        institutions=institution,
        split='val',
        config=config,
        fold=settings['fold'],
        transform='USE_UNPROCESSED_IMAGES',
        manifests=manifests,
    )
    ds_test = ODELIA_Dataset3D(
        path_root=settings['data_dir'],
        institutions=institution,
        split='test',
        config=config,
        fold=settings['fold'],
        transform='USE_UNPROCESSED_IMAGES',
        manifests=manifests,
    )

    return ds_train, ds_val, ds_test


def generate_run_directory(scratch_dir, task_data_name, model_name, local_compare_flag):
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    mode = 'local_compare' if local_compare_flag else 'swarm_learning'
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    return os.path.join(scratch_dir, f"{current_time}_{task_data_name}_{model_name}_{mode}")
