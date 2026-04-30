import logging
import sys
import types
from collections import namedtuple
from pathlib import Path

import pytest
import torch

from conftest import import_module_from_path


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
THREEDCNN_PTL_PATH = SHARED_CUSTOM_DIR / "threedcnn_ptl.py"


class _DummyImage:
    def __init__(self, value: float):
        self.data = torch.full((1, 2, 2, 2), value, dtype=torch.float32)


class _MiniSeries(list):
    def tolist(self):
        return list(self)


class _MiniFrame:
    def __init__(self, rows, columns):
        self._rows = [dict(zip(columns, row)) for row in rows]
        self._columns = list(columns)

    def __getitem__(self, key):
        if isinstance(key, str):
            return _MiniSeries(row[key] for row in self._rows)
        if isinstance(key, list):
            subset_rows = [tuple(row[column] for column in key) for row in self._rows]
            return _MiniFrame(subset_rows, key)
        raise TypeError(f"Unsupported key type: {type(key)!r}")

    def itertuples(self, index=False):
        if index:
            raise NotImplementedError("index=True is not supported in this test stub")
        row_type = namedtuple("MiniFrameRow", self._columns)
        for row in self._rows:
            yield row_type(*(row[column] for column in self._columns))


class _DummyDataset:
    def __init__(self, tmp_path: Path, records, value_by_path=None):
        self.df = _MiniFrame([(uid, institution) for uid, institution, _ in records], ["UID", "Institution"])
        self._tmp_path = tmp_path
        self._value_by_path = {} if value_by_path is None else value_by_path
        for uid, institution, value in records:
            path = self.get_image_path(uid, institution)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self._value_by_path[path] = value

    def get_image_path(self, uid: str, institution: str) -> Path:
        return self._tmp_path / institution / uid / "Sub_1.nii.gz"

    def load_img(self, path_img: Path):
        return _DummyImage(self._value_by_path[path_img])


@pytest.fixture
def _importable_threedcnn_ptl():
    if str(SHARED_CUSTOM_DIR) not in sys.path:
        sys.path.insert(0, str(SHARED_CUSTOM_DIR))

    module_names = [
        "threedcnn_ptl",
        "env_config",
        "data",
        "data.datasets",
        "data.datamodules",
        "models",
        "models.models_config",
    ]
    for mod_name in module_names:
        sys.modules.pop(mod_name, None)

    data_pkg = types.ModuleType("data")
    data_datasets = types.ModuleType("data.datasets")
    data_datamodules = types.ModuleType("data.datamodules")
    env_config = types.ModuleType("env_config")
    models_pkg = types.ModuleType("models")
    models_config = types.ModuleType("models.models_config")

    data_datasets.ODELIA_Dataset3D = object
    data_datamodules.DataModule = object
    env_config.build_odelia_manifests = lambda *args, **kwargs: None
    env_config.generate_run_directory = lambda *args, **kwargs: "/tmp/run"
    env_config.load_environment_variables = lambda: {}
    env_config.prepare_odelia_dataset = lambda *args, **kwargs: None
    env_config.prepare_odelia_dataset_without_augmentation = lambda *args, **kwargs: None
    models_config.create_model = lambda *args, **kwargs: None

    stub_modules = {
        "data": data_pkg,
        "data.datasets": data_datasets,
        "data.datamodules": data_datamodules,
        "env_config": env_config,
        "models": models_pkg,
        "models.models_config": models_config,
    }

    old_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)

    try:
        yield
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
        sys.modules.pop("threedcnn_ptl", None)
        if str(SHARED_CUSTOM_DIR) in sys.path:
            sys.path.remove(str(SHARED_CUSTOM_DIR))


def _import_threedcnn_ptl():
    sys.modules.pop("threedcnn_ptl", None)
    return import_module_from_path("threedcnn_ptl", THREEDCNN_PTL_PATH)


def test_log_data_hash_logs_duplicate_image_data_with_uids(_importable_threedcnn_ptl, tmp_path, caplog):
    threedcnn_ptl = _import_threedcnn_ptl()
    shared_value_by_path = {}

    ds_train = _DummyDataset(
        tmp_path,
        [
            ("ID_005_right", "client_A", 1.0),
            ("ID_005_left", "client_A", 1.0),
        ],
        value_by_path=shared_value_by_path,
    )
    ds_val = _DummyDataset(
        tmp_path,
        [
            ("ID_009_left", "client_A", 2.0),
        ],
        value_by_path=shared_value_by_path,
    )
    ds_test = _DummyDataset(
        tmp_path,
        [
            ("ID_012_left", "client_A", 3.0),
        ],
        value_by_path=shared_value_by_path,
    )

    env_vars = {
        "odelia_hash_cache_dir": str(tmp_path / "hash_cache"),
        "odelia_hash_num_workers": 0,
    }
    logger = logging.getLogger("test_threedcnn_ptl")

    with caplog.at_level(logging.INFO, logger=logger.name):
        threedcnn_ptl.log_data_hash(
            ds_train,
            ds_val,
            ds_test,
            env_vars,
            logger,
            log_dataset_details=True,
        )

    assert "Image data with hash" in caplog.text
    assert "ID_005_left, ID_005_right" in caplog.text
    assert "client_A, client_A" not in caplog.text
