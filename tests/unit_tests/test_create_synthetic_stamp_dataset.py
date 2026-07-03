"""
Unit tests for the STAMP synthetic-data generator's task support (#271).

Only the pure label-column logic (build_label_columns) is tested here — it
needs numpy but not the H5 writing path. The generator module imports h5py at
module load, so the whole test file is skipped when h5py is unavailable.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("h5py")  # generator imports h5py at module level
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = (
    REPO_ROOT
    / "application" / "jobs" / "STAMP_classification" / "app" / "scripts" / "create_synthetic_dataset"
)


@pytest.fixture(scope="module")
def gen():
    p = str(GEN_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    sys.modules.pop("create_synthetic_stamp_dataset", None)
    import create_synthetic_stamp_dataset as g  # noqa: E402
    return g


def _labels(gen, task):
    pids = ["P_000", "P_001", "P_002"]
    classes = ["class_0", "class_1", "class_2"]
    rng = np.random.RandomState(0)
    return gen.build_label_columns(task, pids, classes, rng)


def test_classification_columns(gen):
    fields, rows = _labels(gen, "classification")
    assert fields == ["PATIENT", "Diagnosis"]
    assert [r["Diagnosis"] for r in rows] == ["class_0", "class_1", "class_2"]


def test_survival_columns(gen):
    fields, rows = _labels(gen, "survival")
    assert fields == ["PATIENT", "Time", "Event"]
    # time increases with class index; all positive
    times = [r["Time"] for r in rows]
    assert all(t > 0 for t in times)
    assert times[0] < times[2]
    assert all(r["Event"] in (0, 1) for r in rows)


def test_regression_columns(gen):
    fields, rows = _labels(gen, "regression")
    assert fields == ["PATIENT", "Target"]
    targets = [r["Target"] for r in rows]
    assert all(isinstance(t, float) for t in targets)
    assert targets[0] < targets[2]  # increases with class index


def test_unknown_task_raises(gen):
    with pytest.raises(ValueError):
        _labels(gen, "clustering")


def test_main_rejects_unknown_task(gen, tmp_path):
    with pytest.raises(ValueError):
        gen.main(tmp_path, task="nonsense")
