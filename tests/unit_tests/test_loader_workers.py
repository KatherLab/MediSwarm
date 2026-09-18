"""cap_loader_workers (#574): never more DataLoader workers than the data can feed.

loader_workers.py has no imports at all, so this file needs neither torch nor the
training image and cannot be silently skipped (F10, #565).
"""

import importlib.util
import os

import pytest

_P = os.path.join(os.path.dirname(__file__), "..", "..", "application", "jobs",
                  "_shared", "custom", "loader_workers.py")
_spec = importlib.util.spec_from_file_location("loader_workers_under_test", _P)
loader_workers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loader_workers)
cap = loader_workers.cap_loader_workers


def test_the_mvp_case_that_hit_the_race():
    # 16 workers on 38 volumes (TEST_A_1, 2026-09-11) -> 9
    assert cap(16, 38) == 9


def test_never_binds_on_a_real_site():
    assert cap(16, 17834) == 16      # UKA
    assert cap(16, 190) == 16        # VHIO, the smallest hospital


def test_floor_is_one_worker():
    assert cap(16, 1) == 1
    assert cap(16, 3) == 1


def test_zero_workers_stays_zero():
    """num_workers=0 means in-process loading; the cap must not invent workers."""
    assert cap(0, 38) == 0


def test_unknown_sample_count_leaves_config_alone():
    assert cap(16, None) == 16
    assert cap(16, 0) == 16
    assert cap(16, "not a number") == 16


def test_min_samples_per_worker_is_tunable():
    assert cap(16, 38, min_samples_per_worker=2) == 16
    assert cap(16, 38, min_samples_per_worker=10) == 3
