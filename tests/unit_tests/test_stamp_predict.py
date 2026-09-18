"""
Unit tests for stamp_predict.py (issue #270) — the pure, dependency-light
helpers that don't need torch or STAMP installed:

- discover_checkpoints(): globbing + app_<SITE> parsing + best-first de-dup
- extract_state_dict(): the three checkpoint shapes PTFileModelPersistor/
  Lightning produce
- compute_classification_metrics(): accuracy + AUROC (needs numpy + sklearn;
  importorskip when absent)

The torch/STAMP runtime (inference, model build, dataloader) is exercised by
the deploy-test 2-node run, not here.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "STAMP_classification" / "app" / "custom"


@pytest.fixture(scope="module")
def sp():
    """Import stamp_predict from the STAMP custom dir (module-level imports are stdlib only)."""
    p = str(STAMP_CUSTOM_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    sys.modules.pop("stamp_predict", None)
    import stamp_predict  # noqa: E402
    return stamp_predict


# --------------------------------------------------------------------------- #
#  discover_checkpoints
# --------------------------------------------------------------------------- #

def _make_ckpt(root: Path, site: str, name: str) -> Path:
    d = root / f"app_{site}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_bytes(b"stub")
    return f


def test_discover_checkpoints_finds_all_sites(sp, tmp_path):
    _make_ckpt(tmp_path, "RUMC_1", "FL_global_model.pt")
    _make_ckpt(tmp_path, "RUMC_1", "best_FL_global_model.pt")
    _make_ckpt(tmp_path, "MHA_1", "FL_global_model.pt")

    found = sp.discover_checkpoints(str(tmp_path))
    sites = sorted({c["site"] for c in found})
    assert sites == ["MHA_1", "RUMC_1"]
    # RUMC_1 has both files → 2 records; MHA_1 has 1 → total 3
    assert len(found) == 3
    assert all(c["name"] in sp.CHECKPOINT_NAMES for c in found)


def test_discover_checkpoints_best_only(sp, tmp_path):
    _make_ckpt(tmp_path, "RUMC_1", "FL_global_model.pt")
    _make_ckpt(tmp_path, "RUMC_1", "best_FL_global_model.pt")

    found = sp.discover_checkpoints(str(tmp_path), best_only=True)
    assert len(found) == 1
    assert found[0]["name"] == "best_FL_global_model.pt"


def test_discover_checkpoints_nested_and_dedup(sp, tmp_path):
    # deep NVFlare-style layout: prod/server/job_id/app_SITE/
    nested = tmp_path / "prod_00" / "server" / "job-abc"
    _make_ckpt(nested, "USZ_1", "best_FL_global_model.pt")
    found = sp.discover_checkpoints(str(tmp_path))
    assert [c["site"] for c in found] == ["USZ_1"]
    # de-dup on (site, name)
    assert len({(c["site"], c["name"]) for c in found}) == len(found)


def test_discover_checkpoints_empty(sp, tmp_path):
    assert sp.discover_checkpoints(str(tmp_path)) == []


# --------------------------------------------------------------------------- #
#  extract_state_dict
# --------------------------------------------------------------------------- #

def test_extract_state_dict_model_key(sp):
    sd = {"layer.weight": 1}
    assert sp.extract_state_dict({"model": sd}) is sd


def test_extract_state_dict_state_dict_key(sp):
    sd = {"layer.weight": 1}
    assert sp.extract_state_dict({"state_dict": sd}) is sd


def test_extract_state_dict_bare(sp):
    sd = {"layer.weight": 1, "layer.bias": 2}
    assert sp.extract_state_dict(sd) == sd


def test_extract_state_dict_model_key_prefers_over_state_dict(sp):
    model_sd = {"a": 1}
    assert sp.extract_state_dict({"model": model_sd, "state_dict": {"b": 2}}) is model_sd


def test_extract_state_dict_rejects_non_dict(sp):
    with pytest.raises(ValueError):
        sp.extract_state_dict([1, 2, 3])


# --------------------------------------------------------------------------- #
#  compute_classification_metrics (needs numpy + sklearn)
# --------------------------------------------------------------------------- #

def test_compute_metrics_perfect_binary(sp):
    pytest.importorskip("sklearn")
    y_true = [0, 0, 1, 1]
    y_prob = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    m = sp.compute_classification_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["auroc"] == 1.0
    assert m["num_samples"] == 4
    assert m["num_classes"] == 2


def test_compute_metrics_multiclass(sp):
    pytest.importorskip("sklearn")
    y_true = [0, 1, 2, 0]
    y_prob = [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.6, 0.3, 0.1],
    ]
    m = sp.compute_classification_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert 0.0 <= m["auroc"] <= 1.0
    assert m["num_classes"] == 3


def test_compute_metrics_single_class_auroc_none(sp):
    pytest.importorskip("sklearn")
    y_true = [1, 1, 1]
    y_prob = [[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]]
    m = sp.compute_classification_metrics(y_true, y_prob)
    assert m["auroc"] is None          # undefined with one class
    assert m["accuracy"] == 1.0


def test_compute_metrics_rejects_1d_prob(sp):
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError):
        sp.compute_classification_metrics([0, 1], [0.2, 0.8])


# --------------------------------------------------------------------------- #
#  compute_regression_metrics (#271)
# --------------------------------------------------------------------------- #

def test_compute_regression_perfect(sp):
    pytest.importorskip("sklearn")
    y = [1.0, 2.0, 3.0, 4.0]
    m = sp.compute_regression_metrics(y, y)
    assert m["mse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert m["num_samples"] == 4


def test_compute_regression_with_error(sp):
    pytest.importorskip("sklearn")
    m = sp.compute_regression_metrics([1.0, 2.0, 3.0], [1.5, 2.5, 2.0])
    assert m["mse"] > 0
    assert m["mae"] > 0
    assert m["r2"] is not None


# --------------------------------------------------------------------------- #
#  compute_survival_metrics — Harrell c-index (#271, pure numpy)
# --------------------------------------------------------------------------- #

def test_survival_cindex_perfect(sp):
    # higher risk -> shorter time == perfectly concordant
    m = sp.compute_survival_metrics(times=[1, 2, 3], events=[1, 1, 1], risks=[3.0, 2.0, 1.0])
    assert m["c_index"] == 1.0
    assert m["num_comparable_pairs"] == 3


def test_survival_cindex_worst(sp):
    m = sp.compute_survival_metrics(times=[1, 2, 3], events=[1, 1, 1], risks=[1.0, 2.0, 3.0])
    assert m["c_index"] == 0.0


def test_survival_cindex_ties_half(sp):
    m = sp.compute_survival_metrics(times=[1, 2], events=[1, 1], risks=[5.0, 5.0])
    assert m["c_index"] == 0.5  # one comparable pair, tie counts 0.5


def test_survival_cindex_censoring_excludes_pairs(sp):
    # patient 0 censored (event=0) -> not comparable as the earlier-time case
    m = sp.compute_survival_metrics(times=[1, 2, 3], events=[0, 1, 1], risks=[9.0, 2.0, 1.0])
    # only i=1 (t=2, event) vs j=2 (t=3) is comparable
    assert m["num_comparable_pairs"] == 1
    assert m["c_index"] == 1.0  # risk1=2 > risk2=1


def test_survival_cindex_no_comparable_pairs(sp):
    m = sp.compute_survival_metrics(times=[5, 5], events=[0, 0], risks=[1.0, 2.0])
    assert m["c_index"] is None
    assert m["num_comparable_pairs"] == 0


# --------------------------------------------------------------------------- #
#  compare_federated_vs_local (#275)
# --------------------------------------------------------------------------- #

def test_compare_classification_federated_better(sp):
    fed = {"auroc": 0.90, "accuracy": 0.8}
    local = {"auroc": 0.80, "accuracy": 0.7}
    c = sp.compare_federated_vs_local(fed, local, "classification")
    assert c["comparable"] is True
    assert c["metric"] == "auroc"
    assert c["delta"] == pytest.approx(0.10)
    assert c["federated_at_least_as_good"] is True
    assert c["regression_detected"] is False


def test_compare_classification_regression_detected(sp):
    fed = {"auroc": 0.70}
    local = {"auroc": 0.85}
    c = sp.compare_federated_vs_local(fed, local, "classification")
    assert c["delta"] == pytest.approx(-0.15)
    assert c["federated_at_least_as_good"] is False
    assert c["regression_detected"] is True


def test_compare_regression_lower_is_better(sp):
    # mse: lower is better, so a lower federated mse => positive delta (better)
    fed = {"mse": 0.20}
    local = {"mse": 0.50}
    c = sp.compare_federated_vs_local(fed, local, "regression")
    assert c["metric"] == "mse"
    assert c["higher_is_better"] is False
    assert c["delta"] == pytest.approx(0.30)
    assert c["regression_detected"] is False


def test_compare_survival_cindex(sp):
    c = sp.compare_federated_vs_local({"c_index": 0.75}, {"c_index": 0.60}, "survival")
    assert c["metric"] == "c_index"
    assert c["delta"] == pytest.approx(0.15)


def test_compare_tolerance_absorbs_small_regression(sp):
    fed = {"auroc": 0.78}
    local = {"auroc": 0.80}
    c = sp.compare_federated_vs_local(fed, local, "classification", tolerance=0.05)
    # 0.02 drop is within tolerance -> not flagged
    assert c["regression_detected"] is False
    assert c["federated_at_least_as_good"] is True


def test_compare_incomparable_when_metric_missing(sp):
    c = sp.compare_federated_vs_local({"accuracy": 0.8}, {"auroc": 0.9}, "classification")
    # federated has no auroc and local has no accuracy overlap on the top metric...
    # auroc present only in local, accuracy present only in fed -> no shared metric
    assert c["comparable"] is False


def test_compare_falls_back_to_accuracy(sp):
    # no auroc in either, but accuracy in both -> compares on accuracy
    c = sp.compare_federated_vs_local({"accuracy": 0.9}, {"accuracy": 0.8}, "classification")
    assert c["comparable"] is True
    assert c["metric"] == "accuracy"
    assert c["delta"] == pytest.approx(0.10)
