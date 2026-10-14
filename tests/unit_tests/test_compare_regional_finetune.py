"""Unit tests for the D2.5 regional fine-tuning comparison (#526).

The analysis is paired: both models score the same cases, so the bootstrap must
resample cases jointly and report a CI on the difference. These tests pin that
behaviour, plus the guards that keep an underpowered cohort from being read as a
result.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import import_module_from_path  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "compare_regional_finetune.py"


@pytest.fixture(scope="module")
def mod():
    return import_module_from_path("_cmp_regional", SCRIPT)


def _write(tmp, site, rows):
    d = tmp / site
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "predictions_x.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["uid", "ground_truth", "prediction",
                                           "prob_class_0", "prob_class_1", "prob_class_2"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(uid, gt, p2):
    """A case whose malignant score is p2; the rest is split over the other classes."""
    rest = 1.0 - p2
    return {"uid": uid, "ground_truth": gt, "prediction": 2 if p2 > 0.5 else 0,
            "prob_class_0": rest * 0.7, "prob_class_1": rest * 0.3, "prob_class_2": p2}


def test_auroc_matches_known_value(mod):
    # two positives ranked above two negatives -> perfect separation
    assert mod.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    # perfectly inverted
    assert mod.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    # ties average to chance
    assert mod.auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5


def test_auroc_is_none_when_a_class_is_absent(mod):
    assert mod.auroc([0.1, 0.2], [0, 0]) is None
    assert mod.auroc([0.1, 0.2], [1, 1]) is None


def test_detects_a_real_improvement(mod, tmp_path):
    base, fine = tmp_path / "b", tmp_path / "f"
    # baseline scores malignants no better than chance; fine-tuned separates them
    b = [_row(f"c{i}", 2, 0.4) for i in range(8)] + [_row(f"n{i}", 0, 0.5) for i in range(20)]
    f = [_row(f"c{i}", 2, 0.9) for i in range(8)] + [_row(f"n{i}", 0, 0.1) for i in range(20)]
    _write(base, "RUMC_1", b)
    _write(fine, "RUMC_1", f)
    br, fr = mod.load(str(base), ["RUMC_1"]), mod.load(str(fine), ["RUMC_1"])
    uids = sorted(set(br) & set(fr))
    d, lo, hi, favour = mod.paired_bootstrap(br, fr, uids, "malignant_vs_rest", 400, seed=1)
    assert d > 0.3
    assert lo > 0, "a large real effect should give an interval clear of zero"
    assert favour == 1.0


def test_no_difference_gives_an_interval_spanning_zero(mod, tmp_path):
    base, fine = tmp_path / "b", tmp_path / "f"
    rows = [_row(f"c{i}", 2, 0.6) for i in range(7)] + [_row(f"n{i}", 0, 0.4) for i in range(19)]
    _write(base, "MHA_1", rows)
    _write(fine, "MHA_1", rows)          # identical models
    br, fr = mod.load(str(base), ["MHA_1"]), mod.load(str(fine), ["MHA_1"])
    uids = sorted(set(br) & set(fr))
    d, lo, hi, _ = mod.paired_bootstrap(br, fr, uids, "malignant_vs_rest", 400, seed=1)
    assert d == 0.0
    assert lo <= 0 <= hi


def test_pairing_uses_only_overlapping_cases(mod, tmp_path):
    """A case scored by only one model must not enter the comparison.

    Keys are namespaced by site. Rows returned through the swarm carry a
    row_index that restarts at 0 at every site, so an unnamespaced key would
    pair one site's case with another's and still report a paired result.
    """
    base, fine = tmp_path / "b", tmp_path / "f"
    _write(base, "MHA_1", [_row("a", 2, 0.8), _row("b", 0, 0.2), _row("only_base", 0, 0.9)])
    _write(fine, "MHA_1", [_row("a", 2, 0.8), _row("b", 0, 0.2)])
    br, fr = mod.load(str(base), ["MHA_1"]), mod.load(str(fine), ["MHA_1"])
    uids = sorted(set(br) & set(fr))
    assert uids == ["MHA_1:a", "MHA_1:b"]
    assert "MHA_1:only_base" not in uids


def test_undefined_endpoint_is_reported_not_faked(mod, tmp_path):
    """A cohort with no malignant cases must yield None, never a number."""
    base, fine = tmp_path / "b", tmp_path / "f"
    rows = [_row(f"n{i}", 0, 0.3) for i in range(6)]
    _write(base, "RUMC_1", rows)
    _write(fine, "RUMC_1", rows)
    br, fr = mod.load(str(base), ["RUMC_1"]), mod.load(str(fine), ["RUMC_1"])
    uids = sorted(set(br) & set(fr))
    d, lo, hi, favour = mod.paired_bootstrap(br, fr, uids, "malignant_vs_rest", 100, seed=1)
    assert d is None and lo is None and hi is None and favour is None


def test_multi_site_cohort_pools_cases(mod, tmp_path):
    base, fine = tmp_path / "b", tmp_path / "f"
    for d in (base, fine):
        _write(d, "RUMC_1", [_row("r1", 2, 0.7), _row("r2", 0, 0.2)])
        _write(d, "UMCU_1", [_row("u1", 2, 0.8), _row("u2", 0, 0.1)])
    br = mod.load(str(base), ["RUMC_1", "UMCU_1"])
    assert len(br) == 4, "the Dutch cohort must pool both sites' cases"


# ---- reading rows returned through the swarm ------------------------------

def _write_returned(tmp, site, rows):
    """The layout per_case_predictions writes: <site>.csv, row_index, no uid."""
    with open(tmp / f"{site}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["row_index", "ground_truth",
                                           "prob_class_0", "prob_class_1", "prob_class_2"])
        w.writeheader()
        for i, (gt, p2) in enumerate(rows):
            rest = 1.0 - p2
            w.writerow({"row_index": i, "ground_truth": gt,
                        "prob_class_0": rest * 0.7, "prob_class_1": rest * 0.3,
                        "prob_class_2": p2})


def test_reads_swarm_returned_rows_without_a_uid(mod, tmp_path):
    d = tmp_path / "returned"; d.mkdir()
    _write_returned(d, "UMCU_1", [(2, 0.8), (0, 0.2), (0, 0.1)])
    rows = mod.load(str(d), ["UMCU_1"])
    assert len(rows) == 3
    assert all(k.startswith("UMCU_1:") for k in rows)


def test_row_indices_from_different_sites_do_not_collide(mod, tmp_path):
    """Every site starts its row_index at 0; unnamespaced keys would pair a
    Radboud case with an Utrecht one and report it as a paired comparison."""
    d = tmp_path / "returned"; d.mkdir()
    _write_returned(d, "RUMC_1", [(2, 0.9), (0, 0.1)])
    _write_returned(d, "UMCU_1", [(0, 0.3), (2, 0.7)])
    rows = mod.load(str(d), ["RUMC_1", "UMCU_1"])
    assert len(rows) == 4, "four distinct cases, not two overwritten pairs"
    assert {"RUMC_1:0", "RUMC_1:1", "UMCU_1:0", "UMCU_1:1"} == set(rows)


def test_uid_layout_still_works(mod, tmp_path):
    d = tmp_path / "predicted"
    _write(d, "MHA_1", [_row("case_a", 2, 0.8), _row("case_b", 0, 0.2)])
    rows = mod.load(str(d), ["MHA_1"])
    assert set(rows) == {"MHA_1:case_a", "MHA_1:case_b"}


def test_pairing_across_the_two_layouts_is_possible(mod, tmp_path):
    """Baseline evaluated offline, fine-tuned returned through the swarm."""
    base, fine = tmp_path / "b", tmp_path / "f"
    fine.mkdir()
    _write(base, "MHA_1", [_row(str(i), 2 if i < 3 else 0, 0.5) for i in range(8)])
    _write_returned(fine, "MHA_1", [(2 if i < 3 else 0, 0.9 if i < 3 else 0.1) for i in range(8)])
    br, fr = mod.load(str(base), ["MHA_1"]), mod.load(str(fine), ["MHA_1"])
    uids = sorted(set(br) & set(fr))
    assert len(uids) == 8, "uid '0'..'7' and row_index 0..7 both key to MHA_1:<n>"
    d, lo, hi, favour = mod.paired_bootstrap(br, fr, uids, "malignant_vs_rest", 200, seed=1)
    assert d > 0


# ---- 1.9.0: one returned file per split and model kind ---------------------

def _write_returned_tagged(tmp, site, split, kind, rows, round_num=4):
    """The layout per_case_predictions writes since 1.9.0: '#' provenance lines, then tagged rows."""
    with open(tmp / f"{site}__{split}__{kind}.csv", "w", newline="") as fh:
        fh.write(f"# ODELIA per-case predictions\n# job_id=job-1\n# site={site}\n# split={split}\n"
                 f"# model_kind={kind}\n# round={round_num}\n")
        w = csv.writer(fh)
        w.writerow(["row_index", "ground_truth", "prob_class_0", "prob_class_1", "prob_class_2",
                    "split", "model_kind", "round"])
        for i, (gt, p2) in enumerate(rows):
            rest = 1.0 - p2
            w.writerow([i, gt, rest * 0.7, rest * 0.3, p2, split, kind, round_num])


def test_reads_the_file_of_the_requested_split_and_model_kind(mod, tmp_path):
    d = tmp_path / "returned"; d.mkdir()
    _write_returned_tagged(d, "UMCU_1", "val", "site_local", [(0, 0.1)] * 5)
    _write_returned_tagged(d, "UMCU_1", "val", "final_global", [(1, 0.2)] * 4)
    _write_returned_tagged(d, "UMCU_1", "test", "final_global", [(2, 0.9), (0, 0.3), (2, 0.8)])
    rows = mod.load(str(d), ["UMCU_1"], split="test", model_kind="final_global")
    assert sorted(rows) == ["UMCU_1:0", "UMCU_1:1", "UMCU_1:2"]
    assert [rows[f"UMCU_1:{i}"]["ground_truth"] for i in range(3)] == ["2", "0", "2"]
    assert {r["split"] for r in rows.values()} == {"test"}, "the '#' provenance lines are skipped"


def test_several_returned_files_without_a_choice_is_an_error_not_a_mix(mod, tmp_path):
    d = tmp_path / "returned"; d.mkdir()
    _write_returned_tagged(d, "UMCU_1", "val", "site_local", [(0, 0.1)])
    _write_returned_tagged(d, "UMCU_1", "test", "final_global", [(2, 0.9)])
    with pytest.raises(SystemExit, match="several per-case files for UMCU_1.*choose one with --split"):
        mod.load(str(d), ["UMCU_1"])
    # Either tag alone is enough when it leaves one file.
    assert [r["split"] for r in mod.load(str(d), ["UMCU_1"], split="val").values()] == ["val"]
    assert [r["split"] for r in mod.load(str(d), ["UMCU_1"], model_kind="final_global").values()] == ["test"]


def test_a_pre_190_file_is_read_only_as_local_validation_rows(mod, tmp_path):
    """Up to 1.8 '<site>.csv' held the local model's validation rows; it never stands in for a test split."""
    d = tmp_path / "returned"; d.mkdir()
    _write_returned(d, "MHA_1", [(2, 0.8), (0, 0.2)])
    assert len(mod.load(str(d), ["MHA_1"])) == 2
    assert len(mod.load(str(d), ["MHA_1"], split="val", model_kind="site_local")) == 2
    assert mod.load(str(d), ["MHA_1"], split="test", model_kind="final_global") == {}
    assert mod.load(str(d), ["MHA_1"], model_kind="final_global") == {}


def test_reads_what_the_coordinator_collector_writes(mod, tmp_path, monkeypatch):
    """Round trip: client emission -> PerCasePredictionCollector -> this script, selected by split and kind."""
    pytest.importorskip("nvflare")
    from conftest import SHARED_CUSTOM_DIR
    from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName

    per_case = import_module_from_path("_cmp_per_case", SHARED_CUSTOM_DIR / "per_case_predictions.py")
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")

    class Writer:
        sent = []

        def log_text(self, text, path):
            self.sent.append((path, text))

    writer = Writer()
    per_case.emit_per_case_predictions(writer, [0, 2, 1], [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2]],
                                       split="test", model_kind="final_global", round_num=5)
    per_case.emit_per_case_predictions(writer, [1, 1], [[0.3, 0.4, 0.3]] * 2, split="val",
                                       model_kind="site_local", round_num=4, epoch=9)
    collector = per_case.PerCasePredictionCollector()
    collector.log_info = collector.log_warning = collector.log_error = lambda *args, **kwargs: None
    collector.initialize(None)
    for path, text in writer.sent:
        record = AnalyticsData(key="text", value=text, data_type=AnalyticsDataType.TEXT,
                               sender=LogWriterName.MLFLOW, path=path)
        collector.save(None, record.to_dxo().to_shareable(), "RUMC_1")

    class Ctx:
        def get_job_id(self):
            return "job-7"

        def get_engine(self):
            workspace = type("W", (), {"get_run_dir": lambda self, job_id: str(tmp_path / job_id)})()
            return type("E", (), {"get_workspace": lambda self: workspace})()

    collector.finalize(Ctx())
    out = tmp_path / "job-7" / "cross_site_val" / "per_case_predictions"
    rows = mod.load(str(out), ["RUMC_1"], split="test", model_kind="final_global")
    assert [rows[f"RUMC_1:{i}"]["ground_truth"] for i in range(3)] == ["0", "2", "1"]
    assert float(rows["RUMC_1:1"]["prob_class_2"]) == pytest.approx(0.8)
    assert len(mod.load(str(out), ["RUMC_1"], split="val", model_kind="site_local")) == 2
