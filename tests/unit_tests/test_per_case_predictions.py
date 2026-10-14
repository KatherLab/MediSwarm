"""Unit tests for per-case prediction return (#526, #527, #528).

The properties pinned here are the ones a wrong answer would depend on:
that the opt-in gate actually gates, that nothing identifying is emitted,
that chunks reassemble in order, that a truncated file says so, and (1.9.0)
that rows of different splits or model kinds never end up in the same file.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import SHARED_CUSTOM_DIR, import_module_from_path  # noqa: E402

pytest.importorskip("nvflare")
MODULE = SHARED_CUSTOM_DIR / "per_case_predictions.py"


@pytest.fixture(scope="module")
def mod():
    return import_module_from_path("_per_case_predictions", MODULE)


class _Writer:
    def __init__(self): self.sent = []
    def log_text(self, text, artifact_file_path): self.sent.append((artifact_file_path, text))


class _Workspace:
    def __init__(self, root): self.root = root
    def get_run_dir(self, job_id): return str(self.root / job_id)


class _Engine:
    def __init__(self, root): self.workspace = _Workspace(root)
    def get_workspace(self): return self.workspace


class _Ctx:
    """The two things finalize() needs from an FLContext."""

    def __init__(self, root, job_id="job-42"):
        self.engine, self.job_id = _Engine(root), job_id
    def get_engine(self): return self.engine
    def get_job_id(self): return self.job_id


def _collector(mod, **kwargs):
    c = mod.PerCasePredictionCollector(**kwargs)
    c.logged = []
    c.log_info = lambda ctx, msg, **k: c.logged.append(("info", msg))
    c.log_warning = lambda ctx, msg, **k: c.logged.append(("warning", msg))
    c.log_error = lambda ctx, msg, **k: c.logged.append(("error", msg))
    c.initialize(None)
    return c


def _record(path, text):
    """The shareable MLflowWriter.log_text() puts on the wire."""
    from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName

    return AnalyticsData(key="text", value=text, data_type=AnalyticsDataType.TEXT,
                         sender=LogWriterName.MLFLOW, path=path).to_dxo().to_shareable()


def _send(c, site, writer):
    for path, text in writer.sent:
        c.save(None, _record(path, text), site)


def _read(path):
    lines = Path(path).read_text().splitlines()
    header = dict(line[2:].split("=", 1) for line in lines if line.startswith("# ") and "=" in line)
    rows = [line.split(",") for line in lines if not line.startswith("#")]
    return header, rows[0], rows[1:]


# ---- the opt-in gate ------------------------------------------------------

def test_nothing_is_emitted_without_explicit_opt_in(mod, monkeypatch):
    """Patient-derived data must not leave a site by default."""
    monkeypatch.delenv("ODELIA_RETURN_PER_CASE", raising=False)
    w = _Writer()
    assert mod.emit_per_case_predictions(w, [0, 1], [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1]]) == 0
    assert w.sent == []


def test_a_value_other_than_1_does_not_opt_in(mod, monkeypatch):
    for val in ("0", "true", "yes", ""):
        monkeypatch.setenv("ODELIA_RETURN_PER_CASE", val)
        w = _Writer()
        assert mod.emit_per_case_predictions(w, [0], [[1.0, 0.0, 0.0]]) == 0, f"opted in on {val!r}"


def test_opt_in_emits(mod, monkeypatch):
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")
    w = _Writer()
    n = mod.emit_per_case_predictions(w, [0, 2], [[0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
    assert n == 1 and len(w.sent) == 1
    assert w.sent[0][0].startswith("per_case/val/")


# ---- what leaves the site -------------------------------------------------

def test_rows_carry_no_identifier(mod):
    rows = mod.per_case_rows([0, 1], [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1]])
    assert rows[0].split(",")[0] == "0"
    assert rows[1].split(",")[0] == "1"
    for r in rows:
        assert len(r.split(",")) == 5, "index, ground truth, three probabilities -- nothing else"


def test_row_index_is_positional_only(mod):
    """The index must be a position, not anything carried in from the data."""
    rows = mod.per_case_rows([2, 2, 2], [[0.1, 0.2, 0.7]] * 3)
    assert [r.split(",")[0] for r in rows] == ["0", "1", "2"]


# ---- chunking -------------------------------------------------------------

def test_chunking_never_splits_a_row(mod):
    rows = [f"{i},0,0.100000,0.200000,0.700000" for i in range(500)]
    chunks = mod.chunk_rows(rows, max_bytes=200)
    assert len(chunks) > 1
    rejoined = "".join(chunks).strip().split("\n")
    assert rejoined == rows


def test_a_single_oversized_row_still_goes(mod):
    big = "0,1," + ",".join(["0.123456"] * 500)
    assert mod.chunk_rows([big], max_bytes=10) == [big + "\n"]


def test_empty_input_produces_no_chunks(mod):
    assert mod.chunk_rows([]) == []


# ---- tags travel in the chunk path, never in the rows --------------------

def test_chunk_path_round_trips_every_tag(mod):
    path = mod.chunk_path("test", "final_global", 20, None, 3, 1, 4)
    assert path == "per_case/test/final_global/round_20/epoch_na/emission_3/000001_of_000004.csv"
    assert mod.parse_chunk_path(path) == {"split": "test", "model_kind": "final_global", "round": 20,
                                          "epoch": None, "emission": 3, "seq": 1, "count": 4}


@pytest.mark.parametrize("path", [
    None, "", "per_case/val/000007.csv",                                      # the pre-1.9 layout
    "per_case/train/site_local/round_1/epoch_2/emission_0/000000_of_000001.csv",  # unknown split
    "per_case/val/initial_global/round_1/epoch_2/emission_0/000000_of_000001.csv",  # unknown kind
    "per_case/val/site_local/round_1/epoch_2/emission_0/000001_of_000001.csv",  # seq beyond count
    "per_case/val/site_local/round_x/epoch_2/emission_0/000000_of_000001.csv",
])
def test_malformed_chunk_paths_are_rejected(mod, path):
    assert mod.parse_chunk_path(path) is None


def test_emission_is_tagged_and_numbered(mod, monkeypatch):
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")
    w = _Writer()
    mod.emit_per_case_predictions(w, [0], [[1.0, 0.0, 0.0]], split="val", model_kind="site_local",
                                  round_num=4, epoch=17)
    mod.emit_per_case_predictions(w, [0], [[1.0, 0.0, 0.0]], split="test", model_kind="final_global",
                                  round_num=5)
    first, second = (mod.parse_chunk_path(path) for path, _ in w.sent)
    assert (first["split"], first["model_kind"], first["round"], first["epoch"]) == ("val", "site_local", 4, 17)
    assert (second["split"], second["model_kind"], second["round"]) == ("test", "final_global", 5)
    assert second["emission"] > first["emission"], "later emissions must be recognisable as newer"


def test_unknown_split_or_kind_is_a_programming_error(mod, monkeypatch):
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")
    with pytest.raises(ValueError):
        mod.emit_per_case_predictions(_Writer(), [0], [[1.0, 0.0, 0.0]], split="train")
    with pytest.raises(ValueError):
        mod.emit_per_case_predictions(_Writer(), [0], [[1.0, 0.0, 0.0]], model_kind="best_global")


# ---- server-side reassembly ----------------------------------------------

def _rows(gts, p2):
    return [[1.0 - p2, 0.0, p2] for _ in gts]


def test_one_file_per_split_and_kind_with_only_the_latest_complete_emission(mod, monkeypatch, tmp_path):
    """Two epochs of local validation rows, then the final global's val and test rows: three files, no
    cross-contamination, each from its newest emission, chunks reassembled in sequence order."""
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")
    monkeypatch.setattr(mod, "chunk_rows", lambda rows, max_bytes=0: [r + "\n" for r in rows])  # 1 row/chunk
    w = _Writer()
    mod.emit_per_case_predictions(w, [0, 1, 2], _rows([0, 1, 2], 0.1), "val", "site_local", 3, 10)
    mod.emit_per_case_predictions(w, [0, 1, 2], _rows([0, 1, 2], 0.2), "val", "site_local", 3, 11)
    mod.emit_per_case_predictions(w, [2, 2], _rows([2, 2], 0.7), "val", "final_global", 4)
    mod.emit_per_case_predictions(w, [1, 0, 0, 2], _rows([1, 0, 0, 2], 0.9), "test", "final_global", 4)
    w.sent.reverse()  # the transport gives no ordering guarantee

    c = _collector(mod)
    _send(c, "UMCU_1", w)
    ctx = _Ctx(tmp_path)
    c.finalize(ctx)
    c.finalize(ctx)  # ABOUT_TO_END_RUN and END_RUN: written once

    out = tmp_path / "job-42" / "cross_site_val" / "per_case_predictions"
    assert sorted(p.name for p in out.iterdir()) == [
        "UMCU_1__test__final_global.csv", "UMCU_1__val__final_global.csv", "UMCU_1__val__site_local.csv"]
    header, columns, rows = _read(out / "UMCU_1__val__site_local.csv")
    assert columns == ["row_index", "ground_truth", "prob_class_0", "prob_class_1", "prob_class_2",
                       "split", "model_kind", "round"]
    assert [r[0] for r in rows] == ["0", "1", "2"], "sequence order, not arrival order"
    assert {r[4] for r in rows} == {"0.200000"}, "the epoch-11 emission replaced epoch 10's rows"
    assert {(r[5], r[6], r[7]) for r in rows} == {("val", "site_local", "3")}
    assert (header["job_id"], header["site"], header["split"], header["model_kind"]) == (
        "job-42", "UMCU_1", "val", "site_local")
    assert (header["round"], header["epoch"], header["rows"]) == ("3", "11", "3")
    assert header["complete"].startswith("true")

    _, _, test_rows = _read(out / "UMCU_1__test__final_global.csv")
    assert [r[1] for r in test_rows] == ["1", "0", "0", "2"]
    assert {(r[5], r[6], r[7]) for r in test_rows} == {("test", "final_global", "4")}
    _, _, val_rows = _read(out / "UMCU_1__val__final_global.csv")
    assert len(val_rows) == 2
    assert sum(1 for kind, msg in c.logged if msg.startswith("Wrote")) == 3


def test_an_incomplete_newer_emission_does_not_replace_a_complete_one(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")
    monkeypatch.setattr(mod, "chunk_rows", lambda rows, max_bytes=0: [r + "\n" for r in rows])
    w = _Writer()
    mod.emit_per_case_predictions(w, [0, 1], _rows([0, 1], 0.1), "val", "site_local", 1, 0)
    mod.emit_per_case_predictions(w, [0, 1], _rows([0, 1], 0.2), "val", "site_local", 1, 1)
    del w.sent[-1]  # the newest emission lost a chunk on the way
    c = _collector(mod)
    _send(c, "MHA_1", w)
    c.finalize(_Ctx(tmp_path))
    header, _, rows = _read(tmp_path / "job-42" / "cross_site_val" / "per_case_predictions"
                            / "MHA_1__val__site_local.csv")
    assert {r[4] for r in rows} == {"0.100000"} and header["epoch"] == "0"
    assert header["complete"].startswith("true")


def test_an_incomplete_file_says_so(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")
    monkeypatch.setattr(mod, "chunk_rows", lambda rows, max_bytes=0: [r + "\n" for r in rows])
    w = _Writer()
    mod.emit_per_case_predictions(w, [0, 1, 2], _rows([0, 1, 2], 0.3), "test", "final_global", 2)
    del w.sent[1]
    c = _collector(mod)
    _send(c, "RUMC_1", w)
    c.finalize(_Ctx(tmp_path))
    header, _, rows = _read(tmp_path / "job-42" / "cross_site_val" / "per_case_predictions"
                            / "RUMC_1__test__final_global.csv")
    assert header["complete"].startswith("false") and len(rows) == 2
    assert any("INCOMPLETE" in msg for _, msg in c.logged)


def test_the_per_site_chunk_cap_still_holds(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("ODELIA_RETURN_PER_CASE", "1")
    monkeypatch.setattr(mod, "chunk_rows", lambda rows, max_bytes=0: [r + "\n" for r in rows])
    w = _Writer()
    mod.emit_per_case_predictions(w, list(range(5)), _rows(range(5), 0.5), "val", "final_global", 0)
    c = _collector(mod, max_chunks_per_site=3)
    _send(c, "CAM_1", w)
    c.finalize(_Ctx(tmp_path))
    header, _, rows = _read(tmp_path / "job-42" / "cross_site_val" / "per_case_predictions"
                            / "CAM_1__val__final_global.csv")
    assert len(rows) == 3 and header["complete"].startswith("false")
    assert header["dropped_chunks"].startswith("2")


def test_records_that_are_not_per_case_chunks_are_ignored(mod, tmp_path):
    c = _collector(mod)
    c.save(None, _record("per_case/val/000000.csv", "0,1,0.1,0.2,0.7\n"), "UKA_1")   # pre-1.9 layout
    c.save(None, _record("plots/loss.txt", "x"), "UKA_1")
    c.save(None, "not a shareable", "UKA_1")
    c.finalize(_Ctx(tmp_path))
    assert not (tmp_path / "job-42").exists()
    assert any("unparseable path" in msg for _, msg in c.logged)


def test_row_columns_name_exactly_what_is_sent(mod):
    assert mod.ROW_COLUMNS.split(",") == [
        "row_index", "ground_truth", "prob_class_0", "prob_class_1", "prob_class_2"]
    assert mod.CSV_HEADER == mod.ROW_COLUMNS + ",split,model_kind,round"
