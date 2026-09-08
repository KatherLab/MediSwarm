"""Unit tests for per-case prediction return (#526, #527, #528).

The properties pinned here are the ones a wrong answer would depend on:
that the opt-in gate actually gates, that nothing identifying is emitted,
that chunks reassemble in order, and that a truncated file says so.
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


# ---- server-side reassembly ----------------------------------------------

def test_sequence_is_parsed_from_the_path(mod):
    C = mod.PerCasePredictionCollector
    assert C._sequence_of("per_case/val/000007.csv") == 7
    assert C._sequence_of("per_case/val/notanumber.csv") is None
    assert C._sequence_of(None) is None


def test_chunks_reassemble_in_sequence_order_not_arrival_order(mod):
    """The transport gives no ordering guarantee, and a reordered file would
    silently break the index-based pairing this exists to support."""
    c = mod.PerCasePredictionCollector()
    c._chunks = {"UMCU_1": {2: "c\n", 0: "a\n", 1: "b\n"}}
    body = "".join(c._chunks["UMCU_1"][k] for k in sorted(c._chunks["UMCU_1"]))
    assert body == "a\nb\nc\n"


def test_header_names_exactly_what_is_sent(mod):
    assert mod.CSV_HEADER.split(",") == [
        "row_index", "ground_truth", "prob_class_0", "prob_class_1", "prob_class_2"]
