"""Unit tests for STAMP cross-site class-order verification.

Swarm aggregation averages the classifier head position by position, so if two
sites infer different class orders the run completes and is silently meaningless.
These tests cover the guard that refuses to start such a run.

Regression: a real DECADE Lynch/Sporadic run aborted mid-flight because two sites
spelled the same label with different capitalisation. Comparison must therefore be
case-insensitive but still order-sensitive.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_CUSTOM_DIR = (
    REPO_ROOT / "application" / "jobs" / "STAMP_classification" / "app" / "custom"
)
if str(STAMP_CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(STAMP_CUSTOM_DIR))

import stamp_training as st  # noqa: E402


# ---------------------------------------------------------------------------
# parse_expected_categories
# ---------------------------------------------------------------------------

def test_parse_expected_categories_basic():
    assert st.parse_expected_categories("no,yes") == ["no", "yes"]
    assert st.parse_expected_categories("Lynch syndrome, Sporadic") == [
        "Lynch syndrome", "Sporadic"
    ]


def test_parse_expected_categories_unset_or_blank_is_none():
    assert st.parse_expected_categories(None) is None
    assert st.parse_expected_categories("") is None
    assert st.parse_expected_categories("   ") is None
    assert st.parse_expected_categories(",, ,") is None


# ---------------------------------------------------------------------------
# categories_match
# ---------------------------------------------------------------------------

def test_categories_match_exact():
    assert st.categories_match(["no", "yes"], ["no", "yes"])


def test_categories_match_is_case_insensitive():
    # The regression: sites disagreeing only on capitalisation must still match.
    assert st.categories_match(["Lynch Syndrome", "sporadic"],
                               ["Lynch syndrome", "Sporadic"])


def test_categories_match_ignores_surrounding_whitespace():
    assert st.categories_match([" no ", "yes\t"], ["no", "yes"])


def test_categories_match_is_order_sensitive():
    # Order is the whole point — position determines which output neuron a class
    # maps to, so a reversed order must NOT be treated as equal.
    assert not st.categories_match(["yes", "no"], ["no", "yes"])


def test_categories_match_different_lengths():
    assert not st.categories_match(["no", "yes", "not provided"], ["no", "yes"])


def test_categories_match_none_inputs():
    assert not st.categories_match(None, ["no", "yes"])
    assert not st.categories_match(["no", "yes"], None)


# ---------------------------------------------------------------------------
# extract_categories
# ---------------------------------------------------------------------------

def test_extract_categories_from_attribute():
    model = MagicMock(spec=["categories"])
    model.categories = ["no", "yes"]
    assert st.extract_categories(model) == ["no", "yes"]


def test_extract_categories_from_hparams_mapping():
    model = MagicMock(spec=["hparams"])
    model.hparams = {"categories": [" no", "yes "]}
    assert st.extract_categories(model) == ["no", "yes"]


def test_extract_categories_returns_none_when_unavailable():
    model = MagicMock(spec=[])
    assert st.extract_categories(model) is None


# ---------------------------------------------------------------------------
# verify_class_order
# ---------------------------------------------------------------------------

def _model_with(categories):
    model = MagicMock(spec=["categories"])
    model.categories = categories
    return model


def test_verify_passes_when_order_matches(monkeypatch):
    monkeypatch.delenv(st.ENV_EXPECTED_CATEGORIES, raising=False)
    observed = st.verify_class_order(
        {"num_classes": 2}, _model_with(["Lynch Syndrome", "sporadic"]),
        expected_categories=["Lynch syndrome", "Sporadic"],
    )
    assert observed == ["Lynch Syndrome", "sporadic"]


def test_verify_raises_on_reversed_order():
    with pytest.raises(ValueError, match="Class-order mismatch"):
        st.verify_class_order(
            {"num_classes": 2}, _model_with(["yes", "no"]),
            expected_categories=["no", "yes"],
        )


def test_verify_raises_when_extra_label_widens_the_head():
    # A stray third label ('not provided') is the other real-world failure mode.
    with pytest.raises(ValueError, match="inferred 3 classes"):
        st.verify_class_order(
            {"num_classes": 2}, _model_with(["no", "yes", "not provided"]),
        )


def test_verify_reads_expected_order_from_environment(monkeypatch):
    monkeypatch.setenv(st.ENV_EXPECTED_CATEGORIES, "no, yes")
    st.verify_class_order({"num_classes": 2}, _model_with(["No", "Yes"]))
    monkeypatch.setenv(st.ENV_EXPECTED_CATEGORIES, "yes, no")
    with pytest.raises(ValueError, match="Class-order mismatch"):
        st.verify_class_order({"num_classes": 2}, _model_with(["No", "Yes"]))


def test_verify_is_opt_in_when_env_unset(monkeypatch):
    # Unset => log only, never raise, so existing runs are unaffected.
    monkeypatch.delenv(st.ENV_EXPECTED_CATEGORIES, raising=False)
    assert st.verify_class_order(
        {"num_classes": 2}, _model_with(["anything", "goes"])
    ) == ["anything", "goes"]


def test_verify_returns_none_when_order_unreadable(monkeypatch):
    monkeypatch.delenv(st.ENV_EXPECTED_CATEGORIES, raising=False)
    assert st.verify_class_order({"num_classes": 2}, MagicMock(spec=[])) is None
