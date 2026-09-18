"""Unit tests for the scheduler-horizon resolver (#503).

STAMP sizes its OneCycleLR scheduler once, for max_epochs x total_rounds steps,
while the rounds actually executed come from num_rounds in the job's
config_fed_server.conf. When those disagreed, training died mid-run with
"Tried to step N times. The specified number of total steps is M" — an error
naming neither value. The job's config is now the single source of truth.
"""

import sys
from pathlib import Path

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


def _server_conf(tmp_path, num_rounds):
    p = tmp_path / "config_fed_server.conf"
    p.write_text(
        "format_version = 2\n"
        "workflows = [\n  {\n    args {\n"
        f"      num_rounds = {num_rounds}\n"
        "      start_task_timeout = 1800\n    }\n  }\n]\n"
    )
    return p


# ---------------------------------------------------------------------------
# parse_num_rounds
# ---------------------------------------------------------------------------

def test_parse_num_rounds_reads_the_value(tmp_path):
    assert st.parse_num_rounds(_server_conf(tmp_path, 20).read_text()) == 20


def test_parse_num_rounds_ignores_other_numeric_keys(tmp_path):
    text = "start_task_timeout = 1800\nnum_rounds = 3\nprogress_timeout = 28800\n"
    assert st.parse_num_rounds(text) == 3


def test_parse_num_rounds_absent_or_empty():
    assert st.parse_num_rounds("workflows = []\n") is None
    assert st.parse_num_rounds("") is None
    assert st.parse_num_rounds(None) is None


# ---------------------------------------------------------------------------
# resolve_total_rounds — precedence
# ---------------------------------------------------------------------------

def test_job_config_wins_over_environment(tmp_path):
    # The exact 1.7.0 failure: env said 2, the job ran 20.
    assert st.resolve_total_rounds(
        env_value="2", server_config=_server_conf(tmp_path, 20)
    ) == 20


def test_job_config_used_when_env_unset(tmp_path):
    assert st.resolve_total_rounds(
        env_value=None, server_config=_server_conf(tmp_path, 7)
    ) == 7


def test_env_used_when_job_config_missing(tmp_path):
    missing = tmp_path / "nope.conf"
    assert st.resolve_total_rounds(env_value="5", server_config=missing) == 5


def test_default_when_neither_available(tmp_path):
    missing = tmp_path / "nope.conf"
    assert st.resolve_total_rounds(
        env_value=None, server_config=missing, default=11
    ) == 11


def test_non_integer_env_is_ignored(tmp_path):
    missing = tmp_path / "nope.conf"
    assert st.resolve_total_rounds(
        env_value="not-a-number", server_config=missing, default=9
    ) == 9


def test_agreeing_values_resolve_cleanly(tmp_path):
    assert st.resolve_total_rounds(
        env_value="20", server_config=_server_conf(tmp_path, 20)
    ) == 20


def test_mismatch_is_logged_loudly(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        st.resolve_total_rounds(env_value="2", server_config=_server_conf(tmp_path, 20))
    assert any("disagrees with the job" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# find_server_config
# ---------------------------------------------------------------------------

def test_find_server_config_honours_explicit_path(tmp_path):
    conf = _server_conf(tmp_path, 4)
    assert st.find_server_config(conf) == conf


def test_find_server_config_falls_back_when_explicit_path_missing(tmp_path):
    # A missing explicit path must not be returned; the resolver falls through to
    # its other candidates (which may or may not exist in a test checkout).
    missing = tmp_path / "definitely_missing.conf"
    assert st.find_server_config(missing) != missing
