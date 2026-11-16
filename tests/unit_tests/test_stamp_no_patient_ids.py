"""No patient identifiers in what a STAMP swarm run uploads: its logs and TensorBoard files.

live_sync uploads a client's nohup.out, log.txt and the run's TensorBoard event files to the
monitoring host. STAMP 2.4.0 lists patient identifiers in four data-consistency warnings, and
its Lightning model keeps ``train_patients``/``valid_patients`` as hyperparameters, which the
TensorBoard logger writes into hparams.yaml and into the event files (every patient ID of a
site, with its train/validation assignment).
"""

import argparse
import logging
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

IDS = ["DA-X-0001", "DA-X-0002", "DA-X-0003"]


def _logged(logger, message, *args):
    """What a handler on `logger` receives for one warning."""
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Keep()
    logger.addHandler(handler)
    try:
        logger.warning(message, *args)
    finally:
        logger.removeHandler(handler)
    return records[0]


@pytest.fixture
def stamp_logger():
    logger = logging.getLogger("stamp.test_no_patient_ids")
    logger.propagate = False
    logger.filters.clear()
    st.redact_stamp_identifier_warnings(logger)
    return logger


@pytest.mark.parametrize("message, expected", [
    (f"some patients have no clinical information: {set(IDS)}",
     "some patients have no clinical information: 3 (identifiers not logged)"),
    (f"some patients have no associated slides: {set(IDS[:2])}",
     "some patients have no associated slides: 2 (identifiers not logged)"),
    ("some feature files could not be found: {PosixPath('/data/S/DA-X-0001_A.h5'), PosixPath('/data/S/DA-X-0002.h5')}",
     "some feature files could not be found: 2 (identifiers not logged)"),
    (f"Some patients have no feature file in /data/S/features: {IDS}",
     "Some patients have no feature file in /data/S/features: 3 (identifiers not logged)"),
])
def test_identifier_warnings_keep_only_the_count(stamp_logger, message, expected):
    logged = _logged(stamp_logger, message)
    assert logged == expected
    assert not any(pid in logged for pid in IDS)


def test_percent_style_arguments_are_redacted_too(stamp_logger):
    logged = _logged(stamp_logger, "some patients have no clinical information: %s", set(IDS))
    assert logged == "some patients have no clinical information: 3 (identifiers not logged)"


def test_an_unexpected_format_keeps_only_the_fixed_wording(stamp_logger):
    logged = _logged(stamp_logger, "some patients have no clinical information - DA-X-0001")
    assert logged == "some patients have no clinical information: 0 (identifiers not logged)"


def test_other_messages_are_unchanged(stamp_logger):
    message = "Kept 1741/1741 patients with complete data (100.0%)."
    assert _logged(stamp_logger, message) == message


def test_the_filter_is_attached_once(stamp_logger):
    st.redact_stamp_identifier_warnings(stamp_logger)
    st.redact_stamp_identifier_warnings(stamp_logger)
    assert sum(isinstance(f, st.RedactIdentifiersFilter) for f in stamp_logger.filters) == 1


def test_importing_the_job_code_protects_stamps_own_logger():
    assert any(isinstance(f, st.RedactIdentifiersFilter) for f in logging.getLogger("stamp").filters)


@pytest.mark.parametrize("params", [
    {"train_patients": IDS[:2], "valid_patients": IDS[2:], "max_lr": 1e-4},
    argparse.Namespace(train_patients=IDS[:2], valid_patients=IDS[2:], max_lr=1e-4),
])
def test_patient_lists_become_counts(params):
    clean = st._without_patient_lists(params)
    assert clean["train_patients"] == "2 patients (identifiers not logged)"
    assert clean["valid_patients"] == "1 patients (identifiers not logged)"
    assert clean["max_lr"] == 1e-4
    assert not any(pid in repr(clean) for pid in IDS)


def test_tensorboard_files_carry_no_patient_ids(tmp_path):
    pytest.importorskip("tensorboard")
    tb = st.PatientListFreeTensorBoardLogger(save_dir=tmp_path)
    tb.log_hyperparams({"train_patients": IDS[:2], "valid_patients": IDS[2:], "max_lr": 1e-4})
    tb.finalize("success")
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert any(p.name == "hparams.yaml" for p in written)
    assert any(p.name.startswith("events.out.tfevents") for p in written)
    for path in written:
        data = path.read_bytes()
        assert not any(pid.encode() in data for pid in IDS), path
