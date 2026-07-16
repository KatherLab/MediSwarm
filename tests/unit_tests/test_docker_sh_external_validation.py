"""docker.sh must offer a first-class predict-only external-validation mode (#412).

A center can validate a released global model on its OWN data without joining the
swarm -- no VPN, no aggregation, no network. The capability existed only ad-hoc
(predict.py driven by an internal smoke script); this makes it a documented kit mode:
`--external_validation` runs predict.py on the local test/ext split against a delivered
checkpoint, writing metrics + predictions locally (only the metrics need to be shared).
"""

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"
WRAPPER = REPO_ROOT / "scripts" / "deploy" / "run_external_validation.sh"


@pytest.fixture(scope="module")
def docker_sh():
    template = yaml.safe_load(TEMPLATE.read_text())
    return re.sub(r"\{~~[a-z_]+~~\}", "TESTSITE", template["docker_cln_sh"])


def test_external_validation_flag_is_accepted(docker_sh):
    assert "--external_validation)" in docker_sh


def test_it_runs_predict_only_not_the_swarm(docker_sh):
    """External validation must invoke predict.py, and must NOT join the swarm."""
    branch = docker_sh[docker_sh.index('elif [ -n "$EXTERNAL_VALIDATION" ]'):]
    branch = branch[: branch.index("elif ", 1)]
    assert "scripts/evaluation/predict.py" in branch
    assert "TRAINING_MODE=swarm" not in branch, "external validation must not join the swarm"
    assert "start.sh" not in branch, "external validation must not launch the swarm client"


def test_checkpoint_defaults_to_a_scratch_path(docker_sh):
    """The site drops the delivered model in scratch (mounted at /scratch)."""
    assert 'EV_CHECKPOINT="${CLI_CHECKPOINT:-/scratch/FL_global_model.pt}"' in docker_sh


def test_split_defaults_to_test(docker_sh):
    assert 'EV_SPLIT="${CLI_SPLIT:-test}"' in docker_sh


def test_output_stays_local_under_scratch(docker_sh):
    branch = docker_sh[docker_sh.index('elif [ -n "$EXTERNAL_VALIDATION" ]'):]
    branch = branch[: branch.index("elif ", 1)]
    assert "--output-dir /scratch" in branch, "predictions must be written to the local scratch dir"


def test_external_validation_runs_the_preflight_host_checks(docker_sh):
    """It uses the GPU + reads real data, so it gets the same host preflight as training."""
    assert re.search(
        r'if \[ -n "\$DUMMY_TRAINING" \].*\$EXTERNAL_VALIDATION.*then\s*\n\s*_preflight_host_checks',
        docker_sh,
    )


def test_generated_script_is_valid_bash(docker_sh):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(docker_sh)
        path = handle.name
    result = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_one_command_wrapper_exists_and_is_valid_bash():
    assert WRAPPER.is_file(), "scripts/deploy/run_external_validation.sh missing"
    result = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = WRAPPER.read_text()
    assert "--preflight_check" in text and "--external_validation" in text, (
        "the wrapper must chain preflight then external validation"
    )
