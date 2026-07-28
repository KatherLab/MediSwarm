"""The client `docker.sh` must be able to pass INSTITUTION into the container.

`SITE_NAME` is substituted at kit-generation time and is the NVFlare client identity --
it has to match the certificate, so it cannot be changed per run. But the data loader
reads `/data/$INSTITUTION/...`, and `INSTITUTION` defaults to `SITE_NAME`
(`env_config.py`, `threedcnn_ptl.py`). A site whose data folder differs from its swarm
identity therefore has no way to select it.

That is not hypothetical: UKA registers as `UKA_1` but was asked to train on the curated
`UKA_2` tree. `export INSTITUTION=UKA_2` looked like it should work and silently did
nothing -- only variables named in the template's `--env` list cross the container
boundary, and `INSTITUTION` was not one of them (#478). The run kept reading
`/data/UKA_1`, failed with an empty split, and the error named an institution the site
had not selected. Two people spent a morning on it.

These tests pin the passthrough (and the accompanying same-path data mount, #481) so it
cannot be dropped again.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "docker_config" / "master_template.yml"


def _client_script() -> str:
    return yaml.safe_load(TEMPLATE.read_text())["docker_cln_sh"]


def test_institution_is_forwarded_into_the_container():
    """Only vars explicitly listed as --env reach the container."""
    assert "--env INSTITUTION=" in _client_script()


def test_institution_accepts_a_cli_flag():
    """A CLI flag survives `sudo`; an exported variable does not."""
    script = _client_script()
    assert "--institution)" in script
    assert "CLI_INSTITUTION" in script


def test_cli_institution_takes_precedence_over_the_environment():
    assert "${CLI_INSTITUTION:-${INSTITUTION:-}}" in _client_script()


def test_site_name_remains_the_baked_swarm_identity():
    """INSTITUTION must not be wired to SITE_NAME -- the identity is fixed by the cert."""
    assert "--env SITE_NAME={~~client_name~~}" in _client_script()


def test_data_dir_is_also_mounted_at_its_own_host_path():
    """Absolute symlinks inside the data tree only resolve if that path exists too (#481)."""
    assert "-v $MY_DATA_DIR:$MY_DATA_DIR:ro" in _client_script()
