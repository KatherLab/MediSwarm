"""The two provisioning templates must reference the same NVFlare classes.

`master_template.yml` (ODELIA) and `master_template_STAMP.yml` (DECADE) both describe
the same NVFlare infrastructure -- resource managers, process launchers, log plumbing,
the server/client config. Only the training application differs.

The NVFlare 2.8.0 bump (#392/#462) updated the ODELIA template and missed the STAMP
one, so DECADE kits kept pointing at 2.7.2 paths that no longer exist:

    nvflare.app_common.logging.log_receiver.LogReceiver      (renamed)
    nvflare.app_common.logging.log_sender.ErrorLogSender     (renamed)
    nvflare.ha.dummy_overseer_agent.DummyOverseerAgent       (overseer removed in 2.8.0)

Every DECADE server built from that template died on startup with a ConfigError, and it
was caught only by a 2-node hardware run during the 1.6.0 rollout -- after the image had
been built and pushed. This test makes the next such divergence fail in seconds instead.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ODELIA = REPO_ROOT / "docker_config" / "master_template.yml"
STAMP = REPO_ROOT / "docker_config" / "master_template_STAMP.yml"

PATH_RE = re.compile(r'"path":\s*"(nvflare\.[^"]+)"')


def _nvflare_paths(template: Path) -> set[str]:
    return set(PATH_RE.findall(template.read_text()))


def test_stamp_declares_no_nvflare_class_odelia_does_not():
    """A path only STAMP references is unreviewed against the current backbone.

    Both templates are provisioned against the same image, so any NVFlare class the
    STAMP template names on its own has, by construction, never been exercised by the
    ODELIA deploy test. That is exactly how the 2.8.0 skew survived.
    """
    stamp_only = _nvflare_paths(STAMP) - _nvflare_paths(ODELIA)
    assert not stamp_only, (
        "master_template_STAMP.yml references NVFlare classes that "
        "master_template.yml does not:\n  "
        + "\n  ".join(sorted(stamp_only))
        + "\nIf the backbone moved, update BOTH templates."
    )


def test_no_known_removed_nvflare_paths():
    """Guard the specific 2.8.0 removals, so a revert is caught by name."""
    removed = [
        "nvflare.app_common.logging.log_receiver.LogReceiver",
        "nvflare.app_common.logging.log_sender.ErrorLogSender",
        "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent",
    ]
    for template in (ODELIA, STAMP):
        text = template.read_text()
        for path in removed:
            assert path not in text, (
                f"{template.name} still references {path}, which NVFlare 2.8.0 removed."
            )


def test_overseer_agent_block_is_gone_from_both():
    """NVFlare 2.8.0 dropped the overseer; a leftover block fails server startup."""
    for template in (ODELIA, STAMP):
        assert '"overseer_agent"' not in template.read_text(), (
            f"{template.name} still declares an overseer_agent block; "
            "NVFlare 2.8.0 removed the overseer."
        )


# The NVFlare wiring blocks are backbone configuration, identical for both pipelines --
# only the training application differs between ODELIA and DECADE. Comparing whole
# blocks (not just class paths) is what catches a migration that adds or removes a
# *field*: dropping the overseer left the STAMP clients with no way to find the server,
# and 2.8.0 wants a "target" instead. That is not a class path, so path-parity alone
# missed it and every client died with:
#   RuntimeError: missing 'target' in server config ... provisioned with an older
#   HA-based template
SHARED_CONFIG_BLOCKS = [
    "fed_client",
    "fed_server",
    "local_client_resources",
    "local_server_resources",
    "comm_config",
]


def test_shared_nvflare_config_blocks_are_identical():
    yaml = __import__("pytest").importorskip("yaml")
    odelia = yaml.safe_load(ODELIA.read_text())
    stamp = yaml.safe_load(STAMP.read_text())

    mismatched = [
        key
        for key in SHARED_CONFIG_BLOCKS
        if key in odelia and key in stamp and odelia[key] != stamp[key]
    ]
    assert not mismatched, (
        "These NVFlare config blocks differ between master_template.yml and "
        f"master_template_STAMP.yml: {mismatched}.\n"
        "They configure the shared backbone and must stay in lockstep -- a backbone "
        "migration that updates only one of them ships broken kits to that consortium."
    )
