"""Startup kits are packaged as plain zips for a trusted consortium.

Each kit contains that site's startup/client.key. The ODELIA consortium distributes
kits through a members-only shared folder where every site picks its own, so kits are
NOT encrypted -- a consortium decision (2026-07-16) that reversed the per-site AES
encryption from #449. The archiver still records a sha256 per kit so a site can
confirm which kit it is on, and it must never silently fall back to broken ZipCrypto.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVER = REPO_ROOT / "scripts" / "build" / "_generateStartupKitArchives.sh"


def _script():
    return ARCHIVER.read_text()


def test_archiver_is_valid_bash():
    result = subprocess.run(["bash", "-n", str(ARCHIVER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_kits_are_plain_zips_not_encrypted():
    """The consortium shares a members-only folder; kits ship as plain zips."""
    script = _script()
    assert "openssl enc" not in script, "kits are distributed unencrypted by consortium decision"
    assert "-aes-256-cbc" not in script


def test_no_password_file_is_produced():
    """Per-site passwords were dropped together with the encryption."""
    script = _script()
    assert "kit_passwords.txt" not in script
    assert "openssl rand" not in script


def test_zipcrypto_is_not_used():
    """`zip -e` / `zip -P` is ZipCrypto -- broken; it must not masquerade as protection."""
    script = _script()
    assert not re.search(r"zip\s+[^\n|]*-(e|P)\b", script), "ZipCrypto must not be used"


def test_manifest_publishes_a_checksum_per_kit():
    """So a site can answer 'which kit am I actually on?' against the registry."""
    script = _script()
    assert "sha256sum" in script
    assert "site,version,file,sha256" in script


def test_the_kit_archive_is_named_per_site_and_version():
    """A site must be able to spot its own kit and its version in the shared folder."""
    assert re.search(r'archive="\$\{startupkit\}_\$\{LONG_VERSION\}\.zip"', _script())
