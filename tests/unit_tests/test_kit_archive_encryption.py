"""Startup kits must not be shareable in the clear (they contain a private key).

Each kit contains that site's startup/client.key. They used to be packaged as plain
zips, so putting them anywhere shared would have exposed every site's credentials to
every other site.

Note the archiver deliberately does NOT use a password-protected zip: standard zip
encryption is ZipCrypto, which is broken by a known-plaintext attack and would not
actually protect a private key on a shared link.
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


def test_kits_are_encrypted_with_aes256():
    script = _script()
    assert "-aes-256-cbc" in script, "kits must be AES-256 encrypted"
    assert "-pbkdf2" in script, "a raw password without PBKDF2 is weak key derivation"
    assert "-salt" in script


def test_key_derivation_is_expensive_enough():
    match = re.search(r"PBKDF2_ITER=(\d+)", _script())
    assert match, "PBKDF2 iteration count must be explicit"
    assert int(match.group(1)) >= 100_000, "PBKDF2 iterations too low to resist offline cracking"


def test_zipcrypto_is_not_used():
    """`zip -e` / `zip -P` is ZipCrypto -- broken, and worse than useless here."""
    script = _script()
    assert not re.search(r"zip\s+[^\n|]*-(e|P)\b", script), "ZipCrypto must not be used"


def test_the_plaintext_archive_is_deleted():
    """The intermediate .zip holds the private key; it must not survive."""
    assert re.search(r'rm -f "\$archive"', _script())


def test_each_site_gets_its_own_random_password():
    script = _script()
    assert "openssl rand" in script, "passwords must be randomly generated"
    # generated inside the per-site loop, not once for all sites
    loop_body = script.split("for startupkit in", 1)[1]
    assert "openssl rand" in loop_body, "one shared password would defeat per-site isolation"


def test_password_file_is_not_world_readable():
    assert "umask 077" in _script() or "chmod 600" in _script()


def test_passwords_are_gitignored():
    ignored = (REPO_ROOT / ".gitignore").read_text()
    assert "kit_passwords.txt" in ignored


def test_manifest_publishes_a_checksum_per_kit():
    """So a site can answer 'which kit am I actually on?' against the registry."""
    script = _script()
    assert "sha256sum" in script
    assert "site,version,file,sha256" in script
