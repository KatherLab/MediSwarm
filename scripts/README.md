# MediSwarm Scripts

Utility scripts organized by purpose. Each subdirectory has its own README with detailed documentation.

## Directory Structure

| Directory | Purpose | README |
|-----------|---------|--------|
| [`build/`](build/README.md) | Docker image building, startup kit generation, pretrained weight management | [build/README.md](build/README.md) |
| [`ci/`](ci/README.md) | Integration tests and automated Dockerfile maintenance | [ci/README.md](ci/README.md) |
| [`client_node_setup/`](client_node_setup/README.md) | Client node provisioning: VPN, GPU, Docker, data download | [client_node_setup/README.md](client_node_setup/README.md) |
| [`dev_utils/`](dev_utils/README_dockerfile_update.md) | Developer utilities (Dockerfile version pin management) | [dev_utils/README_dockerfile_update.md](dev_utils/README_dockerfile_update.md) |
| [`evaluation/`](evaluation/README.md) | Model benchmarking and training result evaluation/plotting | [evaluation/README.md](evaluation/README.md) |

## Top-Level Scripts

| Script | Purpose |
|--------|---------|
| `_list_licenses.sh` | Lists pip/apt package licenses and model weight licenses (runs inside Docker) |
| `pr_validation.py` | Runs dummy training and preflight checks for pull request validation |
| `update_config_fed_client.py` | Updates `config_fed_client.conf` with challenge model persistor settings |

---

### `_list_licenses.sh`

Runs inside a Docker container to enumerate all software licenses (pip packages, APT packages, and pretrained model weights). Called by the integration test suite.

### `pr_validation.py`

Validates a pull request by running dummy training and data access preflight checks via Docker startup kits. Requires `SITE_NAME`, `DATADIR`, and `SCRATCHDIR` environment variables.

```bash
SITE_NAME=UKA DATADIR=/path/to/data SCRATCHDIR=/tmp/scratch python scripts/pr_validation.py
```

### `update_config_fed_client.py`

Updates the NVFlare `config_fed_client.conf` file to use a specific challenge model's persistor configuration.

```bash
# List available challenge models:
python scripts/update_config_fed_client.py --list

# Update config for a specific model:
python scripts/update_config_fed_client.py 2BCN_AIM [/path/to/config_fed_client.conf]
```

> **Note**: With the centralized model factory (`models_config.py`), config updates are typically no longer needed -- the factory reads `MODEL_NAME` from environment variables at runtime.
