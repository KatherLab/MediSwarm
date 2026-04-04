# CI Scripts

Scripts for continuous integration testing and automated maintenance.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `runIntegrationTests.sh` | Comprehensive integration test suite for MediSwarm |
| `update_apt_versions.sh` | Automated update of pinned APT package versions in the Dockerfile |

---

## `runIntegrationTests.sh`

Comprehensive integration test suite that exercises the full MediSwarm pipeline inside Docker containers, including model training (standalone, simulation, and swarm modes), startup kit generation, security verification, and license checks.

### Prerequisites

- Docker with GPU support (`nvidia-container-toolkit`)
- Set `GPU_FOR_TESTING` env var to select GPU (default: `all`). Example: `GPU_FOR_TESTING="device=0"`
- Docker image must be built first (the script uses `localhost:5000/odelia:<version>`)

### Usage

```bash
# Run all integration tests:
./scripts/ci/runIntegrationTests.sh

# Run a specific test:
./scripts/ci/runIntegrationTests.sh <test_name>
```

### Available Tests

| Test Name | Description | Approx. Time |
|-----------|-------------|---------------|
| `check_files_on_github` | Verify README and license files are accessible on GitHub | ~10s |
| `run_nvflare_unit_tests` | Run NVFlare's own unit tests inside Docker | ~5min |
| `run_dummy_training_standalone` | Minimal PyTorch training example (standalone) | ~1min |
| `run_dummy_training_simulation_mode` | Minimal training in NVFlare simulation mode | ~2min |
| `run_dummy_training_poc_mode` | Minimal training in proof-of-concept mode | ~2min |
| `run_3dcnn_simulation_mode` | 3D CNN training in NVFlare simulation mode | ~5min |
| `create_startup_kits` | Generate and verify startup kit contents | ~1min |
| `run_list_licenses` | Verify pip/apt/model weight licenses are listed | ~1min |
| `run_docker_gpu_preflight_check` | Dummy training via startup kit (Docker + GPU check) | ~1min |
| `run_data_access_preflight_check` | Data access check with synthetic data (verifies warnings/errors) | ~2min |
| `push_pull_image` | Test Docker image push/pull via local registry | ~3min |
| `check_wrong_startup_kit` | Verify that outdated/invalid certificates are rejected | ~1min |
| `run_dummy_training_in_swarm` | Full swarm training with 2 clients (minimal example) | ~3min |
| `run_3dcnn_local_training` | 3D CNN local training with synthetic data | ~60min |
| `run_3dcnn_training_in_swarm` | Full 3D CNN swarm training with 2 clients | ~60min |
| `kill_server_and_clients` | Kill running server/client Docker containers | instant |
| `all` (default) | Run the full test suite | ~2h+ |

### What It Verifies

- Docker image builds and runs correctly
- GPU access works inside containers
- Synthetic data generation works
- Training completes for minimal and 3D CNN models
- NVFlare provisioning generates valid startup kits
- Startup kit ZIP archives contain required files
- TLS certificates are validated (invalid certs rejected)
- Swarm training with multiple clients completes all rounds
- Model checkpoints (`FL_global_model.pt`, `best_FL_global_model.pt`) are produced
- License information is discoverable for all dependencies

---

## `update_apt_versions.sh`

Automated CI script that updates pinned APT package version numbers in the Dockerfile. Used by the GitHub Actions workflow to keep Dockerfile package versions current.

### How It Works

1. Removes all APT version pins from the Dockerfile
2. Commits the change temporarily
3. Rebuilds the Docker image (capturing the actual installed versions from the build log)
4. Re-adds version pins with the newly resolved versions
5. Reports whether any versions changed

### Usage

This script is designed to run in CI (GitHub Actions). It writes to `$GITHUB_ENV` to communicate results.

```bash
# Typically called by .github/workflows/ — not meant for manual use.
./scripts/ci/update_apt_versions.sh
```

### See Also

- `scripts/dev_utils/README_dockerfile_update.md` for manual Dockerfile version updating
- `scripts/dev_utils/dockerfile_update_removeVersionApt.py` / `dockerfile_update_addAptVersionNumbers.py`
