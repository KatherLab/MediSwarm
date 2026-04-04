# Build Scripts

Scripts for building Docker images, generating NVFlare startup kits, and managing pretrained model weights.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `buildDockerImageAndStartupKits.sh` | Main entry point: builds Docker image and generates startup kits |
| `_buildStartupKits.sh` | Runs NVFlare provisioning inside Docker to create startup kits |
| `_cacheAndCopyPretrainedModelWeights.sh` | Downloads/caches pretrained model weights for inclusion in Docker image |
| `_generateStartupKitArchives.sh` | Creates ZIP archives of each startup kit |
| `_injectLiveSyncIntoStartupKits.sh` | Patches startup kit `docker.sh` with live sync support |
| `getVersionNumber.sh` | Generates version string from `odelia_image.version` + git hash + date |
| `test_and_build_all_models.sh` | Tests challenge model loading and training, then optionally builds startup kits |

> Scripts prefixed with `_` are internal helpers called by `buildDockerImageAndStartupKits.sh`. You normally don't run them directly.

---

## `buildDockerImageAndStartupKits.sh`

Main build script. Creates a clean git archive, builds the Docker image, and generates NVFlare startup kits.

### Prerequisites

- Clean git working tree (no uncommitted changes)
- Docker installed and running
- A project YAML file (e.g., `tests/provision/dummy_project_for_testing.yml`)

### Usage

```bash
# Standard build (no Docker cache):
./scripts/build/buildDockerImageAndStartupKits.sh -p <swarm_project.yml>

# Build with Docker cache (faster for incremental changes):
./scripts/build/buildDockerImageAndStartupKits.sh -p <swarm_project.yml> --use-docker-cache
```

### What It Does

1. Verifies no local git changes exist
2. Creates a clean source tree via `git archive`
3. Injects version numbers into `master_template.yml`
4. Downloads/caches pretrained model weights (`_cacheAndCopyPretrainedModelWeights.sh`)
5. Builds Docker image (`docker build`)
6. Runs NVFlare provisioning to generate startup kits (`_buildStartupKits.sh`)
7. Cleans up temporary files

### Output

- Docker image tagged with version string
- Startup kits in `workspace/<project_name>/prod_XX/`
- Each site gets a ZIP archive ready for distribution

---

## `_buildStartupKits.sh`

Runs NVFlare provisioning inside the Docker container, then injects live sync helpers and generates ZIP archives.

```bash
# Called automatically by buildDockerImageAndStartupKits.sh
./scripts/build/_buildStartupKits.sh <PROJECT.yml> <VERSION> <CONTAINER_NAME>
```

---

## `_cacheAndCopyPretrainedModelWeights.sh`

Downloads (if not already cached) and copies pretrained model weights into the Docker build context.

### Weights Managed

| Model | File | Source |
|-------|------|--------|
| DINOv2 (MST backbone) | `dinov2_vits14_pretrain.pth` | Facebook AI |
| 1DivideAndConquer | `checkpoint_final.pth` | Google Drive |
| 3agaldran | `mvit_v2_s-ae3be167.pth` | PyTorch Hub |

Weights are stored at `/MediSwarm/pretrained_weights/` inside the Docker image (not inside job folders, to avoid NVFlare transferring them over the network).

---

## `_generateStartupKitArchives.sh`

Creates ZIP archives from each startup kit directory.

```bash
# Called automatically by _buildStartupKits.sh
./scripts/build/_generateStartupKitArchives.sh <PROJECT.yml> <VERSION>
```

---

## `_injectLiveSyncIntoStartupKits.sh`

Patches each startup kit's `docker.sh` to include live sync functionality. The original `docker.sh` is preserved as `docker_original.sh`, and the new wrapper adds sync support for both local training and swarm modes.

---

## `getVersionNumber.sh`

Generates a version string by combining:
- Base version from `odelia_image.version` (e.g., `1.2.0`)
- Current date (`YYMMDD`)
- Short git commit hash

```bash
./scripts/build/getVersionNumber.sh
# Output: 1.2.0-dev.260404.abc1234
```

---

## `test_and_build_all_models.sh`

End-to-end testing and build automation for challenge models. Tests model loading in both `preflight_check` and `local_training` modes, then optionally builds startup kits.

### Prerequisites

- Python virtual environment at the configured path
- CUDA GPU available
- `DATA_DIR` pointing to challenge data

### Usage

```bash
# Test default models (1DivideAndConquer, 3agaldran):
./scripts/build/test_and_build_all_models.sh

# Test specific models:
./scripts/build/test_and_build_all_models.sh --models "1DivideAndConquer,2BCN_AIM,3agaldran"

# Skip Docker build:
./scripts/build/test_and_build_all_models.sh --skip-build

# Don't push changes:
./scripts/build/test_and_build_all_models.sh --no-push
```

### Output

- Test logs at `/tmp/test_*_{preflight_check,local_training}.log`
- Build log at `/tmp/build.log`
- Summary table of pass/fail per model
