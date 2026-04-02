# MediSwarm Deploy & Test Workflow

Automated build, deploy, and test pipeline for MediSwarm federated learning experiments.

## Overview

`deploy_and_test.sh` automates the entire workflow for running MediSwarm swarm learning experiments across distributed sites:

1. **Build** — Builds the Docker image and generates per-site startup kits
2. **Push** — Pushes the Docker image to DockerHub
3. **Deploy** — Copies startup kit ZIPs to remote sites via SCP and unzips them
4. **Start server** — Starts the NVFlare server locally
5. **Start clients** — Starts NVFlare clients on remote sites via SSH
6. **Submit** — Launches the admin console and submits a job for federated training
7. **Monitor** — Check status, tail logs, and stop containers across all sites

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| **Docker** | Build and run containers | [docs.docker.com](https://docs.docker.com/get-docker/) |
| **sshpass** | Non-interactive SSH authentication | `sudo apt-get install sshpass` |
| **expect** | Admin console automation (submit only) | `sudo apt-get install expect` |
| **git** | Version number and source management | `sudo apt-get install git` |

You also need:
- DockerHub credentials (for `docker push`)
- SSH access to all remote client sites
- GPU-enabled Docker runtime on client machines

## Configuration

### Site Configuration (`deploy_sites.conf`)

Create a `deploy_sites.conf` file in the repository root. This file is gitignored and contains credentials for each remote site.

**Example:**

```bash
# MediSwarm Deployment Site Configuration
# DO NOT commit this file to git (it is in .gitignore).

# ── Site 1 ─────────────────────────────────────────────
MHA_HOST=192.168.1.100
MHA_USER=flclient
MHA_PASS='your_password'
MHA_SITE_NAME=MHA_1                         # Must match the name in the project YAML
MHA_DATADIR=/home/flclient/data             # Path to training data on the remote machine
MHA_SCRATCHDIR=/home/flclient/scratch       # Writable scratch space for checkpoints
MHA_DEPLOY_DIR=/home/flclient/Odelia        # Where startup kits are deployed
MHA_GPU="device=0"                          # GPU to use

# ── Site 2 ─────────────────────────────────────────────
RSH_HOST=192.168.1.101
RSH_USER=flclient
RSH_PASS='your_password'
RSH_SITE_NAME=RSH_1
RSH_DATADIR=/home/flclient/data
RSH_SCRATCHDIR=/home/flclient/scratch
RSH_DEPLOY_DIR=/home/flclient/Odelia
RSH_GPU="device=0"

# ── Defaults ────────────────────────────────────────────
PROJECT_FILE=application/provision/project_Challenge_test.yml
DEFAULT_JOB=challenge_1DivideAndConquer
ADMIN_USER=admin@example.com
```

Each site needs a block of variables prefixed with the site label (e.g., `MHA_`, `RSH_`). The `SITES` array in `deploy_and_test.sh` (line 63) must list all site labels you define.

## Command Reference

```
./deploy_and_test.sh <command> [args]
```

| Command | Description |
|---------|-------------|
| `build` | Build Docker image + generate startup kits from the project YAML |
| `push` | Push the built Docker image to DockerHub |
| `deploy` | SCP startup kit ZIPs to all remote sites and unzip them |
| `start-server` | Start the NVFlare server container locally |
| `start-clients` | Start NVFlare client containers on all remote sites |
| `submit [job]` | Open admin console and submit a job (default: `$DEFAULT_JOB` from config) |
| `status` | Show running `odelia_swarm_*` containers on all sites |
| `logs <site>` | Tail last 50 lines of logs from a site (`MHA`, `RSH`, or `server`) |
| `stop` | Kill all `odelia_swarm_*` containers on all sites |
| `all [job]` | Full pipeline: build -> push -> deploy -> start-server -> start-clients -> submit |

## Available Jobs

These are the NVFlare jobs that can be submitted:

| Job Name | Model | Description |
|----------|-------|-------------|
| `challenge_1DivideAndConquer` | ResidualEncoder | Divide-and-Conquer classification |
| `challenge_2BCN_AIM` | SwinUNETR | BCN-AIM multi-task learning |
| `challenge_3agaldran` | MViT v2 | Agaldran classification |
| `challenge_4abmil` | CrossModalAttentionABMIL + Swin | Attention-based MIL |
| `challenge_5pimed` | ResNet18 | PIMED classification |
| `ODELIA_ternary_classification` | MST (default) | ODELIA ternary classification |

## Full Pipeline Example

```bash
# 1. One-command full pipeline
./deploy_and_test.sh all challenge_1DivideAndConquer

# 2. Or step by step:

# Build Docker image and startup kits
./deploy_and_test.sh build

# Push image to DockerHub
./deploy_and_test.sh push

# Deploy startup kits to remote sites
./deploy_and_test.sh deploy

# Start server (local)
./deploy_and_test.sh start-server

# Start clients on remote sites
./deploy_and_test.sh start-clients

# Wait a bit for clients to register, then submit
./deploy_and_test.sh submit challenge_1DivideAndConquer

# Monitor
./deploy_and_test.sh status
./deploy_and_test.sh logs MHA
./deploy_and_test.sh logs RSH
./deploy_and_test.sh logs server

# When done
./deploy_and_test.sh stop
```

## Local Testing with docker.sh

Each site's startup kit includes a `docker.sh` script that supports local testing modes. Use the `--job` flag to test individual challenge models without running a full swarm.

### Preflight Check

Quick sanity check (1 epoch) to verify data access and model initialization:

```bash
cd workspace/<project>/prod_NN/<site_name>/startup

# Default model (ODELIA_ternary_classification)
./docker.sh --preflight_check --data_dir /path/to/data --scratch_dir /path/to/scratch --GPU device=0

# Test a specific challenge model
./docker.sh --preflight_check --job challenge_5pimed --data_dir /path/to/data --scratch_dir /path/to/scratch --GPU device=0
```

### Local Training

Full local training (100 epochs, no federation):

```bash
# Default model
./docker.sh --local_training --data_dir /path/to/data --scratch_dir /path/to/scratch --GPU device=0

# Train with a specific challenge model
./docker.sh --local_training --job challenge_2BCN_AIM --data_dir /path/to/data --scratch_dir /path/to/scratch --GPU device=0
```

### All docker.sh Modes

| Flag | Description |
|------|-------------|
| `--dummy_training` | Minimal sanity check for Docker/GPU |
| `--preflight_check` | Verify data access & model init (1 epoch) |
| `--local_training` | Train a local model (100 epochs) |
| `--start_client` | Launch FL client in swarm mode |
| `--list_licenses` | List licenses of installed packages |
| `--interactive` | Drop into interactive container shell |
| `--run_script <cmd>` | Execute a script inside the container |
| `--job <name>` | Select job/model for `--preflight_check` or `--local_training` |

## Troubleshooting

### Build fails

- Ensure Docker daemon is running: `systemctl status docker`
- Check that the project YAML exists: `ls application/provision/project_*.yml`
- Verify `getVersionNumber.sh` returns a valid version

### Cannot connect to remote sites

- Verify SSH connectivity: `sshpass -p 'password' ssh user@host hostname`
- Check that `deploy_sites.conf` has correct host/user/pass
- Ensure the remote machines accept password-based SSH

### Client fails to start

- Check that Docker is installed on the remote machine
- Verify the GPU is available: `remote_exec MHA "nvidia-smi"`
- Ensure the deploy directory has correct permissions

### Job submission fails

- Verify server is running: `./deploy_and_test.sh status`
- Ensure clients have registered (wait 15-30s after starting clients)
- Check server logs: `./deploy_and_test.sh logs server`
- Verify `expect` is installed

### Training errors (preflight_check or local_training)

- Check that the data directory is mounted correctly and contains expected files
- For challenge models, ensure pretrained weights exist in the job's `app/custom/` directory
- Review container logs: `docker logs <container_name>`

### "No reserved resources" error during swarm

- This means the NVFlare scheduler has no GPU resources allocated
- Check `config_fed_server.conf` has the correct `num_gpus` setting
- Restart the server and clients

### MODEL_NAME mismatch

- Each challenge job hardcodes its `MODEL_NAME` in `main.py` to avoid the Docker env var override (`MODEL_NAME=${MODEL_NAME:-MST}` in docker.sh)
- If adding a new model, hardcode the name in both the swarm and local training branches of `main.py`
