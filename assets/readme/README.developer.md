# Usage for MediSwarm and Application Code Developers

## Cloning the Repository

We use a git submodule for a fork of NVFlare, so the MediSwarm repository should be cloned using
 ```bash
 git clone https://github.com/KatherLab/MediSwarm.git --recurse-submodules
 ```

If you have a clone without having initialized the submodule, use the following command in the MediSwarm directory
 ```bash
 git submodule update --init --recursive
 ```

## Environment Variables

Consider setting the following in your `.bashrc`:

```bash
export TMPDIR=<suitable location>
export MEDISWARM_BUILD_CACHE_DIR=<suitable location>
```

Building the MediSwarm Docker image
* creates a temporary directory with a clean clone of the source code in `/tmp` (that is only cleaned up after a successful build). A different location can be configured via `TMPDIR`.
* caches downloaded pre-trained model weights (on the order of 1 GB) if MEDISWARM_BUILD_CACHE_DIR is set, otherwise they are downloaded for each build.

## Cloning the Repository

We use a git submodule for a fork of NVFlare, so the MediSwarm repository should be cloned using
 ```bash
 git clone https://github.com/KatherLab/MediSwarm.git --recurse-submodules
 ```

If you have a clone without having initialized the submodule, use the following command in the MediSwarm directory
 ```bash
 git submodule update --init --recursive
 ```

## Versioning of ODELIA Docker Images

If needed, update the version number in file [odelia_image.version](../../odelia_image.version). It will be used
automatically for the Docker image and startup kits.

## Build the Docker Image and Startup Kits

The Docker image contains all dependencies for administrative purposes (dashboard, command-line provisioning, admin
console, server) as well as for running the 3DCNN pipeline under the pytorch-lightning framework.
The project description specifies the swarm nodes etc. to be used for a swarm training.

 ```bash
 cd MediSwarm
 ./scripts/build/buildDockerImageAndStartupKits.sh -p application/provision/<PROJECT DESCRIPTION.yml>
 ```

1. Make sure you have no uncommitted changes.
   If you have changes, commit them. This ensures that any image built corresponds to a code revision available later.
2. If package versions are no longer available, you may have to check what the current version is and update the
   `Dockerfile` accordingly. Version numbers are hard-coded to avoid issues due to silently different versions being
   installed.
3. After successful build (and after verifying that everything works as expected, i.e., local tests, building startup
   kits, running local trainings in the startup kit), you can manually push the image to DockerHub, provided you have
   the necessary rights. Make sure you are not re-using a version number for this purpose.

## Running Tests

* Build a Docker image for testing using `./scripts/build/buildDockerImageAndStartupKits.sh -p tests/provision/dummy_project_for_testing.yml [--use_docker_cache]`
* If you have multiple GPUs, use `GPU_FOR_TESTING="device=0" (or another device)
* If you have a sliced multiple GPUs, use `GPU_FOR_TESTING="device=0:0" (or another slice)
* Otherwise, leave this environment variable unset to use all GPUs.
* To run only specific tests, look at the options at the end of the script.
  * The whole test suite takes over an hour.

   ```bash
   ./scripts/ci/runIntegrationTests.sh
   ```

You should see output of

1. a check that files are present on github
2. a standalone minimal training run
3. a simulation run of a dummy training with two nodes
4. a proof-of-concept run of a dummy training with two nodes
5. a simulation run of a 3D CNN training using synthetic data with two nodes
6. a set of startup kits being created
7. pushing the Docker image to a local registry and pulling it from there (takes several minutes)
8. a Docker/GPU preflight check using one of the startup kits
9. a data access preflight check using one of the startup kits
10. a local 3D CNN training
11. an outdated client startup kit failing to connect to the server
12. a dummy training run in a swarm consisting of one server and two client nodes
13. a 3D CNN swarm training run in a two-client swarm

If tests fail, you may need to clean up temporary directories or leftover Docker containers.

## Distributing Startup Kits

Distribute the startup kits to the clients.

## Running the Startup Kits

See [README.participant.md](./README.participant.md).

### Configurable Parameters for docker.sh

* The `docker.sh` script run by the swarm participants passes the following environment variables into the container automatically.
* You can override them to customize training behavior.
* Only do this for testing and debugging purposes! The startup kits are designed to ensure that all sites run the same training code, manipulating `docker.sh` might break this.

| Environment Variable | Default         | Description                                                          |
|----------------------|-----------------|----------------------------------------------------------------------|
| `SITE_NAME`          | *from flag*     | Name of your local site, e.g. `TUD_1`, passed via `--start_client`   |
| `DATA_DIR`           | *from flag*     | Path to the host folder that contains your local data                |
| `SCRATCH_DIR`        | *from flag*     | Path for saving training outputs and temporary files                 |
| `GPU_DEVICE`         | `device=0`      | GPU identifier to use inside the container (or `all`)                |
| `MODEL_NAME`         | auto / `MST`    | For `--preflight_check` and `--local_training`, `docker.sh` derives this from `--job` and defaults to `1DivideAndConquer`. `MST` is the fallback when no job-derived model is selected. |
| `INSTITUTION`        | `ODELIA`        | Institution name, used to group experiment logs                      |
| `CONFIG`             | `unilateral`    | Configuration schema for dataset (e.g. label scheme)                 |
| `NUM_EPOCHS`         | `1` (test mode) | Number of training epochs (used in preflight/local training)         |
| `TRAINING_MODE`      | derived         | Internal use. Automatically set based on flags like `--start_client` |
| `ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS` | `0` | Swarm: cadence of the per-round aggregated prediction CSV export (`0`=final round only, `1`=every round, `N`=every Nth round + final). Throttled by default — it runs full train+val inference and dominates round time (#314). |
| `ODELIA_NUM_WORKERS`  | `min(CPU count, 16)` | DataLoader workers. Tune per node: set to roughly the CPU count on IO/CPU-bound nodes for faster rounds; lower it if the host is RAM-constrained (3D loaders are memory-heavy). Measure round-interval impact (#315). |
| `ODELIA_HASH_NUM_WORKERS` | `0` | Worker threads for dataset hashing/dedup (0 = inline). |

These are injected into the container as `--env` variables. You can modify their defaults by editing `docker.sh` or exporting before run:

```bash
export CONFIG=original
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=1 --start_client
```


## Running the Application

1. **CIFAR-10 example:**
   See [README.md](../../application/jobs/cifar10/README.md)
2. **Minimal PyTorch CNN example:**
   See [README.md](../../application/jobs/minimal_training_pytorch_cnn/README.md)
3. **3D CNN for classifying breast tumors:**
   See [README.md](../../application/jobs/ODELIA_ternary_classification/README.md)

### Available Challenge Jobs

| Job Name | Model | Job Directory |
|----------|-------|---------------|
| `challenge_1DivideAndConquer` | ResidualEncoder | `application/jobs/challenge_1DivideAndConquer` |
| `challenge_2BCN_AIM` | SwinUNETR | `application/jobs/challenge_2BCN_AIM` |
| `challenge_3agaldran` | MViT v2 | `application/jobs/challenge_3agaldran` |
| `challenge_4abmil` | CrossModalAttentionABMIL + Swin | `application/jobs/challenge_4abmil` |
| `challenge_5pimed` | ResNet18 | `application/jobs/challenge_5pimed` |

Each challenge job has its own `config_fed_client.conf`; the startup-kit `docker.sh` selects the matching model via `--job`/`MODEL_NAME`.

## Operating & Hardening Swarm Runs

Real multi-site runs surfaced several recurring failure modes; each now has a
fix, a pre-flight check, or both. The full catalogue + operator playbook is in
[`docs/SWARM_FAILURE_MODES.md`](../../docs/SWARM_FAILURE_MODES.md); timeout values
are in [`docs/TIMEOUTS.md`](../../docs/TIMEOUTS.md). Summary:

| Failure | Fix | Where |
|---------|-----|-------|
| Slow site → status-timeout abort | 24 h `max_status_report_interval`/`progress_timeout`/`learn_task_timeout` | job `config_fed_*.conf` |
| Slow site → `MODEL_UNRECOGNIZED` desync | `min_responses_required` = number of participating clients (wait-for-all) | job `config_fed_client.conf` |
| GPU `NVML: Unknown Error` mid-run | switch host Docker to `cgroupfs` driver | `scripts/client_node_setup/fix_docker_cgroupfs.sh` |
| Client won't restart (stale lock) | delete `daemon_pid.fl` before relaunch | pre-flight check auto-clears |
| Read-only cache crash | cache under `/scratch`, not `/data` | pre-flight check fails fast |
| VPN tunnel drop | network/VPN side; 24 h timeout bounds a brief drop | — |

**Pre-flight checks.** `docker.sh` (generated from `docker_config/master_template.yml`,
`_preflight_host_checks`) runs host checks for `--dummy_training`/`--preflight_check`/`--start_client`
(GPU usable in container, cgroup-driver risk, cache path, server reachability, stale lock).
Always have participants re-run dummy + pre-flight before every scheduled run.

**`min_responses_required` must equal the participating-client count.** Wait-for-all
prevents the slow-node desync but means any mid-round drop stalls the round until
`learn_task_timeout`. Set it per run; lower it only if a flaky site must be tolerated.

**Operator diagnostics.** Drive the live server with the admin kit
(`./fl_admin.sh`): `check_status server` (a frozen last-connect = a stale/dead
client), `check_status client`, `list_jobs`, `abort_job <id>` (needs a `y`),
`submit_job <path>`. Post-mortem of a finished/failed job: the run workspace is
in the job store at `/tmp/nvflare/jobs-storage/<job_id>/workspace` (a zip) inside
the server container — `log_error.txt` has the abort reason.

## Contributing Application Code

1. Take a look at application/jobs/minimal_training_pytorch_cnn for a minimal example how pytorch code can be adapted to
   work with NVFlare
2. Take a look at application/jobs/ODELIA_ternary_classification for a more realistic example of pytorch code that can
   run in the swarm
3. Use the local tests to check if the code is swarm-ready
4. TODO more detailed instructions

## Continuous Integration

Tests to be executed after pushing to github are defined in `.github/workflows/pr-test.yaml`.

This largely builds on the integration tests defined above, running those that finish within reasonable time.
