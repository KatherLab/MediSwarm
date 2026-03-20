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

## Versioning of ODELIA Docker Images

If needed, update the version number in file [odelia_image.version](../../odelia_image.version). It will be used
automatically for the Docker image and startup kits.

## Build the Docker Image and Startup Kits

The Docker image contains all dependencies for administrative purposes (dashboard, command-line provisioning, admin
console, server) as well as for running the 3DCNN pipeline under the pytorch-lightning framework.
The project description specifies the swarm nodes etc. to be used for a swarm training.

 ```bash
 cd MediSwarm
 ./buildDockerImageAndStartupKits.sh -p application/provision/<PROJECT DESCRIPTION.yml>
 ```

1. Make sure you have no uncommitted changes.
   If you have changes, commit them. This ensures that any image built corresponds to a code revision available later.
3. If package versions are no longer available, you may have to check what the current version is and update the
   `Dockerfile` accordingly. Version numbers are hard-coded to avoid issues due to silently different versions being
   installed.
4. After successful build (and after verifying that everything works as expected, i.e., local tests, building startup
   kits, running local trainings in the startup kit), you can manually push the image to DockerHub, provided you have
   the necessary rights. Make sure you are not re-using a version number for this purpose.

## Running Tests

* If you have multiple GPUs, use `GPU_FOR_TESTING="device=0" (or another device)
* If you have a sliced multiple GPUs, use `GPU_FOR_TESTING="device=0:0" (or another slice)
* Otherwise, leave this environment variable unset to use all GPUs.
* To run only specific tests, look at the options at the end of the script.
  * The whole test suite takes over an hour.

   ```bash
   ./runIntegrationTests.sh
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
| `MODEL`              | `MST`           | Model architecture, choices: `MST`, `ResNet`                         |
| `INSTITUTION`        | `ODELIA`        | Institution name, used to group experiment logs                      |
| `CONFIG`             | `unilateral`    | Configuration schema for dataset (e.g. label scheme)                 |
| `NUM_EPOCHS`         | `1` (test mode) | Number of training epochs (used in preflight/local training)         |
| `TRAINING_MODE`      | derived         | Internal use. Automatically set based on flags like `--start_client` |

These are injected into the container as `--env` variables. You can modify their defaults by editing `docker.sh` or exporting before run:

```bash
export MODEL=ResNet
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

## Contributing Application Code

* Take a look at application/jobs/minimal_training_pytorch_cnn for a minimal example how pytorch code can be adapted to work with NVFlare
* Take a look at application/jobs/ODELIA_ternary_classification for a more realistic example of pytorch code that can run in the swarm
* If your application code needs additonal/other/newer Python packages than installed via [Dockerfile_ODELIA](../../docker_config/Dockerfile_ODELIA), create and use an adapted Dockerfile for building the Docker image
  * Ensure (by checking against the installation log) that all packages and dependenciese are installed explicitly at pinned versions.
* If your application code needs, e.g., other pre-trained weights in the image, adapt [_cacheAndCopyPretrainedModelWeights.sh](../../_cacheAndCopyPretrainedModelWeights.sh) and [_list_licenses.sh](../../scripts/_list_licenses.sh)

To make sure your code is swarm-compatible and to isolate potential issues, we recommend the following steps.

1. Start with a working version outside the swarm framework in a known environment.
   This way, you have a known-to-work baseline and results you can later compare to.
2. Create a small dataset.
   This avoids data issues and allows faster feedback cycles.
   * For this purpose,
     * either use a subset of your data or
     * write code to create a synthetic dataset similar to [create_synthetic_dataset.py](../../application/jobs/ODELIA_ternary_classification/app/scripts/create_synthetic_dataset/create_synthetic_dataset.py)
       * This will be needed for a self-contained test suite without the need to share any data, but can also be implemented later.
3. Create a git branch of MediSwarm and verify that you can build a Docker image and run the integration tests.
   * This ensures you start with a working version and do not search for issues in your code that are actually problems elsewhere.
   * In the following, you can either start with a subdirectory of [application/jobs/](../../application/jobs/)
     * from scratch
     * adapting `ODELIA_ternary_classification` if your code is similar to what the ODELIA consortium has been doing
     * extending the minimal example `minimal_training_pytorch_cnn` if your code is largely different from the above
4. Make your code Docker-ready.
   * Compare your local system and python environment to what is defined in the MediSwarm [Dockerfile](../../docker_config/Dockerfile_ODELIA) and adapt the Dockerfile to install dependencies of your code.
   * You can drop the package version numbers and do not need to list all dependencies until everything works, but should do so later. This is to avoid changes or incompatibilities due to silently changed versions.
   * See above for instructions on building the Docker image.
   * Adapt the `run_dummy_training_standalone` test in [runIntegrationTests.sh](../../runIntegrationTests.sh), which uses [_run_minimal_example_standalone.sh](../../tests/integration_tests/_run_minimal_example_standalone.sh), to run the code in the Docker container.
   * Debug, commit, rebuild containers, run until this succeeds.
     * Consider running the container interactively for debugging.
5. Make your code NVFlare-ready.
   * NVFlare needs to control the training loop, so your training needs adaptations similar to the minimal and ODELIA ternary classification examples. For details, please consult the NVFlare documentation (for the version used in MediSwarm).
   * NVFlare provides the simulation mode (running separate clients in separate threads) and proof-of-concept mode (running clients in separate processes). These can be used for testing NVFlare-compatibility.
   * Adapt the `run_dummy_training_simulation_mode` or `run_3dcnn_simulation_mode` tests in [runIntegrationTests.sh](../../runIntegrationTests.sh), which uses [_run_3dcnn_simulation_mode.sh](../../tests/integration_tests/_run_3dcnn_simulation_mode.sh), to run your code in simulation mode.
     * Debug, commit, rebuild containers, run until this succeeds.
       * Consider running the container interactively for debugging.
   * Adapt the `run_dummy_training_poc_mode` test in [runIntegrationTests.sh](../../runIntegrationTests.sh), which uses [_run_minimal_example_proof_of_concept_mode.sh](../../tests/integration_tests/_run_minimal_example_proof_of_concept_mode.sh), to run your code in proof-of-concept mode.
     * Debug, commit, rebuild containers, run until this succeeds.
     * If your code runs successfully in simulation mode, you can skip this step, unless later steps fail.
6. Enable local training and local data access preflight check
   * Having adapted the code to be NVFlare-compatible, it should still be able to run outside a swarm and should provide the possibility for swarm participants to check if their data is compatible with the training.
     * A local data access preflight check can simply be a local training for one epoch.
   * Adapt the `docker_cln_sh` section of [master_template.yml](../../docker_config/master_template.yml) to enable these.
   * Debug, commit, rebuild containers, run until this succeeds.
   * Make sure the code stays NVFlare-compatible when making changes.
7. Make sure the code runs in an actual swarm training.
   * Check results against the baseline to make sure the code still trains a useful model.
8. Clean up the implementation
   * Go through the points postponed for later.
   * Prepare a pull request for merging your branch.

TODO iterate instructions and add missing details

## Continuous Integration

Tests to be executed after pushing to github are defined in `.github/workflows/pr-test.yaml`.
This largely builds on the integration tests defined above, running those that finish within reasonable time.
