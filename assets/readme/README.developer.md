# Usage for MediSwarm and Application Code Developers

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
2. If package versions are still not available, you may have to check what the current version is and update the
   `Dockerfile` accordingly. Version numbers are hard-coded to avoid issues due to silently different versions being
   installed.
3. After successful build (and after verifying that everything works as expected, i.e., local tests, building startup
   kits, running local trainings in the startup kit), you can manually push the image to DockerHub, provided you have
   the necessary rights. Make sure you are not re-using a version number for this purpose.

## Running Tests

* If you have multiple GPUs, use `GPU_FOR_TESTING="device=0" (or another device)
* If you have a sliced multiple GPUs, use `GPU_FOR_TESTING="device=0:0" (or another slice)
* Otherwise, leave this environment variable unset to use all GPUs.
* To run only specific tests, look at the options at the end of the script.

   ```bash
   ./runIntegrationTests.sh
   ```

You should see

1. several expected errors and warnings printed from unit tests that should succeed overall, and a coverage report
2. output of a successful simulation run of a dummy training with two nodes
3. output of a successful proof-of-concept run of a dummy training with two nodes
4. output of a successful simulation run of a 3D CNN training using synthetic data with two nodes
5. output of a set of startup kits being generated
6. output of pushing the Docker image to a local registry and pulling it from there (takes several minutes)
7. output of a Docker/GPU preflight check using one of the startup kits
8. output of a data access preflight check using one of the startup kits
9. output of an outdated client startup kit failing to connect to the server
10. output of a dummy training run in a swarm consisting of one server and two client nodes

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

1. Take a look at application/jobs/minimal_training_pytorch_cnn for a minimal example how pytorch code can be adapted to
   work with NVFlare
2. Take a look at application/jobs/ODELIA_ternary_classification for a more realistic example of pytorch code that can
   run in the swarm
3. Use the local tests to check if the code is swarm-ready
4. TODO more detailed instructions

## Continuous Integration

Tests to be executed after pushing to github are defined in `.github/workflows/pr-test.yaml`.
