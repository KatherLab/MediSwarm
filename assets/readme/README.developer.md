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

## Running Local Tests

   ```bash
   ./runTestsInDocker.sh
   ```

You should see

1. several expected errors and warnings printed from unit tests that should succeed overall, and a coverage report
2. output of a successful simulation run with two nodes
3. output of a successful proof-of-concept run run with two nodes
4. output of a set of startup kits being generated
5. output of a dummy training run using one of the startup kits
6. TODO update this to what the tests output now

Optionally, uncomment running NVFlare unit tests in `_runTestsInsideDocker.sh`.

## Distributing Startup Kits

Distribute the startup kits to the clients.

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
2. Take a look at application/jobs/ODELIA_ternary_classification for a more relastic example of pytorch code that can
   run in the swarm
3. Use the local tests to check if the code is swarm-ready
4. TODO more detailed instructions

