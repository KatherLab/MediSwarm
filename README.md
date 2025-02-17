# Introduction
MediSwarm is an open-source project dedicated to advancing medical deep learning through swarm intelligence, leveraging the NVFlare platform. Developed in collaboration with the Odelia consortium, this repository aims to create a decentralized and collaborative framework for medical research and applications.

## Key Features
- **Swarm Learning:** Utilizes swarm intelligence principles to improve model performance and adaptability.
- **NVFlare Integration:** Built on NVFlare, providing robust and scalable federated learning capabilities.
- **Data Privacy:** Ensures data security and compliance with privacy regulations by keeping data local to each institution.
- **Collaborative Research:** Facilitates collaboration among medical researchers and institutions for enhanced outcomes.
- **Extensible Framework:** Designed to support various medical applications and easily integrate with existing workflows.

## Prerequisites
### Hardware recommendations
* 64 GB of RAM (32 GB is the absolute minimum)
* 16 CPU cores (8 is the absolute minimum)
* an NVIDIA GPU with 48 GB of RAM (24 GB is the minimum)
* 8 TB of Storage (4 TB is the absolute minimum)

We demonstrate that the system can run on lightweight hardware like this. For less than 10k EUR, you can configure systems from suppliers like Lambda, Dell Precision, and Dell Alienware.

### Operating System
* Ubuntu 20.04 LTS

# Usage for Developers

## Setup

0. **Clone the repository:**

    ```bash
    git clone https://github.com/KatherLab/MediSwarm.git
    cd MediSwarm
    ```

## Running the Application

1. **CIFAR-10 example:**
   See [cifar10/README.md](application/jobs/cifar10/README.md)
2. **Minimal PyTorch CNN example:**
   See [application/jobs/minimal_training_pytorch_cnn/README.md](application/jobs/minimal_training_pytorch_cnn/README.md)
3. **3D CNN for classifying breast tumors:**
   See [3dcnn_ptl/README.md](application/jobs/3dcnn_ptl/README.md)

### Running Tests

1. Build the testing docker image
   ```bash
   docker build -t nvflare-pt-dev:3dcnn   . -f docker_config/Dockerfile_3dcnn
   docker build -t nvflare-pt-dev:testing . -f docker_config/Dockerfile_testing
   ```
2. Run the Tests via
   ```bash
   ./runTestsInDocker.sh
   ```
3. You should see
   1. several expected errors and warnings printed from unit tests that should succeed overall, and a coverage report
   2. output of a successful simulation run with two nodes
   3. output of a successful proof-of-concept run run with two nodes
4. Optionally, uncomment running NVFlare unit tests in `_runTestsInsideDocker.sh`


## Contributing Application Code

* take a look at application/jobs/minimal_training_pytorch_cnn for a minimal example how pytorch code can be adapted to work with NVFlare
* take a look at application/jobs/3dcnn_ptl for a more relastic example of pytorch code that can run in the swarm
* TODO more detailed instructions

## Setting up a Swarm

* currently described (here)[/application/jobs/3dcnn_ptl/README.md]

# Usage for Swarm Participants

## Setup

1. TODO compute node according to spec, installation of docker, openvpn, …

## Prepare Dataset

* TODO which data is expected in which folder structure + table structure

## Prepare Training Participation

1. TODO steps until startup kit has been extracted

## Run Pre-Flight Check

1. Directories
   ```bash
   export SITE_NAME=<the name of your site>  # TODO should be defined above, also needed for dataset location
   export DATADIR=<path to where the directory $SITE_NAME containing your local data is stored>
   export SCRATCHDIR=<path to where the training can store temporary files>
   ```
2. From the directory where you unpacked the startup kit,
   ```bash
   cd $SITE_NAME/startup
   ```
3. Verify that your Docker/GPU setup is working
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU all --dummy_training
   ```
   * This will pull the Docker image, which might take a while.
   * The “training” itself should take less than minute and does not yield a meaningful classification performance.
4. Verify that your local data can be accessed and the model can be trained locally
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU all --dummy_training
   ```
   * Training time depends on the size of the local dataset
   * TODO update call when handling of the number of epochs has been implemented

## Start Swarm Node

1. From the directory where you unpacked the startup kit
   ```bash
   cd $SITE_NAME/startup  # skip this if you just ran the pre-flight check
   ```
2. Start the client
   ```bash
   rm -rf ../pid.fl ../daemon_pid.fl nohup.out  # clean up potential leftovers from previous run
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU all --start_client
   ```
3. Console output is captured in `nohup.out`, which may have been created by the root user in the container, so make it readable:
   ```bash
   sudo chmod a+r nohup.out
   ```
4. Output files
   * TODO describe

## Run Local Training

1. From the directory where you unpacked the startup kit
   ```bash
   cd $SITE_NAME/startup
   ```
2. Start local training
   ```bash
   /docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU all --local_training
   ```
   * TODO update when handling of the number of epochs has been implemented
3. Output files
   * TODO describe

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Maintainers
[Jeff](https://github.com/Ultimate-Storm)
[Ole Schwen](mailto:ole.schwen@mevis.fraunhofer.de)
[Steffen Renisch](mailto:steffen.renisch@mevis.fraunhofer.de)

# Contributing
Feel free to dive in! [Open an issue](https://github.com/KatherLab/MediSwarm/issues) or submit pull requests.

# Credits
This project utilizes platforms and resources from the following repositories:

- **[NVFLARE](https://github.com/NVIDIA/NVFlare)**: NVFLARE (NVIDIA Federated Learning Application Runtime Environment) is an open-source framework that provides a robust and scalable platform for federated learning applications. We have integrated NVFLARE to efficiently handle the federated learning aspects of our project.

Special thanks to the contributors and maintainers of these repositories for their valuable work and support.

---

For more details about NVFLARE and its features, please visit the [NVFLARE GitHub repository](https://github.com/NVIDIA/NVFlare).
