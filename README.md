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

### Software
* Docker
* openvpn
* git

### Cloning the repository
    ```bash
    git clone https://github.com/KatherLab/MediSwarm.git --recurse-submodules
    ```
* The last argument is necessary because we are using a git submodule for the (ODELIA fork of NVFlare)[https://github.com/KatherLab/NVFlare_MediSwarm]

### VPN
A VPN is necessary so that the swarm nodes can communicate with each other securely across firewalls. For that purpose,
1. Install OpenVPN
   ```bash
   sudo apt-get install openvpn
   ```
2. If you have a graphical user interface(GUI), follow this guide to connect to the VPN: [VPN setup guide(GUI).pdf](assets/VPN%20setup%20guide%28GUI%29.pdf)
3. If you have a command line interface(CLI), follow this guide to connect to the VPN: [VPN setup guide(CLI).md](assets/VPN%20setup%20guide%28CLI%29.md)

# Usage for Swarm Participants
## Setup
1. Make sure your compute node satisfies the specification and has the necessary software installed.
2. Clone the repository and cnnect the client node to the VPN as described above.
3. TODO anything else?

## Prepare Dataset
1. TODO which data is expected in which folder structure + table structure

## Prepare Training Participation
1. Extract startup kit provided by swarm operator

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
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training
   ```
   * This will pull the Docker image, which might take a while.
   * If you have multiple GPUs and 0 is busy, use a different one.
   * The “training” itself should take less than minute and does not yield a meaningful classification performance.
4. Verify that your local data can be accessed and the model can be trained locally
   ```bash
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check
   ```
   * Training time depends on the size of the local dataset

## Start Swarm Node
1. From the directory where you unpacked the startup kit
   ```bash
   cd $SITE_NAME/startup  # skip this if you just ran the pre-flight check
   ```
2. Start the client
   ```bash
   rm -rf ../pid.fl ../daemon_pid.fl nohup.out  # clean up potential leftovers from previous run
   ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client
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

# Usage for MediSwarm and Application Code Developers
## Versioning of ODELIA Docker Images
If needed, update the version number in file (odelia_image.version)[odelia_image.version]. It will be used automatically for the Docker image and startup kits.

## Build the Docker Image
The Docker image contains all dependencies for administrative purposes (dashboard, command-line provisioning, admin console, server) as well as for running the 3DCNN pipeline under the pytorch-lightning framework.
    ```bash
    cd MediSwarm
    ./buildDockerImage.sh
    ```

1. Make sure you have no uncommitted changes.
2. You may need to use `--no-cache` in the `docker build` command if, e.g., the apt repository index cache is out of date and package versions are not found.
3. If package versions are still not available, you may have to check what the current version is and update the `Dockerfile` accordingly. Version numbers are hard-coded to avoid issues due to silently different versions being installed.
4. After successful build (and after verifying that everything works as expected, i.e., local tests, building startup kits, running local trainings in the startup kit), you can manually push the image to DockerHub, provided you have the necessary rights. Make sure you are not re-using a version number for this purpose.

## Running Local Tests
   ```bash
   ./runTestsInDocker.sh
   ```

You should see
1. several expected errors and warnings printed from unit tests that should succeed overall, and a coverage report
2. output of a successful simulation run with two nodes
3. output of a successful proof-of-concept run run with two nodes

Optionally, uncomment running NVFlare unit tests in `_runTestsInsideDocker.sh`.

## Building Startup Kits
   ```bash
   ./runTestsInDocker.sh
   ```
Distribute the startup kits to the clients.

## Running the Application
1. **CIFAR-10 example:**
   See [cifar10/README.md](application/jobs/cifar10/README.md)
2. **Minimal PyTorch CNN example:**
   See [application/jobs/minimal_training_pytorch_cnn/README.md](application/jobs/minimal_training_pytorch_cnn/README.md)
3. **3D CNN for classifying breast tumors:**
   See [3dcnn_ptl/README.md](application/jobs/3dcnn_ptl/README.md)

## Contributing Application Code
1. Take a look at application/jobs/minimal_training_pytorch_cnn for a minimal example how pytorch code can be adapted to work with NVFlare
2. Take a look at application/jobs/3dcnn_ptl for a more relastic example of pytorch code that can run in the swarm
3. Use the local tests to check if the code is swarm-ready
4. TODO more detailed instructions

# Usage for Swarm Operators
## Setting up a Swarm
Production mode is designed for secure, real-world deployments. It supports both local and remote setups, whether on-premise or in the cloud. For more details, refer to the [NVFLARE Production Mode](https://nvflare.readthedocs.io/en/2.4.1/real_world_fl.html).

To set up production mode, follow these steps:

## Edit `/etc/hosts`
Ensure that your `/etc/hosts` file includes the correct host mappings. All hosts need to be able to communicate to the server node.

For example, add the following line (replace `<IP>` with the server's actual IP address):

```plaintext
<IP>    dl3.tud.de dl3
```

## Create Startup Kits
### Via Script (recommended)
1. Use, e.g., the file `application/provision/project_MEVIS_test.yml`, adapt as needed (network protocol etc.)
2. Call `buildStartupKits.sh /path/to/project_configuration.yml` to build the startup kits
3. Startup kits are generated to `workspace/<name configured in the .yml>/prod_00/`
4. Deploy startup kits to the respective server/clients

### Via the Dashboard (not recommended)
```bash
docker run -d --rm \
     --ipc=host -p 8443:8443 \
    --name=odelia_swarm_admin \
    -v /var/run/docker.sock:/var/run/docker.sock \
    <DOCKER_IMAGE> \
    /bin/bash -c "nvflare dashboard --start --local --cred <ADMIN_USER_EMAIL>:<PASSWORD>"
```
using some credentials chosen for the swarm admin account.

Access the dashboard in a web browser at `https://localhost:8443` log in with these credentials, and configure the project:
1. enter project short name, name, description
2. enter docker download link: jefftud/nvflare-pt-dev:3dcnn
3. if needed, enter dates
4. click save
5. Server Configuration > Server (DNS name): <DNS name of server>
6. click make project public

#### Register client per site
Access the dashboard at `https://<DNS name of server>:8443`.

1. register a user
2. enter organziation (corresponding to the site)
3. enter role (e.g., org admin)
4. add a site (note: must not contain spaces, best use alphanumerical name)
5. specify number of GPUs and their memory

#### Approve clients and finish configuration
Access the dashboard at `https://localhost:8443` log in with the admin credentials.
1. Users Dashboard > approve client user
2. Client Sites > approve client sites
3. Project Home > freeze project

## Download startup kits
After setting up the project admin configuration, server and clients can download their startup kits. Store the passwords somewhere, they are only displayed once (or you can download them again).

## Starting a Swarm Training
1. Connect the *server* host to the VPN as described above.
2. Start the *server* startup kit using the respective `startup/docker.sh` script with the option to start the server
3. Provide the *client* startup kits to the swarm participants (be aware that email providers or other channels may prevent encrypted archives)
4. Make sure the participants have started their clients via the respective startup kits, see below
5. Start the *admin* startup kit using the respective `startup/docker.sh` script to start the admin console
6. Deploy a job by `submit_job <job folder>`


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
