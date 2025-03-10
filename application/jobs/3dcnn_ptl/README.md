
# Guide to Run 3D CNN PyTorch Lightning Application

## 1. Run with Docker Environment

To run the 3D CNN application using a Docker container, follow these steps:

```bash
# Set the DATADIR variable
export DATADIR=<your_data_directory>
# Run the Docker container
docker run -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ./docker_config/NVFlare:/workspace/nvflare \
    --gpus=all \
    -v ./:/workspace \
    -v $DATADIR:/data \
    jefftud/nvflare-pt-dev:3dcnn
```

- `DATADIR`: Set this variable to point to your local data directory.
- `--shm-size=16g`: Allocates 16 GB of shared memory.
- `--ipc=host`: Shares the IPC namespace with the host to enable better communication.
- `--ulimit memlock=-1`: Removes memory lock limitations.
- `--ulimit stack=67108864`: Sets the stack size limit to 64 MB.

## 2. Run the FL Simulator

The FL Simulator is a lightweight tool that uses threads to simulate multiple clients. It is useful for quick local testing and debugging. Run the following command to start the simulator:

```bash
nvflare simulator -w /tmp/3dcnn_ptl -n 2 -t 2 application/jobs/3dcnn_ptl -c manual_dl0,manual_dl3
```

- `-w /tmp/3dcnn_ptl`: Specifies the working directory.
- `-n 2`: Sets the number of clients.
- `-t 2`: Specifies the number of threads.
- `-c manual_dl0,manual_dl3`: Selects the manual data loaders.

For more details, refer to the [NVFLARE Quick Start with Simulator](https://nvflare.readthedocs.io/en/2.4.1/getting_started.html#quick-start-with-simulator).

## 3. Run POC Mode

Proof of Concept (POC) mode enables quick local setups on a single machine. The FL server and clients run in separate processes or Docker containers. To run POC mode:

```bash
# With Docker (requires Docker in Docker)
nvflare poc prepare -c manual_dl0 manual_dl3 -d jefftud/nvflare-pt-dev:3dcnn
# Without Docker
nvflare poc prepare -c manual_dl0 manual_dl3

nvflare poc prepare-jobs-dir -j application/jobs/

# Start POC
nvflare poc start
```

For more information on POC mode, see the [NVFLARE POC Commands](https://nvflare.readthedocs.io/en/2.4.1/user_guide/nvflare_cli/poc_command.html).

## 4. Run Production Mode

Production mode is designed for secure, real-world deployments. It supports both local and remote setups, whether on-premise or in the cloud. For more details, refer to the [NVFLARE Production Mode](https://nvflare.readthedocs.io/en/2.4.1/real_world_fl.html).

To set up production mode, follow these steps:

### Edit `/etc/hosts`

Ensure that your `/etc/hosts` file includes the correct host mappings. For example, add the following line (replace `<IP>` with the server's actual IP address):

```plaintext
<IP>    dl3.tud.de dl3
```

### Start the production setup

```bash
docker run -d --rm \
     --ipc=host -p 8443:8443 \
    --name=odelia_swarm_admin \
    -v /var/run/docker.sock:/var/run/docker.sock \
    jefftud/nvflare-pt-dev:nfcore \
    /bin/bash -c "nvflare dashboard --start --local --cred <ADMIN_USER_EMAIL>:<PASSWORD>"
```

using some credentials chosen for the swarm admin account.

Access the dashboard at `https://localhost:8443` log in with these credentials, and configure the project:

* enter project short name, name, description
* enter docker download link: jefftud/nvflare-pt-dev:3dcnn
* if needed, enter dates
* click save
* Server Configuration > Server (DNS name): <DNS name of server>
* click make project public

### Register client per site

Access the dashboard at `https://<DNS name of server>:8443`.

* register a user
* enter organziation (corresponding to the site)
* enter role (e.g., org admin)
* add a site (note: must not contain spaces, best use alphanumerical name)
* specify number of GPUs and their memory

### Approve clients and finish configuration

Access the dashboard at `https://localhost:8443` log in with the admin credentials.
* Users Dashboard > approve client user
* Client Sites > approve client sites
* Project Home > freeze project

### Download startup kits

After setting up the project admin configuration, server and clients can download their startup kits. Store the passwords somewhere, they are only displayed once (or you can download them again).

Start the startup kits using

```bash
./docker.sh
cd startup
./start.sh
```

and enter the information you are prompted for, or look at docker.sh how to pass arguments from the command line.

### Deploying a training

The admin can submit jobs and initiate training by logging in with the admin email and submitting the job folder. For more information, refer to the [NVFLARE Dashboard](https://nvflare.readthedocs.io/en/2.4.1/user_guide/dashboard_ui.html).

### TODO

* merge these instructions with changes from other branches
* check if (and how) the dashboard can be served and accessed via a standard port