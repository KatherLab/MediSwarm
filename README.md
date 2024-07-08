# MediSwarm

## Introduction
MediSwarm is an open-source project dedicated to advancing medical deep learning through swarm intelligence, leveraging the NVFlare platform. Developed in collaboration with the Odelia consortium, this repository aims to create a decentralized and collaborative framework for medical research and applications.

## Key Features
- **Swarm Learning:** Utilizes swarm intelligence principles to improve model performance and adaptability.
- **NVFlare Integration:** Built on NVFlare, providing robust and scalable federated learning capabilities.
- **Data Privacy:** Ensures data security and compliance with privacy regulations by keeping data local to each institution.
- **Collaborative Research:** Facilitates collaboration among medical researchers and institutions for enhanced outcomes.
- **Extensible Framework:** Designed to support various medical applications and easily integrate with existing workflows.

## Install

### Prerequisites
#### Hardware recommendations
* 64 GB of RAM (32 GB is the absolute minimum)
* 16 CPU cores (8 is the absolute minimum)
* an NVIDIA GPU with 48 GB of RAM (24 is the  minimum)
* 8 TB of Storage (4 TB is the absolute minimum)
* We deliberately want to show that we can work with lightweight hardware like this. Here are three quotes for systems like this for less than 10k EUR (Lambda, Dell Precision, and Dell Alienware)

#### Operating System
* Ubuntu 20.04 LTS

## Usage for Developers

### Setup

0. **Clone the repository:**

    ```bash
    git clone https://github.com/KatherLab/MediSwarm.git
    cd MediSwarm
    ```

1. **Run with Docker Environment:**

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

2. **Run Simulator:**

    The FL Simulator is a lightweight tool that uses threads to simulate different clients. This is useful for quick research runs and debugging applications locally.

    ```bash
    nvflare simulator -w /tmp/3dcnn_ptl -n 2 -t 2 application/jobs/3dcnn_ptl -c manual_dl0,manual_dl3
    ```

    For more details on using the simulator, refer to the [NVFLARE Quick Start with Simulator](https://nvflare.readthedocs.io/en/2.4.1/getting_started.html#quick-start-with-simulator).

3. **Run POC Mode:**

    Proof of Concept (POC) mode allows for quick local setups on a single machine, where the FL server and clients run in different processes or Docker containers.

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

4. **Run Production Mode:**

    Production mode is secure and suitable for real-world deployments, supporting local or remote, on-premise or cloud setups. For more information, refer to the [NVFLARE Production Mode](https://nvflare.readthedocs.io/en/2.4.1/real_world_fl.html).

    ```bash

    **Set hosts correctly in `/etc/hosts`:**

    Edit your `/etc/hosts` file to include the following line, replacing `<IP>` with the actual IP address of the server:

    ```plaintext
    <IP>    dl3.tud.de dl3
    ```

    This line maps the hostname `dl3.tud.de` to the IP address of the server, ensuring proper network communication.

    ```bash
    docker run -it --rm \
        --ipc=host \
        -v ./docker_config/NVFlare:/workspace/nvflare \
        -v ./:/workspace \
        -v /var/run/docker.sock:/var/run/docker.sock \
        jefftud/nvflare-pt-dev:nfcore \
        /bin/bash

    nvflare dashboard --port 443 --start -i jefftud/nvflare-pt-dev:nfcore
    ```

    Access the dashboard at `https://localhost:443`. After project admin configuration, clients download the startup kits and run the following commands to join the server:

    For more information on the NVFLARE dashboard, see the [NVFLARE Dashboard](https://nvflare.readthedocs.io/en/2.4.1/user_guide/dashboard_ui.html).

    ```bash
    ./docker.sh
    cd startup
    ./start.sh
    ```

    Admin submits the job and initiates the training by logging in with the admin user email and submitting the job folder.

## Unit Tests

### Create Docker Image

Build docker image nvflare-pt-dev:testing from docker_config/Dockerfile_testing:

```bash
cd docker_setup
docker build . -t nvflare-pt-dev:testing -f Dockerfile_testing
```

### Run Tests

```bash
./runTestsInDocker.sh
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Maintainers
[@Jeff](https://github.com/Ultimate-Storm).

## Contributing
Feel free to dive in! [Open an issue](https://github.com/KatherLab/MediSwarm/issues) or submit PRs.

## Credits

This project utilizes platforms and resources from the following repositories:

- **[NVFLARE](https://github.com/NVIDIA/NVFlare)**: NVFLARE (NVIDIA Federated Learning Application Runtime Environment) is an open-source framework that provides a robust and scalable platform for federated learning applications. We have integrated NVFLARE to handle the federated learning aspects of our project efficiently.

Special thanks to the contributors and maintainers of these repositories for their valuable work and support.

---

For more details about NVFLARE and its features, please visit the [NVFLARE GitHub repository](https://github.com/NVIDIA/NVFlare).
