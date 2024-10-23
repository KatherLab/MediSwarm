
# MediSwarm

## Introduction
MediSwarm is an open-source project dedicated to advancing medical deep learning through swarm intelligence, leveraging the NVFlare platform. Developed in collaboration with the Odelia consortium, this repository aims to create a decentralized and collaborative framework for medical research and applications.

## Key Features
- **Swarm Learning:** Utilizes swarm intelligence principles to improve model performance and adaptability.
- **NVFlare Integration:** Built on NVFlare, providing robust and scalable federated learning capabilities.
- **Data Privacy:** Ensures data security and compliance with privacy regulations by keeping data local to each institution.
- **Collaborative Research:** Facilitates collaboration among medical researchers and institutions for enhanced outcomes.
- **Extensible Framework:** Designed to support various medical applications and easily integrate with existing workflows.

### Prerequisites
#### Hardware recommendations
* 64 GB of RAM (32 GB is the absolute minimum)
* 16 CPU cores (8 is the absolute minimum)
* an NVIDIA GPU with 48 GB of RAM (24 GB is the minimum)
* 8 TB of Storage (4 TB is the absolute minimum)

We demonstrate that the system can run on lightweight hardware like this. For less than 10k EUR, you can configure systems from suppliers like Lambda, Dell Precision, and Dell Alienware.

#### Operating System
* Ubuntu 20.04 LTS

## Usage for Developers

### Setup

0. **Clone the repository:**

    ```bash
    git clone https://github.com/KatherLab/MediSwarm.git
    cd MediSwarm
    ```

1. **Set up VPN Connection:**
  1. Obtain credentials for goodaccess from swarm admin
  2. Either use [envsetup_scripts/setup_vpntunnel.sh ](envsetup_scripts/setup_vpntunnel.sh) from the command line or follow TODO via system settings to set up the VPN connection
  3. Call `ifconfig` and check if a network interface (e.g., tun0) with an IP address in the range 172.24.4.x is available

### Running the Application

2. **Run the minimal CIFAR-10 example:**
   See [cifar10/README.md](application/jobs/cifar10/README.md)

3. **Run the 3D CNN for classifying breast tumors:**
   See [3dcnn_ptl/README.md](application/jobs/3dcnn_ptl/README.md)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Maintainers
[Jeff](https://github.com/Ultimate-Storm)
[Ole Schwen](mailto:ole.schwen@mevis.fraunhofer.de)
[Steffen Renisch](mailto:steffen.renisch@mevis.fraunhofer.de)

## Contributing
Feel free to dive in! [Open an issue](https://github.com/KatherLab/MediSwarm/issues) or submit pull requests.

## Credits
This project utilizes platforms and resources from the following repositories:

- **[NVFLARE](https://github.com/NVIDIA/NVFlare)**: NVFLARE (NVIDIA Federated Learning Application Runtime Environment) is an open-source framework that provides a robust and scalable platform for federated learning applications. We have integrated NVFLARE to efficiently handle the federated learning aspects of our project.

Special thanks to the contributors and maintainers of these repositories for their valuable work and support.

---

For more details about NVFLARE and its features, please visit the [NVFLARE GitHub repository](https://github.com/NVIDIA/NVFlare).
