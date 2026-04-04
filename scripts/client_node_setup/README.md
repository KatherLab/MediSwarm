# Client Node Setup Scripts

Scripts for setting up and verifying participant (client) nodes before joining a MediSwarm swarm training session.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `setup_vpntunnel.sh` | Set up and start the GoodAccess OpenVPN tunnel |
| `gpu_env_setup.sh` | Install and configure NVIDIA container runtime for Docker |
| `test_dockersetup.sh` | Verify Docker installation |
| `test_open_exposed_ports.sh` | Check and open required network ports |
| `get_dataset_gdown.sh` | Download ODELIA dataset from Google Drive |
| `get_dataset_scp.sh` | Download ODELIA dataset via SCP from sentinel node |

---

## `setup_vpntunnel.sh`

Sets up and starts the GoodAccess/OpenVPN tunnel required for swarm communication between sites.

### Prerequisites

- Ubuntu Linux
- `sudo` access
- VPN credentials from the TUD maintainer
- OpenVPN config file in `assets/openvpn_configs/good_access/<host_index>.ovpn`

### Usage

```bash
# First-time setup (installs OpenVPN, prompts for credentials):
./scripts/client_node_setup/setup_vpntunnel.sh -d <host_index> -n

# Subsequent runs (just start the tunnel):
./scripts/client_node_setup/setup_vpntunnel.sh -d <host_index>

# Show help:
./scripts/client_node_setup/setup_vpntunnel.sh -h
```

### Host Index Values

Choose from: `TUD`, `Ribera`, `VHIO`, `Radboud`, `UKA`, `Utrecht`, `Mitera`, `Cambridge`, `Zurich`

### Expected Result

After starting, `hostname -I` should show a VPN IP in the `172.24.4.x` range. If not visible immediately, retry after 10-20 seconds.

### Credentials

Stored at `/etc/openvpn/credentials` (chmod 600). Created during first-time setup (`-n` flag).

---

## `gpu_env_setup.sh`

Installs and configures the NVIDIA container runtime so Docker containers can access the GPU.

### Prerequisites

- NVIDIA GPU drivers already installed (`nvidia-smi` must work)
- Docker installed
- `sudo` access

### Usage

```bash
./scripts/client_node_setup/gpu_env_setup.sh
```

### What It Does

1. Verifies NVIDIA drivers are loaded (`nvidia-smi`)
2. Adds NVIDIA Docker repository
3. Installs `nvidia-container-toolkit` and `nvidia-container-runtime`
4. Restarts Docker service
5. Verifies GPU access from inside a container (`docker run --gpus all ubuntu nvidia-smi`)

---

## `test_dockersetup.sh`

Quick smoke test to verify Docker is installed and the current user can run containers.

### Usage

```bash
./scripts/client_node_setup/test_dockersetup.sh
```

### What It Does

1. Creates the `docker` group if needed
2. Adds current user to the `docker` group
3. Runs `docker run hello-world` to verify

---

## `test_open_exposed_ports.sh`

Checks that all network ports required for swarm learning are open in `iptables`. Automatically opens any closed ports.

### Usage

```bash
./scripts/client_node_setup/test_open_exposed_ports.sh
```

### Ports Checked

| Port | Purpose |
|------|---------|
| 22 | SSH |
| 5814 | HPE SL |
| 30303-30306 | HPE SL peer communication |
| 16000-19000 | Application-specific |

---

## `get_dataset_gdown.sh`

Downloads the ODELIA dataset from Google Drive using `gdown`. Faster than SCP for initial setup.

### Prerequisites

- `pip` (installs `gdown` automatically)
- Internet access to Google Drive

### Usage

```bash
./scripts/client_node_setup/get_dataset_gdown.sh
```

### What It Downloads

- ODELIA breast MRI dataset (ZIP) into the workspace
- Marugoto MRI features (ZIP) into the workspace
- Automatically unzips both

---

## `get_dataset_scp.sh`

Downloads the ODELIA dataset from the sentinel node via SCP.

### Prerequisites

- SSH access to the sentinel node
- Sentinel node IP address

### Usage

```bash
./scripts/client_node_setup/get_dataset_scp.sh -s <SENTINEL_IP>
```

### What It Downloads

Same datasets as `get_dataset_gdown.sh`, but fetched from the sentinel node instead of Google Drive.

---

## Typical Setup Order

For a new client node, run these scripts in order:

1. `test_dockersetup.sh` -- Verify Docker works
2. `gpu_env_setup.sh` -- Set up GPU access in Docker
3. `setup_vpntunnel.sh -d <host> -n` -- One-time VPN setup
4. `setup_vpntunnel.sh -d <host>` -- Start VPN
5. `test_open_exposed_ports.sh` -- Open required ports
6. `get_dataset_gdown.sh` or `get_dataset_scp.sh` -- Download data
