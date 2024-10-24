
# Setting up GoodAccess VPN for ODELIA

Follow the steps below to set up the GoodAccess VPN for ODELIA:

## 1. Obtain VPN Credentials

- Request VPN credentials (including the `.ovpn` configuration file) from the Swarm admin.

## 2. Place VPN Configuration File

- Save your VPN credentials (`.ovpn` file) in this directory.


## 3. Connect to the VPN

You can connect to the VPN in one of two ways:

### Option 1: Using Graphical User Interface (GUI)
- Follow the instructions in [VPN Graphical Setup Guide (PDF)](VPN_graphical_setup_guide.pdf) to set up the VPN through your system's settings.

### Option 2: Using Command Line

- Verify VPN Configuration by opening the `.ovpn` file and check if it contains the following line:
  ```
  auth-user-pass /etc/openvpn/credentials
  ```
  **Note:** This line may be necessary if you are using a script to automate the VPN connection. Please verify whether this applies to your setup.

- Use the provided script to connect from the command line:
  ```bash
  ./envsetup_scripts/setup_vpntunnel.sh
  ```
  The script is located here [setup_vpntunnel.sh](../../../envsetup_scripts/setup_vpntunnel.sh).

## 5. Verify VPN Connection

- Once connected, ensure that a network interface (e.g., `tun0`) is active.
- Verify that it has an IP address in the `172.24.4.x` range by running the following command:
  ```bash
  ifconfig
  ```