# GoodAccess VPN Setup Guide (CLI)

This guide provides step-by-step instructions to set up and connect to GoodAccess VPN on a Linux machine using the command-line interface (CLI).

---

## Step 1: Receive VPN Credentials and Setup Files

You will receive the following via a secure communication channel (e.g., email):
- VPN **username** and **password**
- A setup file (typically in `.zip` format)

---

## Step 2: Unzip the Setup File

1. Locate the received `.zip` file.
2. Extract its contents:
   ```sh
   unzip <setup_file>.zip
   ```
3. Move the `.ovpn` file to the correct directory:
   ```sh
   mv <your_config>.ovpn assets/openvpn_configs/good_access
   ```
4. Open the `.ovpn` file and verify that the following line exists:
   ```
   auth-user-pass /etc/openvpn/credentials
   ```
   - If it does not exist, **manually add it** after `auth-user-pass`.

---

## Step 3: Establish a New VPN Connection

To initiate a VPN connection, run:
```sh
sh envsetup_scripts/setup_vpntunnel.sh -d <host_index> -n
```
Where `<host_index>` is your institute's name. For the ODELIA project, choose from:
- TUD
- Ribera
- VHIO
- Radboud
- UKA
- UMCU
- MHA
- Cambridge
- USZ
- MEVIS

You will be prompted to enter the **username** and **password** provided by TUD.

---

## Step 4: Verify VPN Connection

To confirm that you are connected, check your IP address:
```sh
hostname -I
```
You should see an IP in the range **172.24.4.xx/22**.

---

## Step 5: Reconnect to VPN

If your machine restarts or loses connection, reconnect by running:
```sh
sh envsetup_scripts/setup_vpntunnel.sh
```

The `.ovpn` file assigned to you by TUD is required for re-establishing the connection.

For further troubleshooting, refer to the **VPN Connect Guide**.

---

This guide ensures a smooth setup and reconnection process for GoodAccess VPN via CLI.