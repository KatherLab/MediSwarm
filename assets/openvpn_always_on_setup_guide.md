# OpenVPN Setup on Ubuntu

This guide explains how to set up an OpenVPN connection on **Ubuntu** using a provided `.ovpn` file and credentials, with:

- Automatic reconnection if the VPN drops
- Automatic connection on server reboot
- Secure credential handling
- systemd-based management (no cron jobs)

---
Please remember to replace <Institute_Name> with your actual institute.
Please ask Jeff if you lost your openvpn config file, which is 14357-de-9957-tt8rw.accessgate.cloud.ovpn or your credentials.


## 1. Install OpenVPN

```bash
sudo apt update
sudo apt install -y openvpn
```

---

## 2. Place the `.ovpn` Configuration File

Copy your VPN configuration file to the OpenVPN client directory and rename it:

```bash
sudo cp 14357-de-9957-tt8rw.accessgate.cloud.ovpn \
/etc/openvpn/client/<Institute_Name>.conf
```

OpenVPN expects `.conf` files in this directory for systemd integration.

---

## 3. Create a Credentials File

Create a credentials file so the VPN can connect automatically without interactive login.

```bash
sudo nano /etc/openvpn/client/<Institute_Name>.auth
```

Add the following **two lines**:

```
<VPN_USERNAME>
<VPN_PASSWORD>
```

Secure the file:

```bash
sudo chmod 600 /etc/openvpn/client/<Institute_Name>.auth
sudo chown root:root /etc/openvpn/client/<Institute_Name>.auth
```

---

## 4. Update VPN Configuration for Auto-Reconnect

Edit the configuration file:

```bash
sudo nano /etc/openvpn/client/<Institute_Name>.conf
```

Ensure the following lines are present (add them if missing):

```conf
auth-user-pass /etc/openvpn/client/<Institute_Name>.auth
auth-nocache

persist-key
persist-tun
resolv-retry infinite
keepalive 10 60
```

These options ensure:
- Infinite retry if the server is temporarily unreachable
- Automatic reconnection if the tunnel drops
- Tunnel persistence across reconnects

---

## 5. Enable VPN Auto-Start on Boot

Enable the OpenVPN client service:

```bash
sudo systemctl enable openvpn-client@<Institute_Name>
```

Start the VPN immediately:

```bash
sudo systemctl start openvpn-client@<Institute_Name>
```

---

## 6. Verify VPN Status

Check service status:

```bash
sudo systemctl status openvpn-client@<Institute_Name>
```

Follow logs (useful for debugging):

```bash
journalctl -u openvpn-client@<Institute_Name> -f
```

---

## 7. Confirm VPN Connectivity

Check network interfaces:

```bash
ip a
```

You should see an interface named `tun0` or similar with an IP address starting with `172.24.4.`

---

## 8. Behavior Summary

| Event                     | Result                           |
|---------------------------+----------------------------------|
| VPN connection drops      | Automatically reconnects         |
| VPN server unavailable    | Retries indefinitely             |
| Server reboot             | VPN reconnects on startup        |
| Network delay during boot | systemd retries until successful |

---

## Notes

- Credentials are stored securely with restricted permissions.
- No cron jobs or custom watchdog scripts are required.
- Configuration is suitable for headless servers.

---

## Troubleshooting

### Disconnecting Existing VPN Connections

Some users have experienced that connecting to GoodAccess **disconnects an existing VPN or ssh connection**.
This may happen because OpenVPN is configured to redirect all network traffic through the GoodAccess tunnel, which overrides your local or other VPN routes and may make the machine inaccessible in its local network.

If this occurs, you can prevent the redirection by starting OpenVPN with
```sh
openvpn --config <your_config>.ovpn --pull-filter ignore redirect-gateway
```
This tells the OpenVPN client **not** to override your default gateway, allowing your other VPN or ssh connection to remain active.

**TODO** describe how this can be configured in when starting the VPN as a system service.