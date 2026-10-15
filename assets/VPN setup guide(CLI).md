# GoodAccess VPN Setup Guide (CLI)

This guide sets up the GoodAccess VPN of a swarm node on Linux from the command line. Background,
troubleshooting and the one-tunnel-per-account rule: [docs/VPN_SERVICE.md](../docs/VPN_SERVICE.md).

---

## Step 1: Receive VPN Credentials and Setup Files

You will receive the following via a secure communication channel (e.g., email):
- VPN **username** and **password**
- A setup file (typically in `.zip` format) containing your `.ovpn` file

---

## Step 2: Unzip the Setup File

```sh
unzip <setup_file>.zip
mv <your_config>.ovpn assets/openvpn_configs/good_access/<host_index>.ovpn
```

Use the `.ovpn` file **unchanged**. Earlier versions of this guide asked you to add
`/etc/openvpn/credentials` after `auth-user-pass`; that is no longer needed and stopped the VPN
service from starting (#620). The setup script takes care of the credentials.

Working from a startup kit instead of the repository? Keep the `.ovpn` anywhere and pass it with
`-c <path to your .ovpn>` instead of `-d <host_index>` below.

---

## Step 3: Install the VPN Service

```sh
sudo ./scripts/client_node_setup/setup_vpntunnel.sh -d <host_index> -n -s
```

`<host_index>` is the file name you used in Step 2 (e.g. `Cambridge` for `Cambridge.ovpn`). You
will be prompted for the **username** and **password** provided by TUD. The script installs the
`mediswarm-vpn` systemd service (reconnects automatically, starts at boot), starts it and waits
until the tunnel has its address. If the tunnel does not come up, it prints the reason and the
last OpenVPN messages. Afterwards it offers the health timer: answer `Y`.

If you already run a GoodAccess tunnel on this machine (NetworkManager or a hand-started
`openvpn`), the script does not start a second one: one account allows one tunnel. See
[docs/VPN_SERVICE.md](../docs/VPN_SERVICE.md#one-tunnel-per-account) for how to switch.

---

## Step 4: Verify VPN Connection

```sh
hostname -I                          # shows an address 172.24.4.x
systemctl status mediswarm-vpn       # "active (running)"
```

---

## Step 5: Reconnect to VPN

The service reconnects by itself after a drop and after a reboot. To restart it by hand:
`sudo systemctl restart mediswarm-vpn`.

Without `-s` the script starts OpenVPN in the background once (legacy mode); after a reboot or a
drop you have to run `sudo ./scripts/client_node_setup/setup_vpntunnel.sh -d <host_index>` again.

For further troubleshooting, refer to the VPN Connect Guide on the GoodAccess support page:
[GoodAccess VPN Connect Guide](https://support.goodaccess.com/configuration-guides/linux)

---

## Step 6: Troubleshooting

**No 172.24.4.x address.** Run the diagnostics and send the output to your swarm operator:

```sh
sudo ./scripts/client_node_setup/setup_vpntunnel.sh --diagnose
```

**Connecting to GoodAccess disconnects an existing VPN or ssh connection.**
The gateway redirects all network traffic through the GoodAccess tunnel, which overrides your
local or other VPN routes and may make the machine inaccessible in its local network. To keep
your default route, add this line to `/etc/openvpn/client/mediswarm.conf` and restart the service
(`sudo systemctl restart mediswarm-vpn`), or pass it on the command line of a hand-started
`openvpn`:
```
pull-filter ignore redirect-gateway
```

> **Note:** This behavior was observed by Aitor and Ole after certain OpenVPN updates. The above setting has been effective in resolving the issue.
