# MediSwarm VPN service (GoodAccess)

Every swarm node reaches the coordinator through the GoodAccess VPN: the node gets a fixed
address in `172.24.4.0/22`, the coordinator is `172.24.4.65` (`dl3.tud.de` in `/etc/hosts`).
`scripts/client_node_setup/` (also shipped in the client kits) runs that tunnel as the systemd
service `mediswarm-vpn`, restarted automatically and kept up by a health timer.

## Set up

From a repository checkout or from the `scripts/client_node_setup/` folder of a startup kit:

```bash
# first time: installs OpenVPN, asks for the VPN account and password, installs and starts the service
sudo ./scripts/client_node_setup/setup_vpntunnel.sh -d <Site> -n -s
# from a kit (no assets/ folder there): name your .ovpn file instead of -d
sudo ./scripts/client_node_setup/setup_vpntunnel.sh -c /path/to/<your>.ovpn -n -s
```

Use the `.ovpn` file exactly as GoodAccess delivered it; do not edit its `auth-user-pass` line.
The script writes `/etc/openvpn/client/mediswarm.conf` from it (credentials file, key files
copied to `/etc/openvpn/client/mediswarm.d/`), starts the service and **waits until the tunnel
has its 172.24.4.x address**. Then it offers the health timer (recommended).

| Exit code | Meaning |
|---|---|
| 0 | tunnel up, the address is printed |
| 1 | usage or configuration problem (message printed, nothing started) |
| 3 | another GoodAccess tunnel is already up on this host: nothing was started (see below) |
| 4 | OpenVPN failed (wrong password, unreadable file, ...): reason printed, service stopped and disabled |
| 5 | no address within 90 s (`VPN_WAIT_SECONDS`): reason printed, the service keeps trying |

## One tunnel per account

GoodAccess allows one session per account. A second tunnel with the same account (a
NetworkManager connection, a hand-started `openvpn`, the service) is refused by the gateway or
knocks the first one off, so the two never both work. The setup script and the service therefore
refuse to start while another tunnel holds a 172.24.4.x address; the service logs
`refusing to start: another GoodAccess tunnel ...` and stays stopped.

**Keep the tunnel that works.** To move from a NetworkManager or hand-started tunnel to the
service, do it between runs (a node that leaves the VPN drops out of a running job):

```bash
nmcli connection show --active                 # NetworkManager: note the UUID of the GoodAccess entry
sudo nmcli connection down <UUID>              # or stop your hand-started openvpn
sudo ./scripts/client_node_setup/setup_vpntunnel.sh -d <Site> -s     # add -n the first time
# exit code 0: keep NetworkManager from starting the old tunnel next to the service
sudo nmcli connection modify <UUID> connection.autoconnect no
# any other exit code: bring the old tunnel back and send us the diagnostics
sudo nmcli connection up <UUID>
sudo ./scripts/client_node_setup/setup_vpntunnel.sh --diagnose
```

If you keep NetworkManager or your own tunnel instead, switch the service and its timer off:
`sudo systemctl disable --now mediswarm-vpn-health.timer mediswarm-vpn`.

## Health timer

`vpn_health_monitor.sh --install-timer` (or answering Y in the setup) installs
`mediswarm-vpn-health.timer`, which checks every 30 s and restarts the service only when that can
help. It does nothing while the service is disabled or another tunnel holds the address; it never
interrupts a connection attempt younger than 3 minutes; a missing address or an unreachable
gateway must persist for 3 checks; restarts that do not bring the tunnel back are spaced out (2, 4, 8, 16, then
30 minutes). Before each restart it logs OpenVPN's last error: `journalctl -t mediswarm-vpn-health`.

## Why is there no 172.24.4.x address?

```bash
sudo ./scripts/client_node_setup/setup_vpntunnel.sh --diagnose
```

prints the tunnels on the host, the service state, problems in the installed config and
credentials file (never their content), OpenVPN's last error and the last 40 log lines. Send the
whole output to the swarm operator. Common reasons:

| Reported reason | What to do |
|---|---|
| another GoodAccess tunnel already holds the VPN address | keep one tunnel (see above) |
| the gateway rejected the login (AUTH_FAILED) | rerun with `-n -s` and type account and password again; make sure no other machine uses the account |
| OpenVPN cannot read the credentials file | rerun with `-n -s` |
| OpenVPN rejected the configuration | rerun the setup with the original GoodAccess `.ovpn` |
| no TLS handshake within 60 s | the gateway dropped the session: `.ovpn` of another gateway, or no credentials sent (`--diagnose` shows config problems); otherwise UDP 1194 is blocked |
| OpenVPN cannot read a key or certificate file | rerun the setup with the original `.ovpn`: it copies the files the `.ovpn` names to `/etc/openvpn/client/mediswarm.d/` (the service cannot read files under `/home`) |
| certificate check failed | the `.ovpn` belongs to another gateway: use the current one from GoodAccess |
| the gateway name does not resolve | DNS of the host |

## What went wrong at CAM and RUMC (#620)

At both sites the service started OpenVPN but never got an address, the setup script reported no
error, and the health timer restarted OpenVPN every 30 s. No service log was collected from either
site, so the exact cause there is **not confirmed**. The offline reproduction below shows that the
old scripts (unchanged from July to 1.9.0) produce exactly this picture in each of these cases:

1. **No credentials sent.** In a `.ovpn` without a final newline the old script glued
   `auth-user-pass <file>` to the closing `</tls-auth>` tag, where OpenVPN ignores it. The gateway
   then drops the session silently (it sends no AUTH_FAILED), so OpenVPN keeps running, retries
   every 60 s and never gets an address, and the old health timer restarted it every 30 s (a new
   PID each time, as seen at CAM on 24 Sep).
2. **`auth-user-pass /etc/openvpn/credentials`**, the line our own CLI guide and the good_access
   README told sites to add. `-n -s` stores the credentials elsewhere and the old script kept the
   line, so OpenVPN exited at every start.
3. **Another tunnel with the same account** (CAM's NetworkManager connection, RUMC's
   hand-started tunnel): refused or knocked off by the gateway.
4. **Key files under `/home`** (a NetworkManager export keeps them in `~/.cert/nm-openvpn/`):
   invisible to the service (`ProtectHome=true`).
5. **A password with a backslash** (lost by the old prompt), **`user`/`group` lines** (not
   permitted by the old unit's capabilities), and a **health timer** that restarted any attempt
   taking longer than 30 s and judged `tun0`, which can belong to another tunnel.

All of these are fixed in 1.9.0, and anything else now fails with its reason instead of silently.

## Offline reproduction

`tests/vpn_service/run_vpn_service_repro.sh` builds a GoodAccess look-alike (OpenVPN gateway with a
throwaway PKI, username/password on a shared client certificate, the gateway's push options) and a
site container, runs `setup_vpntunnel.sh -d TestSite -n -s` there with the old (`f799667`) and the
current scripts, and prints what a site would see for each case above. It needs docker and
`/dev/net/tun`, runs without `--privileged` (NET_ADMIN only, on an internal docker network) and
is not part of CI:

```bash
tests/vpn_service/run_vpn_service_repro.sh                 # all scenarios, OpenVPN 2.5 (Ubuntu 22.04)
tests/vpn_service/run_vpn_service_repro.sh --ubuntu 24.04 no_newline happy   # OpenVPN 2.6, two scenarios
```

The unit tests in `tests/unit_tests/test_vpn_service_scripts.py` cover the config generation,
the failure classification, the refusal and the health monitor's rules without docker.
