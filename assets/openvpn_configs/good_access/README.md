# Setting up GoodAccess VPN for ODELIA

1. Obtain credentials for goodaccess from swarm admin
2. Place your VPN credentials (.ovpn file) in this folder
3. Check if the .ovpn file contains the entry `/etc/openvpn/credentials` after `auth-user-pass` TODO is this necessary only when using the script or also for the graphical setup?
4. Open the VPN connection
   1. Either following the instructions in TODO pdf via the system settings dialog
   2. Or from the command line using using [../../../envsetup_scripts/setup_vpntunnel.sh](../../../envsetup_scripts/setup_vpntunnel.sh)
5. Check if a network interface (e.g., tun0) with an IP address in the range 172.24.4.x is available (e.g., using `ifconfig` from the command line)
