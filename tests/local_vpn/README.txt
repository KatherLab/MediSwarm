# Following https://www.digitalocean.com/community/tutorials/how-to-set-up-and-configure-an-openvpn-server-on-ubuntu-20-04
# but on 22.04

Setup
-----
./create_openvpn_certificates.sh builds a docker image and creates certificates and .ovpn config files for the clients specified in _openvpn_certificate_creation.sh
Modify server_config/server.conf and client_configs/client.conf to modify network configuration.
Files to use on the server and client are created in server_config/ and client_configs/

Usage
-----
./openvpn_start.sh builds a docker image and starts OpenVPN server in the docker container.
Modify _openvpn_start.sh for further firewall etc. configuration.

Disclaimer
----------
This configuration is not necessarily secure and should not be re-used unless you know what you are doing.
