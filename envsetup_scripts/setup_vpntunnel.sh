#!/usr/bin/env bash

set -e

ACTION="nochange"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d)  host_index="$2"; shift ;;
        -n)  ACTION="new";;
        -h)  ACTION="help";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ $ACTION = "help" ]]; then
   echo "Usage: setup_vpntunnel.sh -d <host_index> [-n]"
   echo "       -n   one-time setup"
   echo ""
   exit 1
fi

if [ -z "$host_index" ]; then
   echo "Please specify your host index via option -d <host_index>"
   echo "Host index should be chosen from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich]"
   exit 1
fi

if [ $ACTION = "new" ]; then
    # to the initial setup
    echo "Setting up VPN tunnel for swarm learning ..."
    sudo apt-get -y install openvpn

    echo "Please enter your vpn credentials (ask TUD maintainer for the account and password if you don't have the data yet)"
    read -p "vpn account: " vpn_account
    stty -echo
    read -p "vpn password: " vpn_password
    stty echo
    printf '%s\n' $vpn_account $vpn_password | sudo tee /etc/openvpn/credentials > /dev/null
    sudo chmod 600 /etc/openvpn/credentials
fi

if [[ ! -f ./assets/openvpn_configs/good_access/$host_index.ovpn ]]; then
    echo "Configuration file ./assets/openvpn_configs/good_access/$host_index.ovpn not found"
    exit 1
fi

if [[ ! -f /etc/openvpn/credentials ]]; then
    echo "Credentials file /etc/openvpn/credentials not found, please use option -n to create it"
    exit 1
fi

echo "Starting VPN tunnel for swarm learning ..."
sudo nohup openvpn --config ./assets/openvpn_configs/good_access/$host_index.ovpn &
sleep 3
sudo chmod a+r nohup.out

if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above or nohup.out for more details."
    exit 1
fi

hostname -I

echo "You should see an IP address of the form 172.24.4.x in the output above."
echo "If it does not appear, try 'hostname -I' again in 10 or 20 seconds."
