#!/usr/bin/env bash

docker run -d -t --rm -v ./ca_user:/home/ca_user -v ./server_config:/server_config -p 9194:9194/udp --cap-add=NET_ADMIN --privileged --name odelia_testing_openvpnserver odelia_testing_openvpnserver:latest /bin/bash -c "./_openvpn_start.sh && /bin/bash"
