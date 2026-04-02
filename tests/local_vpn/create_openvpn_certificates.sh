#!/usr/bin/env bash

./_build_docker.sh

docker run --rm -v ./ca_user:/home/ca_user -v ./client_configs:/client_configs -v ./server_config:/server_config -p 9194:9194/udp --cap-add=NET_ADMIN --privileged --name odelia_testing_openvpnserver odelia_testing_openvpnserver:latest /bin/bash -c "./_openvpn_certificate_creation.sh"
