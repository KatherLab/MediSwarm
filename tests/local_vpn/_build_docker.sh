#!/usr/bin/env bash

# TODO should this be named "latest"? Do we need to pin versions?
# TODO think about splitting building certificates from running the VPN container

docker build -t odelia_testing_openvpnserver:latest . -f Dockerfile_openvpnserver
