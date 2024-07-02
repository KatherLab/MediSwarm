#!/usr/bin/env bash

# TODO adapt if tests are moved
cd controller/controller

# TODO also test client
# TODO let unittest do test discovery

for testcase in swarm_server_ctl.py swarm_client_ctl.py
do
    coverage run $testcase
    coverage report -m
    rm .coverage
done
