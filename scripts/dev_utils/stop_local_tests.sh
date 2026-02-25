#! /bin/bash

docker stop $(docker ps -q --filter name=odelia_swarm*)
