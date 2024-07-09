#!/usr/bin/env bash

# run NVFlare's tests (takes about 2 minutes)
cd nvflare
./runtest.sh
cd ..

cd controller/controller

python3 -m coverage run -m unittest discover
coverage report -m
rm .coverage
