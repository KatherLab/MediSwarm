#!/usr/bin/env bash

cd controller/controller

python3 -m coverage run -m unittest discover
coverage report -m
rm .coverage
