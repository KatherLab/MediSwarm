#!/usr/bin/env bash

VERSION=`tail -n 1 odelia_image.version`

GIT_SHORT_HASH=`git rev-parse --short HEAD`
DATE=`date "+%y%m%d"`
LONG_VERSION=$VERSION-dev.$DATE.$GIT_SHORT_HASH

echo $LONG_VERSION
