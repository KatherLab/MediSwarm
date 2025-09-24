#!/usr/bin/env bash

set -e

OUTPUT_FOLDER=workspace/`grep "^name: " $1 | sed 's/name: //'`
TARGET_FOLDER=`ls -d $OUTPUT_FOLDER/prod_* | tail -n 1`
LONG_VERSION=$2

# TODO copy from different location

cd $TARGET_FOLDER
for startupkit in `ls .`; do
    cp ../../../tests/local_vpn/client_configs/${startupkit}_client.ovpn ${startupkit}/startup/vpn_client.ovpn
    zip -rq ${startupkit}_$LONG_VERSION.zip $startupkit
    echo "Generated startup kit $TARGET_FOLDER/${startupkit}_$LONG_VERSION.zip"
done
