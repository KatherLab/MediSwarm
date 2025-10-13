#!/usr/bin/env bash

set -e

OUTPUT_FOLDER=workspace/`grep "^name: " $1 | sed 's/name: //'`
TARGET_FOLDER=`ls -d $OUTPUT_FOLDER/prod_* | tail -n 1`
LONG_VERSION=$2

cd $TARGET_FOLDER

for startupkit in `ls .`; do
    VPN_CREDENTIALS_FILE=/vpn_credentials/${startupkit}_client.ovpn
    if [[ -f $VPN_CREDENTIALS_FILE ]]; then
        cp $VPN_CREDENTIALS_FILE ${startupkit}/startup/vpn_client.ovpn
    else
        echo "$VPN_CREDENTIALS_FILE does not exist, omitting VPN credentials for ${startupkit} in startup kit"
    fi
    zip -rq ${startupkit}_$LONG_VERSION.zip $startupkit
    echo "Generated startup kit $TARGET_FOLDER/${startupkit}_$LONG_VERSION.zip"
done
