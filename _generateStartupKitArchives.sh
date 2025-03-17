#!/usr/bin/env bash

OUTPUT_FOLDER=workspace/`grep "^name: " $1 | sed 's/name: //'`
TARGET_FOLDER=`ls -d $OUTPUT_FOLDER/prod_* | tail -n 1`

GIT_SHORT_HASH=`git rev-parse --short HEAD`
DATE=`date "+%y%m%d"`
LONG_VERSION=$2-dev.$DATE.$GIT_SHORT_HASH

cd $TARGET_FOLDER
for startupkit in `ls .`; do
    zip -rq ${startupkit}_$LONG_VERSION.zip $startupkit
    echo "Generated startup kit $TARGET_FOLDER/${startupkit}_$LONG_VERSION.zip"
done
