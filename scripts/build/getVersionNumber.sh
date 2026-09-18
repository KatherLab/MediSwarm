#!/usr/bin/env bash

# A CI run can span local midnight. `date` would then yield a different YYMMDD
# for the step that BUILDS the image and the step that CONSUMES it, so the kit
# build looks up a tag that was never created (#428). Export
# MEDISWARM_IMAGE_VERSION once per run and every caller reuses that value.
if [ -n "${MEDISWARM_IMAGE_VERSION:-}" ]; then
    echo "$MEDISWARM_IMAGE_VERSION"
    exit 0
fi

VERSION=`tail -n 1 odelia_image.version`

GIT_SHORT_HASH=`git rev-parse --short HEAD`
DATE=`date "+%y%m%d"`
LONG_VERSION=$VERSION-dev.$DATE.$GIT_SHORT_HASH

echo $LONG_VERSION
