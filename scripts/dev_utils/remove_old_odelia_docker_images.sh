#!/usr/bin/env bash

export OLD_ODELIA_DOCKER_IMAGES=$(docker image list | grep jefftud/odelia | sed 's|jefftud/odelia *[0-9a-z.-]* *||' | sed 's|  *.*||' | tail -n +2)

echo "All docker images:"

docker image list

echo "The following Docker images are old ODELIA docker images:"

echo $OLD_ODELIA_DOCKER_IMAGES

read -p "Delete these Docker images, unless they have additional tags? (y/n): " answer

if [[ "$answer" == "y" ]]; then
    for image in $OLD_ODELIA_DOCKER_IMAGES; do
        docker rmi $image
    done
fi
