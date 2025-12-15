#!/usr/bin/env bash

# this script is called inside the ODELIA docker containers to list licenses of all pip and apt packages as well as for pre-trained weights

pip-licenses --with-system --with-urls --with-description --format json
distro2sbom -s --format json
grep "DINOv2 code and model weights are released under" /torch_home/hub/facebookresearch_dinov2_main/README.md
