# Third Party Information

The startup kits for all nodes provide an option to list all installed system and Python packages along with the respective licenses. System package information is listed using distro2sbom ([https://pypi.org/project/distro2sbom/](https://pypi.org/project/distro2sbom/)), python package information is listed using pip-licenses ([https://pypi.org/project/pip-licenses/](https://pypi.org/project/pip-licenses/)). License information below is provided to the extent the package listing tools can automatically extract it from the installed packages.
Note that this listing does not distinguish between packages required for the swarm learning framework and packages required for the application code (accompanying the system) to be run in the image. When using different application code, other packages or package versions may be required.

## Pre-Trained Model

* DINOv2 code and model weights are released under the Apache License 2.0.
* URL (model): [https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip](https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip)
* URL (model weights): [https://dl.fbaipublicfiles.com/dinov2/dinov2\_vits14/dinov2\_vits14\_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)

## System Packages

See [Third Party Packages](./ThirdPartyPackages.txt), top part.

## Python Packages

See [Third Party Packages](./ThirdPartyPackages.txt), bottom part.