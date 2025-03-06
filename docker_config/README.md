
# Docker Image Build and Push Instructions

TODO clean up remaining instructions

### 5. Manually Build and Push `nvflare-pt-dev:cifar10` Image

Build the Docker image using the `Dockerfile_cifar10` file and push it to Docker Hub.

```sh
docker build -t nvflare-pt-dev:cifar10 . -f Dockerfile_cifar10
docker tag nvflare-pt-dev:cifar10 jefftud/nvflare-pt-dev:cifar10
docker push jefftud/nvflare-pt-dev:cifar10
```

## Notes

- Ensure you have the necessary permissions to push images to the `jefftud` repository on Docker Hub.
