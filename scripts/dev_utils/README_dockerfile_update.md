# Updating apt Package Versions in Dockerfiles

## Motivation

We use hard-coded version numbers of packages installed via apt and pip to avoid silently getting different versions.
When updated, previous apt packages are sometimes no longer available, so we also need to update the version we install and test if everything still works.
It is tedious to do this manually and it is easy to miss potential new dependencies that should also be installed at a defined version.
These scrips are intended to reduce the number of manual steps.

## Usage

If an update is necessary:

```bash
scripts/dev_utils/dockerfile_update_removeVersionApt.py docker_config/Dockerfile_ODELIA
git commit docker_config/Dockerfile_ODELIA -m 'WIP DO NOT PUSH'
./buildDockerImageAndStartupKits.sh -p tests/provision/dummy_project_for_testing.yml 2>&1 | tee out.txt
scripts/dev_utils/dockerfile_update_addAptVersionNumbers.py docker_config/Dockerfile_ODELIA out.txt
rm out.txt
git commit docker_config/Dockerfile_ODELIA --amend -m 'updated apt versions'
```

If you see the message `No changes You asked to amend the most recent commit, but doing so would make it empty. […]`, there was no update.
Do not amend, but `git reset --hard HEAD~1`.

If any dependencies were installed that are not listed explicitly, those packages are printed and should be added to the Dockerfile.

## Limitations

Updating of pip package versions not implemented yet, this does not seem to be necessary very often.
