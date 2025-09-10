# Usage for Swarm Operators

## Setting up a Swarm

Production mode is designed for secure, real-world deployments. It supports both local and remote setups, whether
on-premise or in the cloud. For more details, refer to
the [NVFLARE Production Mode](https://nvflare.readthedocs.io/en/2.4.1/real_world_fl.html).

To set up production mode, follow these steps:

## Edit `/etc/hosts`

Ensure that your `/etc/hosts` file includes the correct host mappings. All hosts need to be able to communicate to the
server node.

For example, add the following line (replace `<IP>` with the server's actual IP address):

```plaintext
<IP>    dl3.tud.de dl3
```

## Create Startup Kits

### Via Script (recommended)

1. Use, e.g., the file `application/provision/project_MEVIS_test.yml`, adapt as needed (network protocol etc.)
2. Call `buildDockerImageAndStartupKits.sh -p /path/to/project_configuration.yml` to build the Docker image and the startup kits
3. Startup kits are generated to `workspace/<name configured in the .yml>/prod_00/`
4. Deploy startup kits to the respective server/client operators
5. Push the Docker image to the registry

### Via the Dashboard (not recommended)

Build the Docker image as described above.

```bash
docker run -d --rm \
     --ipc=host -p 8443:8443 \
    --name=odelia_swarm_admin \
    -v /var/run/docker.sock:/var/run/docker.sock \
    <DOCKER_IMAGE> \
    /bin/bash -c "nvflare dashboard --start --local --cred <ADMIN_USER_EMAIL>:<PASSWORD>"
```

using some credentials chosen for the swarm admin account.

Access the dashboard in a web browser at `https://localhost:8443` log in with these credentials, and configure the
project:

1. enter project short name, name, description
2. enter docker download link: jefftud/odelia:<version string>
3. if needed, enter dates
4. click save
5. Server Configuration > Server (DNS name): <DNS name of server>
6. click make project public

#### Register client per site

Access the dashboard at `https://<DNS name of server>:8443`.

1. register a user
2. enter organziation (corresponding to the site)
3. enter role (e.g., org admin)
4. add a site (note: must not contain spaces, best use alphanumerical name)
5. specify number of GPUs and their memory

#### Approve clients and finish configuration

Access the dashboard at `https://localhost:8443` log in with the admin credentials.

1. Users Dashboard > approve client user
2. Client Sites > approve client sites
3. Project Home > freeze project

#### Download startup kits

After setting up the project admin configuration, server and clients can download their startup kits. Store the
passwords somewhere, they are only displayed once (or you can download them again).

## Starting a Swarm Training

1. Connect the *server* host to the VPN as described above.
2. Start the *server* startup kit using the respective `startup/docker.sh` script with the option to start the server
3. Provide the *client* startup kits to the swarm participants (be aware that email providers or other channels may
   prevent encrypted archives)
4. Make sure the participants have started their clients via the respective startup kits, see below
5. Start the *admin* startup kit using the respective `startup/docker.sh` script to start the admin console
6. Deploy a job by `submit_job <job folder>`
