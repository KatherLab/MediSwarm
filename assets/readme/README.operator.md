# Usage for Swarm Operators

## Setting up a Swarm

Production mode is designed for secure, real-world deployments. It supports both local and remote setups, whether
on-premise or in the cloud. For more details, refer to
the [NVFLARE Production Mode](https://nvflare.readthedocs.io/en/2.7/real_world_fl.html).

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
  * when adapting the server host name or ports, the server’s `name:`, `fed_learn_port`, and `admin_port` must match the `sp_end_point` in the `overseer_agent` section
2. Call `scripts/build/buildDockerImageAndStartupKits.sh -p /path/to/project_configuration.yml` to build the Docker image and the startup kits
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
6. Log in using the user name configured as "name" of the node of type "admin" (only user name needed, auth happens via certificate)
7. Deploy a job by `submit_job <job folder>`

### Fresh vs Continue ODELIA Jobs

Admin startup kits include `prepare_odelia_job.sh`, which copies an in-image job into the admin kit's mounted
`local/mediswarm_jobs` folder and patches only that copied job. With no custom `--min-*` options, the helper
defaults to the current exact eight-site ODELIA production roster (CAM, VHIO, USZ, RUMC, MHA, RSH, UMCU,
and UKA), and synchronizes mandatory/participating clients plus every quorum value. A production run therefore
cannot silently fall back to a 7-of-8 policy. Use `--strict-clients CLIENT_1,CLIENT_2,...` for a different exact
roster. Supplying `--min-clients`, `--configure-min-clients`, or `--min-responses` opts into a custom test policy.

Fresh start:
```bash
cd <admin-kit>/startup
./prepare_odelia_job.sh --job ODELIA_ternary_classification --warm-start fresh
./docker.sh --no_pull
```
Submit the printed `_fresh` path, for example:
```text
submit_job /fl_admin/local/mediswarm_jobs/ODELIA_ternary_classification_fresh
```

Continue from the last client-local global checkpoint:
```bash
cd <admin-kit>/startup
./prepare_odelia_job.sh --job ODELIA_ternary_classification --warm-start continue
./docker.sh --no_pull
```
Submit the printed `_continue` path, for example:
```text
submit_job /fl_admin/local/mediswarm_jobs/ODELIA_ternary_classification_continue
```

`continue` is strict: each client must have `/scratch/mediswarm_latest_global.pt` available through the same
`--scratch_dir` used by the previous run. If any client is missing that checkpoint, the job aborts with
`WARM_START_REQUIRED_MISSING` instead of silently starting fresh. `fresh` ignores old local checkpoints without
deleting them. Direct `submit_job MediSwarm/application/jobs/...` remains supported for generic/test workflows and
keeps automatic warm-start behavior, but it bypasses the helper's exact-client production guard.

The direct-submit "automatic warm-start" is `warm_start_mode = "auto"` (the shipped default in the challenge job
configs): the first run of a chain finds no mirror and initializes fresh, and every later run warm-starts from
`/scratch/mediswarm_latest_global.pt` if it is present. Use `--warm-start fresh` to force a clean start, and
`--warm-start continue` (strict `require`) to *insist* on resuming.

### Recover an aborted run

If a run aborts part-way (a node crash, a VPN drop, an operator `abort_job`), the latest aggregated global has
already been mirrored to `/scratch/mediswarm_latest_global.pt` on each client at the end of every round, so progress
is not lost. To resume:

1. Confirm each client still has the mirror (it lives on the host `--scratch_dir`, so it survives a container
   restart). From the admin host you can pull every site's mirror with `scripts/collect_swarm_globals.sh`.
2. Re-start the clients and the server using the **same `--scratch_dir`** as the aborted run.
3. Prepare and submit a continue job:
   ```bash
   ./prepare_odelia_job.sh --job <JOB> --warm-start continue
   submit_job /fl_admin/local/mediswarm_jobs/<JOB>_continue
   ```
4. Confirm each client logs `WarmStart: will warm-start from checkpoint /scratch/mediswarm_latest_global.pt`
   (followed by `Loading checkpoint from …`). If a client is missing the mirror, the strict continue aborts with
   `WARM_START_REQUIRED_MISSING` rather than silently restarting — fix that client's `--scratch_dir` and retry.

> **`/scratch` space & persistence.** The mirror is overwritten every round and is as large as the global model
> (≈690 MB for `challenge_1DivideAndConquer`). Make sure each client's `--scratch_dir` has room for it and is on a
> **persistent** host mount (not a tmpfs wiped on reboot), or a `continue` will not find it.
