# Moving your node to the 1.9.3 startup kit

For the person who runs the ODELIA swarm node at a site. It takes about 15 minutes plus the
image download. Your data, your scratch folder and your site's identity in the swarm stay
the same; only the kit folder changes. The examples use `UKA_1` and the folders
`~/odelia_v181` (old kit) and `~/odelia_v193` (new kit): use your site's name and your own
folders.

## Before you start

- Move between runs, when no swarm learning job is running (see the run board).
- Run `nvidia-smi` on the host. If it prints `Driver/library version mismatch`, an automatic
  NVIDIA driver update is waiting for a reboot. Reboot the host first, otherwise the new
  container starts without a GPU.
- Find your current kit: the folder that contains `startup/`, for example `~/odelia_v181/UKA_1`.

## 1. Download

From the ODELIA kits folder, download your site's kit `<SITE>_1.9.3.zip` (for example
`UKA_1_1.9.3.zip`) and `SHA256SUMS.txt`. The kit holds your site's private key: keep it
private and do not forward it.

## 2. Check the download

In the folder that holds both files:

```bash
sha256sum -c --ignore-missing SHA256SUMS.txt
```

It must print `UKA_1_1.9.3.zip: OK`. Anything else: download the kit again, and tell your
TUD contact if it still fails.

## 3. Unzip next to the old kit

Unzip into a new, empty folder, never over the old kit:

```bash
mkdir -p ~/odelia_v193
unzip UKA_1_1.9.3.zip -d ~/odelia_v193
```

This creates `~/odelia_v193/UKA_1/`. Keep the old folder until step 5 shows the node
running from the new kit.

## 4. Move the node

```bash
cd ~/odelia_v193/UKA_1/startup
./docker.sh --migrate_from ~/odelia_v181/UKA_1
```

This is a dry run. It changes nothing and lists what it would do: stop the old client
container and the old kit's monitor upload, carry your settings over from the old
`startup/image.conf` and `startup/sync.conf`, and start the new client with the same data
folder, scratch folder and GPU. If the list looks right, run:

```bash
./docker.sh --migrate_from ~/odelia_v181/UKA_1 --yes --start
```

The first start downloads the 1.9.3 image, which takes a while (several GB). Your data
and the old kit folder are never deleted.

- Run both commands as the same user that started the old client.
- If a swarm learning job is still running in the old container, the command stops and
  changes nothing. Wait until the run is over.
- If it asks for `--data_dir`, `--scratch_dir` or `--GPU` (the old container is already
  gone, so it cannot read them), add them to the same command.

## 5. Check

```bash
./docker.sh --status
docker ps --filter name=odelia_swarm_client
```

`--status` changes nothing. It must show this kit's client running and no client of an
older kit, and `docker ps` must list exactly one client container. If `--status` reports a
problem, send its output to your TUD contact. TUD also sees on the monitor when your node
is back.

On hosts with bash older than 5.2 (Ubuntu 22.04 and older, RHEL 9 and older) the 1.9.3 kit's
`--status` can warn `more than one live_sync daemon runs for this kit (PIDs A B); keep only
PID A` although only one upload runs: B is a helper of A that writes A's log, and it ends
together with A. Nothing to do; `ps -o ppid= -p B` prints A. The next kit no longer counts it.

## What is new for sites

- **One command to move, one to check.** `--migrate_from` carries your settings over from
  the old kit, `--status` shows the state of the node.
- **No key exchange for the monitor upload.** The upload key comes with the kit, so you no
  longer email a public key to TUD.
- **Per-case opt-in carried over.** If your site returns per-case predictions
  (`--env ODELIA_RETURN_PER_CASE=1` in the `MEDISWARM_DOCKER_OPTIONS` line of
  `startup/image.conf`), `--migrate_from` keeps that setting. Nothing changes for sites
  that did not opt in.
- **Model files stay at your site.** Checkpoints are no longer uploaded to the monitor by
  default; logs and metrics still are.
- **The node rejoins on its own after a reboot or power loss.** The container clears its
  stale lock file before it starts.
- **Host helpers come with the kit.** `scripts/client_node_setup/` (VPN service, Docker
  cgroup fix, GPU watchdog) is now in the kit folder, so the commands in the partner
  instructions work from there.

## Keep the NVIDIA driver fixed while runs are scheduled

An automatic driver update replaces the driver files on disk while the old driver stays
loaded. From then on every new container has no GPU until the host reboots. Hold the driver
packages until the runs are over:

```bash
dpkg -l | awk '/^ii/ && $2 ~ /^(nvidia|libnvidia)-/ {print $2}' | xargs -r sudo apt-mark hold
apt-mark showhold
```

After the runs, release them with `sudo apt-mark unhold <packages>`, update, and reboot.

## If `--migrate_from` stops with an error

Send its output to your TUD contact. To move by hand instead:

1. Stop the old client. `docker ps --filter name=odelia_swarm_client` shows its name, then
   `docker rm -f <name>`.
2. Stop the old kit's monitor upload: `pgrep -af live_sync.sh` lists each upload with its
   kit folder; `kill <PID>` the one that names the old kit. With bash older than 5.2 each
   upload is listed twice (the second is its log helper): kill the lower PID, the other ends with it.
3. Copy your own settings, for example a `MEDISWARM_DOCKER_OPTIONS=...` line, from the old
   `startup/image.conf` and `startup/sync.conf` into the new ones.
4. Start the new kit with your usual folders and GPU:

   ```bash
   cd ~/odelia_v193/UKA_1/startup
   ./docker.sh --data_dir <DATA> --scratch_dir <SCRATCH> --GPU device=0 --start_client
   ```

Then check as in step 5.
