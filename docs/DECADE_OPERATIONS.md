# DECADE swarm operations (STAMP)

This guide is for the **swarm operator** of a DECADE run: the person who runs the server and
admin kits, submits the jobs and supports the sites. Sites follow
[`README.participant.STAMP.md`](../assets/readme/README.participant.STAMP.md); the job itself is
described in [`application/jobs/STAMP_classification/README.md`](../application/jobs/STAMP_classification/README.md);
failure modes shared with ODELIA are in [`SWARM_FAILURE_MODES.md`](SWARM_FAILURE_MODES.md).

Everything here was done for real during the November 2026 MSI-High benchmark preparation
(kit 1.7.1, image `jefftud/decade:1.7.0`): moving the server to a stand-in host, setting up the
upload ingest there, running and debugging the 2-round test.

> **This repository is public.** Never put partner data here: no cohort sizes, labels, metrics,
> contact details, host addresses or anything from a site's log beyond a generic error message.
> Run notes with such details stay in the operator's local `workspace/` (git-ignored).

## Contents

1. [The pieces of a DECADE swarm](#1-the-pieces-of-a-decade-swarm)
2. [Moving the server to a stand-in host](#2-moving-the-server-to-a-stand-in-host)
3. [Admin commands without the console](#3-admin-commands-without-the-console)
4. [Reading a site's logs from the admin console](#4-reading-a-sites-logs-from-the-admin-console)
5. [Upload ingest (`mediswarm-upload`) on a host](#5-upload-ingest-mediswarm-upload-on-a-host)
6. [The 2-round test before a benchmark](#6-the-2-round-test-before-a-benchmark)
7. [Failure signatures](#7-failure-signatures)
8. [Questions sites ask](#8-questions-sites-ask)

---

## 1. The pieces of a DECADE swarm

| Piece | What it is | Notes |
|---|---|---|
| Server kit `dl3.tud.de` | NVFlare server, ports 8002 (FL) and 8003 (admin) | The certificates name the server `dl3.tud.de`. Sites reach it over the consortium's Tailscale network with one `/etc/hosts` line for that name. |
| Admin kit | the operator's `project_admin` identity | Submits and aborts jobs; may run shell commands on every site (section 4). |
| Site clients | one container per site, `stamp_swarm_client_<SITE>_<build id>` | Started with `./docker.sh ... --start_client` and left running between runs. |
| Operator-run node | a client kit run by the operator, e.g. for a reference cohort | Same rules as a site. **One identity runs once**: never start the same client kit on two hosts, they push each other out of the swarm. |
| Upload ingest | `mediswarm-upload` account on the server host, `/srv/mediswarm/live/<SITE>/` | Receives the sites' live-sync uploads (heartbeat, logs, metrics summary). Optional for training, needed for monitoring and for collecting the final global models. |
| Run jobs | the STAMP job with the run's settings pinned | The launcher command pins model, feature dimension, classes, class order, training recipe and precision, so a site's environment only supplies data paths and the label column. The roster is strict: `min_clients` = `configure_min_clients` = number of sites, every site in `mandatory_clients` (the benchmark has no fault tolerance). |

## 2. Moving the server to a stand-in host

Use this when the regular server host is unavailable (in November 2026 it was being shipped to
another city) and the consortium should not wait. **The principle: keep the server's name and
its network address.** Then the sites' certificates and `/etc/hosts` lines still match, their
clients reconnect by themselves, and no site has to change anything.

1. **Copy the kits, never rebuild them.** Copy the production server kit and the admin kit
   (same build, same root CA) to the stand-in host. A rebuilt project gets a new CA and every
   site would need a new kit. Keep the regular host's job store for when it comes back.
2. **Resolve the server name inside the containers only.** In `startup/docker.sh` of the server
   and the admin kit:

   ```bash
   NETARG="--net=host --add-host dl3.tud.de:127.0.0.1"
   ```

   Do not edit `/etc/hosts` on a host that a configuration-management tool (Ansible, ...)
   maintains: the next run overwrites it.
3. **Heartbeat timeout 1800 s.** Copy `local/resources.json.default` to `local/resources.json`
   and set `"heart_beat_timeout": 1800`. With the default, a 10-minute network gap at one site
   removes it from a strict run.
4. **Start the server** with `./docker.sh --no_pull --start_server` and check the log for
   `server heartbeat timeout set to 1800`.
5. **Take over the address.** In the Tailscale admin console, move the regular host off the
   server's address and give that address to the stand-in host; turn key expiry off for the
   stand-in node. Check from the stand-in host that every site node answers
   (`tailscale ping <node address>`; with `--accept-dns=false`, ping by address), and ask each
   site to run `./docker.sh --status`, which must show
   `[OK] dl3.tud.de:8002 (FL port) reachable`. Clients that were already running reconnect on
   their own; check `Total clients` in the server log.
6. **Set up the upload ingest** (section 5) before the sites start. Without it, every site's
   live-sync daemon fails its SSH login to the server name every 30 s at first and every 300 s
   later, and the host's SSH log fills with `Invalid user mediswarm-upload`.
7. **Agree a run window** with the stand-in host's administrators: no configuration-management
   runs (they may reload systemd, which can take the GPU away from running containers, see F2,
   or remove packages such as Tailscale), no reboots, `apt-mark hold` for the NVIDIA, Docker and
   Tailscale packages, CI runners on the host paused.
8. **When the regular host returns**, swap back the same way: stop the stand-in server, move
   the address back, start the regular server. The sites again change nothing.

**Relayed connections.** A campus network that blocks outbound UDP forces every Tailscale
connection through a DERP relay (`tailscale netcheck` shows `UDP: false`, `tailscale ping`
never reports a direct path). STAMP models are small, so this works (round trips of 7 to 25 ms
in our runs), but large models would transfer much more slowly.

## 3. Admin commands without the console

[`scripts/admin/flare_admin.py`](../scripts/admin/flare_admin.py) runs one admin command through
NVFlare's FLARE API and prints the result. It needs nvflare, so run it with the swarm's image
and the admin kit mounted:

```bash
KIT=/path/to/admin_kit       # the folder that holds startup/fed_admin.json
JOBS=/path/to/jobs           # must be writable: NVFlare signs a job folder in place
fladmin() {
  docker run --rm --net=host --add-host dl3.tud.de:127.0.0.1 \
    -u "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v "$KIT":/fl_admin -v "$JOBS":/jobs -v "$PWD/scripts/admin/flare_admin.py":/flare_admin.py:ro \
    -w /fl_admin jefftud/decade:1.7.0 python3 /flare_admin.py "$@"
}
fladmin status
```

The `--add-host` is only needed when the admin runs on the server host itself; elsewhere the
host's `/etc/hosts` line does the job.

| Command | Does |
|---|---|
| `status` | Server status and each client's job status. `server_status: stopped` means **no job is running**; the server itself is up. Clients show `no_jobs` between runs. |
| `list` | All jobs with their status: `RUNNING`, `FINISHED:COMPLETED`, `FINISHED:EXECUTION_EXCEPTION`, `FINISHED:ABORTED`, ... |
| `submit /jobs/NAME` | Submits a job folder and prints `JOB_ID ...`. |
| `abort JOB_ID --yes` | Aborts a running job. Without `--yes` it refuses. |
| `download JOB_ID` | Fetches a finished job's server workspace (`log.txt`, `log_error.txt`, ...) into the admin kit's `transfer/`. The server deletes its job folder when the job ends, so this is the way to read a finished job's server log. |
| `grep SITE PATTERN FILE`, `tail SITE FILE -n N`, `ls SITE [PATH]` | Read a site's job log (section 4). |

**Following a run.** The admin kit ships `startup/watch_run.py` (plain Python, no nvflare). It
reads the server's `startup/nohup.out` and prints, per round, when it started and finished, how
many sites finished and which site was last, plus the stall alarms of #542:

```bash
python3 admin_kit/startup/watch_run.py server_kit/startup/nohup.out --follow
```

For a finished job, point it at the downloaded `transfer/JOB_ID/workspace/log.txt`. An aborted
job reads like this (site names replaced):

```
run fbe92494-...
  sites (4): SITE_A, SITE_B, SITE_C, SITE_D   starting client SITE_D
  started 2026-11-26 15:32:49, configured in 8 s, 2 rounds
  round  first start  done at      min  finished  last to finish (train min)
      0  11-26 15:33  running          -    3/4     training: SITE_B 10 min
  rounds complete: 0/2
STATUS ABORTED (exit 1): received failure report from client SITE_B: EXECUTION_EXCEPTION
```

To tell from a script that a job has ended, poll `flare_admin.py list` or use `watch_run.py`'s
exit status. Do not grep the server log for the job id: the line that reports the final status
does not carry it.

## 4. Reading a site's logs from the admin console

When a site's training fails, the server only learns `EXECUTION_EXCEPTION`. The reason is in
the site's own job log. A `project_admin` can read it without asking the site for files: the
default site policy (`local/authorization.json.default`) gives `project_admin` the permission
`"any"`, which includes NVFlare's shell commands (`grep`, `tail`, `ls`, `cat`).

These commands work only **inside the site's client workspace** (its kit folder), never in its
data. The useful files are:

| File in the site's workspace | Contains |
|---|---|
| `JOB_ID/log.txt` | the client's job log, including everything the training subprocess printed (lines from `SubprocessLauncher`), so also its Python traceback |
| `log.txt` | the client's own log across jobs |
| `startup/.mediswarm_sync/live_sync_daemon.log` | the live-sync daemon: SSH and rsync errors |

```bash
fladmin grep SITE_B ERROR JOB_ID/log.txt
fladmin grep SITE_B Traceback JOB_ID/log.txt
fladmin grep SITE_B SubprocessLauncher JOB_ID/log.txt
fladmin tail SITE_B startup/.mediswarm_sync/live_sync_daemon.log -n 40
```

Read only what you need to diagnose a failure, and tell the site what you read and what you
found. Since kit 1.7.1 the STAMP job keeps patient identifiers out of its logs.

## 5. Upload ingest (`mediswarm-upload`) on a host

The sites' `startup/sync.conf` sends uploads to `mediswarm-upload@dl3.tud.de:/srv/mediswarm/live`,
so the account has to exist on whatever host answers as `dl3.tud.de`. The account only
receives uploads through the forced-command guard
[`server_tools/mediswarm_upload_guard.sh`](../server_tools/mediswarm_upload_guard.sh); see
[`server_tools/README.md`](../server_tools/README.md) for why the home is root-owned.

### Requirements on the host

* `/usr/bin/rrsync` that accepts `-no-lock` (tested with rsync 3.2.7, Ubuntu 24.04).
* sshd allows the account: no `AllowUsers`/`AllowGroups` that leave it out, public-key login on.
  Check with `sudo sshd -T -C user=mediswarm-upload,host=x,addr=1.2.3.4`.

### Setup (as root)

```bash
groupadd --system mediswarm-upload
useradd --system -g mediswarm-upload -d /home/mediswarm-upload -M -s /bin/bash \
        -c "MediSwarm live-sync uploads" mediswarm-upload
passwd -l mediswarm-upload                       # no password login
install -d -o root -g mediswarm-upload -m 750 /home/mediswarm-upload /home/mediswarm-upload/.ssh
install -o root -g mediswarm-upload -m 640 authorized_keys /home/mediswarm-upload/.ssh/authorized_keys
install -o root -g root -m 755 server_tools/mediswarm_upload_guard.sh /etc/ssh/mediswarm_upload_guard.sh
install -d -o root -g root -m 755 /srv/mediswarm
install -d -o mediswarm-upload -g mediswarm-upload -m 2750 /srv/mediswarm/live
```

Mode `2750` keeps the uploads away from other users of a shared host (read them with `sudo`);
on a dedicated monitor host that serves them, `2775` is fine.

### Key lines

Write **one site-scoped line per site**, so a site's key can only write into its own folder:

```
restrict,command="/etc/ssh/mediswarm_upload_guard.sh SITE_A" ssh-ed25519 AAAA... SITE_A@decade
```

DECADE kits up to 1.7.1 carry no kit upload key: each site uses the public key it sent by hand,
so take the keys from the previous ingest's `authorized_keys` (where they may still be
unscoped) and rewrite them scoped to their site. Kits that do carry a key (ODELIA 1.9.x) are
registered with `scripts/admin/register_upload_keys.sh` instead. `ssh-keygen -lf authorized_keys`
lists the fingerprints, for comparing with the logins below.

### Checks

1. **Logins.** The sites' daemons retry every 300 s after failures, so allow 5 minutes:

   ```bash
   sudo journalctl _COMM=sshd --since -10min | grep "Accepted publickey for mediswarm-upload"
   ```

   One source per site, each with its key's fingerprint; no new `Invalid user mediswarm-upload`.
2. **Guard decisions.** `sudo journalctl -t mediswarm-upload-guard --since -10min` shows
   `allow-mkdir` and `allow-rsync` per site and no `reject-*`. A site with `allow-mkdir` but
   never `allow-rsync` cannot upload from its side (section 7, signature 4).
3. **Confinement self-test**, no key needed. Each of these must be refused:

   ```bash
   g() { sudo -u mediswarm-upload env SSH_ORIGINAL_COMMAND="$2" SSH_CLIENT="selftest 0 0" \
           /etc/ssh/mediswarm_upload_guard.sh "$1" >/dev/null 2>&1 && echo "ALLOWED $2" || echo "denied  $2"; }
   g SITE_A "mkdir -p '/srv/mediswarm/live/SITE_B/x'"        # another site's folder
   g SITE_A "mkdir -p '/home/mediswarm-upload/.ssh/x'"       # the account's home
   g SITE_A "rm -rf '/srv/mediswarm/live/SITE_A'"            # the site's whole area
   g SITE_A "cat /etc/passwd"                                # anything else
   ```

4. **Files.** `sudo find /srv/mediswarm/live -type f` lists, per site and job,
   `heartbeat.json`, `live_sync_daemon.log`, `log.txt`, `nohup.out` and
   `run_dir/stamp_metrics_summary.csv`; a site's local runs appear under `SITE/local/`.

## 6. The 2-round test before a benchmark

A preflight check at every site is not enough. It does not show that the client was started
with the same options (a different `--data_dir` passes the preflight and fails the run), and it
never runs configuration, result submission and aggregation across sites. Before a benchmark,
submit the run's job with `num_rounds = 2` and the same roster.

**What a healthy test shows:**

1. server log: `successfully configured clients [...]` within seconds of the start (8 s with
   four sites);
2. server log (or `watch_run.py`): per site and round, `start_learn_task` then
   `finished_learn_task`;
3. each site's job log: `got permission from <aggregating site> to submit round ...` and
   `Finished round ...`; the next round then starts from the aggregate;
4. after the last round the best model is broadcast and `flare_admin.py list` shows
   `FINISHED:COMPLETED`.

**Plan the full run from it.** The slowest site sets the pace of every round. Each site's
epochs per round are weighted by its training set size,
`epochs = STAMP_EPOCHS_PER_ROUND × STAMP_EPOCHS_REFERENCE_DATASET_SIZE / train_patients`,
rounded and clamped to `[1, STAMP_EPOCHS_MAX_CAP]`, so the largest site may train only one
epoch per round while a small one trains several. A 20-round run takes roughly 20 times the
slowest site's round plus the model transfers; in our 4-site test the slowest site needed about
7 minutes per round, so a 20-round run fits in an afternoon.

**If the test fails**, the job ends `FINISHED:EXECUTION_EXCEPTION` and the server log names the
site: `received failure report from client SITE: EXECUTION_EXCEPTION`. Read that site's job log
(section 4), tell the site exactly what to change, and repeat the test once its client runs
again. A test needs nothing from the sites while their clients are running.

## 7. Failure signatures

| # | What you see | Where | Meaning | What to do |
|---|---|---|---|---|
| 1 | Run aborts about **600 s** after a round started: `received failure report from client SITE: EXECUTION_EXCEPTION`; the site's job log has `External process has not called flare.init within timeout: 600` | server log, site job log | The site's training subprocess died before it registered with NVFlare (`external_pre_init_timeout = 600`). The other sites train normally meanwhile. | Find the traceback: `fladmin grep SITE Traceback JOB_ID/log.txt`. Seen: `FileNotFoundError: STAMP_SLIDE_TABLE is set but the file does not exist: /data/...`: the client was started with a different `--data_dir` than the preflight. The site removes the container (`docker rm -f stamp_swarm_client_SITE_<id>`), runs `--preflight_check` with exactly the options it uses for the client, then `--start_client`. The same 600 s pattern follows the class-count/order check (`ValueError: Class-order ...`) and the GPU guard (`RuntimeError: STAMP_REQUIRE_GPU=1 ...`). |
| 2 | `Failed to sign job folder: [Errno 30] Read-only file system` on submit | admin | The jobs folder is mounted read-only; NVFlare writes a signature into the job folder. | Mount it writable. |
| 3 | `Invalid user mediswarm-upload from <site address>` every few minutes | SSH log of the server host | The host answering as `dl3.tud.de` has no upload account (typically right after a server move). No effect on training. | Section 5. |
| 4 | Site logs in (`allow-mkdir`) but never uploads (`allow-rsync` missing); its daemon log shows `rsync: --mkpath: unknown option` with `[client=3.1.3]` | guard log, site daemon log | The site's rsync is older than 3.2.3, and the kit's live-sync passes `--mkpath`. Monitoring only. | Until a kit's live-sync handles old rsync, read that site's logs through the admin console (section 4), or the site installs a newer rsync. |
| 5 | A monitoring loop never sees the job end | operator script | The server's final status line carries no job id. | Poll `flare_admin.py list` or use `watch_run.py`. |
| 6 | `server_status: stopped` in `status` | admin | Not an error: no job is running. | None. |

## 8. Questions sites ask

**"Which of my `STAMP_*` settings matter?"** For the swarm run, the run settings the operator
announces (`STAMP_TASK`, `STAMP_MODEL_NAME`, `STAMP_DIM_INPUT`, `STAMP_NUM_CLASSES`,
`STAMP_EXPECTED_CATEGORIES`, `STAMP_GROUND_TRUTH_LABEL`) and the data paths. The run job pins
model, classes and training recipe on its launcher command, which overrides the site's
environment. It also sets `STAMP_PRECISION` empty, so every site trains with the job's
default (16-bit mixed precision on a GPU) whatever the site's environment says.

**"Can I set `STAMP_MAX_EPOCHS` or `STAMP_PRECISION` myself?"** They have no effect on the swarm
run, but they do change the site's `--local_training` baseline, which is then no longer
comparable with the other sites. Delete them, and rerun the local baseline if it already ran
with them.

**"Our labels are `Yes`/`No`, the run expects `no,yes`."** The class order check compares
case-insensitively, so the case does not matter; the order and the number of classes do. A
third value in the label column (for example `not provided`) widens the classifier and stops
the run before training; remove those rows.
