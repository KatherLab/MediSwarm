# ODELIA Swarm — Partner Handbook

**This is the single source of truth for running a swarm node.** It replaces the
weekly emails. Nothing here changes per run; when something *does* change, it changes
here.

Three places, and only three:

| | Where | What it is |
|---|---|---|
| **This handbook** | Google Doc | How to prepare and run your node. Stable. |
| **The board** | Google Sheet | Run date, which kit is ACTIVE, your checklist. Changes per run. |
| **Your kit** | Delivered to you individually | Contains **your private key** — never share it. |

> **You should not need a new kit for a new run.** Kits used to be re-issued almost
> weekly. That was a bug on our side (each build silently invalidated every kit), and
> it is fixed. Your kit now survives our software updates: the training code and the
> swarm backbone live in the Docker image, which `docker.sh` re-pulls on every run.
> Expect a new kit only if your site is added/removed or your certificates expire.

---

## 1. One-time setup

### 1.1 Install your kit
You will receive an **encrypted** kit (`<SITE>_<version>.zip.enc`) and, **separately**,
its password. Kits are encrypted because they contain your site's private key.

```bash
# check you have the kit the board says is ACTIVE
sha256sum <SITE>_<version>.zip.enc          # must match the 'Kit registry' tab

openssl enc -d -aes-256-cbc -pbkdf2 -iter 600000 \
        -in <SITE>_<version>.zip.enc -out <SITE>.zip -pass pass:'<password>'
unzip <SITE>.zip
```

### 1.2 Fix the Docker cgroup driver — *do this first*
Without it the GPU silently disappears from a running container after a routine system
update, mid-run (**F2**).

```bash
sudo ./scripts/client_node_setup/fix_docker_cgroupfs.sh
```
One-time; survives reboots. Safe to re-run.

### 1.3 Bring the VPN up as a service (not a manual/GUI connection)
A hand-started tunnel dies and takes your node with it (**F6**).

```bash
sudo ./scripts/client_node_setup/setup_vpntunnel.sh -d <YourSite> -n -s
sudo ./scripts/client_node_setup/vpn_health_monitor.sh --install-timer
systemctl status mediswarm-vpn mediswarm-vpn-health.timer
```

Then confirm **exactly one** tunnel is up — two tunnels fight over the same routes and
make the node flaky (**F10**):
```bash
ps aux | grep [o]penvpn      # expect ONE, using mediswarm.conf
hostname -I                  # expect a single 172.24.4.x address
```

### 1.4 Data layout
Your images and `annotation.csv` / `split.csv` go under your data directory. `split.csv`
must contain the fold the run uses — if it doesn't, the client now tells you so in
seconds, naming the folds you *do* have.

---

## 2. Before every run

```bash
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --dummy_training
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --preflight_check
```

`--preflight_check` prints a **Pre-run checks** block. Any `[FAIL]` aborts, on purpose:

| Check | If it fails |
|---|---|
| `[PASS] GPU usable inside the container` | run §1.2 (**F2**) |
| `[FAIL] ODELIA_PREPROCESS_CACHE_DIR is under /data` | see the box below (**F4**) |
| `[WARN] Swarm server not reachable` | your VPN is down (§1.3, **F6**) |
| stale `daemon_pid.fl` | cleared automatically (**F3**) |

> ### ⚠️ The one that catches everybody: `$SCRATCHDIR` vs the cache path
> `docker.sh` always mounts what you pass to `--data_dir` at **`/data` read-only**, and
> what you pass to `--scratch_dir` at **`/scratch` writable**. Those are paths *inside
> the container*.
>
> `ODELIA_PREPROCESS_CACHE_DIR` is **also a container path**. So setting it to
> `"$SCRATCHDIR/odelia_preprocess_cache"` — a *host* path — makes it land under the
> read-only `/data`, and the run aborts.
>
> **You do not need to set it at all.** It already defaults to `/scratch/odelia_preprocess_cache`.
> Just make sure your host `$SCRATCHDIR` is writable and **persists between runs** (the
> auto-resume checkpoint lives there).

Then tick your row in the **Site checklist** tab of the board.

---

## 3. On run day

```bash
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --start_client
```
That's it. Leave it running; it will pull the image the run uses, join the swarm, and
train. The board's **Run schedule** tab names the exact image tag for the run.

### Moving to a new software version *without* a new kit
If we ask you to run a specific image:
```bash
echo 'MEDISWARM_IMAGE=jefftud/odelia:<tag>' > startup/image.conf
```
(or `./docker.sh --image jefftud/odelia:<tag> ...` for a one-off). This is why kit
re-issues are now rare.

---

## 4. When something goes wrong

Look up the symptom in the **Known issues** tab. It is cumulative — every problem any
site has hit is there, with the fix. If your symptom isn't listed, send us the output
and we'll add it.

The most common ones:

| You see | It means | Do |
|---|---|---|
| `NVML: Unknown Error`, GPU gone mid-run | cgroup driver (**F2**) | §1.2 |
| `Read-only file system` on startup | cache path (**F4**) | the box in §2 |
| node `deemed disconnected` | VPN dropped (**F6**) | §1.3 |
| `too many unusable inputs … load_error:PermissionError` | **the container can't READ your files (F8)** | Fix file permissions. **Do not delete or exclude the data — it is fine.** |
| VPN service hangs asking for a password | bare `auth-user-pass` (**F9**) | re-run §1.3 |

> **F8 deserves a warning.** The guard's old wording said "fix/exclude the data", and a
> site deleted six perfectly good cases because of it. If the reasons are
> `PermissionError`, **the data is fine and the permissions are wrong.** The message now
> says so explicitly.

---

## 5. Reporting status

Your node uploads its logs and heartbeat automatically (live-sync). If we ask you to
set up the upload key:

```bash
# 1) test first — with the VPN up:
ssh -o BatchMode=yes -o ConnectTimeout=5 mediswarm-upload@dl3.tud.de 'echo ok'
#    prints "ok"  -> nothing to do.
#    fails        -> steps 2 and 3:

# 2) create a key if you don't have one:
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "$(hostname)@mediswarm"

# 3) send us the PUBLIC key line:
cat ~/.ssh/id_ed25519.pub
```
Never send a private key. That account accepts uploads only — it cannot give a shell.
