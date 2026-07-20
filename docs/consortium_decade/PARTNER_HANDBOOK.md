# DECADE Swarm — Partner Handbook

This is the stable "how to run a DECADE node" reference. It changes rarely. The
things that change every run — the **date**, the **ACTIVE kit**, and your **site's
checklist** — live in the **Run Board** (the companion Google Sheet), not here.

DECADE trains a shared model with **swarm learning**: each site trains only on its
own data and only model weights are shared — **no patient data ever leaves your
institution**. We use **STAMP** (computational pathology), which trains on
**pre-extracted feature files (`.h5`)**, not raw slides.

> Do not confuse this with the ODELIA breast-MRI handbook — the steps differ
> (Tailscale not OpenVPN, STAMP env vars, feature files not NIfTI volumes).

---

## 1. One-time setup

### 1.1 Install your kit

Use the kit whose version the Run Board marks **ACTIVE**. Unpack it, then work from
`<SITE_NAME>/startup`.

```bash
cd <SITE_NAME>/startup
```

Your `SITE_NAME` (`UKB_1`, `Mainz_1`, `UKD_1`, `UKHD_1`) is on the Run Board.

> **Do not run `docker.sh` with `sudo`**, and do not run your env-var setup as
> `bash setup.sh`. Both start a *child* shell, so the `STAMP_*` variables you export
> never reach the container and training fails with `KeyError: 'STAMP_CLINI_TABLE'`.
> Use `source setup.sh` (or `. setup.sh`), and plain `./docker.sh …`. If you need
> root for Docker, add yourself to the `docker` group once:
> `sudo usermod -aG docker $USER && newgrp docker`.

### 1.2 Connect Tailscale and map the server

Accept the Tailscale invitation we emailed and connect. Then add the server to
`/etc/hosts`:

```
100.100.101.100  dl3.tud.de  dl3
```

Check it: `ping dl3.tud.de` should reply from `100.100.101.100`. If your node is on
your *personal* tailnet instead of the DECADE one, `ping` fails even with the hosts
line correct — re-accept the invitation.

### 1.3 Share an SSH key (for run-log collection)

During a run a small "live-sync" helper pushes only **logs and metadata** (never
data, never features) to our monitoring host so we can help if a node stalls.

```bash
ssh-keygen -t ed25519 -C "$(hostname)@mediswarm"   # press Enter for defaults
cat ~/.ssh/id_ed25519.pub                            # send THIS line to the operator
```

Never send the private key.

### 1.4 Data layout

```
<SITE_NAME>/
├── features/
│   ├── slide_001.h5        # one .h5 per slide, STAMP 2.5.0, UNI (1024-dim)
│   └── ...
├── clini_table.csv         # patient-ID column + ground-truth column
└── slide_table.csv         # ONLY if a patient has more than one slide
```

- **Extract features with standalone STAMP 2.5.0** (needs Python 3.13). The training
  image runs STAMP 2.4.0 but reads 2.5.0-extracted features. Do **not** use a STAMP
  newer than 2.5.0 without telling us.
- **All sites use the same extractor + dimension:** UNI, `STAMP_DIM_INPUT=1024`.
- **Multi-slide patients:** if any patient has several slides, provide
  `slide_table.csv` (a patient-ID column and a slide-filename column) and set
  `STAMP_SLIDE_TABLE` / `STAMP_FILENAME_LABEL`. Without it, a multi-slide site loads
  **0 patients**.

---

## 2. Before every run

Export your config in the **current** shell (see the sudo/`source` warning above).
The operator sends your site's confirmed `STAMP_NUM_CLASSES` and
`STAMP_GROUND_TRUTH_LABEL` with the target decision.

```bash
export SITE_NAME=<your site, e.g. UKB_1>
export DATADIR=<folder that contains your $SITE_NAME directory>
export SCRATCHDIR=<writable scratch folder>; mkdir -p "$SCRATCHDIR"

export STAMP_CLINI_TABLE="/data/$SITE_NAME/clini_table.csv"   # container path
export STAMP_FEATURE_DIR="/data/$SITE_NAME/features"          # container path
export STAMP_TASK="classification"
export STAMP_MODEL_NAME="vit"
export STAMP_DIM_INPUT="1024"
export STAMP_NUM_CLASSES="<the operator gives you this>"
export STAMP_GROUND_TRUTH_LABEL="<exact column name; quotes OK if it has spaces>"
export STAMP_PATIENT_LABEL="PATIENT"        # or your ID column, e.g. "Slide ID"

# Sanity — neither may print []:
echo "DATADIR=[$DATADIR] SCRATCHDIR=[$SCRATCHDIR]"
env | grep '^STAMP_'
```

Then run these three checks (all run **only on your machine**):

```bash
# 1. Docker + GPU sanity (no data, ~1 min CIFAR CNN)
./docker.sh --scratch_dir "$SCRATCHDIR" --GPU device=0 --dummy_training

# 2. Preflight — 1 epoch on your real data
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --preflight_check

# 3. Plausibility — dataset discrepancies (duplicate IDs, H5 vs table, feature dim)
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --preflight_check --log_dataset_details
```

> Step 3's output can contain **patient IDs** — do **not** send it to us. Just report
> pass/fail and any error message.

**Local baseline** (so we can compare the swarm model against your own):

```bash
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --local_training
```

This runs the full `STAMP_MAX_EPOCHS` (32 by default) — it is a real baseline, not a
smoke test. Re-run it whenever you switch to a new ACTIVE kit.

Tick your rows on the Run Board as you go.

---

## 3. On run day

When the operator says the server is up:

```bash
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --start_client
```

Leave it running. `docker.sh` re-pulls the image, so a software update ships without
a new kit — as long as the certificates in your kit still match the server (see §5).

Watch it:

```bash
docker ps                       # stamp_swarm_client_<SITE>_* should be Up (healthy)
tail -f nohup.out               # training log
```

---

## 4. When something goes wrong

Look up the symptom on the **Known issues** tab of the Run Board. Most site-side
issues are one of: `sudo`/`bash setup.sh` eating your env vars, an empty
`$SCRATCHDIR`, a missing slide table, or the wrong (old) kit.

For anything else: `docker ps`, `nvidia-smi`, `ping dl3.tud.de`, and the tail of
`nohup.out` usually point at it. Send us those — never the `--log_dataset_details`
output.

---

## 5. About kits, updates and certificates (read this once)

**Software updates now reach you automatically.** Your kit contains
`startup/image.conf`, which names a release channel:

```
MEDISWARM_IMAGE=jefftud/decade:current
```

`docker.sh` reads it on every run. When we publish a new version we re-tag
`:current`, and your node picks it up **the next time you start it** — nothing to
install. We never pull or restart anything on your machine: we choose *which*
version, you still choose *when* to run. The Run Board's **Run schedule** tab always
names the exact image a given run uses.

To pin a specific version instead, edit that line (e.g.
`MEDISWARM_IMAGE=jefftud/decade:1.6.0`). For a one-off, pass `--image <ref>` to
`docker.sh`. Re-issuing a kit never overwrites an `image.conf` you have edited.

**Certificates — one last swap.** Kits contain site-specific TLS certificates. Until
now every rebuild regenerated them, which is why you received a new kit almost weekly.
That is fixed: from **1.6.0** onward, rebuilds reuse the same certificates. Because the
fix changes how they are stored, moving to 1.6.0 is the **final forced kit swap** —
after it, a software fix reaches you through the channel above, with no new kit.

The Run Board's **Kit registry** is the source of truth for which kit is ACTIVE and
its `sha256`. Each site's `.zip` has its **own** hash (it contains your certificates),
so check the row for *your* site:

```bash
sha256sum <SITE_NAME>_<version>.zip
```

---

## 6. Reporting status

Use the **Run Board**, not email. Tick your checklist rows, and report bug symptoms
against the Known-issues IDs. The board answers the three questions email kept
losing: *which kit am I on, what do I run, and when is the run?*
