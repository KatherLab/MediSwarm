# Release runbook (ODELIA image + startup kits)

Everything here is manual; nothing in CI builds, pushes or re-tags a release. Each step
was verified on 2026-09-13 against the repository and the live consortium, not assumed.
Written for the 1.8.0 release; the procedure is version-independent. Extended for 1.9.0
(2026-10-14): the durable weights cache and the weights gate (§2), image layering (§3),
image verification (§4), the CVE gate (§5), the coordinator move with its job store on the
host (§8) and the heartbeat check (§10).

## 0. What a release changes, and what it does not

| Artefact | Who consumes it | Changes on release? |
|---|---|---|
| `jefftud/odelia:<version>` | nobody directly | yes, built and pushed |
| `jefftud/odelia:current` | every site whose kit ships `startup/image.conf` (1.6.0 kits and later): `docker.sh` sources it and **pulls on every start** | **yes — this is the only update channel sites have.** It was skipped for 1.7.0, so sites ran the 1.6.0 image until 1.8.0 |
| Startup kits (`workspace/odelia_allsites/prod_NN`) | sites, once, at install | optional — see §9 |
| Coordinator server (agh1, `/home/jeff/deploy_odelia_allsites/dl3.tud.de`) | all sites | recreate it on the new image between runs, job store on the host (§8) |

A kit does **not** need re-issuing for an image release. Jobs carry their own code
(`byoc`), the shared training code travels in the image, and kits only change on
re-provisioning (new site, new ports, new certificates).

Kits provisioned before `image.conf` existed (the **1.5.0** kits) pin the exact image they
were built with and never see `:current`. Such a site needs either a new kit or
`./docker.sh --image jefftud/odelia:<version> --start_client`.

Since 1.9.0 the image also repairs one kit problem itself. Its ENTRYPOINT
(`docker_config/mediswarm_entrypoint.sh`) removes a stale `daemon_pid.fl` / `pid.fl` from
the kit when a client or server container starts in a fresh PID namespace (power loss,
reboot): without that, `sub_start.sh` sees a reused PID, refuses to start and the node stays
offline while the container shows "Up". Kits built from main do this in `docker.sh` (#630);
the 1.8.1 kits in the field get it from the image. Every other command (preflight, local
training, external validation, `--run_script`, the admin console, `docker run ... python3`)
runs unchanged, and `docker run --entrypoint` / `docker exec` bypass it. A site can switch it
off with `MEDISWARM_DOCKER_OPTIONS="-e MEDISWARM_ENTRYPOINT_CLEAR_LOCKS=0"` in `startup/image.conf`.

## 1. Preconditions

- `main` is green **and** the four MVP deploy tests passed on real kits
  (`scripts/deploy/run_deploy_test.sh`, server on dl3, clients dl0+dl2).
- Working tree clean — `buildDockerImageAndStartupKits.sh` refuses local changes.
- `odelia_image.version` bumped and `CHANGELOG.md` written (this is shared with DECADE;
  bumping it moves both consortia's next build).
- The durable build cache (§2) holds every weight, SAM-Med2D included.
- dl0 and dl2 idle: publishing the GitHub release fires `odelia-deploy-test.yml`
  (`release: [published]`), which runs there.

## 2. The durable weights cache and the weights gate

`scripts/build/_cacheAndCopyPretrainedModelWeights.sh` fills the build context from a cache
and checks the SHA-256 of **every** cached file before anything is copied: a truncated
download or a stale file stops the build with the expected and the actual hash. With
`MEDISWARM_BUILD_CACHE_DIR` unset it fills a temporary cache and deletes it afterwards, so
each build downloads again, and SAM-Med2D, which cannot be downloaded, is missing.

The durable cache on agh1 is `/mnt/nvme0n1p1/scratch/jeff/mediswarm_build_cache` (the
variable must be an absolute path). Neither dl0/dl2 nor the CI runners have one.

| In the cache | Model | Comes from | SHA-256 |
|---|---|---|---|
| `pretrained_weights/sam-med2d_b.pth` (2,561,829,701 B) | MST_SAMMed2D | **placed by hand**: published on Google Drive and Baidu only | `921d8c130cc4…` |
| `application/jobs/challenge_1DivideAndConquer/app/custom/models/checkpoint_final.pth` | 1DivideAndConquer | Google Drive, fetched with gdown | `ed686907205f…` |
| `application/jobs/challenge_3agaldran/app/custom/models/mvit_v2_s-ae3be167.pth` | 3agaldran | download.pytorch.org | `ae3be1673308…` |
| `torch_home_cache/hub/checkpoints/dinov2_vits14_pretrain.pth` + the dinov2 hub repo | MST family | dl.fbaipublicfiles.com, GitHub | `b938bf1bc15c…` |
| `hf_home_cache/hub/models--TencentMedicalNet--MedicalNet-Resnet18/` | MedicalNet ResNet18 | huggingface.co | `61224f9317fc…` |

The full hashes are in the script. In the image the three files of `pretrained_weights/`
land in `/MediSwarm/pretrained_weights/` (outside the job folders, which NVFlare ships to
every client on submit). The script also rewrites the ResNet18 `refs/main` on every build to
hold the bare commit hash: before 1.9.0 it had a trailing newline, so the baked cache never
resolved offline.

**Weights gate.** SAM-Med2D is

- **required** (exit 1 with a message saying where to get it) when `MEDISWARM_IMAGE_VERSION`
  names a release, i.e. anything but the `<version>-dev.<date>.<sha>` form, or when
  `MEDISWARM_REQUIRE_SAMMED2D=1`;
- optional with a warning otherwise. CI exports a `-dev.` version for its whole run (#428)
  and its runners hold no copy; `daily-image-verify.yml` also sets `MEDISWARM_REQUIRE_SAMMED2D=0`,
  the explicit switch for a deliberate build without it.

An image without the file cannot run MST_SAMMed2D at all: the model refuses to start
without its pretrained encoder unless `SAM_MED2D_ALLOW_RANDOM_INIT=1` (tests only, #636).
If the cache is lost, take the file from an image that has it and check the hash:

```bash
CACHE=/mnt/nvme0n1p1/scratch/jeff/mediswarm_build_cache
mkdir -p "$CACHE/pretrained_weights"
docker create --name sam jefftud/odelia:1.9.0      # or jefftud/odelia:1.8.1-cam.261009
docker cp sam:/MediSwarm/pretrained_weights/sam-med2d_b.pth "$CACHE/pretrained_weights/" && docker rm sam
sha256sum "$CACHE/pretrained_weights/sam-med2d_b.pth"   # 921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18
```

Everything else is downloaded into the cache by the script when missing.

## 3. Build

```bash
export MEDISWARM_IMAGE_VERSION=1.9.0          # without this the tag is <v>-dev.<date>.<sha> (and SAM-Med2D optional)
export MEDISWARM_BUILD_CACHE_DIR=/mnt/nvme0n1p1/scratch/jeff/mediswarm_build_cache
./scripts/build/buildDockerImageAndStartupKits.sh -p application/provision/project_Odelia_allsites.yml
```

- A release **candidate** is built without kits: add `--build-only` and give it a version of
  its own (the 1.9.0 candidate: `MEDISWARM_IMAGE_VERSION=1.9.0-rc-image-local`, tag
  `jefftud/odelia:1.9.0-rc-image-local`, never pushed). Provisioning the production project
  mints kits, and in a fresh workspace a new CA.
- Never pass `--num-rounds` or `--min-clients` on a release build (1.6.0 shipped with
  `num_rounds = 3` that way).
- Kits land in `workspace/odelia_allsites/prod_NN` (next free number) with
  `kit_manifest.csv` and one `.zip` per participant. The **root CA is reused** from
  `workspace/odelia_allsites/state/cert.json` — NVFlare's `CertBuilder` loads it when
  present — so new kits interoperate with the running server and with every earlier
  generation (prod_00 … prod_03 all share the CA the production server uses). Do not
  delete `state/` and do not provision the production project into a fresh workspace:
  that mints a new CA and every site would need a new kit before it could connect (F9).
- The image records what it is: `ODELIA_IMAGE_VERSION` in its environment and the labels
  `org.opencontainers.image.version`, `org.opencontainers.image.revision` (MediSwarm commit)
  and `mediswarm.nvflare.revision`. Inside a client container `MEDISWARM_VERSION` is the
  **kit's** version, not the image's.

**Layering.** Since 1.9.0 the static content has layers of its own at the bottom of the
image, below pip, apt and the code: `/MediSwarm/pretrained_weights`, `/torch_home`,
`/huggingface_home` and the NVFlare source at `/MediSwarm/docker_config/NVFlare`. Before,
the weights and the NVFlare source rode in the code layer that every commit rebuilds
(1,004 MB compressed in 1.8.1, plus 1,953 MB for SAM-Med2D in the 1.8.1-cam overlay). The
final filesystem is unchanged. What this saves depends on the build cache:

- With `--use-docker-cache` a code-only rebuild recreates only the top layers. The 1.9.0
  candidate, rebuilt after a rebase onto new commits, took 11 s; 19 of its 25 layers were
  identical to the previous build and the changed ones add up to about 12.6 MB. Build a
  hotfix image (1.9.x) this way, on agh1 on top of the 1.9.0 build, and sites download only
  those layers.
- The default (`--no-cache`, fresh apt packages, right for a release like 1.9.0) recreates
  every layer, so do not expect the static layers' digests to match an earlier build.

Before pushing a hotfix, compare its layers with the release it replaces:

```bash
diff <(docker image inspect jefftud/odelia:1.9.0 --format '{{range .RootFS.Layers}}{{println .}}{{end}}') \
     <(docker image inspect jefftud/odelia:1.9.1 --format '{{range .RootFS.Layers}}{{println .}}{{end}}')
```

## 4. Verify the image before it leaves the host

```bash
TMPDIR=<dir with ~25 GB free> ./scripts/build/verify_release_image.sh jefftud/odelia:1.9.0 --trivy
```

It checks, and exits 1 on any failure:

1. the SHA-256 of every shipped weight and the bare-hash ResNet18 `refs/main`;
2. ENTRYPOINT/CMD, `KMP_DUPLICATE_LIB_OK=TRUE` (#596), `ODELIA_IMAGE_VERSION` and the labels;
3. NVFlare's wire stack and the training stack against `--reference` (default
   `jefftud/odelia:1.8.1`, compared only when present locally: 1.8.1 and 1.9.0 images share a
   swarm during the rollout) and that the packages 1.9.0 removed are gone;
4. every module of the job code imports; MST and MST_SAMMed2D build and run on CPU, and the
   MST_SAMMed2D encoder equals the shipped SAM-Med2D checkpoint tensor by tensor;
5. the ResNet18 cache resolves offline as a non-root user;
6. a simulated 1.8.1 client kit with a stale `daemon_pid.fl` starts in a fresh container,
   and a plain command sees the same environment as with the entrypoint bypassed;
7. with `--trivy`: the gate of §5 on a `docker save` of the image (the scanner gets no
   Docker socket). The tar needs ~20 GB under `$TMPDIR`.

Test containers run with `--network none`; only Trivy's database download needs the network.
It ends with `ALL CHECKS PASSED` and the image's layer table.

The 1.9.0 candidate (`jefftud/odelia:1.9.0-rc-image-local`, commit 7866300d, 14 Oct 2026)
passed every check, Trivy 0.70.0 included. Its compressed size is 7.12 GB (1.8.1: 5.46 GB;
1.8.1-cam.261009, which added SAM-Med2D: 7.42 GB). The static layers hold 3.52 GB of
weights, 0.13 GB of Hugging Face cache, 0.09 GB of torch.hub cache and 0.16 GB of NVFlare
source (uncompressed), and the per-commit code layer is 8.2 MB.

## 5. The CVE gate

`.github/workflows/daily-image-verify.yml` builds main every day on dl0/dl2, checks that
it still trains, and fails on any HIGH or CRITICAL finding that has a fix (Trivy,
`ignore-unfixed`). It failed every run from 9 Sep to 14 Oct 2026 (36 of 36, last one
38509424841), always at the Trivy step. For 1.9.0:

- **Removed**: wandb (its bundled wandb-core binary alone had 56 findings) and datasets,
  with the packages only they pulled in (GitPython, gitdb, smmap, docker-pycreds,
  sentry-sdk, setproctitle, pyarrow, multiprocess, dill, xxhash), and pyOpenSSL from the
  base image. Nothing in `application/`, `scripts/`, `tests/` or `docker_config/` imports
  them, and the job code imports without them (§4, step 4). The STAMP/DECADE image
  (`Dockerfile_STAMP`) is separate and unchanged.
- **Bumped**: PyJWT 2.14.0, Werkzeug 3.0.6, pyasn1 0.6.4, pydicom 3.0.2, urllib3 2.8.0,
  Brotli 1.2.0, soupsieve 2.8.4, wheel 0.46.3 (with packaging 24.2), setuptools 80.10.2.
- **Accepted**: the rest, in `.trivyignore`, one reason per ID, reviewed on 2026-10-14 and
  expiring on 2027-04-30, when the gate turns red again. Out of scope for 1.9.0 by decision:
  torch 2.2.2 (CVE-2025-32434 needs a base-image move), monai 1.4.0, the NVFlare wire stack
  (protobuf, msgpack, aiohttp stay as in 1.8.1 while both images share a swarm),
  cryptography 42, fsspec and pillow 10.2.
- The workflow passes the file to Trivy (`trivyignores: .trivyignore`) and builds with
  `MEDISWARM_REQUIRE_SAMMED2D=0`.

When the gate turns red: bump the package if a fixed version is low-risk, otherwise add the
ID under its package with a one-line reason and a new expiry date. Never extend the list
without that review. `verify_release_image.sh <image> --trivy` runs the same gate locally.

## 6. Publish the image

```bash
docker push jefftud/odelia:1.9.0
docker tag  jefftud/odelia:1.9.0 jefftud/odelia:current
docker push jefftud/odelia:current
```

Verify on Docker Hub that the two tags carry the same digest:

```bash
for t in 1.9.0 current; do curl -s "https://hub.docker.com/v2/repositories/jefftud/odelia/tags/$t" | python3 -c "import json,sys; d=json.load(sys.stdin); print('$t', d['digest'])"; done
```

Whether a site has actually moved shows only in its heartbeat (§10).

**Rollback.** Note the digest `:current` points to BEFORE re-tagging it. To go back, point
`:current` at the previous release image again and push; sites pick it up with their next
client restart. Before 1.9.0 that is `1.8.1-cam.261009` (1.8.1 plus the CAM weights), never
plain `1.8.1`, which lacks the SAM-Med2D weights:

```bash
docker pull jefftud/odelia:1.8.1-cam.261009
docker tag  jefftud/odelia:1.8.1-cam.261009 jefftud/odelia:current
docker push jefftud/odelia:current
```

## 7. Tag and release

Tag the commit the image was built from, not whatever `main` is now: the image records it.

```bash
REV=$(docker inspect -f '{{ index .Config.Labels "org.opencontainers.image.revision" }}' jefftud/odelia:1.9.0)
git tag -a v1.9.0 "$REV" -m "ODELIA/MediSwarm v1.9.0"
git push origin v1.9.0
gh release create v1.9.0 --title "v1.9.0 — …" --notes-file <notes.md>
```

A plain tag triggers nothing. **Publishing** the release runs the ODELIA deploy test on
dl0/dl2.

## 8. Move the coordinator server

Do this between runs, with the owner's go-ahead: clients re-register on their own, but a
running job would be lost.

**Why it is more than a restart (#607).** The NVFlare server keeps its job store inside the
container: the server kit's `local/resources.json.default` puts the job store
(`jobs-storage`) and the snapshots (`snapshot-storage`) under `/tmp/nvflare`, and NVFlare 2.8
adds its job submit records beside them. The production container
(`odelia_swarm_server_flserver_a19be57`) has one mount, the kit, so removing it to change
the image deletes every job record (6 jobs on 14 Oct, among them 7a678d9a). Its `docker.sh`
is a 1.6.0 kit script with the image pinned by hand (`DOCKER_IMAGE=jefftud/odelia:1.8.1`);
it ignores `image.conf`.

The 1.9.0 server template (`docker_svr_sh` in `docker_config/master_template.yml`) mounts a
host directory at `/tmp/nvflare` and another at `/archive`, where the MS6 server component
`ms6_global_archive` writes the per-round global models (it falls back to the job workspace
without the mount). Both directories, and the image, can be set in `startup/server.conf`;
`--image <ref>` overrides the image once. The server, unlike a client, never follows
`:current`. After this move a recreate behaves like a restart: the store survives.

```bash
KIT=/home/jeff/deploy_odelia_allsites/dl3.tud.de
C=odelia_swarm_server_flserver_a19be57
NEW=jefftud/odelia:1.9.0
```

**Step 1, no job may be running.** In the admin console `list_jobs` shows only `FINISHED:*`.

**Step 2, export the job store**, keeping an untouched copy as well:

```bash
docker cp $C:/tmp/nvflare - | gzip > ~/nvflare_store_$(date +%y%m%d).tar.gz
mkdir -p $KIT/server_state
docker cp $C:/tmp/nvflare/. $KIT/server_state/
diff <(docker exec $C ls /tmp/nvflare/jobs-storage) <(ls $KIT/server_state/jobs-storage) && echo "job store copied"
```

**Step 3, the archive directory and `server.conf`.** Check the free space first: each
archived round is one global model per run.

```bash
sudo mkdir -p /srv/mediswarm/archive
cat > $KIT/startup/server.conf <<'EOF'
# read by docker.sh (1.9.0 server template), see docs/RELEASE_RUNBOOK.md section 8
MEDISWARM_SERVER_STATE_DIR=/home/jeff/deploy_odelia_allsites/dl3.tud.de/server_state
MEDISWARM_ARCHIVE_DIR=/srv/mediswarm/archive
EOF
```

**Step 4, a new `docker.sh`** rendered from the release image's own template, keeping the
container name that tools and notes use:

```bash
cd $KIT/startup && cp docker.sh docker.sh.bak-1.8.1
docker pull $NEW
docker run --rm --network none --entrypoint python3 $NEW -c '
import os, re, sys, yaml, nvflare.lighter
image, name_suffix = sys.argv[1], sys.argv[2]
t = yaml.safe_load(open(os.path.join(os.path.dirname(nvflare.lighter.__file__), "templates", "master_template.yml")))
s = t["docker_svr_sh"]
for key, value in {"docker_image": image, "admin_port": "8003", "fed_learn_port": "8002"}.items():
    s = s.replace("{~~%s~~}" % key, value)
s = re.sub(r"^(CONTAINER_NAME=odelia_swarm_server_\$\{svr_name\}_)\S+$", r"\g<1>" + name_suffix, s, flags=re.M)
sys.stdout.write(s)' $NEW a19be57 > docker.sh.new
bash -n docker.sh.new && diff docker.sh docker.sh.new
```

The diff may show only: the image, `--image`, the `server.conf` block, the two `-v` mounts
(`SERVER_MOUNTS`), `rm -f ../daemon_pid.fl ../pid.fl` in the container command (#630) and
help text (checked against the production script with the 1.9.0 candidate on 14 Oct). Then
`chmod 755 docker.sh.new && mv docker.sh.new docker.sh`.

**Step 5, recreate the container:**

```bash
docker stop $C && docker rm $C
./docker.sh --start_server     # prints "Job store on the host: ...   archive: ..."
```

**Step 6, verify:**

```bash
docker inspect $C --format '{{.Config.Image}}{{range .Mounts}} {{.Source}}->{{.Destination}}{{end}}'
# jefftud/odelia:1.9.0 .../dl3.tud.de->/startupkit .../server_state->/tmp/nvflare /srv/mediswarm/archive->/archive
```

Within a few minutes `startup/nohup.out` shows `registered client` lines for the sites, and
`list_jobs` in the admin console lists the jobs from before the move.

Rollback: `docker stop $C && docker rm $C && ./docker.sh --image jefftud/odelia:1.8.1 --start_server`.
The store stays on the host, so nothing is lost either way. A later image move needs no
export: set `MEDISWARM_IMAGE=<ref>` in `server.conf` (or pass `--image`), then stop, remove
and `./docker.sh --start_server`.

## 9. Kits and the announcement

- Upload exactly the files named in `prod_NN/SHA256SUMS.txt` (the site client kits) plus that
  file to the kits folder. **Never** upload the server kit (`dl3.tud.de_*.zip`) or the admin kits:
  they carry the coordinator's and the admins' private keys, and the kits folder is readable by
  every site. `upload_authorized_keys.txt` stays on the build host too.

  ```bash
  cd workspace/odelia_allsites/prod_NN && mkdir -p ../upload_NN \
    && awk '{print $2}' SHA256SUMS.txt | xargs -I{} cp {} ../upload_NN/ && cp SHA256SUMS.txt ../upload_NN/
  ```
- Register the kits' upload keys on the upload host BEFORE the kits go out:
  `sudo scripts/admin/register_upload_keys.sh workspace/odelia_allsites/prod_NN/upload_authorized_keys.txt`
  (dry run first with `--dry-run`; server_tools/README.md).
- Sites on a 1.6.0 or later kit: restart once, nothing else. Sites on a 1.5.0 kit: new kit or
  `--image`. Announcement emails are drafted locally as `docs/EMAIL_release_<version>.md` (gitignored, never committed).
- Update the kit registry / run schedule with the exact tag.

## 10. After the release

- **Heartbeat check.** A site's heartbeat reports `image_id` = the image's linux/amd64
  **config** digest, cut to 19 characters (`sha256:` + 12 hex). That is neither the digest
  Docker Hub shows for the tag (the manifest's) nor what `docker images` prints on agh1
  (containerd image store). Resolve it and compare with every site's newest heartbeat:

  ```bash
  scripts/build/heartbeat_image_id.py --live-root /srv/mediswarm/live jefftud/odelia:1.9.0
  ```

  It prints the id to expect, then one line per site, `OK` or `OLD`, with the heartbeat's
  time and kit version (for reference: 1.8.1 is `sha256:cbf6ba7aa574`, 1.8.1-cam
  `sha256:49bdb25cbfcd`). It needs `docker buildx` and access to Docker Hub. A site still
  `OLD` after its next restart has a pinned `image.conf`, a 1.5.0 kit, or never restarted.
- Watch the live monitor's version-skew view for the same thing over time.
- Record the release in `CHANGELOG.md` with the final date.
