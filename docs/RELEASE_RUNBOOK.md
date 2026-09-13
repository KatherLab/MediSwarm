# Release runbook (ODELIA image + startup kits)

Everything here is manual; nothing in CI builds, pushes or re-tags a release. Each step
was verified on 2026-09-13 against the repository and the live consortium, not assumed.
Written for the 1.8.0 release; the procedure is version-independent.

## 0. What a release changes, and what it does not

| Artefact | Who consumes it | Changes on release? |
|---|---|---|
| `jefftud/odelia:<version>` | nobody directly | yes, built and pushed |
| `jefftud/odelia:current` | every site whose kit ships `startup/image.conf` (1.6.0 kits and later): `docker.sh` sources it and **pulls on every start** | **yes — this is the only update channel sites have.** It was skipped for 1.7.0, so sites ran the 1.6.0 image until 1.8.0 |
| Startup kits (`workspace/odelia_allsites/prod_NN`) | sites, once, at install | optional — see §4 |
| Coordinator server (agh1, `/home/jeff/deploy_odelia_allsites/dl3.tud.de`) | all sites | restart it on the new image between runs (§5) |

A kit does **not** need re-issuing for an image release. Jobs carry their own code
(`byoc`), the shared training code travels in the image, and kits only change on
re-provisioning (new site, new ports, new certificates).

Kits provisioned before `image.conf` existed (the **1.5.0** kits) pin the exact image they
were built with and never see `:current`. Such a site needs either a new kit or
`./docker.sh --image jefftud/odelia:<version> --start_client`.

## 1. Preconditions

- `main` is green **and** the four MVP deploy tests passed on real kits
  (`scripts/deploy/run_deploy_test.sh`, server on dl3, clients dl0+dl2).
- Working tree clean — `buildDockerImageAndStartupKits.sh` refuses local changes.
- `odelia_image.version` bumped and `CHANGELOG.md` written (this is shared with DECADE;
  bumping it moves both consortia's next build).
- dl0 and dl2 idle: publishing the GitHub release fires `odelia-deploy-test.yml`
  (`release: [published]`), which runs there.

## 2. Build

```bash
export MEDISWARM_IMAGE_VERSION=1.8.0          # without this the tag is <v>-dev.<date>.<sha>
./scripts/build/buildDockerImageAndStartupKits.sh -p application/provision/project_Odelia_allsites.yml
```

- Never pass `--num-rounds` or `--min-clients` on a release build (1.6.0 shipped with
  `num_rounds = 3` that way).
- Kits land in `workspace/odelia_allsites/prod_NN` (next free number) with
  `kit_manifest.csv` and one `.zip` per participant. The **root CA is reused** from
  `workspace/odelia_allsites/state/cert.json` — NVFlare's `CertBuilder` loads it when
  present — so new kits interoperate with the running server and with every earlier
  generation (prod_00 … prod_03 all share the CA the production server uses). Do not
  delete `state/` and do not provision the production project into a fresh workspace:
  that mints a new CA and every site would need a new kit before it could connect (F9).

## 3. Publish the image

```bash
docker push jefftud/odelia:1.8.0
docker tag  jefftud/odelia:1.8.0 jefftud/odelia:current
docker push jefftud/odelia:current
```

Verify on Docker Hub that the two tags carry the same digest:

```bash
for t in 1.8.0 current; do curl -s "https://hub.docker.com/v2/repositories/jefftud/odelia/tags/$t" | python3 -c "import json,sys; d=json.load(sys.stdin); print('$t', d['digest'])"; done
```

The sites' heartbeats (`/srv/mediswarm/live/<SITE>/…/heartbeat.json`, field `image_id`)
show when each site has actually moved.

## 4. Tag and release

```bash
git tag -a v1.8.0 -m "ODELIA/MediSwarm v1.8.0"
git push origin v1.8.0
gh release create v1.8.0 --title "v1.8.0 — …" --notes-file <notes.md>
```

A plain tag triggers nothing. **Publishing** the release runs the ODELIA deploy test on
dl0/dl2.

## 5. Move the coordinator server

The production server's kit also has `image.conf → :current`, so a restart pulls the new
image. Do this between runs, with the owner's go-ahead — clients reconnect on their own
using their stored tokens, but a running job would be lost.

```bash
cd /home/jeff/deploy_odelia_allsites/dl3.tud.de/startup
docker stop odelia_swarm_server_flserver_a19be57 && docker rm odelia_swarm_server_flserver_a19be57
rm -f ../daemon_pid.fl            # a stale lock makes start.sh refuse to launch
./docker.sh --start_server
```

## 6. Kits and the announcement

- Upload `prod_NN/*.zip` + `SHA256SUMS.txt` the way 1.6.0 was delivered
  (`workspace/UPLOAD_odelia_kits_v1.6.0/` is the template).
- Sites on a 1.6.0 kit: restart once, nothing else. Sites on a 1.5.0 kit: new kit or
  `--image`. The email template for 1.8.0 is `docs/EMAIL_release_1.8.0.md`.
- Update the kit registry / run schedule with the exact tag.

## 7. After the release

- Watch the live monitor's version-skew view: a site still on the old `image_id` after
  its next restart has a pinned `image.conf` or a 1.5.0 kit.
- Record the release in `CHANGELOG.md` with the final date.
