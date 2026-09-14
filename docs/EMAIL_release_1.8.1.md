# Email — ODELIA/MediSwarm 1.8.1: new startup kit for every site, one install

**To:** all ODELIA site contacts (CAM, MHA, RSH, RUMC, UKA, UMCU, USZ, VHIO)
**Subject:** ODELIA software 1.8.1 — please install your new startup kit (10 minutes, same certificates)

---

Hi all,

We have released version 1.8.1 of the ODELIA swarm software, tested end to end on real
startup kits on our own test bed before publishing. This time every site gets a fresh kit so
that all eight nodes are on exactly the same version. **Your certificates do not change**, so
the new kit works with the running coordinator straight away.

**What is new**

1. **Your site's results now reach the coordinator by themselves.** Each site's validation
   metrics (AUROC, accuracy, and how many cases of each class they were computed on) come
   back with the run. Until now we collected them by logging into each site by hand.
2. **Optional per-case predictions.** For the regional fine-tuning comparison and the
   active-learning work we will need, per validation case, the model's three class
   probabilities and the label. This is **off by default**; when a run needs it we will ask you
   first. What leaves the site is a row number, the label and three numbers — no identifiers,
   no image data.
3. **Safer starts and resumes.** A run can no longer start on a site that has not finished
   configuring, and a run that continues from an earlier model checks that the saved weights
   belong to the model being trained.
4. Robustness fixes from our test bed, and the groundwork for the active-learning, robustness
   and privacy deliverables — nothing needed from you for those yet.

**What to do (about 10 minutes, at a convenient moment, not during a run)**

1. Download `<SITE>_1.8.1.zip` from the shared folder *ODELIA Startup Kits / v1.8.1* and check
   it: `sha256sum <SITE>_1.8.1.zip` must match the row for your site in the board's *Kit
   registry* tab.
2. Stop your running client container.
3. Unzip the kit into a **new, empty** folder. If your old kit has a `startup/sync.conf`
   (the log-upload key), copy it into the new kit's `startup/`.
4. From the new kit's `startup/` folder, run the usual preflight and start:
   ```
   ./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --preflight_check
   ./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --start_client
   ```
5. Tick your row in the board's *Site checklist* tab.

That is all. The updated handbook is linked from the board. As before, a full twenty-round run
takes about two days across the eight sites and I will announce each one before it starts.

Best regards,
Jeff

---

## Notes before sending

- Decision 14 Sep: every site installs the 1.8.1 kit (rather than only the two 1.5.0-kit
  sites), so the monitor shows one kit version everywhere. Kits are in
  `workspace/odelia_allsites/prod_05` (17 zips; upload the 15 site kits + `SHA256SUMS.txt`,
  never the server or admin kit) — built with the production root CA, so they coexist with
  the running coordinator and with any site that has not yet switched.
- `:current` and the coordinator already run 1.8.1 (13 Sep 23:08 UTC). Verified before
  publishing: PR validation, a 20-round real-kit run on dl3 + dl0 + dl2, the weekly
  all-models validation on `main`, and the runner-side release check that trained all six
  models.
- Sites that cannot install immediately keep working: 1.6.0 kits follow `:current` and are
  already on 1.8.1 after a restart; only RSH and USZ (1.5.0 kits) are stuck on an old image
  until they install.
