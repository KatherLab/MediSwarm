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

**What to do (the same five steps for every site, about 10 minutes, at a convenient moment,
not during a run)**

1. Download `<SITE>_1.8.1.zip` from the kits folder
   https://drive.google.com/drive/folders/1c3-HaicxgCXukpf5hIlEXWnR4B2N06yS
   and check it: `sha256sum <SITE>_1.8.1.zip` must match your site's row in the *Kit registry*
   tab of the run board
   https://docs.google.com/spreadsheets/d/10_RdWyUulIS7u5Ia5HzC_IiusfIuzhkEd1_c_YU6ml0/edit?gid=1106108057#gid=1106108057
2. Stop your running client container:
   ```
   docker ps --format '{{.Names}}' | grep odelia_swarm_client   # shows the name
   docker stop <that name>
   ```
3. Unzip the kit into a **new, empty** folder. If your current kit has a `startup/sync.conf`
   (the log-upload key), copy it into the new kit's `startup/`. Everything else, including
   your certificates, is already in the new kit.
4. From the new kit's `startup/` folder, run the usual preflight and start, with the same
   data and scratch paths you use today:
   ```
   ./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --preflight_check
   ./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 --start_client
   ```
5. Tick your row in the *Site checklist* tab of the run board (link above).

That is all. The updated partner handbook is here:
https://docs.google.com/document/d/158XgZuXVnYsilfak9IFhELTLb2FBVxKgfuCQDlR_0Aw/edit?tab=t.0
Sections 1.0 and 1.1 are these same steps; section 4 says what to check if the client does
not reconnect. As before, a full twenty-round run takes about two days across the
eight sites and I will announce each one before it starts.

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
- Instructions are identical for all eight sites on purpose (decision 14 Sep): no site is
  singled out in the email. Internally, the two 1.5.0-kit sites (RSH, USZ) are the ones that
  cannot follow `:current` and so are the only ones actually stuck on an old image until they
  install; the 1.6.0-kit sites are already on 1.8.1 after any restart.
- Kits were uploaded to the v1.8.1 Drive folder and the 1.8.1 checksums are in the run
  board's *Kit registry* tab (14 Sep). The handbook link points at the new 1.8.1 handbook
  document.
