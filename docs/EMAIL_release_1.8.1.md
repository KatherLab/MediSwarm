# Email draft — ODELIA/MediSwarm 1.8.1: one client restart at your convenience

**To:** all ODELIA site contacts (CAM, MHA, RSH, RUMC, UKA, UMCU, USZ, VHIO)
**Subject:** ODELIA software 1.8.1 — one client restart at your convenience; new kits only for RSH and USZ

---

Hi all,

We have released version 1.8.1 of the ODELIA swarm software. It was tested end to end on real
startup kits on our own three-machine test bed before publishing. This is what it means for you.

**What is new**

1. **Your site's results now reach the coordinator by themselves.** Each site's validation
   metrics (AUROC, accuracy, and how many cases of each class they were computed on) are
   returned with the run. Until now we collected them by logging into each site by hand.

2. **Optional: per-case predictions.** For the regional fine-tuning comparison and the
   active-learning work we will need, per validation case, the model's three class
   probabilities and the label. This is **off by default**. When a run needs it we will ask you
   first; switching it on is one environment variable at start (`ODELIA_RETURN_PER_CASE=1`).
   What leaves the site is a row number, the label and three numbers — no identifiers, no
   image data.

3. **Safer resumes and starts.** A run that continues from an earlier model checks that the
   saved weights belong to the model being trained and refuses otherwise. And a run can no
   longer start on a site that has not finished configuring — a race we found in our own
   test runs, never at a hospital.

4. Robustness fixes from our test bed, and the groundwork for the active-learning, robustness
   and privacy deliverables — none of which needs anything from you yet.

**What you need to do**

- **CAM, MHA, RUMC, UMCU, VHIO, UKA**: restart your client once at a convenient moment — stop
  the running client container, then start it again as usual with `./docker.sh --start_client`.
  It picks up the new version on start; nothing to download or install. Please do not restart
  while a training run is in progress; I announce runs before they start, as before.

- **RSH and USZ**: your kit does not follow the release channel, so a restart alone would keep
  the old version. Your 1.8.1 kit is in the usual folder. It is a drop-in: same certificates;
  unpack, copy your `sync.conf` over if you have one, and start. Until then you can also start
  the current kit with `./docker.sh --image jefftud/odelia:1.8.1 --start_client`.

That is all. As before, a full twenty-round run takes about two days across the eight sites and
I will announce each one before it starts.

Best regards,
Jeff

---

## Notes before sending

- Supersedes `EMAIL_release_1.8.0.md`, which was never sent: 1.8.1 followed 1.8.0 within a day
  (#576), and both the coordinator and `:current` already run 1.8.1 (13 Sep, 23:08 UTC).
- Verified before publishing: PR validation, a 20-round real-kit run on dl3 + dl0 + dl2 with the
  release candidate (per-site metrics returned for both sites), and the weekly all-models
  validation on `main` (green on rerun; the one failure was the sporadic simulator stall #583,
  two rounds after a normal configure phase).
- Kits for RSH and USZ: `workspace/odelia_allsites/prod_05/{RSH_1,USZ_1}_1.8.1.zip` plus
  `SHA256SUMS.txt`, built with the production root CA (drop-in). Upload to the Drive folder
  "ODELIA Startup Kits / v1.8.1" before sending.
- The 1.5.0-kit sites (RSH, USZ) can alternatively start with
  `./docker.sh --image jefftud/odelia:1.8.1 --start_client` until they install the kit.
