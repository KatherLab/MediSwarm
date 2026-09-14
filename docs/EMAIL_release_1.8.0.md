# Email draft — ODELIA/MediSwarm 1.8.0: what changes at your site (nothing to install for most)

**To:** all ODELIA site contacts (CAM, MHA, RSH, RUMC, UKA, UMCU, USZ, VHIO)
**Subject:** ODELIA software 1.8.0 — one client restart at your convenience; new kits only for RSH and USZ

---

Hi all,

We have released version 1.8.0 of the ODELIA swarm software. It was tested end to end on
real startup kits on our own three-machine test bed before publishing, and this is what it
means for you.

**What is new**

1. **Your site's results now reach the coordinator by themselves.** Each site's validation
   metrics (AUROC, accuracy, and how many cases of each class they were computed on) are
   returned with the run. Until now we collected them by logging into each site by hand,
   which is why the results reports came late.

2. **Optional: per-case predictions.** For the regional fine-tuning comparison and the
   active-learning work we will need, per validation case, the model's three class
   probabilities and the label. This is **off by default**. When a run needs it we will ask
   you first, and switching it on is one environment variable at start
   (`ODELIA_RETURN_PER_CASE=1`). What leaves the site is a row number, the label and three
   numbers — no identifiers, no image data.

3. **Safer resumes.** A run that continues from an earlier model now checks that the saved
   weights really belong to the model being trained, and refuses otherwise. Previously a
   leftover file from a different model could be loaded silently.

4. Robustness fixes found on our test bed, and the groundwork for the active-learning,
   robustness and privacy deliverables — none of which needs anything from you yet.

**What you need to do**

- **CAM, MHA, RUMC, UMCU, VHIO, UKA** (1.6.0 kits): restart your client once at a convenient
  moment — stop the running client container, then start it again as usual with
  `./docker.sh --start_client`. It picks up the new version on start; there is nothing to
  download or install. Please do not restart while a training run is in progress; I will
  announce runs before they start, as before.

- **RSH and USZ** (1.5.0 kits): your kit does not follow the release channel, so a restart
  alone would keep the old version. I will send you a 1.8.0 kit through the usual folder.
  It is a drop-in: same certificates, unpack, copy your `sync.conf` over if you have one, and
  start. Until then you can also start the current kit with
  `./docker.sh --image jefftud/odelia:1.8.0 --start_client`.

That is all. As before, a full twenty-round run takes about two days across the eight sites
and I will announce each one before it starts.

Best regards,
Jeff

---

## Notes before sending

- **Do not send before `:current` is re-tagged and the coordinator server is restarted on
  1.8.0** (`docs/RELEASE_RUNBOOK.md` §3 and §5). A site that restarts before the re-tag
  simply gets the old image again and will believe it has updated.
- Verified 2026-09-13 from the sites' own heartbeats: CAM, MHA, UMCU, VHIO and UKA run 1.6.0
  kits resolving `jefftud/odelia:current` (image id `b8b918541293`, which is the 1.6.0 image —
  `:current` was never moved for 1.7.0). RSH's heartbeat comes from a 1.5.0 kit
  (`/home/asoro/rsh_v150/RSH_1`) although a 1.6.0 RSH record was last seen on 9 Sep — RSH may
  have two installs; ask which one is meant to run. USZ's monitoring feed has been silent
  since 22 Jul (1.5.0 kit). UKA has been unreachable since 4 Sep.
- RUMC's line in the live monitor is currently masked by a deploy-test record from dl0
  (fixed going forward by #573); the real RUMC node is on a 1.6.0 kit per the run history.
- The 1.5.0-kit sites are the only ones that need a kit. New kits come from
  `workspace/odelia_allsites/prod_NN` and reuse the production root CA, so they coexist with
  everyone else's 1.6.0 kits — no re-registration, no server change.
- If UKA's kernel problem (crashes on 6.8.0-138) is still open, say so separately rather
  than here; this email should stay a one-action message.
