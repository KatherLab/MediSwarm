# Reply to Islem Gammoudi (UKB Bonn / UKB_1) — 2026-07-08

**Subject:** Re: DECADE — please complete these checks + send us 2 values before Monday 13 July

Dear Islem,

Thank you — this is exactly the detail we needed, and great to see everything
passing on your side (RTX A6000, 1197 slides, val_loss 0.031 all look excellent).

A few answers and next steps:

**SSH key** — received and **authorized** on our monitoring host. Nothing further
needed; your live-sync will connect automatically during the run once you're on
Tailscale with the `/etc/hosts` entry.

**Startup kit** — please use the **`UKB_1` startup kit we provide** (with the
`docker.sh` wrapper), not the `mediswarm-stamp-1.4.5` source package — the source
is superseded and won't match the swarm. Importantly, we just fixed an issue where
the kit mishandled clinical column names containing a **space**, so I'm sending you
an **updated `UKB_1` kit** that handles your `Slide ID` column directly. Please
unpack it, `cd UKB_1/startup`, and run everything via `./docker.sh`.

**Patient/ID column (`STAMP_PATIENT_LABEL`)** — no need to rename anything: with the
updated kit you can set `STAMP_PATIENT_LABEL="Slide ID"` (keep the quotes). The
values in that column must match your H5 filenames (without `.h5`). If any patient
has multiple slides and you provide a slide table, also set `STAMP_SLIDE_TABLE` and
`STAMP_FILENAME_LABEL`, and let us know.

**Prediction target** — we're finalizing the **single shared target** to be used by
all four sites, because federated training requires everyone to train the *same*
model (same class definitions and count). Both of your options are viable; please
**hold on the target for now** — we'll confirm the agreed one (and its
`STAMP_NUM_CLASSES`) with the whole consortium this week, then you'd point
`STAMP_GROUND_TRUTH_LABEL` at whichever of your columns encodes it.

**Feature extractor** — we're standardizing on **one** extractor across all sites so
the feature dimensions match. **UNI (1024-dim)** is our working assumption, so your
UNI extraction (ready ~July 9) is perfect — please continue it and set
`STAMP_DIM_INPUT=1024`. If the consortium ends up choosing UNI2 (1536) instead
(you already have it), we'll let you know and you'd set `STAMP_DIM_INPUT=1536`.
Your updated stats by Friday are exactly what we need.

So the two things pending on your side: (1) finish the UNI features and re-run
Steps 2–5 with the **new kit's** `docker.sh`, and (2) wait for our confirmation of
the shared target + extractor (this week). Everything else is done.

Thanks again for the careful work — please reach out if anything in the new kit
behaves differently from your source runs.

Best regards,
Jeff
