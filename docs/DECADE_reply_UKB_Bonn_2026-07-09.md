# Reply to Islem Gammoudi (UKB Bonn / UKB_1) — 2026-07-09

**Subject:** Re: DECADE — checks + 2 values before Monday 13 July

Dear Islem,

Thank you — both issues are clear, and the STATISTICS screenshots are very helpful.
Good news: **you do not need to re-extract anything.**

## Issue 1 — STAMP version mismatch: fixed on our side

You diagnosed this correctly. Your features were extracted with STAMP **2.5.0**,
and the training image shipped STAMP 2.4.0, which refuses to read newer features.

We cannot simply upgrade the image to STAMP 2.5.0: STAMP 2.5.0 requires **Python
3.13**, while the swarm-learning stack inside the image (NVFlare + PyTorch)
runs on Python 3.11. Instead we verified that reading 2.5.0-extracted features on
2.4.0 is safe — between the two versions the feature H5 layout is identical and
the **UNI extractor source is byte-identical**, so your features are numerically
exactly what 2.4.0 would have produced. The version check is a conservative guard,
not a format change.

We have therefore updated the image so it accepts features extracted by STAMP up
to 2.5.0 (anything newer still raises, deliberately). We tested precisely your
situation — 2.5.0-extracted features **with a `Slide ID` column** — end to end.

**What you need to do:** use the **new `UKB_1` startup kit** attached
(`UKB_1_1.5.0-dev.260709.8afff27.zip`). It pins the updated image
(`jefftud/decade:1.5.0-dev.260709.8afff27`), which `docker.sh` will pull for you.
Keep your existing UNI (1024-dim) features from STAMP 2.5.0 — no re-extraction.

For the consortium we are standardizing on **extraction with STAMP 2.5.0**, which
is what you already have.

This kit also contains the fix for column names with spaces, so you can set:

```bash
export STAMP_PATIENT_LABEL="Slide ID"
export STAMP_DIM_INPUT=1024
```

## Issue 2 — Tailscale

Nothing is wrong on your machine: your `/etc/hosts` entry (`100.100.101.100
dl3.tud.de`) is correct. The problem is that your node is currently joined to your
**personal** tailnet rather than the DECADE tailnet, so the address is
unreachable.

Jeff is resending the DECADE Tailscale invitation to **islemislem65@gmail.com**.
After you accept it and reconnect, please check:

```bash
tailscale status        # should list the DECADE tailnet / dl3
ping dl3.tud.de         # should reply from 100.100.101.100
```

## Prediction target — still pending

Thank you for both options and the statistics. Because swarm training requires
**all four sites to train the identical model** (same classes, same order), we are
waiting for Mainz, Düsseldorf and Heidelberg to confirm they can label the same
target. We will send the final decision (`STAMP_GROUND_TRUTH_LABEL` +
`STAMP_NUM_CLASSES`) as soon as we have it — most likely early next week. Please
hold on that until then.

## Summary of what to do next

1. Unpack the **new** `UKB_1` startup kit and use its `docker.sh` (not the
   `mediswarm-stamp` source package).
2. Accept the new Tailscale invitation, reconnect, confirm `ping dl3.tud.de`.
3. Re-run Steps 1–5 with the new kit and your existing UNI features
   (`STAMP_DIM_INPUT=1024`, `STAMP_PATIENT_LABEL="Slide ID"`).
4. Await our target confirmation before Monday's run.

Thanks again for the precise reports — the STAMP-version issue would have hit the
other sites too, and you caught it early.

Best regards,
Jeff
