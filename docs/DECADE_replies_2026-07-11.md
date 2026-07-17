# DECADE replies — 2026-07-11 (Mainz + Heidelberg)

---

## A) Reply to Christina Glasner (UM Mainz / `Mainz_1`)

**Subject:** Re: DECADE — slide table bug found and fixed

Dear Christina,

Your instinct was exactly right: **the slide table was not being used at all.** This
was a bug on our side, not a mistake in your setup.

### What was wrong

Our code read `STAMP_SLIDE_TABLE` and passed it to the training step — but the
patient list was built *earlier*, by a STAMP function that takes **no slide table**
and instead assumes each patient has exactly one feature file named
`<patient_id>.h5`. Your H5 files are named per *slide*, so every patient looked
"missing", the patient list came back empty, and training then failed with the
confusing `n_samples=0` error.

So any site with multi-slide patients simply could not train. We reproduced your log
exactly, then fixed it: when `STAMP_SLIDE_TABLE` is set we now use STAMP's real
slide-table path (slide → patient mapping, then pooling each patient's slides into
one bag).

Verified on data shaped like yours: **48 slides → 24 patients**, training proceeds.
We also made the failure mode honest — if no patients can be matched you now get a
message naming the tables, the columns and the fix, instead of `n_samples=0`.

### What you need to do

Use the **new `Mainz_1` startup kit** (attached), keep your `STAMP_SLIDE_TABLE`
export exactly as you had it, and re-run Steps 2 and 3:

```bash
export STAMP_SLIDE_TABLE="/data/Mainz_1/slide_table.csv"
export STAMP_PATIENT_LABEL="PATIENT"      # your patient column, in BOTH tables
export STAMP_FILENAME_LABEL="FILENAME"    # your slide-filename column
```

(Use your real column names if they differ — and remember `/data/...` are paths
*inside* the container.)

Thank you for the precise report — you have now found two genuine bugs for the
consortium.

Best regards,
Jeff

---

## B) Reply to Christian Zöllner (UK Heidelberg / `UKHD_1`)

**Subject:** Re: DECADE — Tailscale invite + the shared target

Dear Christian,

Thanks — SSH key **received and authorized**, nothing further needed there.

### Tailscale

I checked our tailnet: there is currently **no Heidelberg machine registered at all**
(the only DECADE nodes are Bonn, Mainz and Düsseldorf). So nothing was lost — the
node simply was never added, which is why none of your addresses are recognised.

I am sending you a **fresh invitation to `ckobrow@mailbox.org`** (the address you
wrote from). Accept it, connect the Heidelberg machine, then:

```bash
tailscale status          # your node should appear
ping dl3.tud.de           # should answer from 100.100.101.100
```

Make sure `/etc/hosts` contains `100.100.101.100  dl3.tud.de  dl3`. If you'd rather
we invite a different address, just say which.

### "But don't we all need the same training target?"

**Yes — exactly right, and that's the crux.** Swarm learning trains *one* shared
model, so every site must use the identical label with the identical classes in the
identical order. That is why we have been holding this decision.

Good news: your **`Sporadic vs. Familial`** is the same label family Bonn proposed,
so **Heidelberg and Bonn already agree**. Mainz has a different kind of label
(tumour MSI/MMR phenotype). We are resolving this now and will send the final
`STAMP_GROUND_TRUTH_LABEL` + `STAMP_NUM_CLASSES` to everybody.

Two questions for you so we can line the sites up:

1. **`not provided`** — we read this as *missing data*, not a real biological class.
   We would **exclude** those patients rather than train a third class on them. How
   many of your cases are `not provided`, and do you agree?
2. So your usable labels are **`Lynch syndrome` vs `Sporadic`** — how many cases in
   each? (Bonn additionally has a *familial CRC / FAP* group; if you have no such
   cases, the common denominator across sites is the binary **Lynch vs. Sporadic**.)
3. Do you *also* have **MSI status** or **dMMR/pMMR**? That would tell us whether a
   tumour-phenotype target is feasible for everyone (Mainz's proposal).

Once you and Mainz answer, we can fix the target for all four sites.

Best regards,
Jeff
