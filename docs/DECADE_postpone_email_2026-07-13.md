# DECADE consortium — postpone the first swarm run (2026-07-13)

> Send to all four sites. Fill `<YOUR NAME>` / `<OPERATOR_CONTACT>`.
> Attach each site its new kit (see §4) once the final build is ready.

---

**Subject:** DECADE — first swarm run postponed; new startup kits + the one decision we still need

Dear all,

We are **postponing today's first swarm training run.** Two of the four sites are
not yet ready, and — more importantly — we have not yet agreed on the **single
prediction target** that all four sites must share. I would rather delay than burn
a run on a setup we know is incomplete.

Thank you to **Bonn** and **Mainz**: your reports this week found **five real bugs**,
all now fixed. Below is where we stand and exactly what we need.

---

## 1. Why we are postponing

Swarm learning trains **one model together**: every site must train the *same*
architecture on the *same* label with the *same* number of classes. Today:

- **Bonn** is fully ready and configured for **germline syndrome**
  (Lynch / familial FAP / sporadic — 3 classes).
- **Mainz** is now running, and proposes **MSI-High** (binary), a **tumour
  phenotype**.
- **Düsseldorf** and **Heidelberg** have not yet reported their data or settings.

Bonn's and Mainz's labels answer **different biological questions** and cannot be
merged or aggregated. Starting a run with mismatched targets would produce a model
that is meaningless — and because a consortium run has no partial credit (every
site must complete), a single mismatched site invalidates the whole run.

---

## 2. The decision we need — please reply

Mainz has clarified (thank you, Christina, and Sebastian) that a binary
**dMMR vs. pMMR** column and **MSI-High** carry essentially the **same biological
information** — they are two lab routes to the same conclusion. Mainz can supply
**719 slides / 569 patients** on that basis.

**Our proposal: use MSI status (MSI-High vs. not) as the shared target — binary,
`STAMP_NUM_CLASSES=2`.** It is the standard, best-populated, most harmonizable
label in colorectal cancer, and sites can fill it from either MSI or MMR testing.

**Every site, please answer:**

1. Can you provide **MSI status** (MSI-High vs. MSS/MSI-Low), directly or derived
   from dMMR/pMMR? **How many slides and patients per class?**
2. What is the **exact column name** in your `clini_table.csv`, and its possible
   values?
3. If you cannot provide it, tell us now and say what you *can* provide.

Bonn's germline-syndrome labels remain scientifically interesting, but they are a
**different study**. We suggest running MSI first, as it is the target all sites
can realistically share, and revisiting the germline question afterwards.

---

## 3. What Bonn and Mainz found (fixed for everyone)

Both sites hit genuine bugs; every one is fixed in the new kits:

1. **Features extracted with STAMP 2.5.0 were rejected** by the training image
   (Bonn). Now readable — **no re-extraction needed**.
2. **Clinical column names containing spaces** (e.g. `Slide ID`) broke the run
   (Bonn). Fixed.
3. **`sudo` silently discarded your configuration** (Mainz) — the script now
   detects it and tells you what to do. *Do not run `docker.sh` with `sudo`.*
4. **Multi-slide patients loaded 0 patients** (Mainz) — sites using a slide table
   were silently training on nothing. Fixed and verified. **This one is important:
   if any of your patients have more than one slide, you need the new kit.**
5. **Local training (Step 5) ran for only 1 epoch** (Mainz) — so your "baseline"
   was effectively untrained and not comparable to the swarm model. It now trains
   properly (32 epochs by default, `STAMP_MAX_EPOCHS`).

Also fixed: a warning several of you saw — *"Found N module(s) in eval mode at the
start of training"*. This was real: after the first epoch the model kept training
with dropout disabled. Corrected.

**Please re-run Step 5 with the new kit** so your baseline is a fair comparison.

---

## 4. New startup kits — required

| Site | `SITE_NAME` | Kit |
|------|-------------|-----|
| Universitätsklinikum Bonn | `UKB_1` | `UKB_1_1.5.0-dev.260713.6bc06c4.zip` |
| Universitätsmedizin Mainz | `Mainz_1` | `Mainz_1_1.5.0-dev.260713.6bc06c4.zip` |
| Universitätsklinikum Düsseldorf | `UKD_1` | `UKD_1_1.5.0-dev.260713.6bc06c4.zip` |
| Universitätsklinikum Heidelberg | `UKHD_1` | `UKHD_1_1.5.0-dev.260713.6bc06c4.zip` |

**Everyone must switch to the new kit, including Bonn** — the new kits carry fresh
certificates, so an older kit can no longer connect to the server. If you have a
client running, please **stop it** and redeploy:

```bash
cd <SITE_NAME>/startup
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --start_client   # only when we say go
```

Apologies for the extra round-trip, Bonn — this is the cost of the multi-slide and
epoch fixes landing after your kit was issued.

---

## 5. Düsseldorf and Heidelberg — we need to hear from you

Please send:

1. **MSI status** availability + counts per class (see §2).
2. Your **SSH public key** (`ssh-keygen -t ed25519 -C "$(hostname)@mediswarm"`,
   send `~/.ssh/id_ed25519.pub`) to <OPERATOR_CONTACT>.
3. Confirmation that **Tailscale** is connected and `ping dl3.tud.de` works.
4. Pass/fail for **Steps 1–3** with the new kit.

If your data or hardware is not ready, please just say so — that is far more useful
to us than silence.

---

## 6. New date

We will propose a new run date **as soon as we have the target confirmed by all
four sites** (§2). Realistically that means the run happens once §2 is answered and
each site has re-run Steps 2, 3 and 5 on the new kit.

Thank you all — and particularly Islem and Christina, whose careful reports made
the software materially better for every site.

Best regards,
<YOUR NAME>
DECADE Swarm Operator, TU Dresden
