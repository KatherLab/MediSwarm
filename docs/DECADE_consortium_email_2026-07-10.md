# DECADE consortium email — new startup kits + open questions (2026-07-10)

> Send to all four sites. Attach each site its own kit zip from
> `workspace/decade_1.5.0-dev.260710.a295c6b_allsites/prod_00/`.
> Fill `<YOUR NAME>` / `<OPERATOR_CONTACT>` before sending.

---

**Subject:** DECADE — new startup kits (please re-download), answers to your questions, and the one decision we still need

Dear all,

Thank you — especially **Islem (Bonn)** and **Christina (Mainz)** for the very
precise reports this week. Both of you found **real bugs**, and both are now fixed
for everyone. Attached is a **new startup kit for each site**; please switch to it.

A quick reminder on wording: we call this **swarm learning** (each site trains
locally and only model weights are shared — no data ever leaves your institution).

---

## 1. New startup kits — please use these

| Site | `SITE_NAME` | Your kit |
|------|-------------|----------|
| Universitätsklinikum Bonn | `UKB_1` | `UKB_1_1.5.0-dev.260710.a295c6b.zip` |
| Universitätsmedizin Mainz | `Mainz_1` | `Mainz_1_1.5.0-dev.260710.a295c6b.zip` |
| Universitätsklinikum Düsseldorf | `UKD_1` | `UKD_1_1.5.0-dev.260710.a295c6b.zip` |
| Universitätsklinikum Heidelberg | `UKHD_1` | `UKHD_1_1.5.0-dev.260710.a295c6b.zip` |

Unpack it and `cd <SITE_NAME>/startup`. `./docker.sh` pulls the matching container
image for you. **Please discard any earlier kit** — older kits are missing the
fixes below.

**What is fixed in this kit**

1. **`sudo` is now detected.** If the `STAMP_*` variables are not visible, the
   script tells you exactly what to do instead of failing with a Python traceback
   (thanks, Christina — see §2).
2. **Features extracted with STAMP 2.5.0 are now readable.** Previously training
   refused them with *"features were extracted with a newer version of stamp"*
   (thanks, Islem — see §3).
3. **Clinical column names containing spaces** (e.g. `Slide ID`) now work.
4. **Step 1 (dummy training)** and **Steps 2/5 (preflight, local training)** run
   correctly — two packaging bugs are fixed.

---

## 2. Christina's question — "I exported the variable, why can't the script see it?"

Because of **`sudo`**. It resets the environment, so although
`echo $STAMP_CLINI_TABLE` prints the right value in *your* shell, `sudo ./docker.sh`
starts the script with a clean environment and forwards nothing into the container.
We measured this on one of our machines:

| command | `STAMP_*` variables the script can see |
|---|---|
| `sudo ./docker.sh …` | **none** |
| `sudo -E ./docker.sh …` | all of them |
| `./docker.sh …` (no sudo) | all of them |

**Fix — preferred** (run Docker without `sudo`; one-time, then re-login):

```bash
sudo usermod -aG docker $USER
newgrp docker
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check
```

**Fallback:** `sudo -E ./docker.sh …` (preserves your environment).

Sanity check at any time: `env | grep '^STAMP_'`.

### Patients with several slides — use a slide table

If a patient has more than one slide, your `clini_table.csv` has one row per
**patient** while `features/` has one `.h5` per **slide**. A slide table maps them,
so all of a patient's slides are pooled into one bag:

```bash
export STAMP_SLIDE_TABLE="/data/<SITE_NAME>/slide_table.csv"
export STAMP_PATIENT_LABEL="PATIENT"     # patient-ID column (both tables)
export STAMP_FILENAME_LABEL="FILENAME"   # slide-filename column (slide table)
```

`slide_table.csv`, one row per slide:

| PATIENT | FILENAME |
|---------|----------|
| P_001 | slide_001.h5 |
| P_001 | slide_002.h5 |
| P_002 | slide_003.h5 |

`FILENAME` must match the files in your `features/` folder. Use your real column
names and set `STAMP_PATIENT_LABEL` / `STAMP_FILENAME_LABEL` accordingly.

---

## 3. Islem's question — STAMP version for feature extraction

Please **extract features with standalone STAMP 2.5.0** (Bonn already has). The
training container itself runs STAMP 2.4.0, because STAMP 2.5.0 requires Python
3.13 and the swarm stack inside the image does not support that yet. We verified
that the feature files and the UNI extractor are **identical** between 2.4.0 and
2.5.0, and the new kit reads 2.5.0-extracted features. **No re-extraction is
needed.**

Please do **not** use a STAMP newer than 2.5.0 without telling us first.

---

## 4. Agreed technical settings

- **Feature extractor: UNI, `STAMP_DIM_INPUT=1024`** — identical at all sites.
- **Model: `vit`** (default). Also available: `mlp`, `trans_mil`, `linear`.
  `barspoon` is **not** available in this release.
- **Tailscale:** accept the invitation, then add to `/etc/hosts`:
  `100.100.101.100  dl3.tud.de  dl3` and check `ping dl3.tud.de`.
- **Never send us** the output of `--log_dataset_details` — it can contain patient
  IDs. Just tell us pass/fail and any error message.

---

## 5. The one open decision: our shared prediction target

Swarm learning requires **every site to train the identical model** — the same
label, the same number of classes, in the same order. Right now the two proposals
we have are different *kinds* of label:

- **Bonn:** germline syndrome — *Lynch / familial (FAP) / sporadic* (3 classes), or
  *Lynch vs. sporadic* (2 classes).
- **Mainz:** tumour phenotype — *MSI-High* (yes/no, 678 slides), or a *dMMR* column
  with six mixed values (103 slides).

These cannot simply be merged. Two remarks:

1. **`MSI-High` (binary)** looks like the most harmonizable candidate: it is the
   standard, best-populated label.
2. Mainz's `dMMR` column mixes MMR **status** (`pMMR`, `dMMR`) with the **specific
   protein loss** (`MLH1`, `MSH6`, `MLH1, PMS2`, …). As six classes over 103 slides
   it would be very sparse; we would collapse it to **binary dMMR vs pMMR**.

**Please answer, all four sites:**

- Do you have **MSI status** (MSI-High vs. MSS/MSI-Low)? How many slides per class?
- Do you have a clean **binary dMMR vs pMMR** column? How many slides per class?
- If neither, what is the closest label you can provide?

As soon as we have all four answers we will confirm the final
`STAMP_GROUND_TRUTH_LABEL` and `STAMP_NUM_CLASSES` to everyone. **Please hold your
Step 5 (local training) until then** — the class count determines the model's output
layer, so a baseline trained on the wrong target is not comparable.

---

## 6. Where each site stands

| Site | Tailscale | Steps 1–5 | SSH key | Target values |
|------|-----------|-----------|---------|---------------|
| **UKB_1** (Bonn) | ✅ | ✅ (please re-run with the new kit) | ✅ authorized | proposed, pending consortium |
| **Mainz_1** (Mainz) | ✅ | Step 1 ✅, Steps 2/3/5 blocked by `sudo` → now unblocked | ✅ authorized | proposed, pending consortium |
| **UKD_1** (Düsseldorf) | appears connected | **awaiting** | **awaiting** | **awaiting** |
| **UKHD_1** (Heidelberg) | **awaiting** | **awaiting** | **awaiting** | **awaiting** |

**Düsseldorf and Heidelberg** — could you please send us:

1. Your **two values** (`STAMP_GROUND_TRUTH_LABEL`, `STAMP_NUM_CLASSES`) — see §5.
2. Your **SSH public key** (Step 4) for run-log collection:
   `ssh-keygen -t ed25519 -C "$(hostname)@mediswarm"` → send `~/.ssh/id_ed25519.pub`
   to <OPERATOR_CONTACT>. Never the private key.
3. Confirmation that **Tailscale** is connected and `ping dl3.tud.de` works.
4. Pass/fail for **Steps 1–3** with the new kit.

---

## 7. Monday 13 July

Given the open target question, we propose to use Monday for an **infrastructure
dry run**: all four sites start their client, we confirm every node registers with
the server and completes a couple of short swarm rounds. We then start the **real
training run** as soon as the target is agreed and each site has re-run Steps 2–3.
If we do have all four answers before Monday, we will go straight to the real run.

We will send the exact start time on Monday morning.

Thank you all again — the two reports this week genuinely improved the software for
every site.

Best regards,
<YOUR NAME>
DECADE Swarm Operator, TU Dresden
