# DECADE — site registry (data, targets, contacts, status)

**Single source of truth for what each centre can actually provide.**
Update this whenever a site reports new numbers. Last updated: **2026-07-14**.

Swarm learning requires **one shared target**: identical label, identical
`STAMP_NUM_CLASSES`, identical class order at every site. Everything below exists to
answer one question — *which target can all four sites actually supply?*

---

## 1. Contacts

| Site | `SITE_NAME` | Contact | Email |
|------|-------------|---------|-------|
| Universitätsklinikum Bonn | `UKB_1` | Islem Gammoudi | Islem.Gammoudi@ukbonn.de |
| Universitätsmedizin Mainz | `Mainz_1` | Christina Glasner (+ Sebastian, pathology) | glasnerc@uni-mainz.de |
| Universitätsklinikum Düsseldorf | `UKD_1` | Tobias Seraphin (+ Janik) | TobiasPaul.Seraphin@med.uni-duesseldorf.de |
| Universitätsklinikum Heidelberg | `UKHD_1` | Christian Zoellner (GitHub: `Altrabeth`) | ckobrow@mailbox.org |

TU Dresden hosts **server + admin only** (no training client).

---

## 2. Candidate targets — what each site actually has

### Bonn (`UKB_1`) — germline syndrome only. **No MSI.**

- **Has:** germline syndrome, 1174 patients — Lynch **423** / FAP **436** / Sporadic **315**
  - column: `Sporadic vs. Familial`, 3 classes
  - also: `Lynch vs. Sporadic` (binary, 1174 slides)
- **MSI-High / dMMR columns exist but every slide is labelled `not provided`.**
- Only other molecular annotation: *Affected Gene in the Germline*
  (MLH1, MSH2, MSH6, PMS2, EPCAM, APC), plus adenoma dysplasia (low/high grade).
- Checking with Dr. Robert and Dr. Jacob whether any MSI/MMR data exists.
- Note: Lynch patients are *biologically* dMMR/MSI-High, but this is **not** a
  direct MSI label — deriving one would mislabel sporadic MSI-High cases.

### Mainz (`Mainz_1`) — MSI available

| | |
|---|---|
| Column | `MSI-High` |
| Values | `yes` / `no` |
| Cohort | 719 slides, 569 patients |
| **Patients** | **61 positive / 508 negative** |
| **Slides** | **119 positive / 600 negative** |

- Pathologist (Sebastian) confirms binary **dMMR/pMMR ≡ MSI-High** — two lab routes
  to the same conclusion. Mainz supplemented each column from the other.
- Their `dMMR` column mixes MMR *status* with *which gene is affected*
  (`MLH1`, `MSH6`, `MLH1, PMS2`, …) — **do not use as-is**; use `MSI-High`.
- **~11 % positives ⇒ judge on AUROC, not accuracy** (predicting "no" for everyone
  already scores ~89 %).

### Düsseldorf (`UKD_1`) — pending
Returning from holiday; will report **by Wed 2026-07-22 at the latest**.

### Heidelberg (`UKHD_1`) — pending
Still working through the setup steps (see §4).

---

## 3. ⚠️ The open conflict

**Bonn cannot currently supply MSI.** So the proposed shared target (MSI-High,
binary) is **not yet achievable across all four sites**. The realistic options:

| option | target | classes | Bonn | Mainz | notes |
|---|---|---|---|---|---|
| **A** | `MSI-High` | 2 | ❌ not available | ✅ 569 pts | blocked unless Bonn finds MSI/MMR data |
| **B** | Lynch vs. Sporadic | 2 | ✅ 1174 | ❓ | Mainz would need a germline label |
| **C** | dMMR/pMMR | 2 | ❌ `not provided` | ✅ (≡ MSI-High) | same blocker as A |

**Do not** approximate MSI from Lynch status: Lynch is a *germline* syndrome and MSI
is a *tumour* phenotype. Most Lynch tumours are MSI-High, but sporadic MSI-High
(usually MLH1-hypermethylated) exists and would be mislabelled. That is a different
study, not a relabelling.

**Blocked on:** (a) Bonn confirming with Dr. Robert / Dr. Jacob whether any MSI or
MMR data exists; (b) Düsseldorf and Heidelberg reporting.

---

## 4. Readiness

| Site | Tailscale | SSH key authorized | Steps 1–3 | Step 5 baseline | Target |
|------|-----------|--------------------|-----------|-----------------|--------|
| **UKB_1** (Bonn) | ✅ | ✅ | ✅ | ✅ 32 epochs, val_loss 0.350→0.031 (3-class) | germline only |
| **Mainz_1** | ✅ | ✅ | ✅ | ✅ 32 epochs (kit `6bc06c4`) | `MSI-High` ✅ |
| **UKD_1** (Düsseldorf) | ? | ❌ | ❌ | ❌ | pending |
| **UKHD_1** (Heidelberg) | ? | ❌ | Step 1 blocked (empty `$SCRATCHDIR`, see #443) | ❌ | pending |

---

## 5. Agreed technical settings

- **Feature extractor:** UNI, `STAMP_DIM_INPUT=1024`, extracted with **STAMP 2.5.0**.
- **Model:** `vit`. (`barspoon` unavailable — see #432.)
- **Current kit/image:** `jefftud/decade:1.5.0-dev.260713.6bc06c4`
  (kits carry fresh certificates — older kits cannot connect).
- **Server:** `dl3.tud.de` → `100.100.101.100` (Tailscale), ports 8002/8003.
- Sites must **never** send us `--log_dataset_details` output (contains patient IDs).

---

## 6. Bugs found by sites (all fixed)

| # | Reported by | Bug | PR |
|---|---|---|---|
| 1 | Bonn | STAMP 2.5.0-extracted features rejected | #417 |
| 2 | Bonn | column names with spaces (`Slide ID`) broke the run | #413 |
| 3 | Mainz | `sudo` silently discarded all `STAMP_*` config | #433 |
| 4 | Mainz | multi-slide patients loaded **0 patients** | #437 |
| 5 | Mainz | local baseline trained only **1 epoch** | #442 |
| 6 | Mainz | model trained with **dropout disabled** after epoch 1 | #442 |
| 7 | Mainz | live-sync uploads failed — `rrsync` locked the whole upload root, so **one site blocked all others** | #420 |
| 8 | Heidelberg | empty `$SCRATCHDIR` swallowed the next flag; error blamed `--GPU` | #443 |
