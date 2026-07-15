# DECADE — site registry (data, targets, contacts, status)

**Single source of truth for what each centre can actually provide.**
Update this whenever a site reports new numbers. Last updated: **2026-07-15**.

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
| Universitätsklinikum Heidelberg | `UKHD_1` | Christian Zöllner (GitHub: `Altrabeth`) + Nina Nelius | ckobrow@mailbox.org, nina.nelius@med.uni-heidelberg.de |

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
- **Multi-slide site** (719 slides > 569 patients): needs a **slide table**
  (`STAMP_SLIDE_TABLE` + `STAMP_FILENAME_LABEL`), which loaded 0 patients until #437.
  The 61/508 figures are **patients**, 119/600 are **slides** — decide which unit the
  run trains/evaluates on.

### Düsseldorf (`UKD_1`) — pending
Returning from holiday; will report **by Wed 2026-07-22 at the latest**.

### Heidelberg (`UKHD_1`) — MSI available

MSI-High status, **one slide per patient**:

| Cohort | yes (MSI-High) | no | not provided |
|---|---|---|---|
| Stage I–III | **342** | **262** | 845 |
| Adenomas | 1 | 2 | 424 |

- **Usable MSI cohort ≈ 604 patients (342 pos / 262 neg)** from Stage I–III; the
  adenoma cohort has essentially no MSI labels (1/2) and would be dropped.
- Prevalence **~57 % positive** — very different from Mainz's ~11 %. That is a real
  cross-site label-distribution shift (likely cohort selection: Stage I–III vs full).
  Fine for swarm learning, but worth noting for the analysis.
- Exact column name / values still to confirm with Christian.

---

## 3. ⚠️ The open conflict

**Two of three responding sites can supply MSI; Bonn cannot.** MSI-High is now the
clear front-runner for the shared target — the only blocker is Bonn.

| option | target | classes | Bonn | Mainz | Heidelberg | notes |
|---|---|---|---|---|---|---|
| **A** | `MSI-High` | 2 | ❌ all `not provided` | ✅ 569 pts (11% pos) | ✅ ~604 pts (57% pos) | blocked only by Bonn |
| **B** | Lynch vs. Sporadic | 2 | ✅ 1174 | ❓ needs germline | ❓ needs germline | flips the blocker to Mainz+Heidelberg |
| **C** | dMMR/pMMR | 2 | ❌ `not provided` | ✅ (≡ MSI-High) | ❓ | same as A |

Düsseldorf (pending, by Wed 2026-07-22) is the remaining unknown for option A.

**Do not** approximate MSI from Lynch status: Lynch is a *germline* syndrome and MSI
is a *tumour* phenotype. Most Lynch tumours are MSI-High, but sporadic MSI-High
(usually MLH1-hypermethylated) exists and would be mislabelled. That is a different
study, not a relabelling.

**Blocked on:** (a) **Bonn** confirming with Dr. Robert / Dr. Jacob whether any real
MSI/MMR data exists — this is now the only site that cannot do MSI; (b) **Düsseldorf**
reporting (by Wed 2026-07-22). Heidelberg has reported (MSI available).

**If Bonn has no MSI:** the honest choices are (i) run MSI with Mainz + Heidelberg
(+ Düsseldorf if available) and leave Bonn out of the first run, or (ii) pick a
target Bonn shares — but Lynch-vs-sporadic would then require Mainz and Heidelberg to
supply a germline label they have not mentioned. Operator/PI decision.

---

## 4. Readiness

| Site | Tailscale | SSH key authorized | Steps 1–3 | Step 5 baseline | Target |
|------|-----------|--------------------|-----------|-----------------|--------|
| **UKB_1** (Bonn) | ✅ | ✅ | ✅ | ✅ 32 epochs, val_loss 0.350→0.031 (3-class) | germline only |
| **Mainz_1** | ✅ | ✅ | ✅ | ✅ 32 epochs (kit `6bc06c4`) | `MSI-High` ✅ |
| **UKD_1** (Düsseldorf) | ? | ❌ | ❌ | ❌ | pending |
| **UKHD_1** (Heidelberg) | ✅ ping ok | ✅ | Step 1 ✅ (dummy training ran) | ❌ | `MSI-High` ✅ (see §2) |

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
