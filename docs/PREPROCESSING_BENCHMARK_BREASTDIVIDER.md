# BreastDivider preprocessing benchmark (odelia-preprocessing 0.2.1)

Status: 23 September 2026, complete (three train/val folds of the train-from-scratch comparison).
Runs on dl3 (GPU) and dl0 (Duke DICOM conversion).
Scripts: `scripts/preprocessing/benchmark_breastdivider/`.

## Summary

- The package works: 645 exams (102 UMCU challenge, 87 + 456 Duke) cropped with **zero fallback
  cases**, a handful of removed segmentation components, no crashes.
- **Frozen ODELIA swarm models must not be deployed on BreastDivider crops.** On the UMCU
  challenge exams the MST global model drops from 0.874 to 0.751 malignant AUROC and the
  1DivideAndConquer model from 0.984 to 0.806 (paired bootstrap intervals exclude 0). Those models
  were trained on the current crops, and the UMCU exams were part of their training data, so this
  is a train/test geometry mismatch, not a lack of information in the new crops.
- On external Duke exams the same frozen models are unaffected (MST 0.690 vs 0.708,
  1DivideAndConquer 0.885 vs 0.882).
- **Trained from scratch on the same Duke exams, the two formats are equivalent.** MST on the shared
  184-side test set, mean over three train/val folds: 0.892 (BreastDivider) vs 0.883 (current),
  +0.009 with a 95 % interval of [-0.026, +0.045]; the three-model ensembles are 0.900 vs 0.907. The
  fold-0 gain of +0.042 did not replicate (fold 1 +0.003, fold 2 -0.019).
- **Switching crop format without retraining costs about 0.06 AUROC in either direction** (current
  model on BreastDivider crops -0.056, BreastDivider model on current crops -0.060, both intervals
  exclude 0), so the two formats must never be mixed within one model's data.
- Cost: docker + GPU for the segmentation (14.4 GB image), about 1.8 s per exam on the GPU plus
  4.5 s per exam of CPU cropping with 14 workers, against 1.3 s per exam for the current pipeline.
- Recommendation: no accuracy case for switching; the case for BreastDivider is robustness (no
  midline assumption, whole breast in z, handles single-breast exams such as the 467 at RSH). Keep
  the current crops for the October run and for every trained model; if the consortium wants the
  new format for the next data version, decide it on robustness grounds, with a fixed-spacing
  output mode tested first and retraining at every site (section 6).

## 1. What was tested

`odelia-preprocessing-0.2.1.zip` (Fraunhofer MEVIS) replaces step 3 of the ODELIA preprocessing
(bilateral volume to two unilateral 256 x 256 x 32 crops) with a segmentation-based method.

| | Current pipeline (`scripts/preprocessing/step3_unilateral.py`) | BreastDivider package (`BD_main.py`) |
|---|---|---|
| Breast localisation | Fixed midline split; height cropped to 256 rows by the 90 % intensity quantile | nnU-Net breast segmentation (`ykirchhoff/breastdivider:latest`, docker, GPU), then a per-breast bounding box; geometric fallback when the segmentation is untrusted |
| Output grid | 256 x 256 x 32 voxels at a fixed 0.7 x 0.7 x 3 mm | 256 x 256 x 32 voxels, but the box is fitted to the breast and then resized, so the spacing varies per case (example 0.64 x 0.81 x 4.9 mm) |
| z coverage | 96 mm, centred on the scan (CropOrPad) | The whole breast extent (example 156 mm of a 157 mm scan), resampled to 32 slices |
| Chest wall | Included down to the quantile-based cut | Included up to a thorax margin behind the segmented breast |
| Subtraction | Post_1 minus Pre, shifted to >= 0, uint16 | Post_1 minus Pre, signed int16 (irrelevant for the models, the loader z-normalises with a min/max mask) |
| Dependencies | torchio, CPU | docker image 14.4 GB and a GPU for the segmentation, then CPU for the cropping |
| Runtime, 456 Duke exams, 14 workers | 9.6 min (1.3 s per exam) | segmentation 13.4 min on a Quadro RTX 8000 (1.8 s per exam) plus cropping 34 min (4.5 s per exam) |

Input for both pipelines is the bilateral `<root>/data/<exam>/{Pre,Post_1}.nii.gz`. The current
pipeline was re-run for this benchmark (`old_unilateral.py`, same method as steps 2 and 3 of the
repository) so that both variants start from identical volumes.

## 2. Data

| Set | Exams | Sides | Labels (no lesion / benign / malignant) | Role |
|---|---|---|---|---|
| UMCU challenge, all labelled | 102 | 204 | 151 / 20 / 33 | frozen-model comparison; the exams were part of the swarm models' training data |
| UMCU challenge, fold-0 test split | 41 sides | | 31 / 2 / 8 | held out from swarm training, too small to decide anything |
| Duke, fold-0 test patients on disk | 87 | 174 | 84 / 0 / 90 | frozen-model comparison on external data |
| Duke, all annotated patients on disk | 456 | 912 | 436 / 0 / 476 | train from scratch, split from `/mnt/dlhd0/DUKE/metadata`: a fixed test set of 184 sides (87 / 0 / 97) and three train/val folds of 588 / 140 sides |

The Duke download on dl0 is partial: 630 of the 922 mapped patients and 456 of the 651 annotated
ones have both Pre and Post_1 on disk; `Breast_MRI_066` lacks the Post_1 series. That is the only
exam either pipeline skipped.

## 3. Protocol

- Frozen models: the final global models of the consortium runs (`MST_swarm_global_final.pt`,
  `1DivideAndConquer_swarm_global_final.pt`) evaluated with `scripts/evaluation/predict.py` inside
  `jefftud/odelia:current`, with each crop variant bind-mounted at `/data/<SITE>/data_unilateral`.
- From scratch: `TRAINING_MODE=local_training`, MST, 20 epochs, folds 0 to 2 (same test set, the
  train/val partition rotates), batch 1 with 8-step accumulation, 16-bit mixed precision, best
  checkpoint by validation AUROC; 27 min per run on the RTX 8000. Both variants use exactly the
  same uids and split; every model is also tested on the other format.
- Comparison: same cases under both variants, AUROC (malignant vs rest, and macro over classes),
  paired bootstrap over cases (2,000 resamples) for the difference. `compare_variants.py`.

## 4. Results

### 4.1 Frozen swarm models, UMCU challenge exams (204 sides)

| Model | Crops | Accuracy | AUROC malignant | AUROC macro | Difference to current, malignant [95 % CI] |
|---|---|---|---|---|---|
| MST | current (re-run) | 0.735 | 0.874 | 0.813 | |
| MST | BreastDivider | 0.716 | 0.751 | 0.728 | -0.122 [-0.219, -0.034] |
| MST | current (site's existing crops) | 0.740 | 0.880 | 0.816 | +0.006 [-0.009, +0.022] |
| 1DivideAndConquer | current (re-run) | 0.922 | 0.984 | 0.975 | |
| 1DivideAndConquer | BreastDivider | 0.745 | 0.806 | 0.740 | -0.179 [-0.269, -0.099] |
| 1DivideAndConquer | current (site's existing crops) | 0.961 | 0.996 | 0.994 | +0.011 [-0.001, +0.028] |

On the 41 held-out test sides the direction is the same (MST 0.932 to 0.788, 1DivideAndConquer
0.985 to 0.898) but the intervals include 0. The near-perfect numbers of the site's existing crops
show that these exams were seen in training; even a re-run of the same pipeline costs the
1DivideAndConquer model 0.02 macro AUROC, so the frozen models are sensitive to any geometric
change, and the BreastDivider change is large (scale and z coverage).

### 4.2 Frozen swarm models, Duke test patients (174 sides, external data)

| Model | Crops | Accuracy | AUROC malignant | Difference [95 % CI] |
|---|---|---|---|---|
| MST | current | 0.626 | 0.690 | |
| MST | BreastDivider | 0.569 | 0.708 | +0.017 [-0.042, +0.076] |
| 1DivideAndConquer | current | 0.759 | 0.885 | |
| 1DivideAndConquer | BreastDivider | 0.730 | 0.882 | -0.003 [-0.050, +0.044] |

### 4.3 MST trained from scratch on Duke, three folds, shared test set (184 sides)

Malignant AUROC of the best-validation checkpoint of each run; the three folds share the test set
and differ in the train/val partition, so they act as three repeats.

| Trained on | Tested on | Fold 0 | Fold 1 | Fold 2 | Mean | 3-model ensemble |
|---|---|---|---|---|---|---|
| current | current | 0.846 | 0.886 | 0.918 | 0.883 | 0.907 |
| BreastDivider | BreastDivider | 0.887 | 0.889 | 0.899 | 0.892 | 0.900 |
| current | BreastDivider | 0.798 | 0.828 | 0.855 | 0.827 | 0.848 |
| BreastDivider | current | 0.856 | 0.840 | 0.798 | 0.831 | 0.861 |

Paired bootstrap over cases (2,000 resamples) of the mean per-fold difference:

| Comparison | Difference [95 % CI] |
|---|---|
| BreastDivider vs current, each in its own format | +0.009 [-0.026, +0.045] |
| current model, BreastDivider crops vs its own | -0.056 [-0.093, -0.021] |
| BreastDivider model, current crops vs its own | -0.060 [-0.107, -0.013] |

The fold-0 advantage of BreastDivider (+0.042, interval [-0.005, +0.094]) did not hold up: fold 1
+0.003, fold 2 -0.019. The BreastDivider runs are the more consistent three (0.887 to 0.899 against
0.846 to 0.918), which is worth noting but is three numbers. All runs were still improving at
epoch 20, so the absolute values are not the ceiling of either format.

### 4.4 Qualitative

- Segmentation report: 0 untrusted-fallback cases in all three runs; component removals 2 (UMCU),
  0 (Duke 87), 4 across 3 cases (Duke 456).
- Diagnostic panels (`--diagnostic_plots`) show clean left/right separation and boxes that stop at
  the chest wall. The MIPs of the same exam under both pipelines show the intended difference: the
  current crop keeps the breast at fixed scale with black padding laterally and cuts the z range;
  the BreastDivider crop fills the box with the breast, keeps the whole z range and a slice of
  thorax behind the breast.

## 5. Package issues found

1. **False error in `BD_run_seg.py`.** Every batch of 10, 20 or 50 cases is logged as
   `[ERROR] Batch N: container reported no input despite 50 staged file(s)` although the
   segmentation succeeds: `NO_CASES_MARKER = "0 cases in the source folder"` is a substring of
   nnU-Net's `There are 50 cases in the source folder`. Nothing aborts, but a real failure would be
   hidden behind the false alarm. Fix: anchor the match (`There are 0 cases`).
2. **Symlink staging.** A symlink whose target is outside the bind-mounted root is invisible inside
   the container (documented by the package). The benchmark hard-links the bilateral volumes into
   the root instead (`make_layout.py`).
3. **Output spacing is not fixed.** The box is fitted to each breast and resized to
   256 x 256 x 32, so voxel size differs per case and from the 0.7 x 0.7 x 3 mm the ODELIA models
   were trained on. This is a design choice worth discussing: fixed physical spacing keeps lesion
   size comparable across cases; box fitting keeps the whole breast. A `--fixed_spacing` output
   mode would let both be compared on equal terms.
4. **`compute_mips.py` refuses a reference folder without `processing.log`** (provenance check);
   `--allow-version-mismatch` is needed to compare against crops made by another tool.
5. `--diagnostic_plots` roughly doubles the cropping time; fine for a one-off audit, not for a
   site-wide run.

## 6. Recommendation and next steps

1. Keep the current crops for the October benchmark run and for every site that already holds
   the consortium models: those models lose 0.1 to 0.2 AUROC on the new crops (section 4.1).
2. There is no accuracy case for switching: trained from scratch on the same exams the two formats
   are within +0.01 of each other (section 4.3). The case for BreastDivider is robustness: no
   midline assumption, the whole breast kept in z, and single-breast exams handled (RSH has 467
   exams with one breast, which the midline split cannot serve). If the consortium adopts it for the
   next data version, every site re-runs step 3 (docker + GPU, about 6 s per exam) and every model
   is retrained; mixing formats costs 0.06 AUROC (section 4.3).
3. Before adoption, agree with MEVIS on: the fixed-spacing question (item 3 above), the false error
   (item 1), and a version tag in the crop metadata so that a site's crops can be told apart from
   the current format (the loader cannot detect the difference, and mixing formats in one run would
   reproduce the section 4.1 drop silently).
4. Validation still missing: a from-scratch comparison on an ODELIA site with more than a few
   hundred labelled sides (UKA or CAM), run by the site with its own GPU, and the fixed-spacing
   variant of BreastDivider, which is the most likely way to keep its localisation and drop the
   per-case rescale that the cross-format results point at.

## 7. Reproduction

Everything is on dl3 under `/home/swarm/bd_bench` (venv, unzipped package `bdpp/`, `scripts/`,
logs) and `/mnt/swarm_alpha/bd_bench/{challenge,duke,duke_full,eval_out,train_out}`; the Duke
DICOM conversion is on dl0 under `/mnt/dlhd0/bd_bench_duke`. The scripts in
`scripts/preprocessing/benchmark_breastdivider/` are the ones that ran, with their paths:

| Script | Purpose |
|---|---|
| `make_layout.py` | bilateral input layout from the extracted challenge NIfTIs (hard links) |
| `duke_dicom2nifti.py` | Duke DICOM to bilateral NIfTI for a patient subset (SimpleITK, TCIA mapping) |
| `old_unilateral.py` | the current pipeline (steps 2 and 3) runnable on any root |
| `run_challenge_bd.sh`, `run_duke.sh`, `run_duke_full.sh`, `run_duke_folds.sh` | the chains: segmentation, crops, MIPs, evaluation roots, training |
| `eval_variant.sh`, `eval_duke.sh`, `eval_duke_trained.sh`, `train_duke.sh` | frozen-model evaluation, local training and evaluation inside `jefftud/odelia:current` |
| `compare_variants.py` | paired AUROC comparison with bootstrap intervals |
