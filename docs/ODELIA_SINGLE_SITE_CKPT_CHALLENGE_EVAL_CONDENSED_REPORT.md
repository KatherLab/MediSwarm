# ODELIA Checkpoint Evaluation: Condensed Summary
> Presentation-oriented summary. It collapses multiple checkpoints for the same training-source/model family to one externally strongest checkpoint and keeps detailed provenance in the full report.

## Storyline
- The strongest single-site checkpoint on external ODELIA challenge Class-2/Malignant AUROC is **UKA_1DC_epoch25_step57980** with weighted Class-2 AUROC **0.858**.
- **Internal validation** is the source site's local validation split during training; it is useful for checkpoint selection and overfitting diagnosis.
- **External validation** is held-out ODELIA challenge inference across CAM/MHA/RSH/RUMC/UKA/UMCU; this is the main transfer/generalisation endpoint.
- Internal validation and external challenge performance are related but not interchangeable: some high internal-val checkpoints transfer poorly, and some later checkpoints trade AUROC for recall/specificity differently.
- The 2026-05-13 challenge swarm/local package is included as reference internal-validation context for six swarm-trained models; it is not the same endpoint as the single-site external challenge evaluation below.

## Cohort Distributions
External ODELIA challenge cohorts (cases by class):
| Challenge site | Cases |
| --- | --- |
| CAM | n=102; 0=71, 1=11, 2=20 |
| MHA | n=40; 0=26, 1=3, 2=11 |
| RSH | n=40; 0=21, 1=6, 2=13 |
| RUMC | n=14; 0=8, 1=0, 2=6 |
| UKA | n=40; 0=19, 1=17, 2=4 |
| UMCU | n=64; 0=48, 1=11, 2=5 |

Internal validation cohorts used for checkpoint selection:
| Source | Model | Run ID | Validation cases |
| --- | --- | --- | --- |
| USZ | MST | MST_unilateral_2026_04_28_083041 | n=814; 0=450, 1=276, 2=88 |
| USZ | 1DC | 1DivideAndConquer_unilateral_2026_05_12_124440 | n=814; 0=450, 1=276, 2=88 |
| UKA | 1DC | 1DivideAndConquer_unilateral_2026_05_04_082228 | n=4470; 0=1192, 1=2970, 2=308 |
| CAM | 1DC | 1DivideAndConquer_unilateral_2026_04_28_161733 | n=142; 0=120, 1=11, 2=11 |
| MHA | 1DC | 1DivideAndConquer_unilateral_2026_04_22_154631 | n=204; 0=167, 1=22, 2=15 |
| RSH | 5Pimed | challenge_5pimed_unilateral_2026_04_03_182744 | n=87; 0=3, 1=32, 2=52 |
| RUMC | MST | MST_unilateral_2026_04_13_162111 | n=200; 0=199, 1=0, 2=1 |
| RUMC | MST | MST_unilateral_2026_02_18_120355 | n=200; 0=199, 1=0, 2=1 |

## Selected Single-Site Checkpoints
| Source | Model | Selected checkpoint | Internal val C2 AUROC | Internal val ACC | External C2 AUROC | External macro AUROC | External ACC | External C2 recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UKA | 1DC | UKA_1DC_epoch25_step57980 | 0.921 | 0.702 | 0.858 | 0.709 | 0.307 | 0.535 |
| USZ | 1DC | USZ_1DC_epoch14_best | 0.786 | 0.634 | 0.810 | 0.675 | 0.390 | 0.157 |
| MHA | 1DC | MHA_1DC_epoch37_step3876 | 0.808 | 0.828 | 0.795 | 0.658 | 0.710 | 0.404 |
| USZ | MST | USZ_MST_epoch33_best | 0.774 | 0.617 | 0.723 | 0.612 | 0.363 | 0.486 |
| CAM | 1DC | CAM_1DC_epoch52_step6466 | 0.965 | 0.866 | 0.682 | 0.616 | 0.630 | 0.433 |
| RUMC | MST | RUMC_MST_20260218_last | 0.568 | 0.995 | 0.523 | 0.495 | 0.643 | 0.000 |
| RSH | 5Pimed | RSH_5Pimed_last | 0.620 | 0.598 | 0.456 | 0.498 | 0.190 | 0.879 |

Selection rule: one checkpoint per `(training source, model family)`, choosing the highest external ODELIA challenge weighted Class-2 AUROC. This removes repeated checkpoint variants from the presentation view while preserving the main comparison.

## Top External Checkpoints Overall
| Checkpoint | Source | Model | External C2 AUROC | External macro AUROC | External ACC | External C2 recall |
| --- | --- | --- | --- | --- | --- | --- |
| UKA_1DC_epoch25_step57980 | UKA | 1DC | 0.858 | 0.709 | 0.307 | 0.535 |
| USZ_1DC_epoch14_best | USZ | 1DC | 0.810 | 0.675 | 0.390 | 0.157 |
| MHA_1DC_epoch37_step3876 | MHA | 1DC | 0.795 | 0.658 | 0.710 | 0.404 |
| MHA_1DC_epoch34_step3570 | MHA | 1DC | 0.786 | 0.650 | 0.673 | 0.128 |
| UKA_1DC_last | UKA | 1DC | 0.767 | 0.630 | 0.410 | 0.375 |
| USZ_MST_epoch33_best | USZ | MST | 0.723 | 0.612 | 0.363 | 0.486 |
| MHA_1DC_last | MHA | 1DC | 0.721 | 0.612 | 0.630 | 0.288 |
| USZ_1DC_last | USZ | 1DC | 0.716 | 0.643 | 0.413 | 0.393 |

## Internal Validation Summary
| Source | Model | Run ID | Train epochs | Best val ACC | Best val Class-2 AUROC | Last val ACC / C2 AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| USZ | MST | MST_unilateral_2026_04_28_083041 | 100 | e33 / 0.617 | e35 / 0.786 | e99 / 0.557 / 0.711 |
| USZ | 1DC | 1DivideAndConquer_unilateral_2026_05_12_124440 | 100 | e26 / 0.635 | e39 / 0.797 | e99 / 0.588 / 0.752 |
| UKA | 1DC | 1DivideAndConquer_unilateral_2026_05_04_082228 | 100 | e25 / 0.702 | e21 / 0.931 | e99 / 0.650 / 0.861 |
| CAM | 1DC | 1DivideAndConquer_unilateral_2026_04_28_161733 | 100 | e73 / 0.873 | e37 / 0.975 | e99 / 0.852 / 0.940 |
| MHA | 1DC | 1DivideAndConquer_unilateral_2026_04_22_154631 | 100 | e77 / 0.833 | e30 / 0.830 | e99 / 0.819 / 0.667 |
| RSH | 5Pimed | challenge_5pimed_unilateral_2026_04_03_182744 | 25 | e13 / 0.609 | e12 / 0.683 | e24 / 0.598 / 0.620 |
| RUMC | MST | MST_unilateral_2026_04_13_162111 | 1 | e0 / 0.995 | e0 / 0.558 | e0 / 0.995 / 0.558 |
| RUMC | MST | MST_unilateral_2026_02_18_120355 | 100 | e0 / 0.995 | e5 / 0.910 | e99 / 0.995 / 0.568 |

Internal validation curves are available as SVGs under `docs/figures/odelia_single_site_eval/`. The most relevant plots for presentation are the per-run training curves plus the aggregate Class-2 AUROC bar chart.

![External Class-2 AUROC](figures/odelia_single_site_eval/challenge_aggregate_class2_auroc.svg)

## Reference Swarm Context
The 2026-05-13 challenge swarm/local artifact report has complete final artifacts for six models across six sites. Its best **internal** aggregated validation AUROCs were: 4LME_ABMIL 0.900, 3agaldran 0.830, 1DivideAndConquer 0.811, 5Pimed 0.789, MST 0.775, and 2BCN_AIM 0.760. Those numbers are useful for model-family context, but the external single-site checkpoint endpoint in this report is the ODELIA challenge test inference summarized above.

## Files
- Full detailed report: [docs/ODELIA_SINGLE_SITE_CKPT_CHALLENGE_EVAL_REPORT.md](ODELIA_SINGLE_SITE_CKPT_CHALLENGE_EVAL_REPORT.md)
- External per-site metrics: `workspace/odelia_single_site_eval/tables/challenge_summary_metrics.csv`
- External aggregate metrics: `workspace/odelia_single_site_eval/tables/challenge_aggregate_metrics.csv`
- Class distributions: `workspace/odelia_single_site_eval/tables/internal_class_distribution.csv` and `workspace/odelia_single_site_eval/tables/external_challenge_class_distribution.csv`
- Appended source reports: `docs/CHALLENGE_SWARM_LOCAL_TEST_REPORT_20260513.md` and `docs/OLE_SWARM_EVALUATION_ARTIFACT_REPORT.md`
