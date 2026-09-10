# ODELIA Evaluation — Pitfalls

Companion to `SWARM_FAILURE_MODES.md`. That document covers ways a *run* fails; this one covers
ways a *number* misleads. Every entry here actually happened on this project and cost real time or
produced a wrong conclusion.

## Quick reference

| | Pitfall | Tell |
|---|---|---|
| **E1** | Macro AUROC used to rank centres at small class support | One class has single-digit positives |
| **E2** | Checkpoint evaluated with the wrong architecture | Loads silently; metrics look plausible but are meaningless |
| **E3** | Accuracy read as clinical performance | Accuracy healthy while recall for the class that matters is not |
| **E4** | A corrected re-run left the broken one beside it | Two directories for one evaluation; the wrong one has a friendlier name |

---

## E1 — Macro AUROC does not rank centres when class support is small

**What happened (Sep 2026).** The per-site table in the validation report ranked UMCU worst
(macro AUROC 0.602) and RUMC best (0.857), and a recommendation to "investigate UMCU" was written
on that basis. Both were artefacts.

Macro AUROC averages one-vs-rest AUROC over all three classes with equal weight. On the ODELIA
Challenge **test** split the benign class is tiny at several sites, so one third of each site's
headline number was estimated from a handful of positives:

| Site | n | benign cases | benign AUROC | no-lesion | malignant | macro | macro w/o benign |
|---|---|---|---|---|---|---|---|
| UMCU_1 | 41 | 2 | 0.372 | 0.684 | 0.750 | 0.602 | **0.717** |
| CAM_1 | 64 | 2 | 0.315 | 0.856 | 0.960 | 0.710 | **0.908** |
| MHA_1 | 26 | 1 | 0.560 | 0.931 | 0.955 | 0.815 | 0.943 |
| UKA_1 | 26 | 12 | 0.756 | 0.867 | 0.841 | 0.821 | 0.854 |
| RUMC_1 | 8 | 0 | n/a | 0.857 | 0.857 | 0.857 | 0.857 |

**The ordering by macro tracks benign support almost exactly** (0, 1, 2, 2, 12). RUMC "led" only
because it has no benign cases, so two classes were averaged instead of three. CAM was penalised
just as hard as UMCU and nobody noticed, because CAM's other classes were strong enough to mask it.

**It is a split artefact, not a site property.** Benign is 20 of the 204 examinations in UMCU's
challenge slice (~10%); only 2 landed in the 41-case test partition.

**Do instead**

- Report per-class AUROC **with its support**, never a bare macro, when any class is in single digits.
- To compare centres, restrict to the well-populated classes and say so.
- Treat any AUROC with fewer than ~10 positives as descriptive, not comparative.

**Also.** These figures are the **UMCU partition of the challenge test set**, not UMCU's training
data — 41 of 204 curated examinations, against a training contribution of >10,000. Say which when
reporting, or "UMCU's results are weak" will be read as a statement about their data.

## E2 — A checkpoint loaded with the wrong architecture fails silently

**What happened (Aug 2026).** `best_FL_global_model.pt` from job `87c5bbee` was evaluated with
`--model-name MST` when the job had trained `1DivideAndConquer`. `predict.py` loaded the weights
without complaint and produced macro AUROC 0.43, near-constant predictions per centre, and never
once the benign class — which invited a compelling but false diagnosis of scanner-signature
shortcut learning. Re-run against the correct architecture, every metric roughly doubled.

**Do instead.** Read `train_conf` out of the checkpoint and check it against `--model-name` before
trusting any output. The evaluation script should refuse the mismatch rather than proceed.

## E3 — Accuracy hides the operating point

**What happened.** Pooled accuracy 0.752 alongside malignant-vs-rest AUROC 0.887 looked
consistent, but the model called malignant 21 times against 37 true cases and benign 34 times
against 17 — under-calling cancer, the unsafe direction. The discrimination was fine; the decision
threshold was not. The same pattern reproduced independently on Duke: 44 benign calls on a cohort
with zero benign cases.

**Do instead.** Report malignant recall alongside accuracy. A high AUROC with poor recall means the
threshold is wrong, not the model — and that is fixable without retraining.

---

## E4 — A corrected re-run left the broken one in place, under a better name

**What happened (Sep 2026).** The D3.2 acquisition experiment and a follow-up calibration
analysis were both run against `workspace/eval_87c5bbee/results/best/`. That directory holds
the 8-site checkpoint evaluated with `--model-name MST`. The checkpoint is
**1DivideAndConquer** — its own embedded `train_conf` says `ResidualEncoderClsLightning`.
So these are E2 predictions: the weights load partially and produce plausible-looking numbers.

The corrected re-run exists, in `results/fixed/`. It was written **beside** the broken one
rather than replacing it, and `best` names the *checkpoint kind* (best vs final), not the
better result. Reading `best/` as "the best results" is the obvious mistake and it was made.

| | `best/` (MST, wrong) | `fixed/` (1DC, correct) |
|---|---|---|
| Pooled accuracy | 0.533 | **0.752** |
| Malignant AUROC | 0.476 — chance | **0.887** |
| Lesion-vs-none AUROC | 0.375 — below chance | **0.853** |

**What it cost.** Two published conclusions, both false and both confidently argued:

- *"Uncertainty sampling finds fewer malignant cases than random at every budget."* It
  beats random at 5 of 8 budgets, roughly doubling the yield at small budgets (5 vs 2.4 at
  a 10-label budget; 11 vs 6.8 at 30).
- *"The model's confidence carries no information; accuracy falls as confidence rises."*
  Expected calibration error is **0.053**, the fitted temperature is **0.954**, and the one
  apparent reliability inversion is z = 0.12, **p = 0.90 — noise**.

Both are exactly what a mis-loaded checkpoint produces, which is what made them plausible.

**Why it survived review.** The headline model numbers in the deck and the M45 report were
never affected: they come from `analyze_best.py`, which reads `fixed/`. So the deck said
0.887 while the analyses said chance, and nobody reconciled the two. **A single evaluation
having two live answers is the defect**; the wrong analysis was downstream of it.

**Do instead**

1. When you re-run to correct an evaluation, **delete or quarantine the superseded output**.
   Do not leave it next to the corrected one under a plausible name. If it must be kept,
   name it for what is wrong with it (`best_WRONG_ARCH_MST/`), not for the checkpoint.
2. **Refuse chance-level predictions rather than analysing them.** All three consumers now
   do: `scripts/active_learning/run_experiment.py`,
   `scripts/calibration/analyse_calibration.py` and `scripts/evaluation/deck_metrics.py`
   abort if malignant AUROC < 0.60, naming E2 and pointing at `results/fixed`.
3. **Reconcile a new analysis against the numbers already published** from the same
   evaluation before believing it. A one-line check of pooled accuracy against
   `external_summary_best.json` would have caught this in seconds.
4. **Never hardcode the conclusion.** The acquisition script stored its verdict as a fixed
   string; when the numbers reversed, the string did not, and the summary asserted a null
   result on top of data showing the opposite. The verdict is now computed from the yields.

**Reproduce.** `python3 scripts/evaluation/deck_metrics.py --boot 5000` for the headline
numbers with bootstrap intervals, and point any of the three scripts at
`workspace/eval_87c5bbee/results/best` to see the guard fire.

---

## Where this applies next

**Issue #526 (D2.5, regional fine-tuning)** compares regionally fine-tuned models against the
pan-European baseline per region. That comparison runs straight into E1: the per-region splits are
the same small partitions, with the same thin benign class. Use per-class AUROC with support, or
the comparison will measure split composition rather than fine-tuning.

**Issue #527 (D3.2, active learning)** now has a positive result: entropy-based acquisition
beats random at 5 of 8 budgets and roughly doubles the malignant yield where a labelling
budget actually binds. Two caveats belong in the deliverable. The advantage necessarily
vanishes at large budgets, because the selection covers most of the pool and every method
converges — so report the small-budget end and say why. And this measures **acquisition
quality only**: no model has been retrained on the selected cases, which is the question
D3.2 actually asks.
