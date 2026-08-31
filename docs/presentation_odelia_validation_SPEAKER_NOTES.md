# ODELIA Swarm Validation — Speaker Notes

Companion to `presentation_odelia_validation.html` — job `87c5bbee`, 8 sites, 20 rounds, evaluated 27 Aug 2026 on the ODELIA challenge set (both checkpoints) and the Duke cohort.

**Running the deck:** open the HTML in any browser. `←`/`→` or click to navigate, `S` toggles these notes on-screen, `F` fullscreen, `Ctrl+P` exports to PDF.

**Length:** 14 slides, ~1252 words, roughly 9–11 minutes. Leave 10 minutes for questions.

**Meeting ask:** the final slide opens with the preprocessing rollout — that is the decision to land in this meeting, not a status item.

---

## Slide 1 — First successful 8-site, 20-round swarm training on real clinical data

*Title*

Today I am reporting the first completed ODELIA swarm run with eight clinical sites over all twenty planned rounds. The sites trained on real patient data without exchanging that data, and every round retained all eight contributions. I will cover three things: why this attempt finished, how the resulting model performs on held-out data, and what we still need to do before the result is publication-ready.

## Slide 2 — Two findings

*The result in one slide*

There are two headline findings. First, the run finished exactly as intended: eight of eight sites contributed to all twenty rounds, nobody was dropped, and the archived server error log is empty. Previous eight-site attempts had not progressed beyond roughly round eleven. Second, the best aggregated checkpoint generalizes well on the held-out five-centre test split: malignant-versus-rest AUROC is 0.887 and macro AUROC is 0.820. The final-round checkpoint was also evaluated and is close, which I will show in the comparison.

## Slide 3 — Why this run survived when earlier ones did not

*Why now*

This table records the debugging sequence. Two runs at the top used the earlier configuration: one was aborted after about ten hours, and one failed at round three after more than a day. Then two one-round canaries completed. Finally the full job ran for nearly two days and completed. Two technical changes and one procedural change separate the failures from the successful run. I will take the technical changes first, then explain why the shorter feedback loop mattered just as much.

## Slide 4 — What “strict8” fixes

*Fix 1 of 2*

The controller previously stopped waiting when its minimum configuration quorum replied. A site that needed longer to start its job cell could receive peer key exchange before it was ready, fail with “cannot forward req: no path,” and be excluded while the run continued. That allowed a green result to represent only a subset of the consortium. Strict8 requires the exact named eight sites at deployment, configuration and contribution time. In this run, each aggregation contains eight accepted updates; no node is silently pruned.

## Slide 5 — The change that was not technical

*Fix 2 of 2*

The procedural change was to stop testing each hypothesis with a multi-day run. Every fix was first exercised with a one-round canary on the same eight sites and the same large model. The canaries took under three hours each instead of waiting roughly thirty hours for a failure signal. The standing practice should be simple: do not submit a multi-day consortium run until an otherwise identical one-round canary has passed its strict acceptance criteria.

## Slide 6 — The model on data it has never seen

*External validation*

These numbers are for the best aggregated checkpoint on the held-out ODELIA challenge test split: 165 exams across five centres. Malignant versus rest is 0.887, lesion versus none 0.853, macro AUROC 0.820 and accuracy 0.752. Every ground-truth label was checked against the corresponding site annotation file, and all 165 matched. The clinically decisive endpoint—separating malignant from everything else—is also the strongest one.

## Slide 7 — It holds at every centre but one

*External validation*

Broken down by centre, ranking performance is strong at most sites. CAM and MHA both exceed 0.95 on the malignant endpoint. UMCU is the outlier at 0.602 macro AUROC and deserves investigation. RUMC has only eight exams, so that row is descriptive rather than conclusive. Accuracy and AUROC disagree at MHA and UKA. That points to the operating threshold rather than a failure to rank cases, which leads into the next slide.

## Slide 8 — The model under-calls cancer

*Read this carefully*

The predicted and true class mixes show the operating-point problem. The best checkpoint predicts malignant twenty-one times although thirty-seven cases are malignant, and predicts benign thirty-four times although only seventeen are benign. This is the unsafe direction: cancers are being absorbed into the benign category. Yet malignant AUROC is 0.887, so the model ranks malignancy well. The next step is calibration on a separate split, with malignant recall reported alongside accuracy.

## Slide 9 — The same result on a cohort from another continent

*Independent cohort*

The strongest single piece of evidence in this deck. We took the same model and ran it against Duke - a public American breast-MRI cohort, different scanners, different health system, and no relationship to any ODELIA site. The malignant-versus-rest AUROC came back at 0.887. That is not close to our challenge-set figure, it is identical to it. Ranking performance transfers across continents unchanged, which is a much stronger claim than anything the challenge set alone could support. Two caveats. Duke has no benign cases at all, so we get one endpoint rather than a macro average - though it is the endpoint that matters clinically. And look at the red box: the model called benign forty-four times on a cohort that contains zero benign cases, with malignant recall of only 0.662 against that 0.887 AUROC. That is the same threshold problem from three slides ago, reproduced independently. The discrimination travels; the operating point is wrong everywhere. That makes threshold calibration the clearest next win we have.

## Slide 10 — Checkpoint and historical comparison

*Against the record*

This table now includes both retained checkpoints from the completed run. The best checkpoint reaches 0.820 macro and 0.887 malignant AUROC. The final-round checkpoint reaches 0.818 and 0.873, respectively, while improving accuracy from 0.752 to 0.782. So the two checkpoints tell the same broad story, but the model-selection rule still matters. The historical numbers are encouraging rather than a controlled head-to-head: May used a different cohort. The next clean comparison is to re-score the historical swarm and single-site checkpoints on this exact split with the same preprocessing.

## Slide 11 — Does training together actually beat training alone?

*Swarm vs local*

This is the broader May comparison that was requested. Across six model families, the swarm global model beats the best retained single-site model on external macro AUROC in four of six cases, and on the malignant endpoint in five of six. Pimed is an important counterexample: swarm training made it substantially worse, so the benefit is not automatic. The rightmost column shows the recurring generalization problem for local models: an internal AUROC around 0.78 to 0.91 falls to roughly 0.57 to 0.73 externally. The current eight-site row is visually useful but not strictly comparable because it uses 165 exams from five centres, whereas the May rows use 300 from six.

## Slide 12 — How far the consortium had got before

*The previous ceiling*

The previous consortium ceiling needs careful wording. The remembered run involved seven sites and stopped at about round eleven, so it never produced a completed twenty-round model for external scoring. We could not recover its job record or global checkpoint: the server was re-provisioned in July, and the live archive does not contain a matching seven-site job. Therefore I am not assigning it a metric or presenting the round number as independently verified. The action is to recover the original run identifier or meeting record before this claim is used in a paper.

## Slide 13 — Limits of this evaluation

*Honesty*

The limitations are concrete. The external set has only 165 exams, with as few as eight at one centre. Both retained checkpoints are now evaluated: their AUROCs are close, but the best checkpoint has slightly better ranking and the final checkpoint better accuracy, so checkpoint selection must be predefined. Historical local baselines were not recomputed under this exact cohort and pipeline. We are also still waiting for confirmation of RSH data labelling, outstanding since July. Finally, peer-to-peer validation metrics do not reach the coordinator; cross_val_results is empty, so central metric collection remains a prerequisite for a reproducible publication workflow.

## Slide 14 — What comes next

*Future work*

The first future-work decision is to roll out the new preprocessing consistently to every partner. I would put the version, validation gate and rollout schedule first on the next technical-meeting agenda. Then merge the reliability fixes, centralize per-site metrics, calibrate the malignant decision threshold, expand evaluation beyond this small split, and investigate UMCU and the still-unconfirmed RSH labels. The 262-case Duke dataset has now been located on dl0; both checkpoints will be evaluated as soon as the active CI validation releases that GPU. Finally, recompute the local baselines with the same cohort, preprocessing and evaluation code so the swarm-versus-local comparison is genuinely controlled. Happy to take questions.
