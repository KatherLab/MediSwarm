# MS6 white-hat attack: protocol for CAM and RUMC

**Milestone MS6 (T3.3, due M54, June 2027).** The proposal: "We will also set up a white hat
hacker attack to expose and subsequently fix any vulnerabilities of our code." Part A names
CAM and RUMC as the attacking parties. Verification: a preprint plus a journal submission.

This protocol says exactly what TUD provides, what each attacking site runs, what comes
back, and when. It is written so that a site can run every attack on its own machine with
its own data as ground truth. Nothing identifying leaves a site; the results are numbers.

## 1. Roles

| Who | Does |
|---|---|
| TUD (Jeff) | Provides the attack package (§3), a scaffold script per attack, the report template, a kickoff call, the fix after the findings, the re-test and the write-up. |
| CAM, RUMC | Each names one person, runs attacks A and B (C optional) on its own node against its own data, fills the report template, joins two calls. |
| All other sites | Nothing. The attacked model is the consortium model; their data is never touched. |

## 2. What is attacked

The model trained by the October 2026 consortium benchmark (20 rounds, eight sites), as
every site receives it: the final global weights and the global weights after each round.
An attacker inside the consortium has exactly this, plus its own training data and its own
local updates. That is the threat model: **a participating hospital that turns adversarial**,
which in a swarm is every site, because every site receives the model and sends weights.

Already measured at TUD, outputs only (scripts/adversarial/site_inference_attack.py):
from a case's three class probabilities alone, the source hospital can be guessed 60 % of
the time on malignant cases (chance 33 %), not on no-lesion cases. That is a lower bound;
the attacks below have the weights.

## 3. The attack package (from TUD, after the October run)

- `models/global_round_01.pt … global_round_20.pt` and `global_final.pt`: the global model
  after each round, as broadcast to the sites, plus the final model.
- `models/local_single_site_<SITE>.pt`: a model trained on the site's data alone with the
  same recipe, for the comparison in attack A (TUD trains it from the site's own run logs
  if the site prefers not to; otherwise the site trains it, one command).
- Model code and preprocessing: the public MediSwarm repository at the tagged release.
- `attacks/attack_A_membership.py`, `attacks/attack_B_reconstruction.py`,
  `attacks/attack_C_inversion.py`: scaffolds that load the models, run the attack against
  the site's data directory and write the report numbers. Sites may change anything.
- `report_template.md`: the numbers to fill in (§5). No free-text identifiers.

## 4. The attacks

### A. Membership inference (required)

**Question.** Does the shared model reveal whether a given case was in training?

**Setup at the site.** Members: the site's own training cases (fold 0 `train`).
Non-members: the site's held-out `test` cases, which never entered any training. Both are
scored with every global model (rounds 1 to 20 and final) and with the single-site model.

**Attack.** For each case and model, compute the cross-entropy loss of the true label and
the confidence in the predicted class. Three attackers, from weakest to strongest:
1. Loss threshold on the final model (Yeom et al.): predict "member" if the loss is below a
   threshold. Report the attack AUROC over all thresholds.
2. Trajectory attacker: a logistic regression on the 20 per-round losses, fitted on half of
   the cases and evaluated on the other half (five random halves, report the mean).
3. The same two attackers against the single-site model.

**Report.** Attack AUROC with a 1,000-sample bootstrap 95 % interval, balanced accuracy at
the best threshold, member and non-member counts, per class. Chance is 0.5. An interval
that excludes 0.5 is a finding.

**Cost.** Inference only: 21 models over a few hundred cases, hours on one GPU.

### B. Reconstruction from the site's own update (required)

**Question.** Can the update a site sends after a round be turned back into its images?

**Setup.** From the October run the site holds, for at least one round r, the global model
it received (G_r) and the local model it produced (W_r); the scaffold saves both. The
update is Δ = W_r − G_r, which is what the aggregating site sees.

**Attack.** Gradient-inversion in the style of "deep leakage from gradients": optimise a
batch of synthetic volumes so that training G_r on them for one local epoch produces an
update as close as possible to Δ (cosine distance), with total-variation regularisation.
Start at reduced resolution (the scaffold uses 64³) and one batch; scale up only if the
first result is not noise.

**Measure.** For every reconstructed volume, the best normalised cross-correlation and
SSIM against any real training volume of the site, compared with the same statistic for
(a) a mean-image baseline and (b) 100 Gaussian-noise volumes. A reconstruction counts as a
leak when its best match exceeds the 99th percentile of the noise baseline. Report the
distributions and the count of leaks, never the images.

**Cost.** The expensive one: a day on one GPU per configuration tried. If it is not
feasible at the site, say so in the report; that is a valid result about the setting.

### C. Model inversion of the global model (optional)

**Question.** Do class-representative inputs synthesised from the final model resemble real
patients?

**Attack.** Gradient ascent on a random input to maximise the malignant class score of
`global_final.pt`, 50 restarts, with the same similarity measure and baselines as B against
the site's malignant training volumes.

### D. Site identification with the weights (TUD)

TUD repeats the outputs-only probe with white-box access (per-layer activations) on the
challenge set. Not a site task; listed so the report is complete.

## 5. What comes back

One completed `report_template.md` per site, numbers only:

| Attack | Field |
|---|---|
| A | attack AUROC and 95 % CI, balanced accuracy, n members, n non-members, per class; for the final model, the trajectory attacker and the single-site model |
| B | round used, resolution, iterations, best-match NCC and SSIM distribution (min, median, max, 99th percentile), noise-baseline 99th percentile, number of leaks |
| C | same similarity summary as B, or "not run" |
| any | what the site changed in the scaffold, and how long each attack took |

Plus the site's attack scripts, into `scripts/adversarial/ms6/` of the public repository, so
the exercise is reproducible.

## 6. After the findings

TUD proposes the fix for every confirmed leak, implements it in the swarm software (the
candidates are already in the code base: calibrated noise on the update, T3.3; output
rounding and per-case return policy; norm-bounded aggregation, D3.4), and asks the two
sites to repeat the successful attack against the fixed model. The milestone report and the
preprint describe both rounds.

## 7. Timeline

| When | What | Who |
|---|---|---|
| by 10 Oct 2026 | confirm participation and name the contact person | CAM, RUMC |
| Oct 2026 | consortium benchmark run; the scaffold saves G_r and W_r at the attacking sites | all sites |
| by 30 Nov 2026 | attack package sent | TUD |
| first week of Dec 2026 | kickoff call, one hour | all three |
| Jan to Feb 2027 | attacks A and B (C optional) | CAM, RUMC |
| by 28 Feb 2027 | reports back | CAM, RUMC |
| Mar 2027 | fix, re-test call, repeat of the successful attacks | TUD, then CAM, RUMC |
| Apr to May 2027 | preprint draft circulated | TUD with both sites as co-authors |
| Jun 2027 (M54) | milestone verified | consortium |

## 8. Rules

- Data never leaves the site; the report carries counts, metrics and distributions only.
- The scaffolds are read-only on the data directory and write under the scratch directory.
- Anything that worked is a result, anything that did not is also a result; both go in.
- Questions and interim numbers go to Jeff by email; the two calls are the only meetings.
