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

A model trained by the consortium, as every site receives it: the global weights of every
round and the final global. The October 2026 benchmark (job `7a678d9a`, MST, 20 rounds, seven
sites) is the first candidate; it kept only the round-19 and the final global (§9.1). From
1.9.0 on, a run keeps every global at TUD and the per-round models at the attacking sites when
the job switches it on (§9.2, §9.3). An attacker inside the consortium has the globals, plus
its own training data and its own local updates, and in the rounds it aggregates also the
updates of the other sites (§9.4). That is the threat model: **a participating hospital that turns adversarial**,
which in a swarm is every site, because every site receives the model and sends weights.

Already measured at TUD, outputs only (scripts/adversarial/site_inference_attack.py):
from a case's three class probabilities alone, the source hospital can be guessed 60 % of
the time on malignant cases (chance 33 %), not on no-lesion cases. That is a lower bound;
the attacks below have the weights.

## 3. The attack package (from TUD, after the October run)

- `models/`: the globals of the attacked run. For a run with the archive switched on (§9.3):
  `round_00_global.pt … round_20_global.pt`, where `round_rr` is the global every site received
  in round rr (0-based; `round_00` is the initial model, `round_20` the final model of a 20-round
  run), with `manifest.json` and `SHA256SUMS`. For the October run: `best_FL_global_model.pt`
  (the round-19 global) and `FL_global_model.pt` (the final global), see §9.1.
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
scored with every global model in the package and with the single-site model. The trajectory
attacker (2 below) needs the per-round globals, so it runs only on a run with the archive.

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

**Setup.** The site needs, for at least one round r, the global model it received (G_r) and
the local model it sent (W_r). The update Δ = W_r − G_r (over the keys of G_r) is exactly what
that round's aggregating site received (§9.4). The October run holds one such pair, for
r = 19 (§9.1); a run with capture switched on holds one per round (§9.2), together with the
optimizer state at the start of each round if the job asks for it, which lets the scaffold
replay the local update exactly.

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
| Oct 2026 | consortium benchmark run (job `7a678d9a`, seven sites); per-round models were not kept, the round-19 pair exists (§9.1) | all sites |
| Oct 2026 | keep the October round-19 files (§9.1) | CAM, RUMC |
| from 1.9.0 | a run meant for MS6 switches capture and the archive on in its job (§9.2, §9.3); nothing to do at the sites | TUD |
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

## 9. Per-round models: what exists and how they are kept (#612)

### 9.1 October (job `7a678d9a-5fec-473f-9b23-e43c41322f1f`): one pair, round 19

- **G_19 is `best_FL_global_model.pt`.** Every site has it in
  `<kit>/7a678d9a-5fec-473f-9b23-e43c41322f1f/app_<SITE>/`, TUD in the release folder
  `ODELIA_MODELS/release/2026-10_7site_MST_7a678d9a/` (sha256 `7b6eef8f84bff9f13d8ec93dc60424a877d5c6eb777986bc9da449971ad1d8ba`).
  The CAM_1 and RUMC_1 job logs say "Got best from client USZ_1 with metric 0.7304 at round 19":
  the best global is saved when a round's validation of its received global is the best so far,
  and at that moment the persistor holds that round's received global, the one every site
  validated at the start of round 19 (0-based). Consistent with that, the best and the final
  global differ by a relative 1.3e-4 (largest single difference 3.9e-4), one round of updates.
- **W_19 is `last_global_model.ckpt` in the site's run directory.** Despite its name it is the
  site's LOCAL model after its last local training (a copy of Lightning's `last.ckpt`):
  `/scratch/runs/CAM_1/MST_unilateral_fold0_2026_10_13_083235/` and
  `/scratch/runs/RUMC_1/MST_unilateral_fold0_2026_10_13_102840/` ("Last model saved to" in the
  logs). CAM_1 and RUMC_1 trained one local epoch per round, so this is their model at the end of
  round 19, the one they sent to that round's aggregator (USZ_1).
- The two files have different layouts (a Lightning checkpoint and an NVFlare global,
  `{"model": state_dict}`), so the keys must be aligned before Δ is computed.
- The August run `87c5bbee` (1DivideAndConquer) has the same structure (best global from round
  19, RUMC_1's last local checkpoint from epoch 19), a second pair if needed.
- **CAM and RUMC: keep these files.** Copy `last_global_model.ckpt`, `last.ckpt`,
  `epoch=*.ckpt` and the job's `best_FL_global_model.pt` and `FL_global_model.pt` into
  `/scratch/ms6/7a678d9a/` and check `best_FL_global_model.pt` against the sha256 above. If the
  run directory has been cleaned up, the round-19 pair is gone and attack B needs a run with
  capture. Before the package goes out each site confirms that the checkpoint is from epoch 19
  and that Δ is non-zero and of the size of one round's update.

### 9.2 Keeping G_r and W_r in later runs: switched on per job, nothing to do at the sites

From 1.9.0 on, the job code (which travels with every submitted job) can keep per-round models
at named sites. TUD switches it on when building the job, for example
`scripts/admin/build_consortium_job.py ... --ms6-capture-sites CAM_1,RUMC_1 [--ms6-capture-rounds 0,9,19] [--ms6-capture-optimizer]`,
which writes these variables into the job's launcher line. Sites restart nothing and change
nothing.

| Variable | Default | Meaning |
|---|---|---|
| `ODELIA_ROUND_HOOKS` | empty | must list `ms6_capture` (the builder does) |
| `ODELIA_MS6_CAPTURE_SITES` | empty | sites that keep files, or `all`; empty keeps nothing |
| `ODELIA_MS6_CAPTURE_ROUNDS` | `all` | or a list of 0-based rounds |
| `ODELIA_MS6_CAPTURE_DIR` | `/scratch/ms6` | root directory; each job writes to `<root>/<job_id>/`, which a launcher line cannot name in advance |
| `ODELIA_MS6_CAPTURE_OPTIMIZER` | `0` | `1` also keeps the optimizer and LR-scheduler state at the start of each round's local training |

Files at a selected site, in `/scratch/ms6/<job_id>/<SITE>/` (rr = 0-based round):

| File | Content |
|---|---|
| `round_rr_global.pt` | G_r, the global as received, before it is loaded |
| `round_rr_local.pt` | W_r, the local model exactly as sent, after DP clipping and noise when a job adds them |
| `round_rr_optimizer.pt` | optimizer state at the start of the round (only with `ODELIA_MS6_CAPTURE_OPTIMIZER=1`) |
| `manifest.json` | job id, site, model name; per file the round, size, sha256, a tensor digest and the aggregation weight that was sent |

The model files have the layout of a global (`{"model": state_dict}`) and load wherever a
global loads, for example with `--external_validation --checkpoint`. W_r is taken after the
update was sent, so the copy never delays a round. The directory lies outside the run
directory, so live_sync never uploads it; a capture directory inside `$SCRATCH_DIR/runs` is
refused before training starts. Disk per selected site: about 190 MB per round for MST (G_r
and W_r, about 3.9 GB for 20 rounds), about 2.1 GB per round for MST_SAMMed2D (about 44 GB for
20 rounds, so keep selected rounds for that model); the optimizer state adds about twice the
model size per round. Free space is checked before each write; a file that does not fit is
skipped with a WARNING and listed in the manifest, and training goes on. Nothing is kept in
the evaluation-only final round, whose global is the final global.

### 9.3 The TUD archive of every global

With `--archive-globals` the job also sends every global to the coordinator. The aggregating
site of each round streams the global it has just built (client component
`ms6_global_streamer`, in the background, in 1 MB chunks), the starting site streams the
initial global, and the coordinator (server component `ms6_global_archive`) writes
`<archive_root>/<job_id>/round_rr_global.pt` for rr = 0 (initial) up to the number of rounds
(final), plus `manifest.json` (per round the sending site, size, sha256, tensor digest and model
name; missing rounds are flagged when the job ends) and `SHA256SUMS`. `archive_root` is
`/archive`, which the 1.9.0 coordinator mounts from the host; if it cannot be written, the
archive goes to the job's run directory on the server and comes back with `download_job`.

An archived `round_rr_global.pt` and a site's `round_rr_global.pt` hold the same tensors; equal
`tensor_sha256` values in the two manifests show it without loading either file. Cost: one
extra upload of the global per round from that round's aggregator (94 MB for MST, 1.07 GB for
MST_SAMMed2D) and about 2 GB (MST) or 22.5 GB (MST_SAMMed2D) at TUD per 20-round run. A failed
stream is retried and never fails training; the copy then stays in the aggregating site's job
folder (`ms6_outbox`). With the evaluation-only final round switched on, the last two archived
globals carry the same weights, because that round returns zero updates.

### 9.4 Who sees what

- **The aggregating site of a round** (chosen at random every round) receives from each site
  Δ = W_r − G_r over the keys of G_r: the jobs exchange weight differences, and no result
  filter runs in the swarm learn path (the `PercentilePrivacy` entry in the job configurations
  never ran). With it come the aggregation weight `NUM_STEPS_CURRENT_ROUND` (Lightning's
  estimate of the training steps, which at batch size 1 follows from the number of training
  cases) and the site's validation metrics of G_r. Attack B on a site's own update therefore
  measures what any aggregating site could do with another site's update.
- **TUD is a party that holds more than the swarm peers** and is treated as honest but curious
  in the MS6 and T3.3 reports. Besides every global (§9.3), it receives through live_sync, which
  runs on each site host, the site's run directory: logs and TensorBoard files and, in swarm
  mode, also Lightning's `last.ckpt` and `epoch=*.ckpt` (site-local models) and the per-case
  prediction CSVs, including those of the training split. This happened in earlier runs
  (RSH_1 in July 2026, RUMC_1 in April 2026), not for CAM, RUMC and MHA in October; the
  behaviour is intermittent. The MS6 capture files (§9.2) lie outside the uploaded tree and stay
  at the site.
