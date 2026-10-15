# Differential Privacy in MediSwarm (T3.3)

## Summary

Since 1.9.0 an ODELIA swarm job can add calibrated noise to every model update before it leaves a
site. The `dp_noise` round hook clips each site's per-round update to an L2 norm C and adds Gaussian
noise of standard deviation sigma * C inside the training subprocess. A Renyi-DP accountant turns
sigma, delta and the number of released rounds into an epsilon, which is logged and streamed every
round. With a target, the hook refuses a release that would exceed it. A measure-only mode records
update norms to calibrate C.

| | State in 1.9.0 |
|---|---|
| Client-level DP for the model updates (add/remove-site) | implemented: `dp_noise` round hook, in every job on the shared code (`ODELIA_ternary_classification`, `challenge_*`) |
| Per-round epsilon | logged and streamed as `dp/epsilon` |
| Budget halt | `ODELIA_DP_TARGET_EPSILON`: the over-budget release is refused before anything is sent |
| Calibration of C | `ODELIA_DP_MEASURE_ONLY=1` |
| Record-level (per-patient) DP, DP-SGD | not available (Opacus is not in the image) |
| STAMP jobs (and the `minimal_training_pytorch_cnn` test job) | no DP hook: their own training code, no round hooks |
| `PercentilePrivacy` | removed from every job config; it never ran in swarm mode (see below) |
| Utility cost | not measured yet: needs the calibration run, a dl3 rehearsal and the noise-cost run (#593) |

Code: `application/jobs/_shared/custom/dp_noise.py` (mechanism, modes, metrics) and
`application/jobs/_shared/custom/privacy_accounting.py` (accountant, budget, `max_rounds_for_budget`).
Both travel with the job (BYOC), so a DP run needs no image or kit change.

## The guarantee, stated precisely

**Client-level (epsilon, delta)-differential privacy by output perturbation, under add/remove-site
adjacency.** The protected unit is one site's contribution to one round. Neighbouring executions
differ by a site's clipped update versus no contribution (a zero update), so the L2 sensitivity is C
and the noise multiplier is sigma. The guarantee covers the model updates a site sends in training
rounds and everything computed from them: every aggregate, the global model of every round and the
final model are post-processing of the noised updates.

- epsilon is per site and counts that site's releases. Every site releases once per training round,
  so in a normal run all sites carry the same epsilon.
- **Not record-level.** It says nothing about whether one patient was in a site's training set; that
  needs DP-SGD during local training (Phase 3, not started). Do not call a DP run "differentially
  private" without the qualifier "client-level".
- **Adjacency matters for the number.** Under replace-site adjacency (a site's data swapped for any
  other data) the sensitivity is 2C, which is the epsilon of sigma / 2: at sigma 2.4 and delta 1e-5,
  23.61 after 20 rounds instead of 9.84. State the adjacency next to every epsilon.
- The accountant assumes sampling rate 1 (a round uses the whole local dataset). Subsampling would
  only lower epsilon, so the stated number is conservative.

## Threat model: what the guarantee does not cover

The guarantee holds against the other sites (in swarm mode the aggregating site receives each site's
update) and against anyone who holds a global model. It does not cover these channels:

| Channel | What leaves the site | Status in a DP job |
|---|---|---|
| live_sync (kit side, not part of the job) | local checkpoints (`last.ckpt`, `epoch=*.ckpt`, the un-noised local model), TensorBoard event files and logs, uploaded to TUD | still uploaded: by the 1.8.1 kits and by the kit template in this repository (`kit_live_sync/live_sync.sh`) at the time of writing |
| Metric stream to the coordinator | validation metrics of the global and the local model (`val/*`, ...), `dp/update_norm`, `dp/clipped` | computed on local data without noise |
| `NUM_STEPS_CURRENT_ROUND` in each update's metadata | a dataset-size proxy, used as the aggregation weight | constant 1 with `ODELIA_DP_EQUAL_WEIGHTS=1` |
| Per-case predictions | validation rows of the local model | DP jobs set `ODELIA_RETURN_PER_CASE=0`; the hook refuses to start otherwise |
| The round-0 global | a warm start from a model trained on partner data is not private | start fresh (`warm_start_mode = "fresh"`) or from public weights |

**TUD (coordinator and live monitor) is therefore inside the trust boundary of the current setup.**
Through live_sync it receives each site's un-noised local checkpoints and logs. The DP claim of a run
made with these kits is: client-level DP against the other sites and against users of the global
models, not against TUD. A claim that includes TUD needs live_sync stopped at every site for the DP
run (or a kit that skips local checkpoints), and the metric stream limited to private values.

## How the mechanism works

NVFlare's `FLCallback` sends `pl_module.state_dict()` in `on_train_end`. The hook's Lightning callback
sits immediately before `FLCallback` (round hooks, `round_hooks.insert_before_fl_callback`). In
`on_train_end` of every training round it rewrites the model in place, and `FLCallback` sends the
result:

```
U  = W - G                       W: local model after training, G: global received this round,
                                 over the trainable floating-point parameters that G contains
U' = U * min(1, C / ||U||_2)     one L2 norm over all those tensors together
W' = G + U' + N(0, (sigma*C)^2)  independent noise per coordinate
```

With `params_transfer_type = "DIFF"` (every ODELIA job template) the client API transmits
`W' - G = U' + noise`.

- **Other state-dict entries** that G also has (buffers of any dtype, frozen parameters) are copied
  back from G, bit-exact for integer buffers, so no local change to them leaves the site. Integer
  buffers are never noised or cast. For MST this changes nothing: all 23,533,443 parameters are
  trainable, and its only float buffer, `slice_fusion.rotary_pos_emb.inv_freq`, is a constant. For a
  model with BatchNorm the running statistics never leave the site, so the global keeps the round-0
  statistics for the whole run (the hook warns at start).
- **Entries G does not have** are not sent under DIFF transfer. One is `_class_weight`, the class
  weights computed from the site's label counts: the persistor's model has none. Under a FULL
  transfer they would be sent without noise, so the hook refuses a release in that case.
- **Evaluation-only final round** (`ODELIA_FINAL_EVAL_ROUND=1`): the received global goes back
  unchanged, with no noise and no release counted.
- **Precision:** training uses `precision="16-mixed"`, which is autocast only; the parameters stay
  float32 (Lightning's `MixedPrecision` does not convert the module). The noise is computed in
  float32 on CPU. On MST one release takes under a second.
- **Noise source:** one torch CPU generator per training subprocess, seeded from `os.urandom`. It is
  never seeded from `ODELIA_SEED` or the global RNG. It is a Mersenne Twister, not a cryptographic
  generator, and floating-point side channels of the Gaussian sampler are out of scope.
- **Counting:** the release of round r is counted as release `max(previous count + 1, r + 1)`. After
  every release the count is saved in `dp_noise_state.json` in the working directory of the training
  subprocess, which is the job's app directory on the client. A training subprocess restarted within
  the job continues that count, so a round repeated after its update was sent counts as a new
  release. A count file that cannot be read, or that was written with another sigma, C or delta, stops
  the job; failing to save it only logs a warning (the round-index bound still holds).
- **Budget:** with `ODELIA_DP_TARGET_EPSILON` the hook raises `PrivacyBudgetExceeded`, before anything
  is sent: when the job starts if its training rounds cannot fit the target (instead of failing after
  hours of training), before each round's training, and again immediately before the noised update
  would be written. A refused release ends the job with an error. CCWF cannot end a run early from
  inside a client, so the job builder must keep the number of rounds within
  `max_rounds_for_budget(sigma, delta, target)`.
- **Order with other hooks:** hook callbacks run in `ODELIA_ROUND_HOOKS` order. With
  `ms6_capture,dp_noise` the MS6 capture keeps the un-noised local model W; with `dp_noise,ms6_capture`
  it keeps the sent W'.

## Configuration

Set in the launcher line of `config_fed_client.conf` (the variables reach the training subprocess
only):

```
script = "ODELIA_ROUND_HOOKS=dp_noise ODELIA_DP_SIGMA=2.4 ODELIA_DP_CLIP_NORM=<C> ODELIA_DP_DELTA=1e-5 ODELIA_DP_TARGET_EPSILON=10 ODELIA_DP_EQUAL_WEIGHTS=1 ODELIA_RETURN_PER_CASE=0 KEY_METRIC=val/AUC_ROC python3 custom/{app_script}  {app_config} "
```

| Variable | Default | Meaning |
|---|---|---|
| `ODELIA_ROUND_HOOKS` | "" | must list `dp_noise` |
| `ODELIA_DP_SIGMA` | unset | noise multiplier sigma (> 0); switches the noise on |
| `ODELIA_DP_CLIP_NORM` | unset | L2 clip norm C of the per-round update; required with `ODELIA_DP_SIGMA` |
| `ODELIA_DP_DELTA` | 1e-5 | delta of the accountant |
| `ODELIA_DP_TARGET_EPSILON` | unset | refuse a release that would take epsilon past this value |
| `ODELIA_DP_MEASURE_ONLY` | 0 | 1 = no clipping, no noise, stream the update norms (calibration, no guarantee) |
| `ODELIA_DP_EQUAL_WEIGHTS` | 0 | 1 = report `NUM_STEPS_CURRENT_ROUND=1`, so aggregation is unweighted |
| `ODELIA_RETURN_PER_CASE` | site env | must be 0 in a DP job |

The modes are exclusive and the parsing is strict, because a mistake must never mean updates
leaving without the promised noise:

- noise: `ODELIA_DP_SIGMA` + `ODELIA_DP_CLIP_NORM` (optional delta, target, equal weights);
- measure: `ODELIA_DP_MEASURE_ONLY=1` (optional `ODELIA_DP_CLIP_NORM` to see how often a candidate C
  would clip); sigma, delta and target are refused;
- equal weights only: `ODELIA_DP_EQUAL_WEIGHTS=1` alone, no noise: the aggregation of a DP run without
  noise, for a matched non-DP baseline.

Anything else stops the job before training: an unknown `ODELIA_DP_*` name, an empty value, sigma
without C, noise and measure together, `dp_noise` listed without any mode, or the never-wired name
`PRIVACY_EPSILON_TARGET`. Any `ODELIA_DP_*` variable without `dp_noise` in `ODELIA_ROUND_HOOKS` also
stops the job (round hooks safety net). Hooks run only in `TRAINING_MODE=swarm`.

### Building a DP job

`scripts/admin/build_consortium_job.py` writes a DP job from one command (`--dp-sigma`,
`--dp-clip-norm`, `--dp-delta`, `--dp-target-epsilon`, `--dp-measure-only`, `--dp-equal-weights`):

- the launcher variables above, with `dp_noise` as the last round hook (the noise must be the final
  transformation of the outgoing update) and `ODELIA_RETURN_PER_CASE=0`;
- it refuses more training rounds than `max_rounds_for_budget(sigma, delta, target)` allows, using that
  function from the job's own `privacy_accounting.py`; an evaluation-only final round comes on top and
  spends no budget;
- it refuses `--warm-start continue`, because the accountant counts the releases of one job only and a
  chained run would under-state epsilon. Start fresh, or from a named source that was not trained on
  partner data (otherwise the round-0 global is outside the guarantee);
- a per-job global mirror (`latest_global_path`, default `/scratch/mediswarm_latest_global_<name>.pt`),
  so the DP global does not become the warm start of later non-DP jobs.

The consortium utility run of November 2026 (sigma 2.4, clip norm 1.0, epsilon at most 10 at delta
1e-5 over 20 rounds) is the preset `dp_s24` of `scripts/admin/odelia_run_plan.py`
([ODELIA_RUN_PLAN_2026-Q4.md](ODELIA_RUN_PLAN_2026-Q4.md)).

`params_transfer_type = "DIFF"` comes from the templates. The builder does not change the server's
per-site metric collector, which keeps only the validation and test metric prefixes, so the `dp/*`
values reach the coordinator in the metric stream but are not written to `cross_val_results.json`.
They are in every site's log and TensorBoard files.

Build DP jobs from a repository checkout, or with the 1.9.0 image's job copy: a job built from an
older image's copy contains no `dp_noise.py`, and `ODELIA_ROUND_HOOKS=dp_noise` then stops it at start.

## Calibrating C

C sets both the clipping bound and the noise (std sigma * C), so it has to match the size of a typical
update. Too small biases every update towards zero; too large wastes budget on noise.

1. Run the planned model, epochs per round and number of rounds with `ODELIA_DP_MEASURE_ONLY=1` on
   **public data** (for example the dl3 four-client harness). A C chosen from partner data would itself
   depend on that data. The measure run sends the updates without noise and has no guarantee.
2. Each round logs `DP measure-only, round r: update L2 norm ...`, streams `dp/update_norm`, and the
   job end logs the median per site.
3. Set C near the median of the per-round update norms, so about half of the updates get clipped.
4. Optionally repeat with `ODELIA_DP_CLIP_NORM=<candidate>`: `dp/clipped` then shows whether each
   update would have been clipped.
5. Record C, the data and the model here and in the job.

| Date | Model | Data | Rounds | Median update norm | C chosen |
|---|---|---|---|---|---|
| (not yet measured) | MST | | | | |

## Accounting

`compute_epsilon(sigma, rounds, delta)` at sigma 2.4, delta 1e-5 (add/remove-site):

| Rounds | 1 | 2 | 3 | 4 | 5 | 10 | 14 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| epsilon | 1.77 | 2.60 | 3.26 | 3.84 | 4.36 | 6.51 | 7.94 | 8.28 | 9.84 |

`max_rounds_for_budget(2.4, 1e-5, target)`: 20 rounds for target 10, 14 for 8, 2 for 3. Over 20
rounds at delta 1e-5, epsilon <= 10 / 3 / 1 needs sigma 2.4 / 6.7 / 18.1.

```python
from privacy_accounting import compute_epsilon, max_rounds_for_budget
compute_epsilon(2.4, 20, 1e-5)        # 9.84
max_rounds_for_budget(2.4, 1e-5, 8.0) # 14
```

Until 1.9.0, `PrivacyAccountant.budget_exhausted()` became true only after the over-budget release had
been counted: at target 8 it tripped after release 15 (epsilon 8.28). It is now true as soon as the
next release no longer fits, and `check_release()` refuses that release. The accountant no longer reads
`PRIVACY_EPSILON_TARGET` from the environment; the target is passed explicitly.

## Metrics

| Key | Mode | Meaning |
|---|---|---|
| `dp/epsilon` | noise | epsilon after this round's release |
| `dp/update_norm` | noise, measure | L2 norm of U before clipping |
| `dp/clipped` | noise, measure (with C) | 1 if the update was (would be) scaled down to C |
| `dp/noise_std` | noise | sigma * C |
| `dp/released_norm` | noise | L2 norm of U' + noise, what leaves the site |

The step of each record is the round. They go to every Lightning logger of the trainer: the local
TensorBoard log and, in a swarm run, the NVFlare metric stream to the coordinator. Each release also
writes one log line (`DP release, round r: ...`), and the job end writes a summary. `dp/update_norm`
and `dp/clipped` are computed before the noise and are outside the guarantee.

## Expected utility

This is an inference, not a measurement. MST has d = 23,533,443 trainable parameters. Each site's noise
vector has an L2 norm of about sigma * C * sqrt(d), about 11,600 * C at sigma 2.4: about 11,600 times
the norm of the clipped update. Averaging over the sites lowers the per-coordinate noise std to about
sigma * C / sqrt(n_eff); n_eff is about 3.4 with size weighting, where UKA carries about half the
weight, and 7 with equal weights. A collapse of utility after the first rounds is the expected outcome
at sigma 2.4. Rehearse on dl3 (noise versus no noise, a few rounds) before using consortium time.
Options if utility collapses:

- fewer rounds (3 rounds cost epsilon 3.26, 5 rounds 4.36);
- DP on a reduced trainable subset, for example a frozen backbone (frozen parameters are not
  noised, so the noise scales with the trainable part only), with a matched non-DP baseline;
- record-level DP-SGD (Opacus is not in the image).

## PercentilePrivacy: what it was and why it is gone

Until 1.9.0 every job's `config_fed_client.conf` listed NVFlare's `PercentilePrivacy`
(`percentile=10, gamma=0.01`) as a `task_result_filter` on the task `train`. It zeroes the smallest 10 %
of each weight diff and clips every coordinate to +/-0.01, with no noise and no epsilon (Shokri &
Shmatikov, CCS 2015). It carried no formal guarantee anyway.

**It never ran in a swarm job.** CCWF dispatches only `swarm_*` task names, and its client controller
runs the learn task through a direct `learn_executor.execute(...)` call (`execute_learn_task` in
`nvflare/app_common/ccwf/client_ctl.py`). NVFlare applies task filters only to tasks it dispatches by
name, looked up by the exact key `<task>/<direction>` (`ClientRunner._do_process_task`,
`TaskController.broadcast_and_wait`), so a filter on `train` is never looked up. The filter logs a line
at INFO on every call. 768 mirrored client logs with 2,655 training rounds contain none, including all
20 rounds of the October 2026 run (job 7a678d9a) at MHA_1.

1.9.0 removes it from every job config. This changes no result, because it never ran. Every swarm
result to date, the October run included, was produced without it, and statements that the updates
were privacy-filtered were incorrect. A unit test
(`tests/unit_tests/test_job_configs_no_inert_result_filters.py`) fails if a swarm job wires a result
filter on its learn task again. For the same reason, any NVFlare DXO filter (`SVTPrivacy`, or the
`CalibratedNoiseFilter` sketched in earlier versions of this document) would not run in a swarm job
either: the noise has to be added inside the training subprocess, which is what `dp_noise` does. The
kit templates' `privacy.json.sample` (NVFlare's stock sample, inactive unless renamed) is unchanged.

## Roadmap

- Phase 1, assessment (v1.3.0): done.
- Phase 2, calibrated noise and accounting: mechanism, per-round epsilon and budget refusal in 1.9.0.
  Open: calibrate C (dl3 measure run), dl3 rehearsal of utility against sigma, then the consortium
  noise-cost run (#593).
- STAMP: utility with and without noise needs a DP hook in STAMP's own training loop; not in 1.9.0.
- Phase 3, record-level DP (DP-SGD): not started; Opacus would have to enter the image.
- Phase 4, formal verification, third-party audit and DPIA documentation: not started.

## References

1. Shokri & Shmatikov, "Privacy-Preserving Deep Learning", CCS 2015 (PercentilePrivacy basis)
2. Abadi et al., "Deep Learning with Differential Privacy", CCS 2016 (DP-SGD)
3. McMahan et al., "Learning Differentially Private Recurrent Language Models", ICLR 2018
   (client-level DP by clipping and noising per-client updates)
4. Mironov, "Renyi Differential Privacy", CSF 2017 (RDP composition)
5. Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy", NeurIPS 2020, and
   Balle et al., "Hypothesis Testing Interpretations and Renyi Differential Privacy", AISTATS 2020
   (the RDP to (epsilon, delta) conversion used here)
6. Mironov, "On Significance of the Least Significant Bits for Differential Privacy", CCS 2012
   (floating-point side channels)
7. Dwork & Roth, "The Algorithmic Foundations of Differential Privacy", 2014
