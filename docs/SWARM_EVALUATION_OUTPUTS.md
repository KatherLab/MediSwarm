# Swarm evaluation outputs: which model each number belongs to

> **Scope.** What a swarm job reports about its models from release 1.9.0 on: the metric prefixes the
> sites stream, the owners in `cross_val_results.json`, the per-case files on the coordinator, the local
> CSVs at each site, and the evaluation-only final round (`ODELIA_FINAL_EVAL_ROUND=1`) that scores the
> final global at every site. Companion to [SWARM_MODEL_KINDS.md](./SWARM_MODEL_KINDS.md) (which
> checkpoint files a run leaves) and [EVALUATION_PITFALLS.md](./EVALUATION_PITFALLS.md).

## 1. Why this changed

Up to 1.8 a site validated the global it received with the same `val/*` metric names as the local
validation inside its training epochs, and the coordinator kept the last value per name under the owner
`swarm_global`. The published per-site numbers were therefore each site's **local** model after its last
epoch, labelled as the global (October run 7a678d9a, see
[ODELIA_RUN_2026-10_MST.md](./ODELIA_RUN_2026-10_MST.md)). The final global itself was never evaluated
at the sites, and returned per-case rows of different splits or models would have overwritten each other.

## 2. Streamed metrics and `cross_val_results.json`

Each model streams under its own prefix, and `PerSiteMetricCollector` files each prefix under its own
owner in `cross_site_val/cross_val_results.json` (`{site: {owner: {metric: value}}}`):

| Prefix | Model | When | Owner in `cross_val_results.json` |
|---|---|---|---|
| `val/` | the site's own model | validation inside every local epoch | `site_local_model` (last epoch of the last training round) |
| `global_val/` | the global the site received | start of every training round | `swarm_global_received` (last training round) |
| `final_val/`, `final_test/` | the job's final global | evaluation-only round, validation and test split | `swarm_global` |

- Every owner carries `<prefix>/round` (0-based round its values come from) and `_last_step`. Support
  counts travel with every metric (`*/support_class{i}`, `*/n`, see E1 in EVALUATION_PITFALLS.md).
- `swarm_global` exists only when the job ran the evaluation-only round; without it the final global is
  not evaluated at the sites. It never holds local-model values any more.
- `train/*` values are streamed but not published (the collector keeps only the prefixes above).
  Validation and test losses are streamed per epoch only: at batch size 1 a per-step validation loss is
  one case's loss, i.e. per-case data without the opt-in.
- CCWF best-global selection is unchanged. `IntimeModelSelector` reads `val/AUC_ROC` from the metrics of
  the first validation of each round (`FLCallback` sends them as the result's `initial_metrics`). The
  training code still records `val/AUC_ROC` there for the received global but does not stream it
  (`BasicClassifier.set_metric_labels`); renaming it would switch best-global selection off silently.

## 3. Per-case predictions (opt-in, `ODELIA_RETURN_PER_CASE=1`)

The coordinator writes one file per site, split and model kind:

```
cross_site_val/per_case_predictions/<SITE>__<split>__<model_kind>.csv
```

| `split` | `model_kind` | Rows | Sent |
|---|---|---|---|
| `val` | `site_local` | the local model's validation predictions | after every local epoch; the newest complete emission is kept (last epoch of the last training round) |
| `val` | `final_global` | the final global's validation predictions | evaluation-only round |
| `test` | `final_global` | the final global's test predictions | evaluation-only round |

Each file starts with `#` provenance lines (job id, site, split, model kind, round, epoch, rows, whether
every chunk arrived), then the columns
`row_index,ground_truth,prob_class_0,prob_class_1,prob_class_2,split,model_kind,round`. Read it with
`pandas.read_csv(path, comment="#")`. `row_index` is the position in the site's own (unshuffled) split, the
same case for every model of that split, so two models pair by `(site, split, row_index)`; no identifier
leaves the site. `scripts/evaluation/compare_regional_finetune.py` selects a file with `--split` and
`--model-kind` and refuses to guess when a site has several. Up to 1.8 the job wrote one `<SITE>.csv`,
which held the local model's validation rows (`val`/`site_local`).

## 4. Local CSVs in the site's run directory

| File | Model | Split | Written |
|---|---|---|---|
| `site_model_gt_and_classprob_train.csv`, `..._validation.csv` | local model | train / val | after every local epoch (no header, `epoch` column) |
| `aggregated_model_gt_and_classprob_train.csv`, `..._validation.csv` | received global | train / val | rounds selected by `ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS` (default: final round only) |
| `final_global_model_gt_and_classprob_validation.csv`, `..._test.csv` | final global | val / test | evaluation-only round; header `round,row_index,ground_truth,prob_class_*` |
| `last_site_model_gt_and_classprob_test.csv`, `best_site_model_gt_and_classprob_test.csv` | local last / best checkpoint | test | `finalize_training` at the end of the job |

Every export runs the model in eval mode and restores each submodule's own mode afterwards (#618), so
BatchNorm and Dropout models export what they would predict at inference. None of these files carries a
uid (kit_live_sync uploads `*_model_gt_and_classprob_*.csv`). Since 1.9.0 the challenge models 2-5 also
write the two local test CSVs (#627).

## 5. The evaluation-only final round

Enable it in the job: `ODELIA_FINAL_EVAL_ROUND=1` in the launcher line of `config_fed_client.conf` and
`num_rounds = training rounds + 1` in `config_fed_server.conf` (the job builder sets both). In the last
round every site:

1. validates the received global, which is the final global of the job (`final_val/*`);
2. evaluates it on its test split (`final_test/*`);
3. writes `final_global_model_gt_and_classprob_{validation,test}.csv` and, if opted in, returns the
   per-case rows (`final_global`); rows and metrics come from the same forward passes;
4. runs the aggregated-model export if `ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS` selects this round
   (by default it does: the final-round export moves here, the last training round no longer pays for it);
5. runs `finalize_training` (local best/last checkpoint test CSVs);
6. sends the received weights back unchanged, with `NUM_STEPS_CURRENT_ROUND=1` and the metrics of step 1.

Nothing trains in that round, so there is no update for DP noise and no local model to capture. Round
hooks see `after_receive` (with `ctx.is_eval_only_round=True`) and `on_end`, never `before_fit` or
`after_round`.

**Why the final global stays identical.** The jobs exchange weight differences
(`params_transfer_type = "DIFF"`). The client API turns the full parameters sent in step 6 into
`sent - received` against the very task of this round (`ModelRegistry._prepare_param_diff`), which is an
exact zero for every tensor. `InTimeAccumulateWeightedAggregator` averages zeros to zero and
`SimpleModelShareableGenerator` adds that zero to the round's global, so `FL_global_model.pt` is the
global the sites evaluated (bitwise, apart from a stored `-0.0` becoming `+0.0`). The selector weighs
sites equally in every round (`weigh_by_local_iter` is off), so the evaluation round's `val/AUC_ROC`
competes for best global on the same footing as every earlier round.

**What can fail the round.** Only step 1: it loads the global, and the result must carry its metrics.
Steps 2-5 are outputs and run best effort: a failure is logged with its traceback, the site then has no
`final_test/*` values and no test files, and the weights still go back, so one unreadable test image does
not hold the last round until the learn-task timeout. Up to 1.8 `finalize_training` ran after the last
send, where its failure could not hold up the job either.

**A job with only this round** (`num_rounds = 1`) scores the round-0 global, e.g. a warm-start baseline,
with exactly the code that scores the fine-tuned model. It has no local checkpoints, so step 5 only logs
that there is nothing to export.

## 6. Export switches and stage timings

| Variable | Default | Effect |
|---|---|---|
| `ODELIA_FINAL_EVAL_ROUND` | `0` | `1` = the last round is the evaluation-only round (section 5) |
| `ODELIA_EXPORT_TRAIN_PREDICTIONS` | `1` | `0` = no train-split pass in the per-epoch and in the aggregated-model export; validation rows are still written and returned |
| `ODELIA_PREDICTION_EXPORT_EVERY_N_ROUNDS` | `0` | aggregated-model export: `0` final round only, `1` every round, `N` every Nth round plus the final one |

Both switches accept only `0`/`1` (also `true`/`false`, `yes`/`no`, `on`/`off`); anything else stops the
job at start rather than silently training or evaluating the wrong round. They are launcher variables:
the job builder sets them for every site.

Every stage is logged as `[STAGE] Start: <name>` / `[STAGE] Done: <name> in <s>s` (`Failed` on an
exception), followed by an `ODELIA execution stage timings:` summary per round:

| Round | Stages | Cost driver |
|---|---|---|
| training | `Validation` (received global), `Aggregated prediction export` (selected rounds), `Fit` | `Fit` includes the per-epoch CSV callback: one inference pass over the train split and one over the validation split after every epoch |
| evaluation-only | `Final global validation`, `Final global test`, `Final global prediction CSVs and per-case return`, `Aggregated prediction export` (default on), `Local checkpoint test export (finalize)`, `Return received global` | one validation and one test pass, the export's train and validation pass, two test passes for the local checkpoints |

Measured costs of the exports (#542): the per-epoch train+val export took 12 of 34 minutes of a `Fit`
at UKA_1 with MST (October), and 21 (August) to 63-135 (September) minutes per epoch at UMCU_1 with
1DivideAndConquer; the final-round aggregated export took 600 s at UKA_1 and 682 s at UMCU_1 (October).
In September UMCU_1 never started its final-round training because it was still in that export. The
train-split pass is the bulk of both, and its CSV (augmented, randomly ordered, no uid) is of little use
for analysis: set `ODELIA_EXPORT_TRAIN_PREDICTIONS=0` for consortium runs with large sites.
