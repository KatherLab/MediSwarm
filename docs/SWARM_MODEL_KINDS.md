# Swarm training: model kinds saved per site

> **Scope.** This doc explains the **three distinct model artifacts** that NVFlare-based swarm training in this repo persists per site, where each one is written, what it semantically represents, and how each performs empirically. It also gives concrete guidance on which artifact to pick for deployment.
>
> The empirical numbers below come from the 3-site Duke IID swarm run on 2026-04-26 (`MST` job, 20 rounds, Cosmos server + dd-dl0/dl2/dl3 clients). See [DUKE_IID_SWARM_REPORT.md](./DUKE_IID_SWARM_REPORT.md) for the full run report.

## 1. The three model kinds

For every (site × job) pair the swarm leaves **three checkpoint files** on disk:

| Kind | Filename | Saved by | Path on the client machine | Format |
|------|----------|----------|----------------------------|--------|
| **GLOBAL FINAL** | `FL_global_model.pt` | NVFlare's `PTFileModelPersistor` | `<DEPLOY_DIR>/<site>/<job_id>/app_<site>/FL_global_model.pt` | bare `state_dict` (`*.pt`) |
| **LOCAL BEST**   | `epoch=N-step=M.ckpt` | PyTorch Lightning's `ModelCheckpoint(save_top_k=1, monitor="val/ACC")` (declared in [`base_model.py`](../application/jobs/_shared/custom/models/base_model.py)) | `<SCRATCH_DIR>/runs/<site>/<job>_<config>_<ts>/epoch=N-step=M.ckpt` | full Lightning `*.ckpt` (state_dict + optimizer + scheduler + loop state) |
| **LOCAL LAST**   | `last.ckpt` | PyTorch Lightning auto-save | same dir as LOCAL BEST | full Lightning `*.ckpt` |

A swarm with N sites × J jobs therefore produces **N × J × 3 = 3·N·J** checkpoint files in total. For our 3-site MST run that's 9 files (3 per site).

### Semantics

- **GLOBAL FINAL.** The aggregated model after the final round of swarm training, persisted by NVFlare's standard `PTFileModelPersistor`. It is the same object across all sites for a clean run — the swarm hands the latest aggregated weights back to every participant for persistence. **In our 2026-04-26 MST run, all three sites' `FL_global_model.pt` files were byte-identical** (`md5sum` matched), as expected.
- **LOCAL BEST.** Per site, a Lightning-managed checkpoint of the model state at the local epoch where this site's *own* validation accuracy peaked (`val/ACC`), monitored across **all rounds** since the trainer was started. The optimizer and Lightning trainer state from that exact step are also preserved. Because each site has its own validation split and trains for a different number of local epochs per round, **LOCAL BEST is genuinely site-specific** and the three files differ across sites.
- **LOCAL LAST.** Per site, a Lightning auto-saved snapshot of the model state at the *very last* local training epoch on that site. Since each round overwrites this file, in practice it captures the post-round weights that this site held just before the next aggregation cycle. **It is also genuinely site-specific** — different across sites — but tends to be closer to the global aggregated weights than LOCAL BEST is, because it always reflects the most recent round's update.

### What is **not** saved

- There is **no `best_FL_global_model.pt`** in this configuration. The `PTFileModelPersistor` here is configured to track only the latest aggregated model, not a global-validation-tracked best. The collector script searches for both names, but only the latest is ever found.
- Optimizer / scheduler state is **only** present in the Lightning `*.ckpt` files. The NVFlare `*.pt` is a bare `state_dict` and cannot be used to resume training without restarting the optimizer.

### Inference: which file format does `predict.py` accept?

`scripts/evaluation/predict.py` handles both formats:

- For `FL_global_model.pt` (default) — pass `--checkpoint-type state_dict` (or omit; default).
- For Lightning `.ckpt` — pass `--checkpoint-type lightning`. The script internally pulls `ckpt["state_dict"]`.

When pointed at a swarm workspace via `--workspace`, the script auto-discovers `app_*/FL_global_model.pt` and (if present) `app_*/best_FL_global_model.pt`. To evaluate Lightning files, pass them explicitly via `--checkpoint /path/file.ckpt ...`.

## 2. Empirical comparison (MST, 2026-04-26 run)

We collected all three kinds for all three sites (9 files; one of those — GLOBAL FINAL — is shared across sites, so 4 unique models in total) and ran them through the same downstream evaluation:

- **Duke held-out test** (binary malignant-vs-not, 262 UIDs from 131 patients).
- **ODELIA challenge** at 6 institutions (`CAM`, `MHA`, `RSH`, `RUMC`, `UKA`, `UMCU`) — full ternary `{0=No, 1=Benign, 2=Malignant}` with **Class 2 (Malignant) AUROC** as the clinical headline metric.

### Duke held-out test (binary)

| Kind | AUROC | Accuracy | F1 | Recall (Sens) | Specificity |
|------|-------|----------|-----|---------------|-------------|
| **GLOBAL FINAL** | **0.8952** | 0.8092 | 0.8120 | 0.7883 | 0.8320 |
| node_A LOCAL BEST | 0.8604 | 0.8015 | 0.8116 | 0.8175 | 0.7840 |
| node_A LOCAL LAST | 0.8777 | 0.8053 | 0.8061 | 0.7737 | 0.8400 |
| node_B LOCAL BEST | 0.8775 | 0.8015 | 0.7937 | 0.7299 | 0.8800 |
| node_B LOCAL LAST | 0.8879 | 0.8053 | 0.8061 | 0.7737 | 0.8400 |
| node_C LOCAL BEST | 0.8846 | 0.8015 | 0.8045 | 0.7810 | 0.8240 |
| node_C LOCAL LAST | 0.8900 | 0.8282 | 0.8276 | 0.7883 | 0.8720 |

### ODELIA — Class 2 (Malignant) AUROC per institution

| Kind | CAM | MHA | RSH | RUMC | UKA | UMCU | **mean** |
|------|-----|-----|-----|------|-----|------|----------|
| **GLOBAL FINAL** | 0.9585 | 0.6897 | 0.7322 | 0.7917 | 0.8542 | 0.8373 | **0.8106** |
| node_A LOCAL BEST | 0.9244 | 0.6207 | 0.7436 | 0.7917 | 0.7431 | 0.7831 | 0.7678 |
| node_A LOCAL LAST | 0.9421 | 0.6991 | 0.7664 | 0.6250 | 0.8472 | 0.7288 | 0.7681 |
| node_B LOCAL BEST | 0.9579 | 0.7210 | 0.6895 | 0.6667 | 0.7917 | 0.7831 | 0.7683 |
| node_B LOCAL LAST | 0.9622 | 0.6708 | 0.6866 | 0.8333 | 0.8403 | 0.7492 | 0.7904 |
| node_C LOCAL BEST | 0.9506 | 0.6583 | 0.6809 | 0.6667 | 0.8125 | 0.8339 | 0.7671 |
| node_C LOCAL LAST | 0.9512 | 0.6928 | 0.8063 | 0.6042 | 0.8542 | 0.7356 | 0.7741 |

### ODELIA — 3-class macro AUROC per institution

| Kind | CAM | MHA | RSH | RUMC | UKA | UMCU | **mean** |
|------|-----|-----|-----|------|-----|------|----------|
| **GLOBAL FINAL** | 0.7869 | 0.5553 | 0.6398 | — | 0.7119 | 0.6444 | 0.6677 |
| node_A LOCAL BEST | 0.7242 | 0.6815 | 0.5989 | — | 0.5489 | 0.6101 | 0.6327 |
| node_A LOCAL LAST | 0.7650 | 0.5383 | 0.6563 | — | 0.6560 | 0.5573 | 0.6346 |
| node_B LOCAL BEST | 0.8294 | 0.6393 | 0.5697 | — | 0.6549 | 0.5833 | 0.6553 |
| node_B LOCAL LAST | 0.8064 | 0.5839 | 0.6170 | — | 0.7010 | 0.6419 | 0.6700 |
| node_C LOCAL BEST | 0.7714 | 0.5961 | 0.6524 | — | 0.6123 | 0.6317 | 0.6528 |
| node_C LOCAL LAST | 0.7903 | 0.6579 | 0.7164 | — | 0.6963 | 0.5654 | **0.6853** |

(RUMC's macro AUROC is undefined because RUMC test labels contain only classes 0 and 2 — no Benign — so torchmetrics can't compute a 3-class macro. Per-class AUROCs are still well-defined and shown above.)

## 3. Findings

### a. GLOBAL FINAL is the strongest default
- Beats every LOCAL kind on the in-distribution **Duke** held-out test (0.8952 AUROC vs ≤0.8900 for the best LOCAL).
- Has the highest mean Class-2 AUROC on ODELIA across the 6 institutions (0.8106 vs ≤0.7904 for any LOCAL).

### b. LOCAL LAST beats LOCAL BEST, consistently
On every site, on every dataset, **LOCAL LAST ≥ LOCAL BEST** for the headline metrics. The Lightning "best by val/ACC" pick is captured *early* in training (e.g. node_A picked epoch 32 of 120), before the model has seen many federated aggregations. By the *last* epoch the model has absorbed many more rounds of cross-site information and generalises better — even though the local val/ACC may be noisier.

**Implication.** `monitor="val/ACC"` selecting an early-round checkpoint is, in our setup, a misleading proxy for downstream test performance. If a "best" is desired, monitoring something like a moving-average of val/ACC across recent rounds, or simply preferring `last.ckpt`, is more reliable.

### c. Per-institution, sometimes a LOCAL LAST beats GLOBAL FINAL on cross-domain data

This is the most surprising finding. On 4 of 6 ODELIA institutions, *some* LOCAL LAST snapshot edges out the GLOBAL FINAL on Class-2 AUROC:

| Institution | GLOBAL FINAL | Best LOCAL | Winner |
|-------------|--------------|------------|--------|
| CAM   | 0.9585 | node_B_LAST 0.9622 | LOCAL by +0.004 |
| MHA   | 0.6897 | node_B_BEST 0.7210 | LOCAL by +0.031 |
| RSH   | 0.7322 | node_C_LAST 0.8063 | LOCAL by +0.074 |
| RUMC  | 0.7917 | node_B_LAST 0.8333 | LOCAL by +0.042 |
| UKA   | 0.8542 | node_C_LAST 0.8542 | tie |
| UMCU  | 0.8373 | node_C_BEST 0.8339 | GLOBAL by +0.003 |

Plausible explanation: federated aggregation smoothes per-site idiosyncrasies. A site whose local data happens to be closer to a particular target institution's distribution can briefly hold weights better suited for that institution than the post-aggregation global model. **You can't pick the "right" LOCAL ahead of time** — which one wins on which institution is essentially noise — but it gives a hint that there's still site-specific signal that the simple FedAvg aggregation isn't perfectly preserving.

### d. The 3 GLOBAL FINAL files are byte-identical
Confirmed by `md5sum` for our MST run. This is the expected behaviour for a clean swarm run where the persistor flushes the same final aggregated state across sites. If a swarm is **early-stopped mid-flight** (as we did for the 1DivideAndConquer run), the per-site `FL_global_model.pt` files will *differ*, because each site's persistor has only seen the rotating global model up to whenever that site last received it.

## 4. Practical guidance

1. **Default → use GLOBAL FINAL** (`<DEPLOY_DIR>/<site>/<job_id>/app_<site>/FL_global_model.pt`). It's the strongest single picker for in-distribution performance and the most consistent across cross-domain institutions.
2. **If `predict.py` reports a "best by val" Lightning checkpoint as superior, double-check on test data** — it's likely an artefact of an early epoch peaking on a small noisy val set. Compare against `last.ckpt` from the same run before believing the result.
3. **For cross-domain deployment** (e.g. running an ODELIA-trained model on UMCU images), prefer GLOBAL FINAL as the safer-on-average choice; ensemble across the 3 LOCAL LAST checkpoints if you can afford it — that recovers most of the per-institution upside without having to know in advance which site's snapshot best matches the target.
4. **If you early-stop a swarm run**, the per-site GLOBAL FINAL files differ. Pick the *freshest* by `mtime` (it's the latest aggregated state), or do a clean `abort_job` from the NVFlare admin console (which lets the persistor flush a final aggregated model on every site simultaneously, restoring the byte-identical invariant).
5. **Don't rely on `best_FL_global_model.pt`** in this repo's setup — it isn't being saved. If you want a global-validation-tracked best, you'd need to add a `BestModelSelector` filter in the swarm app config (out of scope here).

## 5. Reproducing this analysis

The 6 Lightning checkpoints are at `workspace/deploy_test_results/MST_lightning_checkpoints/node_{A,B,C}_{best,last}.ckpt`. The eval orchestrator that produced the per-kind metrics is `mst_lightning_eval.sh` (kept outside the repo). To rebuild the comparison tables from raw eval JSON: `python3 build_kind_comparison.py` writes `workspace/deploy_test_results/duke_iid_eval/kind_comparison.{json,md}`.

Per-checkpoint raw outputs:
- **GLOBAL FINAL**: `workspace/deploy_test_results/duke_iid_eval/MST/<dataset>/predictions_node_*_latest{,_binary}.{csv,json}`
- **Lightning kinds**: `workspace/deploy_test_results/duke_iid_eval/MST_lightning/<dataset>/predictions_node_<X>_<best|last>_<best|single>{,_binary}.{csv,json}`

(The double `_best_best` / `_last_single` suffix in Lightning filenames comes from `predict.py` appending the per-checkpoint *kind* it inferred from the path.)
