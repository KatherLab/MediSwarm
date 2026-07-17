# External validation (predict-only node)

A center can validate a released ODELIA global model on **its own data** without joining
the swarm — no VPN, no aggregation, no network traffic. Predictions and per-sample
outputs stay on the node; **only the aggregate metrics need to be shared back.** This is
the privacy-preserving way to add an external-validation site (#412).

## What the node needs

1. The ODELIA Docker image (pulled automatically by `docker.sh`, like any node).
2. The delivered global model checkpoint (e.g. `best_FL_global_model.pt`).
3. Local data with a **`test`** (or `ext`) split in `split.csv` — the same layout a
   training site uses (`annotation.csv` / `split.csv` under the data dir).

An external-validation node does **not** need a provisioned swarm kit: it never
authenticates to the server, so it needs no certificates. Any current ODELIA kit works
(its `docker.sh` carries the mode), or ship just `docker.sh` + `predict.py` in the image.

## One command

Put the delivered model in your scratch dir (mounted at `/scratch`), then:

```bash
scripts/deploy/run_external_validation.sh \
    --data_dir    "$DATADIR" \
    --scratch_dir "$SCRATCHDIR" \
    --model_name  <released model name> \
    --split test
# optional: --checkpoint /scratch/<file>.pt   (default: /scratch/FL_global_model.pt)
#           --GPU device=0   --kit_dir <kit>/startup   --skip_preflight
```

It runs the **data-integrity + GPU preflight** first (rejects a corrupt/degenerate
dataset before predicting — the guard that caught a site's corrupt inputs), then the
prediction.

### Or directly, via the kit

```bash
# 1) integrity gate
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 \
            --model_name <name> --preflight_check
# 2) predict-only external validation
./docker.sh --data_dir "$DATADIR" --scratch_dir "$SCRATCHDIR" --GPU device=0 \
            --model_name <name> --split test --external_validation
```

## Outputs (all local, in your scratch dir)

| File | Contents | Share back? |
|---|---|---|
| `prediction_results.json` | aggregate metrics (accuracy, AUC-ROC, per-class F1, confusion matrix) | **yes — this is all you send** |
| `predictions_*.csv` | per-sample predictions | no — keep on the node |

Because the metric set matches the training-site evaluation, the external numbers are
directly comparable to the swarm's internal results.

## Notes

- `--model_name` must match the released model's architecture, or the checkpoint won't
  load. The board's kit registry names the model for each released global model.
- Data is mounted **read-only** at `/data`; scratch is writable at `/scratch`. Nothing
  leaves the container except what you choose to share from `prediction_results.json`.
