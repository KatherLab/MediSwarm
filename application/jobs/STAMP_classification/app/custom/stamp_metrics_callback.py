"""Per-epoch metrics and per-patient prediction CSV callbacks for STAMP.

Provides two Lightning callbacks:

1. **STAMPMetricsSummaryCallback** — appends one row per epoch to a summary
   CSV with columns: epoch, train_loss, val_loss, val_auroc, learning_rate.

2. **STAMPPredictionCallback** — runs the model over train and validation
   dataloaders at each epoch end and writes per-patient ground-truth and
   predicted-probability CSVs (same pattern as ODELIA's
   ``GT_PredProb_Output_Callback`` but adapted for STAMP's bag-of-features
   dataloader format).

Both callbacks are optional — enable via ``prepare_training()`` in
``stamp_training.py``.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Optional

import torch

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File names for CSV output (parallel to ODELIA naming convention)
# ---------------------------------------------------------------------------
FILENAME_METRICS_SUMMARY = "stamp_metrics_summary.csv"
FILENAME_GT_PREDPROB_SITE_TRAIN = "stamp_gt_predprob_site_model_train.csv"
FILENAME_GT_PREDPROB_SITE_VALIDATION = "stamp_gt_predprob_site_model_validation.csv"
FILENAME_GT_PREDPROB_AGGREGATED_TRAIN = "stamp_gt_predprob_aggregated_model_train.csv"
FILENAME_GT_PREDPROB_AGGREGATED_VALIDATION = "stamp_gt_predprob_aggregated_model_validation.csv"


# ---------------------------------------------------------------------------
# AUROC helper (pure Python — no sklearn/torch dependency, unit-testable in CI)
# ---------------------------------------------------------------------------
# STAMP's Lightning model does not expose AUROC under a ``callback_metrics`` key
# we can rely on, so ``val_auroc`` in the summary CSV was always blank (#492).
# We instead compute AUROC directly from the validation predictions the
# prediction callback already gathers each epoch.

def _binary_auroc(scores, labels):
    """AUROC for one binary problem via the rank (Mann-Whitney U) statistic.

    ``scores``: iterable of floats (predicted score for the positive class).
    ``labels``: iterable of 0/1 (1 = positive).
    Ties in ``scores`` receive averaged ranks, matching sklearn's
    ``roc_auc_score``. Returns ``None`` if either class is absent.
    """
    scores = list(scores)
    labels = list(labels)
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based average rank for the tie group
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    sum_pos_ranks = sum(ranks[idx] for idx, l in enumerate(labels) if l == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _true_class_index(gt):
    """Normalise a ground-truth entry (scalar index or one-hot list) to an int."""
    if isinstance(gt, (list, tuple)):
        best_i, best_v = 0, gt[0]
        for i, v in enumerate(gt):
            if v > best_v:
                best_i, best_v = i, v
        return best_i
    return int(gt)


def macro_ovr_auroc(ground_truths, prob_rows):
    """Macro one-vs-rest AUROC.

    ``ground_truths``: list of ground-truth entries (int class index or one-hot list).
    ``prob_rows``: list of per-class probability lists (one row per sample).

    Returns the unweighted mean of each class's one-vs-rest AUROC over the classes
    for which both positives and negatives are present, or ``None`` if none qualify.
    For the binary case this is the standard AUROC.
    """
    if not prob_rows:
        return None
    n_classes = len(prob_rows[0])
    true_idx = [_true_class_index(gt) for gt in ground_truths]
    per_class = []
    for c in range(n_classes):
        scores = [row[c] for row in prob_rows]
        labels = [1 if t == c else 0 for t in true_idx]
        auc = _binary_auroc(scores, labels)
        if auc is not None:
            per_class.append(auc)
    if not per_class:
        return None
    return sum(per_class) / len(per_class)


# ---------------------------------------------------------------------------
# 1. Per-epoch metrics summary
# ---------------------------------------------------------------------------

class STAMPMetricsSummaryCallback(Callback):
    """Write a per-epoch metrics summary CSV.

    Columns: epoch, train_loss, val_loss, val_auroc, learning_rate

    The CSV is created with a header on the first write and appended to on
    subsequent epochs (survives across ``trainer.fit()`` calls in swarm mode).
    """

    HEADER = ["epoch", "train_loss", "val_loss", "val_auroc", "learning_rate"]

    def __init__(self, output_dir: Path, metric_holder=None):
        super().__init__()
        self.csv_path = output_dir / FILENAME_METRICS_SUMMARY
        self._header_written = self.csv_path.exists()
        # Optional object exposing ``last_val_auroc`` (populated by the
        # prediction callback). Used as a fallback when STAMP does not log
        # AUROC into ``callback_metrics`` — which is the usual case (#492).
        self.metric_holder = metric_holder

    def _write_header_if_needed(self):
        if not self._header_written:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)
            self._header_written = True

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Append metrics row at the end of each training epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        train_loss = metrics.get("training_loss")
        val_loss = metrics.get("validation_loss")
        learning_rate = metrics.get("learning_rate")

        # AUROC — prefer a value STAMP logged into callback_metrics; otherwise
        # fall back to the AUROC computed by the prediction callback (#492).
        val_auroc = None
        # "validation_auroc" is the key STAMP actually logs (confirmed from a real
        # 2-node run); the others are kept as fallbacks for other STAMP versions.
        for key in ("validation_auroc", "val_auroc", "val_MulticlassAUROC"):
            v = metrics.get(key)
            if v is not None:
                val_auroc = v
                break
        if val_auroc is None and self.metric_holder is not None:
            val_auroc = getattr(self.metric_holder, "last_val_auroc", None)

        # Convert tensors to Python floats
        def _to_float(v):
            if v is None:
                return ""
            return v.item() if hasattr(v, "item") else float(v)

        row = [
            epoch,
            _to_float(train_loss),
            _to_float(val_loss),
            _to_float(val_auroc),
            _to_float(learning_rate),
        ]

        self._write_header_if_needed()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 2. Per-patient ground-truth and predicted-probability CSV
# ---------------------------------------------------------------------------

class STAMPPredictionCallback(Callback):
    """Write per-patient predictions to CSV after each training epoch.

    For each patient in the train and validation dataloaders, records:
    ``epoch, patient_id, ground_truth, pred_class, prob_0, prob_1, ..., prob_N``

    This mirrors ODELIA's ``GT_PredProb_Output_Callback`` but adapted for
    STAMP's dataloader format where each batch is a bag of tile features for
    one or more patients.

    STAMP's Lightning models log ``validation_loss`` and optionally AUROC,
    but don't expose per-patient predictions via callback_metrics.  This
    callback runs inference explicitly.
    """

    def __init__(
        self,
        train_dl: torch.utils.data.DataLoader,
        valid_dl: torch.utils.data.DataLoader,
        output_dir: Path,
        metric_holder=None,
    ):
        super().__init__()
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.csv_train = output_dir / FILENAME_GT_PREDPROB_SITE_TRAIN
        self.csv_valid = output_dir / FILENAME_GT_PREDPROB_SITE_VALIDATION
        # Latest validation-set AUROC of the locally trained (site) model.
        self.last_val_auroc = None
        # Optional object whose ``last_val_auroc`` we also update, so the summary
        # callback and NVFlare model selection can read it (#492 / #493).
        self.metric_holder = metric_holder

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Run inference on train/val sets and write prediction CSVs.

        Restores the module's original train/eval mode afterwards: ``_write_predictions``
        switches the model to ``eval()``, and Lightning does **not** put it back before
        the next epoch. Leaking eval mode meant every epoch after the first (and every
        swarm round after the first, since NVFlare calls ``fit()`` per round) trained
        with dropout disabled — which Lightning reports as
        "Found N module(s) in eval mode at the start of training".
        """
        epoch = trainer.current_epoch
        was_training = pl_module.training

        try:
            self._write_predictions(pl_module, self.train_dl, epoch, self.csv_train)
            self._write_predictions(pl_module, self.valid_dl, epoch, self.csv_valid)
            # Publish val AUROC to Lightning's metric store so NVFlare's
            # IntimeModelSelector (key_metric=val_auroc) can select the best
            # round instead of always keeping the last (#493). Fail-safe: any
            # error here leaves selection at its prior (no-metric) behaviour.
            if self.last_val_auroc is not None:
                try:
                    pl_module.log("val_auroc", float(self.last_val_auroc),
                                  prog_bar=False, on_epoch=True)
                except Exception as le:
                    logger.debug(f"could not log val_auroc metric: {le}")
        except Exception as e:
            # Don't crash training if prediction CSV writing fails
            logger.warning(f"STAMPPredictionCallback failed at epoch {epoch}: {e}")
        finally:
            if was_training:
                pl_module.train()

    @torch.no_grad()
    def _write_predictions(
        self,
        model: Any,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        csv_path: Path,
    ) -> None:
        """Run model inference on a dataloader and append predictions to CSV.

        STAMP dataloaders yield batches where each sample contains:
        - ``bags``: Tensor of tile features, shape (batch, bag_size, dim)
        - ``targets``: Ground-truth class index or survival label
        - Possibly ``patient_id`` or metadata depending on STAMP version

        The model's ``forward()`` returns logits which we convert to
        probabilities via softmax.
        """
        model.eval()
        device = next(model.parameters()).device

        write_header = not csv_path.exists()
        rows = []

        for batch_idx, batch in enumerate(dataloader):
            # STAMP 2.4.0 BagDataset yields tuples: (bags, targets, ...)
            # or dicts depending on the collate function.  Handle both.
            # STAMP 2.4.0's BagDataset collate yields a 4-tuple:
            #   (bags, coords, bag_sizes, targets)
            # Older/other formats may yield (bags, targets[, patient_ids]) or a
            # dict; handle all of them.
            coords = None
            patient_ids = None
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 4:
                    bags, coords, _bag_sizes, targets = batch[0], batch[1], batch[2], batch[3]
                elif len(batch) == 3:
                    bags, targets, patient_ids = batch[0], batch[1], batch[2]
                else:
                    bags, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                bags = batch.get("bags", batch.get("features"))
                coords = batch.get("coords")
                targets = batch.get("targets", batch.get("labels"))
                patient_ids = batch.get("patient_ids", batch.get("patient_id"))
            else:
                logger.warning(f"Unexpected batch type: {type(batch)}")
                continue

            if bags is None or targets is None:
                continue

            bags = bags.to(device)
            targets = targets.to(device)
            if coords is not None and hasattr(coords, "to"):
                coords = coords.to(device)

            # Forward pass. STAMP's Lightning module wraps the network in
            # ``model.model`` and invokes it as ``model.model(bags, coords=coords,
            # mask=None)`` (see LitTileClassifier.predict_step). Mirror that; fall
            # back to a plain positional call for non-STAMP models.
            net = getattr(model, "model", model)
            try:
                logits = net(bags, coords=coords, mask=None)
            except Exception:
                try:
                    logits = model(bags)
                except Exception as e:
                    logger.warning(f"Could not run forward pass: {e}")
                    return

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1).cpu()
            targets_cpu = targets.cpu()

            batch_size = probs.shape[0]
            for i in range(batch_size):
                gt = targets_cpu[i].item() if targets_cpu[i].dim() == 0 else targets_cpu[i].tolist()
                prob_list = probs[i].tolist()
                pred_class = probs[i].argmax().item()

                # Patient ID: use from batch if available, otherwise use batch index
                if patient_ids is not None:
                    if isinstance(patient_ids, (list, tuple)):
                        pid = patient_ids[i]
                    elif hasattr(patient_ids, '__getitem__'):
                        pid = patient_ids[i]
                    else:
                        pid = f"batch{batch_idx}_sample{i}"
                else:
                    pid = f"batch{batch_idx}_sample{i}"

                rows.append([epoch, pid, gt, pred_class] + prob_list)

        # Write to CSV
        if rows:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    n_classes = len(rows[0]) - 4  # epoch, pid, gt, pred_class, then probs
                    header = ["epoch", "patient_id", "ground_truth", "pred_class"]
                    header += [f"prob_{i}" for i in range(n_classes)]
                    writer.writerow(header)
                writer.writerows(rows)

            logger.debug(f"Wrote {len(rows)} predictions to {csv_path}")

        # For the validation set, compute AUROC from the just-gathered predictions
        # (row layout: [epoch, patient_id, ground_truth, pred_class, prob_0, ...]).
        # This is the site (locally-trained) model's val AUROC for this epoch.
        if rows and csv_path == getattr(self, "csv_valid", None):
            try:
                ground_truths = [r[2] for r in rows]
                prob_rows = [r[4:] for r in rows]
                auroc = macro_ovr_auroc(ground_truths, prob_rows)
                if auroc is not None:
                    self.last_val_auroc = auroc
                    holder = getattr(self, "metric_holder", None)
                    if holder is not None:
                        holder.last_val_auroc = auroc
            except Exception as e:
                logger.debug(f"AUROC computation failed: {e}")
