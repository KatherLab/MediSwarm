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
# 1. Per-epoch metrics summary
# ---------------------------------------------------------------------------

class STAMPMetricsSummaryCallback(Callback):
    """Write a per-epoch metrics summary CSV.

    Columns: epoch, train_loss, val_loss, val_auroc, learning_rate

    The CSV is created with a header on the first write and appended to on
    subsequent epochs (survives across ``trainer.fit()`` calls in swarm mode).
    """

    HEADER = ["epoch", "train_loss", "val_loss", "val_auroc", "learning_rate"]

    def __init__(self, output_dir: Path):
        super().__init__()
        self.csv_path = output_dir / FILENAME_METRICS_SUMMARY
        self._header_written = self.csv_path.exists()

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

        # AUROC — try multiple keys STAMP might log
        val_auroc = None
        for key in ("val_auroc", "val_MulticlassAUROC"):
            v = metrics.get(key)
            if v is not None:
                val_auroc = v
                break

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
    ):
        super().__init__()
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.csv_train = output_dir / FILENAME_GT_PREDPROB_SITE_TRAIN
        self.csv_valid = output_dir / FILENAME_GT_PREDPROB_SITE_VALIDATION

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Run inference on train/val sets and write prediction CSVs."""
        epoch = trainer.current_epoch

        try:
            self._write_predictions(pl_module, self.train_dl, epoch, self.csv_train)
            self._write_predictions(pl_module, self.valid_dl, epoch, self.csv_valid)
        except Exception as e:
            # Don't crash training if prediction CSV writing fails
            logger.warning(f"STAMPPredictionCallback failed at epoch {epoch}: {e}")

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
            if isinstance(batch, (list, tuple)):
                bags = batch[0]
                targets = batch[1]
                # Patient IDs may be in batch[2] if available
                patient_ids = batch[2] if len(batch) > 2 else None
            elif isinstance(batch, dict):
                bags = batch.get("bags", batch.get("features"))
                targets = batch.get("targets", batch.get("labels"))
                patient_ids = batch.get("patient_ids", batch.get("patient_id"))
            else:
                logger.warning(f"Unexpected batch type: {type(batch)}")
                continue

            if bags is None or targets is None:
                continue

            bags = bags.to(device)
            targets = targets.to(device)

            # Forward pass — model returns logits
            try:
                logits = model(bags)
            except Exception:
                # Some STAMP models have different forward signatures
                # Fall back to passing as keyword argument
                try:
                    logits = model(features=bags)
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
