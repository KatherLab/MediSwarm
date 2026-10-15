"""al_subset: round hook for D3.2 budgeted active learning (release 1.9.0).

Listed as ``al_subset`` in ODELIA_ROUND_HOOKS. The round-hook safety net (round_hooks.FEATURE_VARIABLES) maps
every variable below to this module: a job that sets one of them without listing al_subset stops at
load_hooks() instead of silently training on the full split. The pins a warm-started selection run usually
carries as well (ODELIA_WARM_START_SHA256, ODELIA_SEED) are applied by reproducibility.py, which load_hooks
adds by itself and runs first, so the round-0 global is verified before anything here scores with it.

Variables (training subprocess, set on the launcher line)
--------------------------------------------------------
ODELIA_AL_STRATEGY        entropy | margin | random | class_matched_random (al_selection.py). Enables selection.
ODELIA_AL_BUDGET          cases selected per site (required with a strategy).
ODELIA_TRAIN_INCLUDE_LIST path of a UID list at the site (one UID per line, '#' comments): train on exactly
                          these cases instead of selecting (strategy 'include_list'). Replaying the
                          selected_uids.txt of an earlier run gives that run's training set.
ODELIA_AL_POOL_SPLIT      train (default) | test: the split scored as the pool. The selected cases are trained
                          on with the training augmentation; with 'test', the local test split then contains
                          training cases (logged and recorded; local test metrics are no longer held out).
                          'val' is refused: it drives checkpoint selection and the swarm's best-global choice.
ODELIA_AL_SEED            seed of the random arms (default 0). Sorted by UID first, so row order never matters.
ODELIA_AL_SELECT_ROUND    round whose received global scores the pool (default 0). Earlier rounds train on the
                          full training split; this and every later round train on the selection.
ODELIA_AL_CLASS_WEIGHTS   subset (default): recompute the loss class weights from the selected cases;
                          keep: keep the weights computed at setup from the full training split.

What happens at the selection round (before_fit, after the received global was validated)
------------------------------------------------------------------------------------------
1. The model must hold the received global: FLCallback loads it in on_validation_start of the round's first
   validate(), which main.py runs before before_fit. This is checked tensor by tensor, so scoring can never
   silently use last round's local weights.
2. The pool split is scored once in eval mode, fp32, no augmentation (the validation transform), batch 1,
   like scripts/evaluation/predict.py. An input the dataset cannot read is returned by its loader as the
   neighbouring case; such positions are detected (UID mismatch) and left out of the pool.
3. The selection replaces the DataModule's training set in place (every holder of the DataModule, e.g. the
   per-epoch prediction export, sees it), the cached train loader is dropped, the loader worker count is
   capped for the small set (#574; validation and test loaders keep theirs), the class weights follow
   ODELIA_AL_CLASS_WEIGHTS, and the epochs per round are recomputed with compute_weighted_epochs (so
   EPOCHS_PER_ROUND / EPOCHS_REFERENCE_DATASET_SIZE / EPOCHS_MAX_CAP still pin them). NVFlare's FLCallback
   carries max_epochs from round to round (max_epochs_per_round + epochs done so far); both values are set
   here so every later round trains the new count.
4. Records. Case identifiers never leave the site: the full record (UIDs, scores, labels) and a replayable
   selected_uids.txt go to $SCRATCH_DIR/odelia_al/<job_id>/<site>/, outside the run directory, because
   live_sync uploads the whole run directory when it stops (kit_live_sync/live_sync.sh, final_sync). The run
   directory gets al_selection_summary.json with aggregates only. Streamed metrics (aggregates only):
   al/n_selected, al/class_count_<k>, al/mean_entropy (not for an include list), al/selection_digest (first
   48 bits of the selection digest as a number). The coordinator keeps them only if the job's
   PerSiteMetricCollector lists the "al/" prefix. Log lines carry counts, never UIDs (nohup.out is uploaded).

Aggregation weight. The hook's Lightning callback (round_steps_callback) sends the optimizer steps a site
actually trained in the round as NUM_STEPS_CURRENT_ROUND, in every round of the job. NVFlare's default,
trainer.estimated_stepping_batches, is the current loader's steps per epoch times the fit loop's cumulative
max_epochs: with a selection in round s > 0 it kept the s rounds of per-site weighted epochs trained before
the selection, so sites that trained the same subset were weighted differently (SP-1). Equal weights stay
dp_noise's ODELIA_DP_EQUAL_WEIGHTS: with it set, the callback is not installed.
"""

import copy
import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import al_selection
import reproducibility
import state_digest
from loader_workers import cap_loader_workers
from reproducibility import release_loader
from round_hooks import RoundHook

ENV_STRATEGY = "ODELIA_AL_STRATEGY"
ENV_BUDGET = "ODELIA_AL_BUDGET"
ENV_POOL_SPLIT = "ODELIA_AL_POOL_SPLIT"
ENV_AL_SEED = "ODELIA_AL_SEED"
ENV_SELECT_ROUND = "ODELIA_AL_SELECT_ROUND"
ENV_CLASS_WEIGHTS = "ODELIA_AL_CLASS_WEIGHTS"
ENV_INCLUDE_LIST = "ODELIA_TRAIN_INCLUDE_LIST"
ENV_FINAL_EVAL_ROUND = "ODELIA_FINAL_EVAL_ROUND"
ENV_DP_EQUAL_WEIGHTS = "ODELIA_DP_EQUAL_WEIGHTS"  # dp_noise's; read here only to leave NUM_STEPS to it

AL_PREFIX = "ODELIA_AL_"
AL_VARIABLES = (ENV_STRATEGY, ENV_BUDGET, ENV_POOL_SPLIT, ENV_AL_SEED, ENV_SELECT_ROUND, ENV_CLASS_WEIGHTS)
# Variables that only refine a selection; on their own they would silently do nothing.
SELECTION_MODIFIERS = (ENV_POOL_SPLIT, ENV_AL_SEED, ENV_SELECT_ROUND, ENV_CLASS_WEIGHTS)

POOL_SPLITS = ("train", "test")
CLASS_WEIGHTS_SUBSET = "subset"
CLASS_WEIGHTS_KEEP = "keep"
CLASS_WEIGHT_POLICIES = (CLASS_WEIGHTS_SUBSET, CLASS_WEIGHTS_KEEP)
MAX_SEED = reproducibility.MAX_SEED

RECORD_DIR = "odelia_al"
RECORD_FILE = "selection_record.json"
UID_LIST_FILE = "selected_uids.txt"
INCLUDE_LIST_ERROR_FILE = "include_list_errors.json"
SUMMARY_FILE = "al_selection_summary.json"

# What FLCallback sends as the update's aggregation weight (nvflare MetaKey.NUM_STEPS_CURRENT_ROUND); it reads
# the LightningModule attribute __fl_meta__ and only fills this key itself when the key is missing.
FL_META_ATTRIBUTE = "__fl_meta__"
NUM_STEPS_KEY = "NUM_STEPS_CURRENT_ROUND"

_LOG = logging.getLogger(__name__)


class ALConfigError(ValueError):
    """The al_subset variables are inconsistent; raised at load_hooks(), before any data is touched."""


# ---------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------


@dataclass
class ALConfig:
    strategy: Optional[str] = None  # None: no selection
    budget: Optional[int] = None
    pool_split: str = "train"
    al_seed: int = 0
    select_round: int = 0
    class_weights: str = CLASS_WEIGHTS_SUBSET
    include_list_path: Optional[str] = None
    include_uids: Optional[List[str]] = None
    include_list_sha256: Optional[str] = None
    final_eval_round: bool = False
    dp_equal_weights: bool = False  # ODELIA_DP_EQUAL_WEIGHTS=1: dp_noise sends NUM_STEPS=1 (no steps callback)

    @property
    def selects(self) -> bool:
        return self.strategy is not None


def _value(env: Mapping[str, str], name: str) -> Optional[str]:
    """Stripped value, None when absent. Present but empty is an error: a template value got lost."""
    if name not in env:
        return None
    value = str(env[name]).strip()
    if not value:
        raise ALConfigError(f"{name} is set but empty; give it a value or remove it from the launcher line")
    return value


def _int(env: Mapping[str, str], name: str, default: Optional[int], minimum: int = 0, maximum: Optional[int] = None):
    raw = _value(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ALConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ALConfigError(f"{name} must be {bounds}, got {value}")
    return value


def read_uid_list(path: str) -> List[str]:
    """UIDs from a list file: one per line, blank lines and '#' comments ignored, order kept.

    Raises on a missing or empty file and on duplicates (a list with repeats is not the list that was meant).
    Messages never quote UIDs: they end up in logs that leave the site.
    """
    if not os.path.isfile(path):
        raise ALConfigError(f"{ENV_INCLUDE_LIST}={path!r}: no such file at this site")
    uids = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            uid = line.strip()
            if uid and not uid.startswith("#"):
                uids.append(uid)
    if not uids:
        raise ALConfigError(f"{ENV_INCLUDE_LIST}={path!r} lists no UIDs")
    duplicates = len(uids) - len(set(uids))
    if duplicates:
        raise ALConfigError(f"{ENV_INCLUDE_LIST}={path!r} repeats {duplicates} UID(s)")
    return uids


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_config(env: Mapping[str, str]) -> ALConfig:
    """Read and cross-check every al_subset variable; raise ALConfigError on anything inconsistent."""
    unknown = sorted(name for name in env if name.startswith(AL_PREFIX) and name not in AL_VARIABLES)
    if unknown:
        raise ALConfigError(
            f"Unknown variable(s) {', '.join(unknown)}; al_subset understands {', '.join(AL_VARIABLES)} "
            f"(a misspelt variable would otherwise leave its default in place)"
        )

    config = ALConfig()
    config.final_eval_round = str(env.get(ENV_FINAL_EVAL_ROUND, "0")).strip() == "1"
    config.dp_equal_weights = str(env.get(ENV_DP_EQUAL_WEIGHTS, "0")).strip() == "1"
    strategy = _value(env, ENV_STRATEGY)
    include_path = _value(env, ENV_INCLUDE_LIST)
    if strategy is not None:
        strategy = strategy.lower()
        if strategy not in al_selection.STRATEGIES + (al_selection.STRATEGY_INCLUDE_LIST,):
            raise ALConfigError(
                f"{ENV_STRATEGY}={strategy!r} is not one of "
                f"{', '.join(al_selection.STRATEGIES + (al_selection.STRATEGY_INCLUDE_LIST,))}"
            )
    if include_path is not None:
        if strategy not in (None, al_selection.STRATEGY_INCLUDE_LIST):
            raise ALConfigError(
                f"{ENV_INCLUDE_LIST} and {ENV_STRATEGY}={strategy} both choose the training cases; set only one"
            )
        strategy = al_selection.STRATEGY_INCLUDE_LIST
    elif strategy == al_selection.STRATEGY_INCLUDE_LIST:
        raise ALConfigError(f"{ENV_STRATEGY}=include_list needs {ENV_INCLUDE_LIST} (the UID list file)")

    budget = _int(env, ENV_BUDGET, None, minimum=1)
    if strategy is None:
        stray = [name for name in (ENV_BUDGET,) + SELECTION_MODIFIERS if name in env]
        if stray:
            raise ALConfigError(
                f"{', '.join(stray)} set, but neither {ENV_STRATEGY} nor {ENV_INCLUDE_LIST} selects any cases"
            )
        return config

    config.strategy = strategy
    config.pool_split = (_value(env, ENV_POOL_SPLIT) or "train").lower()
    if config.pool_split not in POOL_SPLITS:
        raise ALConfigError(
            f"{ENV_POOL_SPLIT}={config.pool_split!r} is not one of {', '.join(POOL_SPLITS)} ('val' is refused: it "
            f"selects the local best checkpoint and the swarm's best global, so it must not contain training cases)"
        )
    config.al_seed = _int(env, ENV_AL_SEED, 0, 0, MAX_SEED)
    config.select_round = _int(env, ENV_SELECT_ROUND, 0, 0)
    config.class_weights = (_value(env, ENV_CLASS_WEIGHTS) or CLASS_WEIGHTS_SUBSET).lower()
    if config.class_weights not in CLASS_WEIGHT_POLICIES:
        raise ALConfigError(
            f"{ENV_CLASS_WEIGHTS}={config.class_weights!r} is not one of {', '.join(CLASS_WEIGHT_POLICIES)}"
        )

    if strategy == al_selection.STRATEGY_INCLUDE_LIST:
        config.include_list_path = include_path
        config.include_uids = read_uid_list(include_path)
        config.include_list_sha256 = _file_sha256(include_path)
        if budget is not None and budget != len(config.include_uids):
            raise ALConfigError(
                f"{ENV_BUDGET}={budget} but {ENV_INCLUDE_LIST} lists {len(config.include_uids)} UIDs; "
                f"is it the intended list?"
            )
        config.budget = len(config.include_uids)
    else:
        if budget is None:
            raise ALConfigError(f"{ENV_STRATEGY}={strategy} needs {ENV_BUDGET} (cases to select per site)")
        config.budget = budget
    return config


# ---------------------------------------------------------------------------------------------------
# Datasets and the DataModule
# ---------------------------------------------------------------------------------------------------


def _reset_bad_input_state(dataset) -> None:
    # Per-dataset bookkeeping of unreadable inputs; a copy must not share the original's sets.
    if hasattr(dataset, "_bad_input_count"):
        dataset._bad_input_count = 0
    if hasattr(dataset, "_bad_input_logged"):
        dataset._bad_input_logged = set()
    if hasattr(dataset, "_bad_input_reasons"):
        dataset._bad_input_reasons = {}


def _uid_series(dataset):
    df = getattr(dataset, "df", None)
    if df is None or "UID" not in df.columns:
        raise RuntimeError(
            f"al_subset needs a dataset with a UID column (ODELIA_Dataset3D); got {type(dataset).__name__}"
        )
    return df["UID"].astype(str)


def label_column(dataset) -> str:
    labels = list(getattr(dataset, "labels", None) or ["Lesion"])
    if len(labels) != 1:
        raise RuntimeError(f"al_subset supports single-label datasets (CONFIG=unilateral*), got labels {labels}")
    return labels[0]


def labels_by_uid(dataset) -> Dict[str, Optional[int]]:
    """UID -> class index of every case in ``dataset``; raises if a UID occurs twice in the split."""
    uids = _uid_series(dataset).tolist()
    if len(uids) != len(set(uids)):
        split = getattr(dataset, "split", "?")
        raise RuntimeError(f"{len(uids) - len(set(uids))} duplicate UID(s) in the {split} split")
    values = dataset.df[label_column(dataset)].tolist()
    result = {}
    for uid, value in zip(uids, values):
        try:
            result[uid] = None if value is None or value != value else int(value)  # value != value: NaN
        except (TypeError, ValueError):
            result[uid] = None
    return result


def num_classes_of(dataset, model=None) -> Optional[int]:
    counts = getattr(dataset, "class_labels_num", None)
    if counts:
        return int(sum(counts))
    weights = getattr(model, "_class_weight", None)
    return int(weights.numel()) if weights is not None else None


def eval_view(dataset, transform):
    """Shallow copy of ``dataset`` that applies ``transform`` (the validation pipeline: no augmentation).

    An unreadable case is served as its neighbour here, also when the pool is the test split (an evaluation
    split, which otherwise stops at such a case): score_pool detects the substitution by UID and leaves the
    case out of the pool.
    """
    view = copy.copy(dataset)
    view.transform = transform
    view.resample_bad_inputs = True
    _reset_bad_input_state(view)
    return view


def restrict_dataset(dataset, uids: Sequence[str], transform=None):
    """Shallow copy of an ODELIA_Dataset3D holding exactly ``uids`` (with ``transform`` if given).

    Filtering the dataframe (rather than wrapping in torch's Subset) keeps len(), .df and the label columns
    working for everything downstream, and keeps the unreadable-input fallback inside the selected cases.
    """
    keep = [str(uid) for uid in uids]
    keep_set = set(keep)
    if len(keep_set) != len(keep):
        raise RuntimeError(f"selection repeats {len(keep) - len(keep_set)} UID(s)")
    uid_series = _uid_series(dataset)
    subset_df = dataset.df[uid_series.isin(keep_set).values].reset_index(drop=True)
    found = set(subset_df["UID"].astype(str))
    if found != keep_set or len(subset_df) != len(keep):
        raise RuntimeError(
            f"selection does not match the {getattr(dataset, 'split', '?')} split: {len(keep_set - found)} UID(s) "
            f"missing, {len(subset_df) - len(found)} repeated"
        )
    view = copy.copy(dataset)
    view.df = subset_df
    view.item_pointers = subset_df.index.tolist()
    if transform is not None:
        view.transform = transform
    _reset_bad_input_state(view)
    return view


def _split_dataset(data_module, split: str):
    attribute = {"train": "ds_train", "val": "ds_val", "test": "ds_test"}[split]
    dataset = getattr(data_module, attribute, None)
    if dataset is None:
        raise RuntimeError(f"al_subset: the DataModule has no {split} dataset to use as the pool")
    return dataset


def _check_data_module(data_module) -> None:
    if getattr(data_module, "weights", None) is not None:
        raise RuntimeError("al_subset cannot restrict a DataModule that samples with per-case weights (weights=...)")
    if getattr(data_module, "num_train_samples", None) is not None:
        raise RuntimeError("al_subset cannot restrict a DataModule with a fixed num_train_samples")


def replace_training_set(data_module, dataset) -> Tuple[int, int]:
    """Make ``dataset`` the DataModule's training set, in place. Returns (workers before, workers after).

    In place, because the per-epoch prediction export and anything else holding the DataModule must see the
    new set too. The cached train loader is dropped (DataModule.train_dataloader() returns a cached loader,
    and Lightning asks for it again at every fit()). The worker count is capped for the small set (#574);
    the validation and test loaders are built first so they keep the configured count.
    """
    _check_data_module(data_module)
    old_loader = getattr(data_module, "_train_loader", None)
    if old_loader is not None:
        release_loader(old_loader)
    for split, method in (("ds_val", "val_dataloader"), ("ds_test", "test_dataloader")):
        if getattr(data_module, split, None) is not None and hasattr(data_module, method):
            getattr(data_module, method)()  # builds and caches the loader; no worker starts until it is iterated
    data_module.ds_train = dataset
    data_module._train_loader = None
    before = int(getattr(data_module, "num_workers", 0) or 0)
    after = cap_loader_workers(before, len(dataset))
    data_module.num_workers = after
    return before, after


# ---------------------------------------------------------------------------------------------------
# Model: received-global check, scoring, class weights, epochs
# ---------------------------------------------------------------------------------------------------


@contextmanager
def eval_mode(model):
    """model.eval() for the block; afterwards every submodule gets back its own training flag (#618)."""
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield model
    finally:
        for module, training in flags:
            module.training = training


def require_received_global(model, params: Mapping) -> int:
    """Raise unless ``model`` holds exactly the received global ``params``; return the number of tensors compared."""
    import torch

    local = model.state_dict()
    compared = 0
    for key, value in params.items():
        if key not in local or state_digest.is_ignored_key(key):
            continue
        mine = local[key].detach().cpu()
        theirs = torch.as_tensor(value).detach().cpu()
        if mine.shape != theirs.shape or not torch.equal(mine, theirs.to(mine.dtype)):
            raise RuntimeError(
                f"al_subset: at before_fit the model does not hold the received global (first difference at "
                f"'{key}'). Scoring the pool now would rank it with the wrong weights. NVFlare's FLCallback loads "
                f"the global on the round's first validate(); was the validation skipped?"
            )
        compared += 1
    if compared == 0:
        raise RuntimeError("al_subset: the received global shares no parameter names with the model")
    return compared


def _scoring_worker_init(_worker_id: int) -> None:
    # Same thread cap the DataModule's workers use (ODELIA_THREADS_PER_WORKER): 16 workers x all cores each
    # would oversubscribe the node.
    import torch

    threads = max(1, int(os.environ.get("ODELIA_THREADS_PER_WORKER", "1")))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)


def score_pool(model, dataset, labels: Mapping[str, Optional[int]], device=None, num_workers: int = 0):
    """Class probabilities of every readable case in ``dataset``: ([(uid, probs, label)], unusable UIDs).

    Eval mode, no_grad, fp32, batch 1, in dataset order. The dataset's fallback for an unreadable input
    returns the neighbouring case under the neighbour's UID; positions whose UID is not the expected one are
    recorded as unusable instead of being scored twice.
    """
    import torch
    from torch.utils.data import DataLoader

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    expected = [str(dataset.df.loc[pointer, "UID"]) for pointer in dataset.item_pointers]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=_scoring_worker_init if num_workers > 0 else None,
    )
    to_probs = getattr(model, "logits2probabilities", None) or (lambda logits: torch.softmax(logits, dim=1))
    rows, unusable = [], []
    model.to(device)
    try:
        with eval_mode(model), torch.no_grad():
            for position, batch in enumerate(loader):
                uid = batch["uid"][0]
                uid = uid if isinstance(uid, str) else str(uid)
                if position >= len(expected) or uid != expected[position]:
                    if position < len(expected):
                        unusable.append(expected[position])
                    continue
                probs = to_probs(model(batch["source"].to(device))).detach().float().cpu()[0].tolist()
                rows.append((uid, probs, labels.get(uid)))
    finally:
        release_loader(loader)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, unusable


def class_weights_from_labels(labels: Sequence[Optional[int]], num_classes: int) -> List[float]:
    """n / (num_classes * count_k) per class, 1.0 for a class absent from ``labels``.

    The formula threedcnn_ptl.set_up_data_module applies to the full training split.
    """
    counts = al_selection.class_counts(labels, num_classes)
    n = sum(counts.values())
    return [n / (num_classes * counts[k]) if counts.get(k, 0) > 0 else 1.0 for k in range(num_classes)]


def apply_class_weights(model, labels: Sequence[Optional[int]], num_classes: Optional[int], policy: str, log=None):
    """Set the model's ``_class_weight`` buffer per ``policy``; return the weights now in use (or None)."""
    import torch

    buffer = getattr(model, "_class_weight", None)
    if buffer is None:
        (log or _LOG).warning("al_subset: the model has no _class_weight buffer; the loss stays unweighted")
        return None
    if policy == CLASS_WEIGHTS_KEEP:
        return [float(w) for w in buffer.detach().cpu().tolist()]
    if not num_classes:
        raise RuntimeError("al_subset: number of classes unknown; cannot recompute class weights")
    weights = torch.tensor(class_weights_from_labels(labels, num_classes), dtype=buffer.dtype)
    if tuple(weights.shape) != tuple(buffer.shape):
        raise RuntimeError(
            f"al_subset: class-weight buffer has shape {tuple(buffer.shape)}, expected {tuple(weights.shape)}"
        )
    with torch.no_grad():
        buffer.copy_(weights.to(buffer.device))  # in place: the registered buffer stays the one the loss reads
    return [float(w) for w in weights.tolist()]


def find_fl_callback(trainer):
    for callback in getattr(trainer, "callbacks", None) or []:
        if type(callback).__name__ == "FLCallback":
            return callback
    return None


def set_epochs_per_round(trainer, epochs: int) -> None:
    """Train ``epochs`` local epochs in this round and in every later one.

    FLCallback (nvflare/app_opt/lightning/api.py) records max_epochs_per_round from trainer.max_epochs at the
    end of its first round and afterwards sets fit_loop.max_epochs = max_epochs_per_round + epochs done so
    far. Setting both here works before round 0 ends (nothing recorded yet) and after it.
    """
    trainer.fit_loop.max_epochs = int(trainer.current_epoch) + int(epochs)
    fl_callback = find_fl_callback(trainer)
    if fl_callback is not None and hasattr(fl_callback, "max_epochs_per_round"):
        fl_callback.max_epochs_per_round = int(epochs)


def _num_steps_key() -> str:
    try:
        from nvflare.app_common.abstract.fl_model import MetaKey

        return MetaKey.NUM_STEPS_CURRENT_ROUND
    except ImportError:
        return NUM_STEPS_KEY


def set_round_steps_meta(pl_module, steps: int) -> None:
    """Make FLCallback send ``steps`` as NUM_STEPS_CURRENT_ROUND (the update's aggregation weight).

    A copy of the module's __fl_meta__ dict gets the key (other entries are kept), as dp_noise does it.
    """
    meta = getattr(pl_module, FL_META_ATTRIBUTE, None)
    meta = dict(meta) if isinstance(meta, dict) else {}
    meta[_num_steps_key()] = int(steps)
    setattr(pl_module, FL_META_ATTRIBUTE, meta)


def round_steps_callback(ctx=None, log=None):
    """Lightning callback that reports the optimizer steps of the round's fit as the aggregation weight (SP-1).

    FLCallback otherwise sends trainer.estimated_stepping_batches: the current train loader's optimizer steps
    per epoch times the fit loop's max_epochs, which FLCallback keeps cumulative across rounds. That is
    proportional to a site's work only while its loader and epochs never change. A selection in round s > 0
    changes both: from then on every site trains a subset of the same size, yet the number still counts the
    s rounds of weighted epochs trained on the full split (different at every site), now multiplied by the
    subset's steps per epoch. InTimeAccumulateWeightedAggregator then weighted sites that did identical work
    differently. trainer.global_step counts optimizer steps, the unit of estimated_stepping_batches (gradient
    accumulation included), so its difference across one fit is the work of this round, comparable between
    sites; in rounds where every site's history is alike it gives the same relative weights as before.

    Inserted before FLCallback (round_hooks), so on_train_end runs before the model is sent. dp_noise's
    callback, later in the list, still overrides the key with ODELIA_DP_EQUAL_WEIGHTS=1; with that variable
    set the hook does not install this callback at all, so no hook order can undo the equal weights.
    """
    from pytorch_lightning.callbacks import Callback  # deferred: this module stays light to import

    log = log or _LOG

    class ALRoundStepsCallback(Callback):
        def __init__(self):
            super().__init__()
            self.start_step: Optional[int] = None
            self.reported: List[int] = []  # NUM_STEPS_CURRENT_ROUND of every fit, in order

        def on_train_start(self, trainer, pl_module) -> None:
            self.start_step = int(trainer.global_step)

        def on_train_end(self, trainer, pl_module) -> None:
            if self.start_step is None:
                return
            steps = int(trainer.global_step) - self.start_step
            self.start_step = None
            if steps < 1:
                # Weight 0 for every site would divide by zero in the aggregator; 1 = the smallest real weight.
                log.warning(f"al_subset: this round's fit took {steps} optimizer steps; reporting weight 1")
                steps = 1
            set_round_steps_meta(pl_module, steps)
            self.reported.append(steps)
            round_text = f"round {ctx.current_round}" if ctx is not None else "this round"
            log.info(f"al_subset: {round_text} aggregation weight NUM_STEPS_CURRENT_ROUND={steps} (optimizer steps "
                     f"trained in the round)")

    return ALRoundStepsCallback()


def _default_epochs_fn():
    import threedcnn_ptl  # already imported by main.py; deferred so this module stays light to import

    return threedcnn_ptl.compute_weighted_epochs


# ---------------------------------------------------------------------------------------------------
# Records and metrics
# ---------------------------------------------------------------------------------------------------


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def record_dir(ctx) -> Path:
    """Site-local directory for records that contain UIDs: $SCRATCH_DIR/odelia_al/<job>/<site>/.

    Outside the run directory on purpose: live_sync uploads the whole run directory when it stops.
    """
    scratch = (ctx.env_vars or {}).get("scratch_dir") or os.environ.get("SCRATCH_DIR")
    if not scratch:
        raise RuntimeError("al_subset: SCRATCH_DIR is not set; nowhere site-local to keep the selection record")
    run = ctx.job_id or (Path(ctx.path_run_dir).name if ctx.path_run_dir else "unknown_run")
    return Path(scratch) / RECORD_DIR / run / (ctx.site_name or "unknown_site")


def selection_metrics(summary: Mapping) -> Dict[str, float]:
    """The aggregates streamed to the coordinator. Never UIDs."""
    metrics = {
        "al/n_selected": int(summary["n_selected"]),
        "al/selection_digest": int(summary["selection_digest_number"]),
    }
    for k, count in sorted(summary["class_counts"].items(), key=lambda kv: int(kv[0])):
        metrics[f"al/class_count_{int(k)}"] = int(count)
    if summary.get("mean_entropy") is not None:
        metrics["al/mean_entropy"] = float(summary["mean_entropy"])
    return metrics


def stream_metrics(trainer, metrics: Mapping[str, float], log=None) -> int:
    """Log ``metrics`` through every Lightning logger (ClientLogger streams to the coordinator). Best effort."""
    sent = 0
    step = int(getattr(trainer, "global_step", 0) or 0)
    for logger in getattr(trainer, "loggers", None) or []:
        try:
            logger.log_metrics(dict(metrics), step=step)
            sent += 1
        except Exception as exc:  # a metric stream must not cost the round (as for ClientLogger itself)
            (log or _LOG).warning(f"al_subset: could not log selection metrics to {type(logger).__name__}: {exc}")
    return sent


# ---------------------------------------------------------------------------------------------------
# The selection
# ---------------------------------------------------------------------------------------------------


def apply_selection(
    ctx, config: ALConfig, global_digest: Optional[str] = None, epochs_fn=None, device=None, log=None
) -> Dict:
    """Select at this round and restrict training to the selection (module docstring, steps 1-4).

    Returns the UID-free summary (also written to the run directory and kept in ctx.state["al_subset"]).
    """
    log = log or ctx.logger or _LOG
    started = time.perf_counter()
    data_module = ctx.data_module
    _check_data_module(data_module)
    pool = _split_dataset(data_module, config.pool_split)
    labels = labels_by_uid(pool)
    if not labels:
        raise RuntimeError(f"al_subset: the pool ({config.pool_split} split) is empty")
    num_classes = num_classes_of(pool, ctx.model)
    params = getattr(ctx.input_model, "params", None) or {}
    if global_digest is None and params:
        global_digest = state_digest.state_digest(params)
    directory = record_dir(ctx)

    unusable: List[str] = []
    scored = None
    if config.strategy == al_selection.STRATEGY_INCLUDE_LIST:
        missing = [uid for uid in config.include_uids if uid not in labels]
        if missing:
            _write_json(directory / INCLUDE_LIST_ERROR_FILE, {"pool_split": config.pool_split, "not_in_pool": missing})
            raise RuntimeError(
                f"{ENV_INCLUDE_LIST}: {len(missing)} of {len(config.include_uids)} UIDs are not in this site's "
                f"{config.pool_split} split (fold/data version?). Site-local list: "
                f"{directory / INCLUDE_LIST_ERROR_FILE}"
            )
        pool_rows = [(uid, None, label) for uid, label in labels.items()]
        uids = list(config.include_uids)
        selection = al_selection.Selection(
            strategy=config.strategy,
            budget=config.budget,
            seed=config.al_seed,
            uids=uids,
            stats=al_selection.selection_statistics(pool_rows, uids, num_classes),
        )
    else:
        require_received_global(ctx.model, params)
        workers = cap_loader_workers(int(getattr(data_module, "num_workers", 0) or 0), len(labels))
        scored, unusable = score_pool(ctx.model, eval_view(pool, data_module.ds_val.transform), labels, device, workers)
        if not scored:
            raise RuntimeError(f"al_subset: no readable case in the pool ({config.pool_split} split)")
        selection = al_selection.select(scored, config.budget, config.strategy, config.al_seed, num_classes)
    stats = selection.stats

    training_set = restrict_dataset(pool, selection.uids, transform=data_module.ds_train.transform)
    # A training set, also when the pool is the test split: an unreadable case is resampled, as in ds_train.
    training_set.resample_bad_inputs = True
    workers_before, workers_after = replace_training_set(data_module, training_set)
    selected_labels = [labels[uid] for uid in selection.uids]
    class_weights = apply_class_weights(ctx.model, selected_labels, num_classes, config.class_weights, log)
    epochs = int((epochs_fn or _default_epochs_fn())(len(training_set), ctx.site_name or ""))
    set_epochs_per_round(ctx.trainer, epochs)
    elapsed = time.perf_counter() - started

    summary = {
        "schema": "odelia_al_selection_summary/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site": ctx.site_name,
        "job_id": ctx.job_id,
        "round": ctx.current_round,
        "strategy": config.strategy,
        "budget": config.budget,
        "seed": config.al_seed,
        "pool_split": config.pool_split,
        "pool_size": stats["pool_size"],
        "pool_class_counts": stats["pool_class_counts"],
        "pool_unusable": len(unusable),
        "n_selected": stats["n_selected"],
        "class_counts": stats["class_counts"],
        "mean_entropy": stats["mean_entropy"],
        "selection_digest": stats["digest"],
        "selection_digest_number": stats["digest_number"],
        "global_digest": global_digest,
        "include_list_sha256": config.include_list_sha256,
        "class_weight_policy": config.class_weights,
        "class_weights": class_weights,
        "epochs_per_round": epochs,
        "loader_workers": {"configured": workers_before, "training_subset": workers_after},
        "seconds": round(elapsed, 2),
        "record": str(directory / RECORD_FILE),
    }

    record = dict(summary)
    record["schema"] = "odelia_al_selection_record/1"
    record["notice"] = "Site-local: contains case identifiers. Do not upload or share."
    record["include_list_path"] = config.include_list_path
    record["selected_uids"] = list(selection.uids)
    record["unusable_pool_uids"] = unusable
    rank = {uid: i for i, uid in enumerate(selection.uids)}
    if scored is not None:
        record["pool"] = [
            {
                "uid": uid,
                "label": label,
                "probs": probs,
                "entropy": al_selection.entropy(probs),
                "margin_uncertainty": al_selection.margin_uncertainty(probs),
                "selected": uid in rank,
                "rank": rank.get(uid),
            }
            for uid, probs, label in sorted(scored)
        ]
    else:
        record["pool"] = [
            {"uid": uid, "label": labels[uid], "selected": uid in rank, "rank": rank.get(uid)} for uid in sorted(labels)
        ]
    _write_json(directory / RECORD_FILE, record)
    _write_text(
        directory / UID_LIST_FILE,
        f"# {config.strategy} budget={config.budget} seed={config.al_seed} pool={config.pool_split} "
        f"job={ctx.job_id} site={ctx.site_name} digest={stats['digest']}\n"
        + "".join(f"{uid}\n" for uid in selection.uids),
    )
    if ctx.path_run_dir:
        _write_json(Path(ctx.path_run_dir) / SUMMARY_FILE, summary)

    stream_metrics(ctx.trainer, selection_metrics(summary), log)
    ctx.state.setdefault("al_subset", {})["selection"] = summary

    if config.budget > stats["pool_size"]:
        log.warning(
            f"al_subset: budget {config.budget} exceeds the pool ({stats['pool_size']} cases); the whole pool is used"
        )
    if unusable:
        log.warning(
            f"al_subset: {len(unusable)} unreadable pool case(s) left out of the selection "
            f"(UIDs in the site-local record)"
        )
    if config.pool_split != "train":
        log.warning(
            f"al_subset: the pool is the {config.pool_split} split, so {stats['n_selected']} of its cases are now "
            f"training cases; local {config.pool_split} metrics of this run are not held out"
        )
    mean_entropy = "n/a" if stats["mean_entropy"] is None else f"{stats['mean_entropy']:.4f}"
    log.info(
        f"al_subset: round {ctx.current_round} selection {config.strategy} (budget {config.budget}, seed "
        f"{config.al_seed}, pool {config.pool_split}): {stats['n_selected']} of {stats['pool_size']} cases, classes "
        f"{stats['class_counts']}, mean entropy {mean_entropy}, digest {stats['digest'][:12]}. Training restricted to "
        f"them: {epochs} epoch(s)/round, class weights {class_weights} ({config.class_weights}), loader workers "
        f"{workers_before}->{workers_after}, {elapsed:.1f}s. Site-local record: {directory / RECORD_FILE}"
    )
    return summary


# ---------------------------------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------------------------------


class ALSubsetHook(RoundHook):
    name = "al_subset"

    def __init__(self, config: ALConfig, logger=None, epochs_fn=None, device=None):
        self.config = config
        self.logger = logger or _LOG
        self.epochs_fn = epochs_fn  # tests inject; default threedcnn_ptl.compute_weighted_epochs
        self.device = device
        self.first_round: Optional[int] = None
        self.summary: Optional[Dict] = None
        self.steps_callback = None

    def _log(self, ctx):
        return ctx.logger or self.logger

    def lightning_callbacks(self, ctx) -> list:
        if self.config.dp_equal_weights:
            self._log(ctx).info(f"al_subset: {ENV_DP_EQUAL_WEIGHTS}=1, so dp_noise sets the aggregation weight")
            return []
        self.steps_callback = round_steps_callback(ctx, self._log(ctx))
        return [self.steps_callback]

    def after_receive(self, ctx) -> None:
        if self.first_round is None:
            self.first_round = ctx.current_round
            self._check_round_plan(ctx)

    def before_fit(self, ctx) -> None:
        # No before_fit is called in the evaluation-only final round (main.py); the flag is checked anyway so
        # the selection can never run on the final global.
        if self.summary is not None or ctx.is_eval_only_round:
            return
        if ctx.current_round is None or ctx.current_round < self.config.select_round:
            return
        if ctx.current_round > self.config.select_round:
            raise RuntimeError(
                f"al_subset: selection round {self.config.select_round} passed without a selection (this process "
                f"first saw round {self.first_round}); training on the full split would not be the experiment"
            )
        self.summary = apply_selection(ctx, self.config, _verified_round0_digest(ctx), self.epochs_fn, self.device,
                                       self._log(ctx))

    def _check_round_plan(self, ctx) -> None:
        if ctx.current_round is not None and ctx.current_round > self.config.select_round:
            raise RuntimeError(
                f"al_subset: this process started in round {ctx.current_round}, after the selection round "
                f"{self.config.select_round}"
            )
        total = ctx.total_rounds
        if total:
            training_rounds = total - (1 if self.config.final_eval_round else 0)
            if self.config.select_round >= training_rounds:
                raise RuntimeError(
                    f"{ENV_SELECT_ROUND}={self.config.select_round}, but the job trains only {training_rounds} "
                    f"round(s) (0-based): the selection would never run"
                )


def _verified_round0_digest(ctx) -> Optional[str]:
    """The digest reproducibility.py verified for this round's global (round 0 only), so it is not hashed twice."""
    if ctx.current_round != 0:
        return None
    return (ctx.state.get(reproducibility.STATE_KEY) or {}).get("round0_global_digest")


def create_hook(env, logger):
    config = parse_config(env)
    if not config.selects:
        (logger or _LOG).info("al_subset is listed but none of its variables is set; not active")
        return None
    return ALSubsetHook(config, logger)
