#!/usr/bin/env python3
"""Build a consortium job directory for the ODELIA swarm in one command.

Every consortium run so far was staged by hand: copy a job template, strip the test
folders, run patch_warm_start_job.py, then edit MODEL_NAME in two places (#541), the
deploy_map (#543) and the job name, and record the code version in a notebook. Each
manual step has failed at least once. This tool does all of it, refuses requests that
cannot work, validates the result and records what was built.

What it does, in order:

1. Takes the base job (``ODELIA_ternary_classification`` or
   ``challenge_1DivideAndConquer``) from a source tree: this checkout (``repo``), a git
   ref (``git:v1.9.0``), a release image (``image:jefftud/odelia:1.9.0``) or any
   MediSwarm tree (``dir:<path>``). The copy resolves the ``app/custom`` symlink (the
   job code travels with the job) and drops ``app/tests`` and ``app/scripts``.
2. Applies the strict roster with ``patch_warm_start_job.py``: every site mandatory,
   ``min_clients = configure_min_clients = min_responses_required = number of sites``
   (F8 in docs/SWARM_FAILURE_MODES.md), ``participating_clients`` and ``num_rounds``.
3. Sets ``meta.conf``: the job name and ``deploy_map = ["server", <sites>]``.
4. Pins ``MODEL_NAME`` and ``CONFIG`` in the launcher line AND in the persistor
   (``model_name``, ``data_config``; for the MST-family factory also the model's own
   ``model_name`` arg). The launcher's KEY=VALUE prefix reaches only the training
   subprocess; the persistor runs in the client job process, which sees the container's
   environment instead, so a launcher-only pin lets the starting client seed round 0
   with the site's default model (#541).
5. Writes the feature switches of the 1.9.0 contract into the launcher line
   (``ODELIA_FINAL_EVAL_ROUND``, ``ODELIA_RETURN_PER_CASE``, ``ODELIA_MS6_*``,
   ``ODELIA_DP_*``, ``ODELIA_AL_*``, ...) and derives ``ODELIA_ROUND_HOOKS`` from them.
6. Configures the warm start (fresh, continue from a mirror, or a named source file at
   the starting client) and gives the experiment its own mirror path, so one run never
   overwrites the global another run continues from.
7. Validates the job (configs parse, pins agree, roster consistent, deploy_map has the
   server, no VHIO_1 unless listed, the job's own code supports every switch), then
   writes ``job_manifest.json`` (git commit, dirty flag, all flags, sha256 of every
   custom file, expected image).

Round hook order. ``ODELIA_ROUND_HOOKS`` lists modules in a fixed order:
``al_subset, ms6_capture, dp_noise``. Hooks run in list order, and each hook's Lightning
callbacks are inserted directly before FLCallback in that same order (contract 3.1), so:

* ``al_subset`` first: it replaces the training set (``ctx.data_module``) before any
  other hook or the trainer reads it, so MS6 metadata and the step count that weights
  the aggregation both describe the subset that was actually trained.
* ``ms6_capture`` before ``dp_noise``: its callback sees the site's own model W_r at the
  end of training, before the update is clipped and noised.
* ``dp_noise`` last: clipping and noise must be the final transformation of the
  outgoing weights. Its callback then sits directly before FLCallback, so no other hook
  can change the update after the noise was added (which would void the accounting).

Exit status: 0 on success, 2 for a refused request or a job that failed validation.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
BUILDER_REPO_ROOT = HERE.parents[1]


def _load_sibling(name: str):
    """Import a module that sits next to this script, however this script was loaded."""
    module_name = f"_mediswarm_admin_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses look their module up in sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


hocon = _load_sibling("hocon_lite")
patcher = _load_sibling("patch_warm_start_job")

# ── Constants of the 1.9.0 contract ───────────────────────────────────────

# The seven sites of the October 2026 benchmark; VHIO_1 is excluded until its node can
# read all of its data (docs/ODELIA_RUN_2026-10_MST.md).
DEFAULT_SITES = ("CAM_1", "MHA_1", "RSH_1", "RUMC_1", "UKA_1", "UMCU_1", "USZ_1")

MODEL_FACTORY_MST = "models.models_config.create_model"
BASE_JOBS: Dict[str, Dict[str, Any]] = {
    "ODELIA_ternary_classification": {
        # create_model(model_name=...) picks the architecture, so it gets the pin too.
        "model_factory": MODEL_FACTORY_MST,
        "fixed_model": None,
    },
    "challenge_1DivideAndConquer": {
        # This factory always builds the 1DivideAndConquer network and takes no
        # model_name; any other model needs ODELIA_ternary_classification.
        "model_factory": "models.challenge.1DivideAndConquer.model.create_model",
        "fixed_model": "1DivideAndConquer",
    },
}
CONFIG_CHOICES = ("unilateral", "unilateral_BD", "original")
AL_STRATEGIES = ("entropy", "random", "class_matched_random")
HOOK_ORDER = ("al_subset", "ms6_capture", "dp_noise")
HOOK_FOR_PREFIX = (
    ("ODELIA_AL_", "al_subset"),
    ("ODELIA_TRAIN_INCLUDE_LIST", "al_subset"),
    ("ODELIA_MS6_", "ms6_capture"),
    ("ODELIA_DP_", "dp_noise"),
)
CLIENT_STREAMER_ID = "ms6_global_streamer"
SERVER_ARCHIVE_ID = "ms6_global_archive"
LEGACY_MIRROR = "/scratch/mediswarm_latest_global.pt"
DEFAULT_IMAGE_REPO = "jefftud/odelia"
MANIFEST_NAME = "job_manifest.json"
MANIFEST_SCHEMA = "mediswarm.job_manifest/1"

CLIENT_CONF = Path("app/config/config_fed_client.conf")
SERVER_CONF = Path("app/config/config_fed_server.conf")
META_CONF = Path("meta.conf")

# Arguments PTFileModelPersistor accepts; WarmStartablePTFileModelPersistor forwards
# **kwargs to it, so these count as accepted when its __init__ has **kwargs.
PT_FILE_PERSISTOR_ARGS = {
    "exclude_vars",
    "model",
    "global_model_file_name",
    "best_global_model_file_name",
    "source_ckpt_file_full_name",
    "filter_id",
    "load_weights_only",
    "allow_numpy_conversion",
}

# Launcher variables owned by a dedicated flag. --env may not set them, so a job can
# never carry two different answers (e.g. --model MST with --env MODEL_NAME=...).
MANAGED_ENV = {
    "MODEL_NAME": "--model",
    "CONFIG": "--config",
    "KEY_METRIC": "the job template (it must match the model selector)",
    "ODELIA_ROUND_HOOKS": "the feature flags (and --extra-hook)",
    "ODELIA_FINAL_EVAL_ROUND": "--final-eval/--no-final-eval",
    "ODELIA_EXPORT_TRAIN_PREDICTIONS": "--export-train-predictions",
    "ODELIA_RETURN_PER_CASE": "--per-case",
    "ODELIA_SEED": "--seed",
    "ODELIA_WARM_START_SHA256": "--warm-start-digest",
    "ODELIA_MS6_CAPTURE_SITES": "--ms6-capture-sites",
    "ODELIA_MS6_CAPTURE_ROUNDS": "--ms6-capture-rounds",
    "ODELIA_MS6_CAPTURE_OPTIMIZER": "--ms6-capture-optimizer",
    "ODELIA_DP_SIGMA": "--dp-sigma",
    "ODELIA_DP_CLIP_NORM": "--dp-clip-norm",
    "ODELIA_DP_DELTA": "--dp-delta",
    "ODELIA_DP_TARGET_EPSILON": "--dp-target-epsilon",
    "ODELIA_DP_MEASURE_ONLY": "--dp-measure-only",
    "ODELIA_DP_EQUAL_WEIGHTS": "--dp-equal-weights",
    "ODELIA_AL_STRATEGY": "--al-strategy",
    "ODELIA_AL_BUDGET": "--al-budget",
    "ODELIA_AL_POOL_SPLIT": "--al-pool-split",
    "ODELIA_AL_SEED": "--al-seed",
    "ODELIA_AL_SELECT_ROUND": "--al-select-round",
    "ODELIA_TRAIN_INCLUDE_LIST": "--train-include-list",
}

# Variables read by plain job code (not by a hook module). If the job's code does not
# mention one of them, the switch would be silently ignored at every site, so the
# builder refuses (checked only for active values; a pinned default is harmless).
CODE_READ_ENV = {
    "ODELIA_FINAL_EVAL_ROUND": lambda v: v != "0",
    "ODELIA_EXPORT_TRAIN_PREDICTIONS": lambda v: True,
    "ODELIA_RETURN_PER_CASE": lambda v: True,
    "ODELIA_SEED": lambda v: True,
    "ODELIA_WARM_START_SHA256": lambda v: True,
    "ODELIA_ROUND_HOOKS": lambda v: bool(v),
}

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SITE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The launcher line is split with shlex and NVFlare substitutes {placeholders} in it,
# so values are limited to characters that need no quoting and are not braces.
ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_./,:@+%=-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PATH_RE = re.compile(r"^/[A-Za-z0-9_./@+%=,:-]+$")
LAUNCHER_TOKEN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# Same order grid and conversion as privacy_accounting.py (rdp_gaussian/rdp_to_dp/
# compute_epsilon). Used only when that module cannot be imported here (for example a
# job built from a tree whose privacy_accounting.py needs torch).
RDP_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


class BuildError(Exception):
    """A request the builder refuses, or a job that failed validation."""


# ── Small helpers ─────────────────────────────────────────────────────────


def _fmt_number(value: float) -> str:
    """Shortest text that parses back to the same float (2.4, 1e-05, 10.0)."""
    return repr(float(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(cmd: Sequence[str], what: str, **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(list(cmd), check=True, capture_output=True, **kwargs)
    except FileNotFoundError as exc:
        raise BuildError(f"{what}: {cmd[0]} is not installed") from exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        raise BuildError(f"{what} failed: {' '.join(cmd)}\n{err.strip()}") from exc


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_sites(value: str) -> Tuple[str, ...]:
    try:
        sites = patcher.parse_strict_clients(value)
    except ValueError as exc:
        raise BuildError(f"--sites: {exc}") from exc
    bad = [s for s in sites if not SITE_RE.match(s)]
    if bad:
        raise BuildError(f"--sites: invalid site name(s) {bad}")
    return tuple(sites)


def parse_env_pairs(pairs: Sequence[str]) -> "OrderedDict[str, str]":
    env: "OrderedDict[str, str]" = OrderedDict()
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not ENV_KEY_RE.match(key):
            raise BuildError(f"--env {pair!r}: expected KEY=VALUE with KEY like ODELIA_FOO")
        if key in MANAGED_ENV:
            raise BuildError(f"--env {key}: set by {MANAGED_ENV[key]}, not by --env")
        if not ENV_VALUE_RE.match(value):
            raise BuildError(
                f"--env {key}: value {value!r} must be non-empty and use only letters, digits and _./,:@+%=- "
                "(the launcher line is split on whitespace and NVFlare substitutes {...})"
            )
        if key in env:
            raise BuildError(f"--env {key} given twice")
        env[key] = value
    return env


def derive_hooks(env: Dict[str, str], extra: Sequence[str] = ()) -> List[str]:
    """Hook modules required by the launcher variables, in the contract's order.

    Extra modules (future features) go after al_subset and ms6_capture but before
    dp_noise, which must stay the last transformation of the outgoing update.
    """
    needed = {hook for key in env for prefix, hook in HOOK_FOR_PREFIX if key.startswith(prefix)}
    needed |= {hook for hook in extra if hook in HOOK_ORDER}
    head = [hook for hook in HOOK_ORDER if hook in needed and hook != "dp_noise"]
    middle: List[str] = []
    for hook in extra:
        if hook not in HOOK_ORDER and hook not in middle:
            middle.append(hook)
    tail = ["dp_noise"] if "dp_noise" in needed else []
    return head + middle + tail


# ── DP accounting ─────────────────────────────────────────────────────────


def local_epsilon(sigma: float, rounds: int, delta: float) -> float:
    """(epsilon, delta) after ``rounds`` Gaussian releases; copy of compute_epsilon."""
    if rounds <= 0:
        return 0.0
    if sigma <= 0:
        return float("inf")
    best = float("inf")
    for alpha in RDP_ALPHAS:
        if alpha <= 1:
            continue
        rdp = rounds * alpha / (2.0 * sigma**2)
        eps = rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        best = min(best, eps)
    return best


def local_max_rounds(sigma: float, target_epsilon: float, delta: float, limit: int = 100000) -> int:
    """Largest number of rounds whose epsilon stays within the target (epsilon grows with rounds)."""
    if local_epsilon(sigma, 1, delta) > target_epsilon:
        return 0
    lo, hi = 1, 1
    while hi < limit and local_epsilon(sigma, hi * 2, delta) <= target_epsilon:
        hi *= 2
    hi = min(hi * 2, limit)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if local_epsilon(sigma, mid, delta) <= target_epsilon:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _import_job_module(custom_dir: Path, name: str):
    path = custom_dir / f"{name}.py"
    if not path.is_file():
        return None, f"{path.name} not in the job code"
    module_name = f"_mediswarm_job_{name}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # the job module may need packages this host lacks
        sys.modules.pop(module_name, None)
        return None, f"import failed ({type(exc).__name__}: {exc})"
    return module, ""


def _call_max_rounds(fn, sigma: float, target: float, delta: float) -> Optional[int]:
    """Call the job's max_rounds_for_budget by parameter name (its signature is not fixed)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    kwargs = {}
    for pname, param in params.items():
        lname = pname.lower()
        if "sigma" in lname or "noise" in lname:
            kwargs[pname] = sigma
        elif "delta" in lname:
            kwargs[pname] = delta
        elif "eps" in lname or "target" in lname or "budget" in lname:
            kwargs[pname] = target
        elif param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return None
    if len(kwargs) < 3:
        return None
    result = fn(**kwargs)
    return int(result)


def dp_budget(plan: "Plan", custom_dir: Optional[Path]) -> Dict[str, Any]:
    """Epsilon after the training rounds and, with a target, the round cap.

    The cap comes from the job's own privacy_accounting.max_rounds_for_budget when it
    can be imported (the code the sites will run); otherwise from the local copy of the
    same RDP formula. Only training rounds release an update: the evaluation-only final
    round sends the received global back unchanged, so it spends no budget.
    """
    dp = plan.dp
    sigma = dp.get("sigma")
    if sigma is None:
        return {}
    delta = dp["delta"]
    rounds = plan.training_rounds
    report: Dict[str, Any] = {
        "sigma": sigma,
        "clip_norm": dp.get("clip_norm"),
        "delta": delta,
        "training_rounds": rounds,
        "epsilon_after_training_rounds": round(local_epsilon(sigma, rounds, delta), 4),
        "guarantee": "client-level (epsilon, delta)-DP by output perturbation, add/remove-site adjacency; "
        "not record-level",
    }
    target = dp.get("target_epsilon")
    if target is None:
        return report
    local_cap = local_max_rounds(sigma, target, delta)
    cap, source = local_cap, "builder copy of the RDP Gaussian formula in privacy_accounting.py"
    if custom_dir is not None:
        module, why = _import_job_module(custom_dir, "privacy_accounting")
        fn = getattr(module, "max_rounds_for_budget", None) if module else None
        if fn is None:
            source += f" (job's privacy_accounting.max_rounds_for_budget unavailable: {why or 'not defined'})"
        else:
            job_cap = _call_max_rounds(fn, sigma, target, delta)
            if job_cap is None:
                source += " (could not map the arguments of the job's max_rounds_for_budget)"
            else:
                cap, source = job_cap, "privacy_accounting.max_rounds_for_budget (job code)"
                if job_cap != local_cap:
                    plan.warnings.append(
                        f"DP round cap: the job's max_rounds_for_budget says {job_cap}, the builder's copy says "
                        f"{local_cap}; using the job's value"
                    )
    report.update({"target_epsilon": target, "max_training_rounds": cap, "accountant": source})
    if rounds > cap:
        raise BuildError(
            f"DP budget: {rounds} training rounds exceed the budget epsilon <= {target} at sigma={sigma}, "
            f"delta={delta}: epsilon after {rounds} rounds = {local_epsilon(sigma, rounds, delta):.3f}; "
            f"at most {cap} rounds fit ({source})"
        )
    return report


# ── The request, normalised ───────────────────────────────────────────────


@dataclass
class Plan:
    name: str
    base_job: str
    sites: Tuple[str, ...]
    model: str
    config: str
    training_rounds: int
    final_eval: bool
    starting_client: Optional[str]
    warm_start: str  # fresh | continue | source
    persistor_mode: str  # warm_start_mode written to the persistor
    source_ckpt: str
    source_sha256: Optional[str]
    warm_digest: Optional[str]
    mirror_path: str
    launcher_env: "OrderedDict[str, str]"
    hooks: List[str]
    archive_globals: bool
    archive_root: Optional[str]
    dp: Dict[str, Any]
    per_case: str
    warnings: List[str] = field(default_factory=list)

    @property
    def total_rounds(self) -> int:
        return self.training_rounds + (1 if self.final_eval else 0)


def _parse_rounds_list(value: str, training_rounds: int, flag: str) -> str:
    if value.strip().lower() == "all":
        return "all"
    rounds = set()
    for part in value.split(","):
        part = part.strip()
        if not re.fullmatch(r"\d+", part):
            raise BuildError(f"{flag}: {part!r} is not a 0-based round number (or use 'all')")
        rounds.add(int(part))
    bad = sorted(r for r in rounds if r >= training_rounds)
    if bad:
        raise BuildError(
            f"{flag}: round(s) {bad} do not exist; training rounds are 0..{training_rounds - 1} "
            "(the evaluation-only final round captures nothing)"
        )
    return ",".join(str(r) for r in sorted(rounds))


def _check_path(value: str, flag: str) -> str:
    if not PATH_RE.match(value or ""):
        raise BuildError(f"{flag}: {value!r} must be an absolute path without spaces or quotes")
    return value


def make_plan(args: argparse.Namespace) -> Plan:
    """Validate the flags against each other and decide every value the job gets."""
    warnings: List[str] = []
    if not NAME_RE.match(args.name or ""):
        raise BuildError(f"--name {args.name!r}: use letters, digits and _.- (it names the folder and the mirror)")
    base = BASE_JOBS[args.base_job]
    sites = parse_sites(args.sites)
    if args.rounds is None or args.rounds < 1:
        raise BuildError("--rounds must be a positive number of training rounds")
    rounds = args.rounds

    model = args.model
    if base["fixed_model"]:
        if model and model != base["fixed_model"]:
            raise BuildError(
                f"--model {model}: {args.base_job} always builds {base['fixed_model']}; "
                "use --base-job ODELIA_ternary_classification for other models"
            )
        model = base["fixed_model"]
    if not model:
        raise BuildError(f"--model is required with --base-job {args.base_job} (e.g. MST, MST_SAMMed2D)")
    if not ENV_VALUE_RE.match(model):
        raise BuildError(f"--model {model!r}: invalid model name")

    starting = args.starting_client
    if starting is not None and starting not in sites:
        raise BuildError(f"--starting-client {starting} is not in --sites")

    # Warm start.
    warm = (args.warm_start or "fresh").strip()
    source_sha = None
    if warm == "fresh":
        persistor_mode = "fresh"
    elif warm == "continue":
        persistor_mode = "require"
        if not args.mirror_path:
            raise BuildError(
                "--warm-start continue reads the mirror an earlier run wrote: pass that path with --mirror-path "
                "(the per-name default only exists if an earlier job had the same name)"
            )
    elif warm.startswith("source:"):
        rest = warm[len("source:") :]
        path, sep, sha = rest.rpartition(":")
        if not sep or not path:
            raise BuildError("--warm-start source:<path at the starting client>:<sha256 of that file>")
        _check_path(path, "--warm-start source path")
        sha = sha.strip().lower()
        if not SHA256_RE.match(sha):
            raise BuildError(f"--warm-start source: {sha!r} is not a sha256 (64 hex digits)")
        if starting is None:
            raise BuildError(
                "--warm-start source needs --starting-client: only the starting client's persistor seeds round 0, "
                "so it must be the site that holds the file"
            )
        if not args.warm_start_digest:
            raise BuildError(
                "--warm-start source needs --warm-start-digest (state digest of the checkpoint, contract 3.4): every "
                "site checks the round-0 global against it, which is what catches a missing or different file"
            )
        # 'auto' at every site: the starting client loads the file (verified by
        # source_sha256), the others hold no copy and initialise fresh, which the round-0
        # global then overwrites. 'require' would stop every site without the file.
        persistor_mode, source_sha, warm = "auto", sha, "source"
        source_ckpt = path
    else:
        raise BuildError(f"--warm-start {warm!r}: expected fresh, continue or source:<path>:<sha256>")

    digest = None
    if args.warm_start_digest:
        digest = args.warm_start_digest.strip().lower()
        if not SHA256_RE.match(digest):
            raise BuildError("--warm-start-digest must be 64 hex digits (state_digest.py output)")
        if warm == "fresh":
            raise BuildError("--warm-start-digest makes no sense with --warm-start fresh (round 0 is a new model)")

    mirror = args.mirror_path or f"/scratch/mediswarm_latest_global_{args.name}.pt"
    _check_path(mirror, "--mirror-path")
    if not mirror.endswith(".pt"):
        raise BuildError(f"--mirror-path {mirror}: expected a .pt file")
    if mirror == LEGACY_MIRROR:
        warnings.append(
            f"mirror {LEGACY_MIRROR} is shared by every job built before 1.9.0; another run may overwrite it"
        )
    if warm == "source":
        if source_ckpt == mirror:
            raise BuildError(
                "--warm-start source path equals the mirror path: this run's result would overwrite its own source"
            )
    else:
        source_ckpt = mirror

    # Feature switches -> launcher variables.
    env: "OrderedDict[str, str]" = OrderedDict()
    env["MODEL_NAME"] = model
    env["CONFIG"] = args.config
    env["ODELIA_FINAL_EVAL_ROUND"] = "1" if args.final_eval else "0"

    dp: Dict[str, Any] = {}
    dp_active = any(
        v is not None for v in (args.dp_sigma, args.dp_clip_norm, args.dp_delta, args.dp_target_epsilon)
    ) or (args.dp_measure_only or args.dp_equal_weights)
    if dp_active:
        if args.dp_sigma is None and not args.dp_measure_only:
            raise BuildError("DP options need --dp-sigma (noise) or --dp-measure-only (calibration run)")
        if args.dp_sigma is not None:
            if args.dp_sigma <= 0:
                raise BuildError("--dp-sigma must be positive")
            if args.dp_clip_norm is None:
                raise BuildError("--dp-sigma needs --dp-clip-norm: noise is calibrated to the clip norm C")
        if args.dp_clip_norm is not None and args.dp_clip_norm <= 0:
            raise BuildError("--dp-clip-norm must be positive")
        if args.dp_target_epsilon is not None:
            if args.dp_sigma is None:
                raise BuildError("--dp-target-epsilon needs --dp-sigma")
            if args.dp_measure_only:
                raise BuildError("--dp-target-epsilon with --dp-measure-only: a run without noise has no budget")
            if args.dp_target_epsilon <= 0:
                raise BuildError("--dp-target-epsilon must be positive")
        if args.dp_delta is not None:
            if args.dp_sigma is None:
                raise BuildError("--dp-delta needs --dp-sigma")
            if not 0 < args.dp_delta < 1:
                raise BuildError("--dp-delta must be between 0 and 1")
        if args.per_case == "site":
            raise BuildError(
                "--per-case site with DP: per-case rows would leave the sites without noise; DP jobs force "
                "ODELIA_RETURN_PER_CASE=0"
            )
        if warm == "continue":
            raise BuildError(
                "DP with --warm-start continue: the accountant counts this job's rounds only, so continuing a chain "
                "would under-state epsilon; start fresh or from a named source"
            )
        if args.dp_sigma is not None:
            dp = {
                "sigma": float(args.dp_sigma),
                "clip_norm": float(args.dp_clip_norm),
                "delta": float(args.dp_delta if args.dp_delta is not None else 1e-5),
                "target_epsilon": float(args.dp_target_epsilon) if args.dp_target_epsilon is not None else None,
            }
            env["ODELIA_DP_SIGMA"] = _fmt_number(dp["sigma"])
            env["ODELIA_DP_CLIP_NORM"] = _fmt_number(dp["clip_norm"])
            env["ODELIA_DP_DELTA"] = _fmt_number(dp["delta"])
            if dp["target_epsilon"] is not None:
                env["ODELIA_DP_TARGET_EPSILON"] = _fmt_number(dp["target_epsilon"])
        elif args.dp_clip_norm is not None:
            env["ODELIA_DP_CLIP_NORM"] = _fmt_number(args.dp_clip_norm)
        if args.dp_measure_only:
            env["ODELIA_DP_MEASURE_ONLY"] = "1"
            dp["measure_only"] = True
        if args.dp_equal_weights:
            env["ODELIA_DP_EQUAL_WEIGHTS"] = "1"
            dp["equal_weights"] = True

    per_case = args.per_case or ("off" if dp_active else "site")
    if per_case == "off":
        env["ODELIA_RETURN_PER_CASE"] = "0"
    if args.export_train_predictions is not None:
        env["ODELIA_EXPORT_TRAIN_PREDICTIONS"] = args.export_train_predictions
    if args.seed is not None:
        if args.seed < 0:
            raise BuildError("--seed must be >= 0")
        env["ODELIA_SEED"] = str(args.seed)
    if digest:
        env["ODELIA_WARM_START_SHA256"] = digest

    # MS6.
    if args.ms6_capture_sites:
        raw = args.ms6_capture_sites.strip()
        if raw.lower() == "all":
            capture_sites = "all"
        else:
            listed = parse_sites(raw)
            missing = [s for s in listed if s not in sites]
            if missing:
                raise BuildError(f"--ms6-capture-sites {missing} are not in --sites")
            capture_sites = ",".join(listed)
        env["ODELIA_MS6_CAPTURE_SITES"] = capture_sites
        env["ODELIA_MS6_CAPTURE_ROUNDS"] = _parse_rounds_list(
            args.ms6_capture_rounds or "all", rounds, "--ms6-capture-rounds"
        )
        if args.ms6_capture_optimizer:
            env["ODELIA_MS6_CAPTURE_OPTIMIZER"] = "1"
    elif args.ms6_capture_rounds or args.ms6_capture_optimizer:
        raise BuildError("--ms6-capture-rounds/--ms6-capture-optimizer need --ms6-capture-sites")
    if args.archive_root and not args.archive_globals:
        raise BuildError("--archive-root needs --archive-globals")
    if args.archive_root:
        _check_path(args.archive_root, "--archive-root")

    # Active learning / training subset.
    if args.al_strategy or args.al_budget is not None:
        if not args.al_strategy or args.al_budget is None:
            raise BuildError("--al-strategy and --al-budget go together")
        if args.al_budget < 1:
            raise BuildError("--al-budget must be a positive number of cases per site")
        if args.train_include_list:
            raise BuildError(
                "--al-strategy selects on site, --train-include-list imposes a fixed list: choose one of them"
            )
        select_round = args.al_select_round if args.al_select_round is not None else 0
        if not 0 <= select_round < rounds:
            raise BuildError(f"--al-select-round {select_round}: training rounds are 0..{rounds - 1}")
        al_seed = args.al_seed if args.al_seed is not None else 0
        if al_seed < 0:
            raise BuildError("--al-seed must be >= 0")
        pool = args.al_pool_split or "train"
        if not ENV_VALUE_RE.match(pool):
            raise BuildError(f"--al-pool-split {pool!r}: invalid split name")
        env["ODELIA_AL_STRATEGY"] = args.al_strategy
        env["ODELIA_AL_BUDGET"] = str(args.al_budget)
        env["ODELIA_AL_POOL_SPLIT"] = pool
        env["ODELIA_AL_SEED"] = str(al_seed)
        env["ODELIA_AL_SELECT_ROUND"] = str(select_round)
    elif args.al_pool_split or args.al_seed is not None or args.al_select_round is not None:
        raise BuildError("--al-pool-split/--al-seed/--al-select-round need --al-strategy and --al-budget")
    if args.train_include_list:
        env["ODELIA_TRAIN_INCLUDE_LIST"] = _check_path(args.train_include_list, "--train-include-list")

    user_env = parse_env_pairs(args.env or [])
    for key, value in user_env.items():
        env[key] = value

    for hook in args.extra_hook or []:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", hook):
            raise BuildError(f"--extra-hook {hook!r}: expected a module name under custom/")
    hooks = derive_hooks(env, args.extra_hook or [])
    if hooks:
        # Placed right after the pins so a reviewer sees it next to MODEL_NAME/CONFIG.
        items = list(env.items())
        env = OrderedDict(items[:3] + [("ODELIA_ROUND_HOOKS", ",".join(hooks))] + items[3:])

    return Plan(
        name=args.name,
        base_job=args.base_job,
        sites=sites,
        model=model,
        config=args.config,
        training_rounds=rounds,
        final_eval=bool(args.final_eval),
        starting_client=starting,
        warm_start=warm,
        persistor_mode=persistor_mode,
        source_ckpt=source_ckpt,
        source_sha256=source_sha,
        warm_digest=digest,
        mirror_path=mirror,
        launcher_env=env,
        hooks=hooks,
        archive_globals=bool(args.archive_globals),
        archive_root=args.archive_root,
        dp=dp,
        per_case=per_case,
        warnings=warnings,
    )


# ── Source trees ──────────────────────────────────────────────────────────


@dataclass
class SourceTree:
    root: Path
    info: Dict[str, Any]


def _git(root: Path, *args: str) -> str:
    return _run(["git", "-C", str(root), *args], f"git {' '.join(args)}").stdout.decode().strip()


def git_state(root: Path, base_job: str) -> Dict[str, Any]:
    """Commit and dirty flags of a checkout; a non-git tree records nulls."""
    try:
        commit = _git(root, "rev-parse", "HEAD")
    except BuildError:
        return {"git_commit": None, "git_describe": None, "dirty": None, "job_code_dirty": None}
    describe = _git(root, "describe", "--tags", "--always", "--dirty")
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    # Untracked files under the job code ARE copied into the job, so they count here.
    job_paths = [f"application/jobs/{base_job}", "application/jobs/_shared"]
    job_dirty = bool(_git(root, "status", "--porcelain", "--", *job_paths))
    return {"git_commit": commit, "git_describe": describe, "dirty": dirty, "job_code_dirty": job_dirty}


def _image_version(root: Path) -> Optional[str]:
    path = root / "odelia_image.version"
    if not path.is_file():
        return None
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return lines[-1] if lines else None


def _extract_tar(data: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(dest)


def _normalise_image(ref: str) -> str:
    return ref if ("/" in ref or ":" in ref) else f"{DEFAULT_IMAGE_REPO}:{ref}"


def materialise_source(spec: str, base_job: str, workdir: Path) -> SourceTree:
    """Return a tree that holds application/jobs/<base_job> and application/jobs/_shared."""
    if spec == "repo":
        root = BUILDER_REPO_ROOT
        info = {"kind": "repo", "path": str(root), **git_state(root, base_job)}
    elif spec.startswith("dir:"):
        root = Path(spec[4:]).expanduser().resolve()
        if not root.is_dir():
            raise BuildError(f"--source {spec}: not a directory")
        info = {"kind": "dir", "path": str(root), **git_state(root, base_job)}
    elif spec.startswith("git:"):
        ref = spec[4:]
        if not ref:
            raise BuildError("--source git:<ref> needs a ref, e.g. git:v1.9.0")
        commit = _git(BUILDER_REPO_ROOT, "rev-parse", "--verify", f"{ref}^{{commit}}")
        data = _run(
            ["git", "-C", str(BUILDER_REPO_ROOT), "archive", "--format=tar", commit, "application/jobs",
             "odelia_image.version"],
            f"git archive {ref}",
        ).stdout
        root = workdir / "source"
        root.mkdir()
        _extract_tar(data, root)
        info = {"kind": "git", "ref": ref, "git_commit": commit, "git_describe": None, "dirty": False,
                "job_code_dirty": False}
    elif spec.startswith("image:"):
        image = _normalise_image(spec[6:])
        inspect_out = _run(["docker", "image", "inspect", image], f"docker image inspect {image} (pull it first)")
        meta = json.loads(inspect_out.stdout.decode())[0]
        root = workdir / "source"
        (root / "application").mkdir(parents=True)
        cid = _run(["docker", "create", image], f"docker create {image}").stdout.decode().strip()
        try:
            _run(["docker", "cp", f"{cid}:/MediSwarm/application/jobs", str(root / "application")], "docker cp jobs")
            try:
                _run(["docker", "cp", f"{cid}:/MediSwarm/odelia_image.version", str(root)], "docker cp version")
            except BuildError:
                pass
        finally:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        info = {
            "kind": "image",
            "image": image,
            "image_id": meta.get("Id"),
            "repo_digests": meta.get("RepoDigests") or [],
            "labels": (meta.get("Config") or {}).get("Labels") or {},
            "git_commit": ((meta.get("Config") or {}).get("Labels") or {}).get("org.opencontainers.image.revision"),
            "dirty": None,
            "job_code_dirty": None,
        }
    else:
        raise BuildError(f"--source {spec!r}: expected repo, git:<ref>, image:<ref> or dir:<path>")
    for needed in (f"application/jobs/{base_job}", "application/jobs/_shared"):
        if not (root / needed).exists():
            raise BuildError(f"source {spec}: {needed} is missing")
    info["image_version_file"] = _image_version(root)
    return SourceTree(root, info)


# ── Assembling the job ────────────────────────────────────────────────────


def split_launcher(script: str) -> Tuple[List[Tuple[str, str]], str]:
    """Leading KEY=VALUE tokens of the launcher command, and the command after them."""
    tokens: List[Tuple[str, str]] = []
    pos = 0
    while True:
        match = re.match(r"\s*(\S+)", script[pos:])
        if not match:
            break
        token = LAUNCHER_TOKEN_RE.match(match.group(1))
        if not token:
            break
        if any(q in token.group(2) for q in "'\"\\"):
            raise BuildError(f"launcher token {match.group(1)!r} is quoted; refusing to rewrite it")
        tokens.append((token.group(1), token.group(2)))
        pos += match.end()
    return tokens, script[pos:].lstrip(" ")


def render_launcher(script: str, env: Dict[str, str]) -> str:
    """Set ``env`` on the launcher line: existing keys keep their place, new keys follow."""
    tokens, command = split_launcher(script)
    if not command.strip():
        raise BuildError("launcher script has no command after its KEY=VALUE tokens")
    seen = set()
    out = []
    for key, value in tokens:
        if key in seen:
            continue  # a duplicated token in the template collapses to one
        seen.add(key)
        out.append((key, env.get(key, value)))
    for key, value in env.items():
        if key not in seen:
            out.append((key, value))
    return " ".join(f"{k}={v}" for k, v in out) + " " + command


def _copy_job(source: SourceTree, base_job: str, dest: Path) -> None:
    shutil.copytree(
        source.root / "application" / "jobs" / base_job,
        dest,
        symlinks=False,  # resolve app/custom -> ../../_shared/custom (cp -L)
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    for sub in ("app/tests", "app/scripts"):
        target = dest / sub
        if target.is_dir():
            shutil.rmtree(target)
    custom = dest / "app" / "custom"
    if not custom.is_dir() or custom.is_symlink():
        raise BuildError(f"{base_job}: app/custom did not copy as a directory")


def _persistor_args(root):
    return hocon.component_args("persistor")(root)


def _swarm_workflow_args(root):
    wf = hocon.find_in_list(root, "workflows", "swarm_controller")
    return hocon.get_value(wf, "args") if wf is not None else None


def assemble_job(plan: Plan, source: SourceTree, job_dir: Path) -> None:
    _copy_job(source, plan.base_job, job_dir)
    base = BASE_JOBS[plan.base_job]

    # 1. Strict roster, rounds and warm_start_mode with the existing patcher.
    try:
        patcher.patch_job_dir(
            job_dir,
            plan.persistor_mode,
            num_rounds=plan.total_rounds,
            strict_clients=",".join(plan.sites),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise BuildError(f"patch_warm_start_job: {exc}") from exc

    # 2. meta.conf: name and an explicit deploy_map (the server must be in it, #543).
    meta = hocon.ConfText((job_dir / META_CONF).read_text(), "meta.conf")
    meta.set_value(hocon.root_object, "name", plan.name)
    meta.set_value(hocon.nested("deploy_map"), "app", ["server", *plan.sites], what="deploy_map")
    (job_dir / META_CONF).write_text(meta.text)

    # 3. Server config: starting client and the MS6 archive switch.
    server = hocon.ConfText((job_dir / SERVER_CONF).read_text(), "config_fed_server.conf")
    if plan.starting_client:
        server.set_value(
            _swarm_workflow_args,
            "starting_client",
            plan.starting_client,
            what="the swarm_controller workflow args",
            comment="pinned by build_consortium_job.py",
        )
    archive = hocon.component(server.root, SERVER_ARCHIVE_ID)
    if archive is not None:
        server.set_value(hocon.component_args(SERVER_ARCHIVE_ID), "enabled", plan.archive_globals)
        if plan.archive_root:
            server.set_value(hocon.component_args(SERVER_ARCHIVE_ID), "archive_root", plan.archive_root)
    elif plan.archive_globals:
        raise BuildError(
            f"--archive-globals: the {plan.base_job} template has no '{SERVER_ARCHIVE_ID}' server component "
            "(1.9.0 job templates carry it, disabled)"
        )
    (job_dir / SERVER_CONF).write_text(server.text)

    # 4. Client config: launcher line, persistor pins and warm start, MS6 streamer switch.
    client = hocon.ConfText((job_dir / CLIENT_CONF).read_text(), "config_fed_client.conf")
    launcher_args = hocon.component_args("launcher")(client.root)
    script_node = hocon.get_value(launcher_args, "script") if launcher_args is not None else None
    if script_node is None or script_node.kind != "scalar" or not isinstance(script_node.value, str):
        raise BuildError("config_fed_client.conf: the launcher component has no script line")
    client.set_value(hocon.component_args("launcher"), "script", render_launcher(script_node.value, plan.launcher_env))

    persistor = _persistor_args(client.root)
    if persistor is None:
        raise BuildError("config_fed_client.conf: no 'persistor' component with args")
    model_path = hocon.get_value(hocon.get_value(persistor, "model"), "path")
    if model_path is None or model_path.value != base["model_factory"]:
        raise BuildError(
            f"{plan.base_job}: persistor model factory is {getattr(model_path, 'value', None)!r}, expected "
            f"{base['model_factory']!r}; the template changed and the builder needs updating"
        )
    pin_note = "pinned by build_consortium_job.py (#541): the persistor does not see the launcher env"
    client.set_value(_persistor_args, "source_ckpt_file_full_name", plan.source_ckpt)
    client.set_value(_persistor_args, "latest_global_path", plan.mirror_path)
    if plan.source_sha256:
        client.set_value(_persistor_args, "source_sha256", plan.source_sha256, comment="sha256 of the source file")
    client.set_value(_persistor_args, "model_name", plan.model, comment=pin_note)
    client.set_value(_persistor_args, "data_config", plan.config, comment="pinned with CONFIG on the launcher line")
    if base["model_factory"] == MODEL_FACTORY_MST:
        client.set_value(
            hocon.nested("model", "args", base=_persistor_args),
            "model_name",
            plan.model,
            what="the persistor's model args",
            comment="pinned so the initial global matches the trainers' MODEL_NAME (#541)",
        )
    streamer = hocon.component(client.root, CLIENT_STREAMER_ID)
    if streamer is not None:
        client.set_value(hocon.component_args(CLIENT_STREAMER_ID), "enabled", plan.archive_globals)
    elif plan.archive_globals:
        raise BuildError(
            f"--archive-globals: the {plan.base_job} template has no '{CLIENT_STREAMER_ID}' client component"
        )
    (job_dir / CLIENT_CONF).write_text(client.text)


# ── Does the job's own code support what the job asks for? ──────────────


def persistor_accepted_args(custom_dir: Path) -> Optional[set]:
    """Arguments WarmStartablePTFileModelPersistor accepts, read from its source."""
    path = custom_dir / "warm_continue.py"
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "WarmStartablePTFileModelPersistor":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = item.args
                    names = {a.arg for a in args.args + args.kwonlyargs} - {"self"}
                    if args.kwarg is not None:
                        names |= PT_FILE_PERSISTOR_ARGS
                        # Arguments taken out of **kwargs by name are accepted too.
                        for sub in ast.walk(item):
                            if (
                                isinstance(sub, ast.Call)
                                and isinstance(sub.func, ast.Attribute)
                                and sub.func.attr in ("pop", "get")
                                and isinstance(sub.func.value, ast.Name)
                                and sub.func.value.id == args.kwarg.arg
                                and sub.args
                                and isinstance(sub.args[0], ast.Constant)
                                and isinstance(sub.args[0].value, str)
                            ):
                                names.add(sub.args[0].value)
                    return names
            return set()
    return None


def _python_corpus(custom_dir: Path, subdirs: Sequence[str] = ()) -> str:
    roots = [custom_dir / s for s in subdirs] if subdirs else [custom_dir]
    chunks = []
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.py")) if root.is_dir() else []
        for path in files:
            try:
                chunks.append(path.read_text(errors="replace"))
            except OSError:
                pass
    return "\n".join(chunks)


def _quoted_in(text: str, word: str) -> bool:
    return f'"{word}"' in text or f"'{word}'" in text


def code_support_problems(plan: Plan, custom_dir: Path) -> List[str]:
    """Reasons the job's own code would ignore or reject what this job asks for."""
    problems: List[str] = []
    accepted = persistor_accepted_args(custom_dir)
    needed = {"model_name", "data_config"} | ({"source_sha256"} if plan.source_sha256 else set())
    if accepted is None:
        problems.append("custom/warm_continue.py does not define WarmStartablePTFileModelPersistor")
    else:
        missing = sorted(needed - accepted)
        if missing:
            problems.append(
                f"the job's WarmStartablePTFileModelPersistor does not accept {missing} (contract 3.6; needs 1.9.0 "
                "job code): every site would fail at configure"
            )

    if BASE_JOBS[plan.base_job]["model_factory"] == MODEL_FACTORY_MST:
        models_config = _python_corpus(custom_dir, ["models/models_config.py"])
        if not _quoted_in(models_config, plan.model):
            problems.append(f"model {plan.model!r} is not known to the job's models/models_config.py")

    data_code = _python_corpus(custom_dir, ["data", "env_config.py"])
    if not _quoted_in(data_code, plan.config):
        problems.append(f"CONFIG {plan.config!r} is not known to the job's data loader (custom/data, env_config.py)")

    corpus = _python_corpus(custom_dir)
    for key, value in plan.launcher_env.items():
        is_active = CODE_READ_ENV.get(key)
        if is_active is not None and is_active(value) and key not in corpus:
            problems.append(f"{key}={value}: the job's code never reads {key}, so the setting would be ignored")
        elif is_active is None and key.startswith("ODELIA_") and key not in MANAGED_ENV and key not in corpus:
            plan.warnings.append(f"--env {key}: not mentioned in the job's code; check the name")

    if plan.hooks:
        if not (custom_dir / "round_hooks.py").is_file():
            problems.append("ODELIA_ROUND_HOOKS is set but the job code has no round_hooks.py")
        if "round_hooks" not in _python_corpus(custom_dir, ["main.py"]):
            problems.append("ODELIA_ROUND_HOOKS is set but the job's main.py does not load round hooks")
        for hook in plan.hooks:
            if not ((custom_dir / f"{hook}.py").is_file() or (custom_dir / hook / "__init__.py").is_file()):
                problems.append(f"round hook module {hook!r} is not in the job code (custom/{hook}.py)")
    return problems


# ── Validation of the finished directory ─────────────────────────────────


def load_conf(path: Path) -> Tuple[Dict[str, Any], str, List[str]]:
    """Parsed config (pyhocon when installed, as NVFlare reads it) and lint findings."""
    text = path.read_text()
    lint: List[str] = []
    builtin: Optional[Dict[str, Any]] = None
    try:
        tree = hocon.parse(text)
        builtin = hocon.to_python(tree)
        lint = [f"{path.name}: key set twice: {k}" for k in hocon.duplicate_keys(tree)]
    except hocon.ConfError as exc:
        lint = [f"{path.name}: {exc}"]
    try:
        reference = hocon.parse_with_pyhocon(text)
    except Exception as exc:  # pyhocon's own parse error
        raise BuildError(f"{path}: pyhocon cannot parse the result: {exc}") from exc
    if reference is not None:
        return reference, "pyhocon", lint
    if builtin is None:
        raise BuildError(f"{path}: cannot parse the result: {'; '.join(lint)}")
    return builtin, "hocon_lite", lint


def _strings(value: Any):
    """Every key and string value in a parsed config."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def _component(conf: Dict[str, Any], comp_id: str) -> Optional[Dict[str, Any]]:
    hits = [c for c in conf.get("components", []) if isinstance(c, dict) and c.get("id") == comp_id]
    return hits[0] if len(hits) == 1 else None


def _swarm_client_args(conf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in conf.get("executors", []):
        executor = item.get("executor", {}) if isinstance(item, dict) else {}
        if str(executor.get("path", "")).endswith("SwarmClientController"):
            return executor.get("args", {})
    return None


def validate_job(plan: Plan, job_dir: Path) -> Dict[str, Any]:
    """Check the written job against the plan; raise BuildError listing every failure."""
    failures: List[str] = []
    checks: List[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        (checks if ok else failures).append(label + (f": {detail}" if detail and not ok else ""))

    meta, parser_name, lint_meta = load_conf(job_dir / META_CONF)
    server, _, lint_server = load_conf(job_dir / SERVER_CONF)
    client, _, lint_client = load_conf(job_dir / CLIENT_CONF)
    for finding in lint_meta + lint_server + lint_client:
        failures.append(finding)
    checks.append(f"configs parse ({parser_name})")

    sites = list(plan.sites)
    n = len(sites)
    check(meta.get("name") == plan.name, "meta name", f"{meta.get('name')!r}")
    app = (meta.get("deploy_map") or {}).get("app")
    check(app == ["server", *sites], "deploy_map is ['server', <sites>]", f"{app}")
    check(isinstance(app, list) and "server" in app, "deploy_map includes the server")
    check(isinstance(app, list) and "@ALL" not in app, "deploy_map has no @ALL")
    check(meta.get("min_clients") == n, "meta min_clients", f"{meta.get('min_clients')} != {n}")
    check(meta.get("mandatory_clients") == sites, "meta mandatory_clients", f"{meta.get('mandatory_clients')}")

    wf = [w for w in server.get("workflows", []) if isinstance(w, dict) and w.get("id") == "swarm_controller"]
    wf_args = wf[0].get("args", {}) if len(wf) == 1 else {}
    check(bool(wf_args), "server has one swarm_controller workflow")
    check(wf_args.get("num_rounds") == plan.total_rounds, "num_rounds", f"{wf_args.get('num_rounds')}")
    check(wf_args.get("min_clients") == n, "server min_clients (strict)", f"{wf_args.get('min_clients')}")
    check(wf_args.get("configure_min_clients") == n, "configure_min_clients (F8)", f"{wf_args.get('configure_min_clients')}")
    check(wf_args.get("participating_clients") == sites, "participating_clients", f"{wf_args.get('participating_clients')}")
    if plan.starting_client:
        check(wf_args.get("starting_client") == plan.starting_client, "starting_client", f"{wf_args.get('starting_client')}")
    check(wf_args.get("starting_client") in (None, "", *sites), "starting_client is a participant")

    swarm = _swarm_client_args(client)
    check(swarm is not None, "client has a swarm client controller")
    if swarm is not None:
        check(swarm.get("min_responses_required") == n, "min_responses_required", f"{swarm.get('min_responses_required')}")

    launcher = _component(client, "launcher") or {}
    script = (launcher.get("args") or {}).get("script", "")
    try:
        tokens, command = split_launcher(script)
    except BuildError as exc:
        tokens, command = [], ""
        failures.append(str(exc))
    keys = [k for k, _ in tokens]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    check(not dupes, "launcher tokens unique", f"{dupes}")
    token_env = dict(tokens)
    for key, value in plan.launcher_env.items():
        check(token_env.get(key) == value, f"launcher {key}", f"{token_env.get(key)!r} != {value!r}")
    check("python3" in command and "{app_script}" in command, "launcher still runs the app script", command)

    persistor = (_component(client, "persistor") or {}).get("args", {})
    check(persistor.get("model_name") == plan.model, "persistor model_name pin", f"{persistor.get('model_name')}")
    check(persistor.get("data_config") == plan.config, "persistor data_config pin", f"{persistor.get('data_config')}")
    model_cfg = persistor.get("model") or {}
    if model_cfg.get("path") == MODEL_FACTORY_MST:
        pinned = (model_cfg.get("args") or {}).get("model_name")
        check(pinned == plan.model, "persistor model args model_name (#541)", f"{pinned}")
    check(token_env.get("MODEL_NAME") == persistor.get("model_name"), "MODEL_NAME pins agree")
    check(token_env.get("CONFIG") == persistor.get("data_config"), "CONFIG pins agree")
    check(persistor.get("warm_start_mode") == plan.persistor_mode, "warm_start_mode", f"{persistor.get('warm_start_mode')}")
    check(persistor.get("latest_global_path") == plan.mirror_path, "mirror path", f"{persistor.get('latest_global_path')}")
    check(persistor.get("source_ckpt_file_full_name") == plan.source_ckpt, "warm-start source path")
    if plan.warm_start == "source":
        check(persistor.get("source_sha256") == plan.source_sha256, "source_sha256")
        check(persistor.get("source_ckpt_file_full_name") != persistor.get("latest_global_path"), "source != mirror")

    # Round hooks: listed modules match the variables (mirrors load_hooks' safety net).
    listed = [h for h in token_env.get("ODELIA_ROUND_HOOKS", "").split(",") if h]
    check(listed == plan.hooks, "ODELIA_ROUND_HOOKS", f"{listed} != {plan.hooks}")
    for prefix, hook in HOOK_FOR_PREFIX:
        if any(k.startswith(prefix) for k in token_env):
            check(hook in listed, f"{prefix}* needs hook {hook}")
    if listed:
        check(listed[-1] == "dp_noise" or "dp_noise" not in listed, "dp_noise is the last hook")

    streamer = _component(client, CLIENT_STREAMER_ID)
    archive = _component(server, SERVER_ARCHIVE_ID)
    if streamer is not None or archive is not None or plan.archive_globals:
        s_on = bool((streamer or {}).get("args", {}).get("enabled", False))
        a_on = bool((archive or {}).get("args", {}).get("enabled", False))
        check(s_on == plan.archive_globals and a_on == plan.archive_globals, "MS6 archive components",
              f"streamer={s_on} archive={a_on}")

    if plan.per_case == "off":
        check(token_env.get("ODELIA_RETURN_PER_CASE") == "0", "per-case return forced off")

    if "VHIO_1" not in plan.sites:
        # Parsed values only: a template comment that mentions VHIO_1 is harmless.
        hits = [name for name, conf in (("meta.conf", meta), ("server", server), ("client", client))
                if any("VHIO_1" in s for s in _strings(conf))]
        check(not hits, "no VHIO_1 unless listed", f"found in {hits}")

    for sub in ("app/tests", "app/scripts"):
        check(not (job_dir / sub).exists(), f"{sub} removed")
    custom = job_dir / "app" / "custom"
    check(custom.is_dir() and not custom.is_symlink(), "app/custom is a real directory")

    if failures:
        raise BuildError("job failed validation:\n  - " + "\n  - ".join(failures))
    return {"parser": parser_name, "passed": checks}


# ── Manifest ──────────────────────────────────────────────────────────────


def file_hashes(job_dir: Path, sub: str) -> "OrderedDict[str, str]":
    base = job_dir / sub
    out: "OrderedDict[str, str]" = OrderedDict()
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        out[str(path.relative_to(job_dir))] = _sha256_file(path)
    return out


def tree_digest(hashes: Dict[str, str]) -> str:
    digest = hashlib.sha256()
    for rel, sha in sorted(hashes.items()):
        digest.update(f"{rel}\0{sha}\n".encode())
    return digest.hexdigest()


def build_manifest(
    plan: Plan,
    args: argparse.Namespace,
    argv: Sequence[str],
    source: SourceTree,
    job_dir: Path,
    expected_image: str,
    validation: Dict[str, Any],
    dp_report: Dict[str, Any],
    code_checks: str,
) -> Dict[str, Any]:
    custom = file_hashes(job_dir, "app/custom")
    configs = OrderedDict(
        (str(p), _sha256_file(job_dir / p)) for p in (META_CONF, SERVER_CONF, CLIENT_CONF)
    )
    flags = {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(args).items())}
    return OrderedDict(
        [
            ("schema", MANIFEST_SCHEMA),
            ("job_name", plan.name),
            ("base_job", plan.base_job),
            ("built_at_utc", _utc_now()),
            ("builder", {"script": "scripts/admin/build_consortium_job.py", **git_state(BUILDER_REPO_ROOT, plan.base_job)}),
            ("source", source.info),
            ("expected_image", expected_image),
            ("argv", list(argv)),
            ("flags", flags),
            ("sites", list(plan.sites)),
            ("starting_client", plan.starting_client),
            ("model", plan.model),
            ("config", plan.config),
            ("training_rounds", plan.training_rounds),
            ("final_eval_round", plan.final_eval),
            ("num_rounds", plan.total_rounds),
            ("launcher_env", plan.launcher_env),
            ("round_hooks", plan.hooks),
            (
                "warm_start",
                {
                    "mode": plan.warm_start,
                    "warm_start_mode": plan.persistor_mode,
                    "source_ckpt_file_full_name": plan.source_ckpt,
                    "source_sha256": plan.source_sha256,
                    "state_digest": plan.warm_digest,
                    "latest_global_path": plan.mirror_path,
                },
            ),
            ("per_case_return", plan.per_case),
            ("ms6_archive_globals", plan.archive_globals),
            ("dp", dp_report or None),
            ("code_checks", code_checks),
            ("warnings", plan.warnings),
            ("validation", validation),
            ("config_sha256", configs),
            ("custom_tree_sha256", tree_digest(custom)),
            ("custom_files_sha256", custom),
        ]
    )


# ── Command line ──────────────────────────────────────────────────────────

EXAMPLES = """\
examples:
  # October recipe on 1.9.0: MST, 7 sites, 20 training rounds + the evaluation-only round
  build_consortium_job.py --name odelia_mst_20r_7site --model MST --rounds 20 \\
      --out-dir <admin kit>/local/mediswarm_jobs

  # CAM model, built from the release image, per-case return off
  build_consortium_job.py --name odelia_cam_20r --model MST_SAMMed2D --rounds 20 --per-case off \\
      --source image:jefftud/odelia:1.9.0 --out-dir <dir>

  # DP noise-cost run, refused if 20 rounds exceed epsilon 10
  build_consortium_job.py --name odelia_mst_dp_s24 --model MST --rounds 20 --dp-sigma 2.4 \\
      --dp-clip-norm 1.0 --dp-target-epsilon 10 --out-dir <dir>

  # D3.2 arm: warm start from the August global held by UKA_1, entropy selection of 40 cases
  build_consortium_job.py --name d32_entropy_b40 --base-job challenge_1DivideAndConquer --rounds 5 \\
      --warm-start source:/scratch/odelia_87c5bbee_global.pt:<file sha256> --warm-start-digest <state digest> \\
      --starting-client UKA_1 --al-strategy entropy --al-budget 40 --out-dir <dir>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_consortium_job.py",
        description="Build a validated ODELIA consortium job directory (1.9.0 contract).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    p.add_argument("--name", required=True, help="job name: meta.conf name, folder name and default mirror suffix")
    p.add_argument("--base-job", default="ODELIA_ternary_classification", choices=sorted(BASE_JOBS))
    p.add_argument("--sites", default=",".join(DEFAULT_SITES), help="ordered CSV roster (default: the 7 active sites)")
    p.add_argument("--model", help="MODEL_NAME (required for ODELIA_ternary_classification)")
    p.add_argument("--config", default="unilateral", choices=CONFIG_CHOICES, help="data format, pinned as CONFIG")
    p.add_argument("--rounds", type=int, required=True, help="training rounds N")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--final-eval", dest="final_eval", action="store_true", default=True,
                       help="default: add an evaluation-only round (num_rounds=N+1, ODELIA_FINAL_EVAL_ROUND=1)")
    group.add_argument("--no-final-eval", dest="final_eval", action="store_false")
    p.add_argument("--starting-client", help="site whose persistor seeds round 0 (required for a source warm start)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra launcher variable (repeatable)")
    p.add_argument("--per-case", choices=("site", "off"),
                   help="site: each site's image.conf decides (default); off: force ODELIA_RETURN_PER_CASE=0")
    p.add_argument("--export-train-predictions", choices=("0", "1"), help="ODELIA_EXPORT_TRAIN_PREDICTIONS")
    p.add_argument("--seed", type=int, help="ODELIA_SEED")
    p.add_argument("--extra-hook", action="append", default=[], metavar="MODULE",
                   help="additional round-hook module under custom/ (placed before dp_noise)")

    ms6 = p.add_argument_group("MS6 per-round models (#612)")
    ms6.add_argument("--ms6-capture-sites", help="CSV of sites that keep G_r/W_r, or 'all'")
    ms6.add_argument("--ms6-capture-rounds", help="'all' (default) or CSV of 0-based training rounds")
    ms6.add_argument("--ms6-capture-optimizer", action="store_true", help="also keep the optimizer state")
    ms6.add_argument("--archive-globals", action="store_true",
                     help=f"enable {CLIENT_STREAMER_ID}/{SERVER_ARCHIVE_ID} (per-round globals to the coordinator)")
    ms6.add_argument("--archive-root", help="archive_root of the server component (default in the template)")

    dp = p.add_argument_group("differential privacy (T3.3, #593); implies --per-case off")
    dp.add_argument("--dp-sigma", type=float, help="noise multiplier")
    dp.add_argument("--dp-clip-norm", type=float, help="L2 clip norm C of the per-round update")
    dp.add_argument("--dp-delta", type=float, help="delta of the accountant (default 1e-5)")
    dp.add_argument("--dp-target-epsilon", type=float, help="budget: the build is refused if N rounds exceed it")
    dp.add_argument("--dp-measure-only", action="store_true", help="log update norms, add no noise")
    dp.add_argument("--dp-equal-weights", action="store_true", help="unweighted aggregation (NUM_STEPS=1)")

    al = p.add_argument_group("training subset / active learning (D3.2)")
    al.add_argument("--al-strategy", choices=AL_STRATEGIES)
    al.add_argument("--al-budget", type=int, help="cases selected per site")
    al.add_argument("--al-pool-split", help="split scored as the pool (default train)")
    al.add_argument("--al-seed", type=int, help="seed of the random arms (default 0)")
    al.add_argument("--al-select-round", type=int, help="round whose received global scores the pool (default 0)")
    al.add_argument("--train-include-list", help="UID list at the site that restricts training")

    ws = p.add_argument_group("warm start")
    ws.add_argument("--warm-start", default="fresh", help="fresh | continue | source:<path at starting client>:<sha256>")
    ws.add_argument("--warm-start-digest", help="state digest every site checks the round-0 global against")
    ws.add_argument("--mirror-path", help="per-experiment mirror (default /scratch/mediswarm_latest_global_<name>.pt)")

    out = p.add_argument_group("source and output")
    out.add_argument("--source", default="repo", help="repo | git:<ref> | image:<ref> | dir:<path> (default repo)")
    out.add_argument("--expected-image", help="image the job is meant to run on (default from the source)")
    out.add_argument("--out-dir", type=Path, help="parent directory; the job is written to <out-dir>/<name>")
    out.add_argument("--dry-run", action="store_true", help="build and validate in a temporary directory only")
    out.add_argument("--force", action="store_true", help="replace an existing <out-dir>/<name>")
    out.add_argument("--no-code-checks", action="store_true",
                     help="skip the checks that the job's own code supports each setting (recorded in the manifest)")
    out.add_argument("--json", action="store_true", help="print the manifest as JSON on stdout")
    return p


def summary_lines(plan: Plan, job_path: Optional[Path], manifest: Dict[str, Any]) -> List[str]:
    lines = [
        f"job        {plan.name}  (base {plan.base_job}, source {manifest['source'].get('kind')})",
        f"sites      {len(plan.sites)}: {', '.join(plan.sites)}"
        + (f"  starting client {plan.starting_client}" if plan.starting_client else ""),
        f"rounds     {plan.training_rounds} training" + (" + 1 evaluation-only" if plan.final_eval else "")
        + f" -> num_rounds = {plan.total_rounds}",
        f"model      MODEL_NAME={plan.model} CONFIG={plan.config} (launcher and persistor)",
        f"warm start {plan.warm_start} (warm_start_mode={plan.persistor_mode}); mirror {plan.mirror_path}",
        "launcher   " + " ".join(f"{k}={v}" for k, v in plan.launcher_env.items()),
        f"hooks      {','.join(plan.hooks) or '-'}",
        f"image      {manifest['expected_image']}",
        f"custom     sha256 {manifest['custom_tree_sha256'][:16]}... ({len(manifest['custom_files_sha256'])} files)",
    ]
    if manifest.get("dp"):
        dp = manifest["dp"]
        line = f"dp         sigma={dp.get('sigma')} C={dp.get('clip_norm')} delta={dp.get('delta')} " \
               f"epsilon({plan.training_rounds})={dp.get('epsilon_after_training_rounds')}"
        if dp.get("target_epsilon") is not None:
            line += f" target={dp['target_epsilon']} cap={dp['max_training_rounds']} rounds"
        lines.append(line)
    for warning in plan.warnings:
        lines.append(f"WARNING    {warning}")
    if job_path is not None:
        lines.append(f"written    {job_path}")
        if "/local/mediswarm_jobs" in str(job_path):
            lines.append(f"submit     submit_job /fl_admin/local/mediswarm_jobs/{plan.name}   (inside the admin kit)")
    else:
        lines.append("dry run    nothing written")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(argv)
    try:
        if not args.dry_run and args.out_dir is None:
            raise BuildError("--out-dir is required (or use --dry-run)")
        plan = make_plan(args)
        final_path = None
        if not args.dry_run:
            out_dir = args.out_dir.expanduser().resolve()
            if not out_dir.is_dir():
                raise BuildError(f"--out-dir {out_dir} does not exist")
            final_path = out_dir / plan.name
            if final_path.exists() and not args.force:
                raise BuildError(f"{final_path} exists; pick another --name or pass --force")

        with tempfile.TemporaryDirectory(prefix="mediswarm_job_build_") as tmp:
            workdir = Path(tmp)
            source = materialise_source(args.source, plan.base_job, workdir)
            if args.dry_run:
                job_dir = workdir / "job" / plan.name
            else:
                # Built next to the destination and renamed at the end, so a refused
                # build never leaves a half-patched job where the admin looks for jobs.
                job_dir = Path(tempfile.mkdtemp(prefix=f".{plan.name}.building.", dir=str(final_path.parent)))
                job_dir.rmdir()
            try:
                job_dir.parent.mkdir(parents=True, exist_ok=True)
                assemble_job(plan, source, job_dir)
                custom_dir = job_dir / "app" / "custom"
                if args.no_code_checks:
                    code_checks = "skipped (--no-code-checks)"
                    plan.warnings.append("code-support checks skipped: the job may ask for features its code lacks")
                else:
                    problems = code_support_problems(plan, custom_dir)
                    if problems:
                        raise BuildError(
                            "the job's code does not support this job:\n  - " + "\n  - ".join(problems)
                            + f"\n(source: {args.source}; build from 1.9.0 code, or pass --no-code-checks if you "
                            "know better)"
                        )
                    code_checks = "passed"
                dp_report = dp_budget(plan, custom_dir)
                validation = validate_job(plan, job_dir)
                version = source.info.get("image_version_file")
                expected_image = args.expected_image or (
                    source.info.get("image") or (f"{DEFAULT_IMAGE_REPO}:{version}" if version else "unknown")
                )
                manifest = build_manifest(
                    plan, args, argv, source, job_dir, expected_image, validation, dp_report, code_checks
                )
                (job_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
                if final_path is not None:
                    if final_path.exists():
                        shutil.rmtree(final_path)
                    os.rename(job_dir, final_path)
            except BaseException:
                if final_path is not None and job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
                raise
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    text = "\n".join(summary_lines(plan, final_path, manifest))
    if args.json:
        print(text, file=sys.stderr)
        print(json.dumps(manifest, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
