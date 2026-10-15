#!/usr/bin/env python3
"""The ODELIA consortium runs of October to November 2026 as named job presets.

Each run of the Q4 2026 schedule (docs/ODELIA_RUN_PLAN_2026-Q4.md) is one preset here: a fixed
set of build_consortium_job.py arguments. The coordinator builds a run with one command, so the job
that is submitted is the one reviewed in this file, not a command line typed on the day. All
validation stays in the builder: this tool only assembles its arguments and calls it, so a preset
the builder refuses is refused here with the builder's own reason.

Overrides, for a site that cannot join and for the dl3 test bed:

* ``--exclude-sites VHIO_1`` drops sites from the preset's roster (VHIO_1 joins only once its
  node can read all of its data). A site the roster does not contain is reported and ignored.
* ``--sites TEST_A_1,TEST_B_1`` replaces the roster (test bed); --exclude-sites applies after it.
* ``--starting-client`` and ``--ms6-capture-sites`` replace the preset's site-bound settings.
  When the roster no longer contains a site a preset names there, the request is refused until
  the override names another: the warm start's file sits at the starting client, and the MS6
  captures are planned for named sites, so neither moves silently.

The job name is the preset's stem plus ``_<n>site``: a 7-site build of a preset never shares its
name, nor the per-name mirror /scratch/mediswarm_latest_global_<name>.pt, with the 8-site build.

Jobs are built from the release image by default (``--source image:jefftud/odelia:1.9.1``), so
the job code shipped with the job is the code of the image the sites run. ``--source repo``
builds from this checkout (development and unit tests).

Examples:
  odelia_run_plan.py --list
  odelia_run_plan.py --show cam --exclude-sites VHIO_1
  odelia_run_plan.py --build cam --exclude-sites VHIO_1 --out-dir <admin kit>/local/mediswarm_jobs
  # dl3 test bed: same preset, test sites, a starting client that holds the start model
  odelia_run_plan.py --build d32_entropy_b20 --sites TEST_A_1,TEST_B_1 --starting-client TEST_A_1 \\
      --source repo --out-dir <dir>

Exit status: 0 on success, 2 for a refused request (here or in the builder).
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
BUILDER_SCRIPT = "scripts/admin/build_consortium_job.py"


def _load_builder():
    """build_consortium_job.py, under the name its own sibling loader uses."""
    module_name = "_mediswarm_admin_build_consortium_job"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, HERE / "build_consortium_job.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


bcj = _load_builder()

# ── The schedule's constants ─────────────────────────────────────────────

# The eight hospitals. VHIO_1 is listed: every preset is meant for all of them, and
# --exclude-sites VHIO_1 builds the run without it until its data access is fixed.
SITES = ("CAM_1", "MHA_1", "RSH_1", "RUMC_1", "UKA_1", "UMCU_1", "USZ_1", "VHIO_1")

RELEASE_IMAGE = "jefftud/odelia:1.9.1"
DEFAULT_SOURCE = f"image:{RELEASE_IMAGE}"
ODELIA_JOB = "ODELIA_ternary_classification"
CHALLENGE_JOB = "challenge_1DivideAndConquer"

# The August 2026 pan-European model: final global of job 87c5bbee (1DivideAndConquer, 8 sites),
# FL_global_model.pt in ODELIA_MODELS/release/2026-08_8site_1DivideAndConquer_87c5bbee/ on the
# coordinator. The starting client keeps a copy as <scratch dir>/odelia_87c5bbee_global.pt, which
# its container sees under /scratch.
AUGUST_MODEL_PATH = "/scratch/odelia_87c5bbee_global.pt"
# sha256sum of the file, as in the release folder's SHA256SUMS (the starting client's persistor checks it).
AUGUST_MODEL_SHA256 = "ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d"
# state_digest.checkpoint_state_digest() of the file (digest format 1), computed on 15 Oct 2026 inside
# jefftud/odelia:1.9.1. Every site checks the round-0 global against it (ODELIA_WARM_START_SHA256).
AUGUST_MODEL_STATE_DIGEST = "cdbfab5748a025f3a88216747ac57208d764a93f9e416e90d9c1063f1ea6b51e"

# What has to be in place before a job is submitted (shown by --show; the plan doc has the details).
NEEDS_KIT = "every site on the 1.9.0 kit, running jefftud/odelia:1.9.1 via :current (heartbeat image_id)"


def _needs_start_model(site: str) -> str:
    return f"{site} holds the August model at {AUGUST_MODEL_PATH} (sha256 {AUGUST_MODEL_SHA256[:12]}...)"


def _needs_per_case(sites: Sequence[str]) -> str:
    return f"per-case return on (ODELIA_RETURN_PER_CASE=1 in image.conf) at {', '.join(sites)}"


# ── Presets ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Preset:
    key: str
    stem: str  # job name without the _<n>site suffix
    title: str
    base_job: str = ODELIA_JOB
    model: Optional[str] = None  # None: the base job fixes the model (challenge_1DivideAndConquer)
    config: str = "unilateral"
    rounds: int = 20  # training rounds; the builder adds the evaluation-only round
    sites: Tuple[str, ...] = SITES
    starting_client: Optional[str] = None
    august_warm_start: bool = False
    ms6_capture_sites: Tuple[str, ...] = ()
    flags: Tuple[str, ...] = ()  # further builder flags, verbatim
    needs: Tuple[str, ...] = (NEEDS_KIT,)


def _d32(strategy: str, budget: int) -> Preset:
    return Preset(
        key=f"d32_{strategy}_b{budget}",
        stem=f"odelia_d32_{strategy}_b{budget}_5r",
        title=f"D3.2 active learning (#592): {strategy} selection of {budget} cases per site, from the August model",
        base_job=CHALLENGE_JOB,
        rounds=5,
        starting_client="UMCU_1",
        august_warm_start=True,
        flags=("--al-strategy", strategy, "--al-budget", str(budget), "--al-pool-split", "train", "--al-seed", "0"),
        needs=(NEEDS_KIT, _needs_start_model("UMCU_1")),
    )


PRESETS: "OrderedDict[str, Preset]" = OrderedDict(
    (p.key, p)
    for p in (
        Preset(
            key="check",
            stem="odelia_check_mst_1r",
            title="Check job: MST, one training round, every site",
            model="MST",
            rounds=1,
        ),
        Preset(
            key="cam",
            stem="odelia_cam_sammed2d_20r",
            title="Run A, CAM model: MST_SAMMed2D, MS6 capture at CAM_1 and RUMC_1, every global archived at TUD",
            model="MST_SAMMed2D",
            ms6_capture_sites=("CAM_1", "RUMC_1"),
            flags=("--export-train-predictions", "0", "--archive-globals"),
            needs=(NEEDS_KIT, "coordinator archive mounted at /archive (1.9.0 server kit)"),
        ),
        Preset(
            key="vhio",
            stem="odelia_vhio_medicalnet_20r",
            title="Run B, VHIO model: MedicalNet (ResNet-34 weights in the 1.9.1 image)",
            model="MedicalNet",
        ),
        Preset(
            key="d25_dutch",
            stem="odelia_d25_dutch_5r",
            title="D2.5 regional fine-tuning (#526), Dutch cohort: the August model on RUMC_1 and UMCU_1",
            base_job=CHALLENGE_JOB,
            rounds=5,
            sites=("RUMC_1", "UMCU_1"),
            starting_client="UMCU_1",
            august_warm_start=True,
            needs=(NEEDS_KIT, _needs_start_model("UMCU_1"), _needs_per_case(("RUMC_1", "UMCU_1"))),
        ),
        Preset(
            key="d25_greek",
            stem="odelia_d25_greek_5r",
            # Kept as planned: the builder refuses it (a swarm needs two sites); see the plan doc.
            title="D2.5 regional fine-tuning (#526), Greek cohort: MHA_1 alone. BLOCKED: a swarm job needs 2 sites",
            base_job=CHALLENGE_JOB,
            rounds=5,
            sites=("MHA_1",),
            starting_client="MHA_1",
            august_warm_start=True,
            needs=(NEEDS_KIT, _needs_start_model("MHA_1"), _needs_per_case(("MHA_1",))),
        ),
        _d32("entropy", 20),
        _d32("entropy", 40),
        _d32("random", 20),
        _d32("random", 40),
        Preset(
            key="dp_s24",
            stem="odelia_dp_s24_mst_20r",
            title="T3.3 DP utility (#593): MST, sigma 2.4, clip 1.0, epsilon <= 10 at delta 1e-5; per-case off",
            model="MST",
            flags=("--dp-sigma", "2.4", "--dp-clip-norm", "1.0", "--dp-delta", "1e-5", "--dp-target-epsilon", "10"),
        ),
        Preset(
            key="bd",
            stem="odelia_bd_1dac_20r",
            title="Run C, BreastDivider crops: 1DivideAndConquer on CONFIG=unilateral_BD, strict format check",
            base_job=CHALLENGE_JOB,
            config="unilateral_BD",
            # A site whose crops do not look like BreastDivider output stops at setup instead of training
            # on the wrong format (docs/BREASTDIVIDER_SITE_GUIDE.md, "For coordinators").
            flags=("--env", "ODELIA_STRICT_DATA_FORMAT=1"),
            needs=(NEEDS_KIT, "data_unilateral_BD/ and metadata_unilateral_BD/ at every site, checked with "
                              "reconcile_odelia_dataset.py and a preflight check (BREASTDIVIDER_SITE_GUIDE.md)"),
        ),
    )
)


# ── From a preset to the builder's arguments ─────────────────────────────


class PlanError(Exception):
    """An override that does not fit the preset."""


@dataclass
class Request:
    preset: Preset
    argv: List[str]  # builder arguments, without the output flags
    sites: Tuple[str, ...]
    notes: List[str]


def _csv(value: Optional[str]) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def get_preset(key: str) -> Preset:
    if key not in PRESETS:
        raise PlanError(f"unknown preset {key!r}; one of: {', '.join(PRESETS)}")
    return PRESETS[key]


def builder_request(
    key: str,
    sites: Optional[str] = None,
    exclude_sites: Optional[str] = None,
    starting_client: Optional[str] = None,
    ms6_capture_sites: Optional[str] = None,
    name: Optional[str] = None,
    source: str = DEFAULT_SOURCE,
) -> Request:
    """The build_consortium_job.py arguments of preset ``key`` with the overrides applied."""
    preset = get_preset(key)
    notes: List[str] = []
    roster = _csv(sites) if sites is not None else list(preset.sites)
    if sites is not None and not roster:
        raise PlanError("--sites is empty")
    excluded = _csv(exclude_sites)
    absent = [s for s in excluded if s not in roster]
    if absent:
        notes.append(f"{', '.join(absent)} not in the roster of {key}; nothing to exclude")
    roster = [s for s in roster if s not in excluded]
    if not roster:
        raise PlanError(f"{key}: --exclude-sites removed every site")

    start = starting_client or preset.starting_client
    if not starting_client and start and start not in roster:
        why = (
            f"it holds the start model at {AUGUST_MODEL_PATH}" if preset.august_warm_start else "the preset pins it"
        )
        raise PlanError(
            f"{key}: the starting client {start} is not in the roster ({','.join(roster)}); {why}. "
            "Pass --starting-client <a site of the roster>"
            + (" that holds the same file" if preset.august_warm_start else "")
        )

    if ms6_capture_sites is not None:
        if not preset.ms6_capture_sites:
            raise PlanError(f"{key} captures no MS6 models; --ms6-capture-sites only replaces a preset's capture sites")
        capture = ms6_capture_sites
    else:
        missing = [s for s in preset.ms6_capture_sites if s not in roster]
        if missing:
            raise PlanError(
                f"{key}: MS6 capture site(s) {', '.join(missing)} not in the roster ({','.join(roster)}); "
                "pass --ms6-capture-sites with sites of the roster"
            )
        capture = ",".join(preset.ms6_capture_sites)

    argv = [
        "--name", name or f"{preset.stem}_{len(roster)}site",
        "--base-job", preset.base_job,
    ]
    if preset.model:
        argv += ["--model", preset.model]
    argv += ["--config", preset.config, "--rounds", str(preset.rounds), "--sites", ",".join(roster)]
    if start:
        argv += ["--starting-client", start]
    if preset.august_warm_start:
        argv += [
            "--warm-start", f"source:{AUGUST_MODEL_PATH}:{AUGUST_MODEL_SHA256}",
            "--warm-start-digest", AUGUST_MODEL_STATE_DIGEST,
        ]
    if capture:
        argv += ["--ms6-capture-sites", capture]
    argv += list(preset.flags)
    argv += ["--source", source]
    return Request(preset=preset, argv=argv, sites=tuple(roster), notes=notes)


def command_line(request: Request, out_dir: Optional[str] = None) -> str:
    """The builder command for ``request``, as it would be typed in the repository root."""
    out = shlex.quote(out_dir) if out_dir else "<out-dir>"
    return f"python3 {BUILDER_SCRIPT} {shlex.join(request.argv)} --out-dir {out}"


# ── Command line ─────────────────────────────────────────────────────────


def list_lines() -> List[str]:
    lines = []
    for preset in PRESETS.values():
        req = builder_request(preset.key)
        model = preset.model or preset.base_job.replace("challenge_", "")
        sites = "all 8" if preset.sites == SITES else ",".join(preset.sites)
        extras = []
        if preset.starting_client:
            extras.append(f"start {preset.starting_client}")
        if preset.august_warm_start:
            extras.append("warm start 87c5bbee")
        if preset.ms6_capture_sites:
            extras.append(f"ms6 capture {','.join(preset.ms6_capture_sites)}")
        if preset.flags:
            extras.append(" ".join(preset.flags))
        lines.append(f"{preset.key:<17} {preset.title}")
        lines.append(
            f"{'':<17} {req.argv[1]}: {model}, CONFIG={preset.config}, {preset.rounds}+1 rounds, sites {sites}"
            + (f"; {', '.join(extras)}" if extras else "")
        )
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="odelia_run_plan.py",
        description="Build the ODELIA Q4 2026 consortium jobs from named presets (docs/ODELIA_RUN_PLAN_2026-Q4.md).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets: " + ", ".join(PRESETS),
    )
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list the presets and their settings")
    action.add_argument("--show", metavar="PRESET", help="print the exact build_consortium_job.py command")
    action.add_argument("--build", metavar="PRESET", help="build the job with build_consortium_job.py")
    p.add_argument("--sites", help="replace the preset's roster (ordered CSV), e.g. the dl3 test sites")
    p.add_argument("--exclude-sites", help="CSV of sites to drop from the roster, e.g. VHIO_1")
    p.add_argument("--starting-client", help="replace the preset's starting client (it must hold the start model)")
    p.add_argument("--ms6-capture-sites", help="replace the preset's MS6 capture sites (CSV or 'all')")
    p.add_argument("--name", help="job name (default: the preset's stem + _<n>site)")
    p.add_argument("--source", default=DEFAULT_SOURCE, help=f"builder --source (default {DEFAULT_SOURCE})")
    p.add_argument("--out-dir", help="builder --out-dir: the job is written to <out-dir>/<name>")
    p.add_argument("--dry-run", action="store_true", help="builder --dry-run: build and validate, write nothing")
    p.add_argument("--force", action="store_true", help="builder --force: replace an existing <out-dir>/<name>")
    p.add_argument("--json", action="store_true", help="builder --json: print the manifest")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.list:
        print("\n".join(list_lines()))
        return 0
    key = args.show or args.build
    try:
        request = builder_request(
            key,
            sites=args.sites,
            exclude_sites=args.exclude_sites,
            starting_client=args.starting_client,
            ms6_capture_sites=args.ms6_capture_sites,
            name=args.name,
            source=args.source,
        )
    except PlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for note in request.notes:
        print(f"NOTE: {note}", file=sys.stderr)

    if args.show:
        print(f"# {request.preset.key}: {request.preset.title}")
        for need in request.preset.needs:
            print(f"# before submitting: {need}")
        print(command_line(request, args.out_dir))
        return 0

    out = []
    if args.out_dir:
        out += ["--out-dir", args.out_dir]
    for flag in ("dry_run", "force", "json"):
        if getattr(args, flag):
            out.append("--" + flag.replace("_", "-"))
    return bcj.main(request.argv + out)


if __name__ == "__main__":
    raise SystemExit(main())
