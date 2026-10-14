#!/usr/bin/env python3
"""Watch a swarm run from the coordinator's server log and raise the two #542 alarms.

Input: the coordinator's ``startup/nohup.out`` (every job of every server session, with
colour codes) or one job's ``workspace/log.txt`` from ``download_job``. The server sees
each client's status reports (``updated status of client <SITE> on round <R>: ...,
action=<A>``), which is enough to follow a p2p swarm without access to the sites.

Report: per round, when it started and finished, how many sites finished, and which
site finished last (with its training minutes); then which site was slowest over the
run. Two alarms, each with its own exit status:

* "all rounds done, not finished": every site finished the last training round, but the
  run did not end within ``--finish-grace`` minutes. In #542 a job sat RUNNING for 4.5 h
  after its rounds; until the job ends, cross_site_val/ is not written and the models
  are not published, and the server's own progress timeout is 24 h.
* "no progress": no status change for ``--stall`` minutes while the run is not finished.
  The server's progress_timeout is 24 h, so a hung transfer is otherwise noticed a day
  late.

Exit status (distinct, so a cron job or the deploy harness can act on it):

  0  finished cleanly, or still running with recent progress
  1  aborted (FATAL_SYSTEM_ERROR, abort request) or ended before its last round
  2  usage error, unreadable log, or no matching run in the log
  3  all training rounds done but not finished within --finish-grace minutes (#542)
  4  no progress for --stall minutes

Timestamps are the coordinator's clock; the container logs in UTC, so "now" is taken in
UTC unless ``--log-clock local``. A site whose clock is off (RUMC_1 runs ~2 h ahead)
does not matter: only the server's own timestamps are used.

Examples:
  watch_run.py /home/jeff/deploy_odelia_allsites/dl3.tud.de/startup/nohup.out
  watch_run.py nohup.out --job-id 7a678d9a --json
  watch_run.py nohup.out --follow --interval 300 --finish-grace 60 --stall 180
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

EXIT_OK = 0
EXIT_ABORTED = 1
EXIT_USAGE = 2
EXIT_FINISH_OVERDUE = 3
EXIT_NO_PROGRESS = 4

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,(?P<ms>\d{1,6}))? - (?P<logger>\S+) - (?P<level>[A-Z]+) - "
    r"(?P<rest>.*)$"
)
CTX_RE = re.compile(r"^\[(?P<ctx>[^\]]*)\] - (?P<msg>.*)$")
STATUS_RE = re.compile(
    r"updated status of client (?P<site>\S+) on round (?P<round>-?\d+|None): timestamp=[^,]*, "
    r"action=(?P<action>[^,]+), all_done=(?P<done>True|False)"
)
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# Cheap prefilter: an 88 MB nohup.out is mostly transfer and cell chatter.
KEYWORDS = (
    "updated status of client",
    "Server runner",
    "Using participating clients",
    "Starting client:",
    "Workflow Config:",
    "Got ALL_DONE",
    "finished on all clients",
    "FATAL_SYSTEM_ERROR",
    "asked to abort",
    "FaultTolerant: pruned",
    "deemed disconnected",
    "client configuration took",
    "Try to abort job",
)
FINISH_ACTIONS = {"finished_learn_task"}
# Actions that can only happen after a round's aggregation, i.e. after every site trained.
POST_AGGREGATION_ACTIONS = {"better_aggregation", "broadcast_best_result", "broadcast_last_result",
                            "finished_broadcast_last_result"}


def _parse_ts(text: str, ms: Optional[str]) -> _dt.datetime:
    ts = _dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if ms:
        ts = ts.replace(microsecond=int(ms.ljust(6, "0")[:6]))
    return ts


def _ctx_dict(ctx: str) -> Dict[str, str]:
    out = {}
    for part in ctx.split(","):
        key, sep, value = part.strip().partition("=")
        if sep:
            out[key] = value
    return out


@dataclass
class StatusEvent:
    ts: _dt.datetime
    site: str
    round: Optional[int]
    action: str
    all_done: bool


@dataclass
class Run:
    job_id: str
    first_ts: _dt.datetime
    last_ts: _dt.datetime
    sites: List[str] = field(default_factory=list)
    starting_client: Optional[str] = None
    num_rounds: Optional[int] = None
    started_ts: Optional[_dt.datetime] = None
    configure_seconds: Optional[float] = None
    events: List[StatusEvent] = field(default_factory=list)
    all_done_ts: Optional[_dt.datetime] = None
    workflow_done_ts: Optional[_dt.datetime] = None
    finished_ts: Optional[_dt.datetime] = None
    abort_reasons: List[str] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    disconnects: List[Tuple[_dt.datetime, str]] = field(default_factory=list)
    last_progress_ts: Optional[_dt.datetime] = None

    def progressed(self, ts: _dt.datetime) -> None:
        if self.last_progress_ts is None or ts > self.last_progress_ts:
            self.last_progress_ts = ts


class LogState:
    """Incremental parse of a server log; feed lines in file order."""

    def __init__(self, cutoff: Optional[_dt.datetime] = None) -> None:
        self.runs: Dict[str, Run] = {}
        self.order: List[str] = []
        self.current: Optional[str] = None  # run that unscoped lines are attributed to
        self.lines = 0
        # With --now, lines after that moment are ignored, so a post-mortem shows what a
        # watcher would have reported at the time.
        self.cutoff = cutoff

    def _run(self, job_id: str, ts: _dt.datetime) -> Run:
        run = self.runs.get(job_id)
        if run is None:
            run = Run(job_id, ts, ts)
            self.runs[job_id] = run
            self.order.append(job_id)
        run.last_ts = max(run.last_ts, ts)
        return run

    def feed(self, raw: str) -> None:
        self.lines += 1
        if not any(k in raw for k in KEYWORDS):
            return
        line = ANSI_RE.sub("", raw).rstrip("\n")
        match = LINE_RE.match(line)
        if not match:
            return
        ts = _parse_ts(match.group("ts"), match.group("ms"))
        if self.cutoff is not None and ts > self.cutoff:
            return
        rest = match.group("rest")
        ctx_match = CTX_RE.match(rest)
        ctx = _ctx_dict(ctx_match.group("ctx")) if ctx_match else {}
        msg = ctx_match.group("msg") if ctx_match else rest
        job_id = ctx.get("run")
        if job_id and UUID_RE.fullmatch(job_id):
            run = self._run(job_id, ts)
            if run.finished_ts is None:
                self.current = job_id
        else:
            run = self.runs.get(self.current) if self.current else None
            if "Try to abort job" in msg:
                ids = UUID_RE.findall(msg)
                if ids and ids[0] in self.runs:
                    target = self.runs[ids[0]]
                    target.abort_reasons.append("abort requested (admin abort_job or server shutdown)")
                return
            if run is None:
                return
            run.last_ts = max(run.last_ts, ts)
        self._handle(run, ts, msg)

    def _handle(self, run: Run, ts: _dt.datetime, msg: str) -> None:
        status = STATUS_RE.search(msg)
        if status:
            rnd = status.group("round")
            event = StatusEvent(
                ts,
                status.group("site"),
                None if rnd == "None" else int(rnd),
                status.group("action").strip(),
                status.group("done") == "True",
            )
            run.events.append(event)
            run.progressed(ts)
            return
        if "Server runner starting" in msg:
            run.started_ts = run.started_ts or ts
        elif msg.startswith("Using participating clients:"):
            run.sites = _literal(msg.split(":", 1)[1], list) or run.sites
        elif msg.startswith("Starting client:"):
            run.starting_client = msg.split(":", 1)[1].strip() or None
        elif msg.startswith("Workflow Config:"):
            cfg = _literal(msg.split(":", 1)[1], dict) or {}
            if isinstance(cfg.get("num_rounds"), int):
                run.num_rounds = cfg["num_rounds"]
            if not run.sites and isinstance(cfg.get("cwf.clients"), list):
                run.sites = list(cfg["cwf.clients"])
            if not run.starting_client and cfg.get("cwf.start_client"):
                run.starting_client = cfg["cwf.start_client"]
        elif msg.startswith("client configuration took"):
            nums = re.findall(r"[0-9.]+", msg)
            run.configure_seconds = float(nums[0]) if nums else None
            run.progressed(ts)
        elif msg.startswith("Got ALL_DONE"):
            run.all_done_ts = run.all_done_ts or ts
            run.progressed(ts)
        elif "finished on all clients" in msg:
            run.workflow_done_ts = run.workflow_done_ts or ts
            run.progressed(ts)
        elif "Server runner finished" in msg:
            run.finished_ts = ts
            run.progressed(ts)
            if self.current == run.job_id:
                self.current = None
        elif "FATAL_SYSTEM_ERROR" in msg:
            reason = msg.split("received:", 1)[1].strip() if "received:" in msg else msg
            run.abort_reasons.append(reason)
        elif "asked to abort" in msg:
            if not run.abort_reasons:
                run.abort_reasons.append("abort requested")
        elif "FaultTolerant: pruned" in msg:
            names = _literal(msg.split("pruned", 1)[1].split("(", 1)[0], list) or []
            run.pruned.extend(n for n in names if n not in run.pruned)
        elif "deemed disconnected" in msg:
            # Logged without a run tag; keep it only for a participant of the active run.
            site = re.search(r"Client (\S+) is deemed disconnected", msg)
            if site and (not run.sites or site.group(1) in run.sites):
                run.disconnects.append((ts, site.group(1)))


def _literal(text: str, kind: type):
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, kind) else None


# ── Analysis ──────────────────────────────────────────────────────────────


@dataclass
class RoundReport:
    round: int
    first_start: Optional[_dt.datetime]
    done_at: Optional[_dt.datetime]
    complete: bool
    finished: int
    expected: int
    slowest: Optional[str] = None
    slowest_train_min: Optional[float] = None
    train_min: Dict[str, float] = field(default_factory=dict)
    training: Dict[str, float] = field(default_factory=dict)  # site -> minutes so far (incomplete round)
    not_started: List[str] = field(default_factory=list)

    @property
    def duration_min(self) -> Optional[float]:
        if self.first_start and self.done_at:
            return (self.done_at - self.first_start).total_seconds() / 60.0
        return None


def _minutes(a: _dt.datetime, b: _dt.datetime) -> float:
    return (b - a).total_seconds() / 60.0


def analyse_rounds(run: Run, now: _dt.datetime) -> List[RoundReport]:
    sites = [s for s in run.sites if s not in run.pruned] or sorted({e.site for e in run.events})
    by_round: Dict[int, List[StatusEvent]] = {}
    for event in run.events:
        if event.round is not None:
            by_round.setdefault(event.round, []).append(event)
    if not by_round:
        return []
    last_round = max(by_round)
    if run.num_rounds is not None:
        last_round = min(max(last_round, 0), run.num_rounds - 1)
    reports = []
    for rnd in range(0, last_round + 1):
        events = by_round.get(rnd, [])
        starts: Dict[str, _dt.datetime] = {}
        finishes: Dict[str, _dt.datetime] = {}
        post_aggregation: Optional[_dt.datetime] = None
        for event in events:
            if event.action == "start_learn_task":
                starts.setdefault(event.site, event.ts)
            elif event.action in FINISH_ACTIONS:
                finishes.setdefault(event.site, event.ts)
            elif event.action in POST_AGGREGATION_ACTIONS:
                post_aggregation = post_aggregation or event.ts
        next_starts = [e.ts for e in by_round.get(rnd + 1, []) if e.action == "start_learn_task"]
        # The round's global exists once any site starts the next round, or once an
        # aggregator reports past aggregation (always the case for the last round), or
        # once a client reports ALL_DONE. Not "Workflow ... finished on all clients":
        # the server logs that on an abort too (job 513c8988 ended that way mid-round).
        candidates = [t for t in (min(next_starts) if next_starts else None, post_aggregation) if t]
        if run.all_done_ts and rnd == last_round:
            candidates.append(run.all_done_ts)
        round_end = min(candidates) if candidates else None
        complete = round_end is not None or (sites and all(s in finishes for s in sites))
        report = RoundReport(
            round=rnd,
            first_start=min(starts.values()) if starts else None,
            done_at=None,
            complete=bool(complete),
            finished=len([s for s in sites if s in finishes]),
            expected=len(sites),
        )
        if complete:
            # A site with no finish report finished just before the aggregation: its
            # finished_learn_task was overwritten by better_aggregation before the next
            # status report, which happens to the aggregator when it is the last to train.
            ends = {s: finishes.get(s, round_end) for s in sites if finishes.get(s, round_end) is not None}
            if ends:
                slowest = max(sorted(ends), key=lambda s: ends[s])
                report.done_at = ends[slowest]
                report.slowest = slowest
                report.finished = len(sites)
                for site, end in ends.items():
                    if site in starts:
                        report.train_min[site] = round(_minutes(starts[site], end), 2)
                report.slowest_train_min = report.train_min.get(slowest)
        else:
            for site in sites:
                if site in starts and site not in finishes:
                    report.training[site] = round(_minutes(starts[site], run.finished_ts or now), 1)
                elif site not in starts:
                    report.not_started.append(site)
        reports.append(report)
    return reports


@dataclass
class Verdict:
    code: int
    status: str
    detail: str
    rounds_complete: int
    all_rounds_done_at: Optional[_dt.datetime]


def judge(run: Run, reports: List[RoundReport], now: _dt.datetime, finish_grace_min: float,
          stall_min: float) -> Verdict:
    complete = [r for r in reports if r.complete]
    n_complete = len(complete)
    total = run.num_rounds
    all_done_at = None
    if total is not None and n_complete >= total and reports:
        all_done_at = reports[total - 1].done_at
    if run.finished_ts is not None:
        if run.abort_reasons:
            return Verdict(EXIT_ABORTED, "ABORTED", "; ".join(dict.fromkeys(run.abort_reasons)), n_complete, all_done_at)
        if total is not None and n_complete < total:
            return Verdict(EXIT_ABORTED, "ENDED EARLY",
                           f"server runner finished after {n_complete} of {total} rounds", n_complete, all_done_at)
        detail = f"finished at {run.finished_ts:%Y-%m-%d %H:%M:%S}"
        if all_done_at:
            detail += f", {_minutes(all_done_at, run.finished_ts):.1f} min after the last round"
        return Verdict(EXIT_OK, "FINISHED", detail, n_complete, all_done_at)
    if run.abort_reasons:
        return Verdict(EXIT_ABORTED, "ABORTING", "; ".join(dict.fromkeys(run.abort_reasons)), n_complete, all_done_at)
    if all_done_at is not None:
        waited = _minutes(all_done_at, now)
        if waited > finish_grace_min:
            return Verdict(
                EXIT_FINISH_OVERDUE,
                "ALL ROUNDS DONE, NOT FINISHED",
                f"last round done at {all_done_at:%Y-%m-%d %H:%M:%S}, {waited:.0f} min ago (grace {finish_grace_min:g} "
                "min); results are published only when the job ends (#542): check the final-result broadcast, "
                "then consider abort_job",
                n_complete,
                all_done_at,
            )
    last = run.last_progress_ts or run.last_ts
    idle = _minutes(last, now)
    if idle > stall_min:
        return Verdict(
            EXIT_NO_PROGRESS,
            "NO PROGRESS",
            f"no status change since {last:%Y-%m-%d %H:%M:%S} ({idle:.0f} min, limit {stall_min:g} min)",
            n_complete,
            all_done_at,
        )
    if all_done_at is not None:
        return Verdict(EXIT_OK, "FINISHING", f"all rounds done at {all_done_at:%H:%M:%S}; waiting for the end of run",
                       n_complete, all_done_at)
    return Verdict(EXIT_OK, "RUNNING", f"last progress {idle:.0f} min ago", n_complete, all_done_at)


def slowest_summary(reports: List[RoundReport]) -> Dict[str, Any]:
    complete = [r for r in reports if r.complete and r.slowest]
    last_counts: Dict[str, int] = {}
    per_site: Dict[str, List[float]] = {}
    for report in complete:
        last_counts[report.slowest] = last_counts.get(report.slowest, 0) + 1
        for site, minutes in report.train_min.items():
            per_site.setdefault(site, []).append(minutes)
    medians = {s: round(statistics.median(v), 1) for s, v in per_site.items()}
    if not last_counts:
        return {"site": None, "rounds_last": 0, "rounds": 0, "median_train_min": medians}
    site = max(sorted(last_counts), key=lambda s: last_counts[s])
    return {"site": site, "rounds_last": last_counts[site], "rounds": len(complete), "last_counts": last_counts,
            "median_train_min": medians}


# ── Output ────────────────────────────────────────────────────────────────


def _hm(ts: Optional[_dt.datetime]) -> str:
    return f"{ts:%m-%d %H:%M}" if ts else "-"


def text_report(run: Run, reports: List[RoundReport], verdict: Verdict, slow: Dict[str, Any]) -> str:
    sites = [s for s in run.sites if s not in run.pruned]
    out = [f"run {run.job_id}"]
    out.append(
        f"  sites ({len(sites)}): {', '.join(sites) or '?'}"
        + (f"   starting client {run.starting_client}" if run.starting_client else "")
    )
    if run.pruned:
        out.append(f"  pruned: {', '.join(run.pruned)}")
    started = run.started_ts or run.first_ts
    cfg = f", configured in {run.configure_seconds:.0f} s" if run.configure_seconds is not None else ""
    out.append(f"  started {started:%Y-%m-%d %H:%M:%S}{cfg}, {run.num_rounds if run.num_rounds is not None else '?'} rounds")
    out.append("  round  first start  done at      min  finished  last to finish (train min)")
    for r in reports:
        if r.complete:
            dur = f"{r.duration_min:6.1f}" if r.duration_min is not None else "     -"
            last = f"{r.slowest} ({r.slowest_train_min:.1f})" if r.slowest_train_min is not None else (r.slowest or "-")
            out.append(f"  {r.round:5d}  {_hm(r.first_start):11s}  {_hm(r.done_at):11s} {dur}  {r.finished:3d}/{r.expected:<3d}   {last}")
        else:
            busy = ", ".join(f"{s} {m:.0f} min" for s, m in sorted(r.training.items(), key=lambda kv: -kv[1]))
            waiting = f"; not started: {', '.join(r.not_started)}" if r.not_started else ""
            out.append(f"  {r.round:5d}  {_hm(r.first_start):11s}  {'running':11s}      -  {r.finished:3d}/{r.expected:<3d}   "
                       f"training: {busy or '-'}{waiting}")
    if slow.get("site"):
        medians = ", ".join(f"{s} {m}" for s, m in sorted(slow["median_train_min"].items(), key=lambda kv: -kv[1]))
        out.append(f"  slowest site: {slow['site']} finished last in {slow['rounds_last']} of {slow['rounds']} rounds")
        out.append(f"  median training minutes per round: {medians}")
    if slow.get("note"):
        out.append(f"  note: {slow['note']}")
    for ts, site in run.disconnects:
        out.append(f"  deemed disconnected: {site} at {ts:%Y-%m-%d %H:%M:%S}")
    total = run.num_rounds if run.num_rounds is not None else "?"
    out.append(f"  rounds complete: {verdict.rounds_complete}/{total}")
    out.append(f"STATUS {verdict.status} (exit {verdict.code}): {verdict.detail}")
    return "\n".join(out)


def json_report(run: Run, reports: List[RoundReport], verdict: Verdict, slow: Dict[str, Any]) -> Dict[str, Any]:
    def iso(ts):
        return ts.strftime("%Y-%m-%dT%H:%M:%S") if ts else None

    return {
        "job_id": run.job_id,
        "sites": run.sites,
        "pruned": run.pruned,
        "starting_client": run.starting_client,
        "num_rounds": run.num_rounds,
        "started": iso(run.started_ts or run.first_ts),
        "finished": iso(run.finished_ts),
        "last_progress": iso(run.last_progress_ts),
        "rounds": [
            {
                "round": r.round,
                "complete": r.complete,
                "first_start": iso(r.first_start),
                "done_at": iso(r.done_at),
                "duration_min": round(r.duration_min, 2) if r.duration_min is not None else None,
                "finished": r.finished,
                "expected": r.expected,
                "slowest": r.slowest,
                "slowest_train_min": r.slowest_train_min,
                "train_min": r.train_min,
                "training_min_so_far": r.training,
                "not_started": r.not_started,
            }
            for r in reports
        ],
        "slowest_site": slow,
        "disconnects": [[iso(ts), site] for ts, site in run.disconnects],
        "abort_reasons": run.abort_reasons,
        "status": verdict.status,
        "detail": verdict.detail,
        "rounds_complete": verdict.rounds_complete,
        "all_rounds_done_at": iso(verdict.all_rounds_done_at),
        "exit_code": verdict.code,
    }


# ── Command line ──────────────────────────────────────────────────────────


def select_run(state: LogState, job_id: Optional[str]) -> Optional[Run]:
    if job_id:
        hits = [j for j in state.order if j.startswith(job_id)]
        return state.runs[hits[-1]] if hits else None
    # The most recently started run (the one being watched, in practice).
    return state.runs[state.order[-1]] if state.order else None


def _now(args: argparse.Namespace) -> _dt.datetime:
    if args.now:
        return _dt.datetime.strptime(args.now, "%Y-%m-%d %H:%M:%S")
    if args.log_clock == "local":
        return _dt.datetime.now()
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


def evaluate(state: LogState, args: argparse.Namespace) -> Tuple[int, str, Optional[Run]]:
    run = select_run(state, args.job_id)
    if run is None:
        what = f"run {args.job_id}" if args.job_id else "any run"
        return EXIT_USAGE, f"no {what} in {args.log}", None
    now = _now(args)
    reports = analyse_rounds(run, now)
    verdict = judge(run, reports, now, args.finish_grace, args.stall)
    slow = slowest_summary(reports)
    medians = slow.get("median_train_min") or {}
    if medians and max(medians.values()) > args.stall:
        site = max(medians, key=medians.get)
        slow["note"] = (
            f"{site} trains {medians[site]} min per round (median), longer than --stall {args.stall:g}: "
            "status changes only at round boundaries, so raise --stall for this run"
        )
    if args.json:
        return verdict.code, json.dumps(json_report(run, reports, verdict, slow), indent=2), run
    return verdict.code, text_report(run, reports, verdict, slow), run


def read_lines(path: str, offset: int = 0) -> Tuple[Iterable[str], int]:
    with open(path, "r", errors="replace") as fh:
        fh.seek(offset)
        data = fh.read()
        return data.splitlines(keepends=True), fh.tell()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="watch_run.py",
        description="Per-round progress, slowest site and the #542 alarms from a coordinator server log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit: 0 ok, 1 aborted, 2 usage/no run, 3 all rounds done but not finished, 4 no progress",
    )
    p.add_argument("log", help="server log: startup/nohup.out or a job's workspace/log.txt")
    p.add_argument("--job-id", help="run to report (prefix is enough); default: the last run in the log")
    p.add_argument("--finish-grace", type=float, default=60.0,
                   help="minutes after the last round before 'not finished' fires (default 60)")
    p.add_argument("--stall", type=float, default=180.0,
                   help="minutes without a status change before 'no progress' fires (default 180)")
    p.add_argument("--now", help="evaluate as of 'YYYY-mm-dd HH:MM:SS' in the log's clock (post-mortems, tests)")
    p.add_argument("--log-clock", choices=("utc", "local"), default="utc", help="clock of the log timestamps")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.add_argument("--follow", action="store_true", help="re-read the log until the run ends or an alarm fires")
    p.add_argument("--interval", type=float, default=300.0, help="seconds between reads with --follow")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.finish_grace < 0 or args.stall <= 0:
        print("ERROR: --finish-grace must be >= 0 and --stall > 0", file=sys.stderr)
        return EXIT_USAGE
    if args.now:
        try:
            _dt.datetime.strptime(args.now, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("ERROR: --now must look like 2026-10-13 23:36:41", file=sys.stderr)
            return EXIT_USAGE
    cutoff = _now(args) if args.now else None
    state = LogState(cutoff)
    offset = 0
    last_text = None
    while True:
        try:
            size = os.path.getsize(args.log)
            if size < offset:  # truncated or replaced: start over
                state, offset = LogState(cutoff), 0
            lines, offset = read_lines(args.log, offset)
        except OSError as exc:
            print(f"ERROR: cannot read {args.log}: {exc}", file=sys.stderr)
            return EXIT_USAGE
        for line in lines:
            state.feed(line)
        code, text, run = evaluate(state, args)
        if not args.follow:
            print(text, file=sys.stdout if run is not None else sys.stderr)
            return code
        if text != last_text:
            print(text, flush=True)
            last_text = text
        # Keep following until the run ends or an alarm fires; with no run yet (watch
        # started before submit_job), keep waiting for it to appear.
        if run is not None and (code != EXIT_OK or run.finished_ts is not None):
            return code
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
