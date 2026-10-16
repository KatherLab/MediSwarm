#!/usr/bin/env python3
"""Fleet readiness: does every ODELIA site run the expected kit and image, right now?

For each site (default: the eight ODELIA hospital sites) this reads the newest heartbeat that
the site's live_sync uploaded to the coordinator and prints one line: kit version, image (tag
and short id, and whether that id is the expected image), heartbeat age, status, job, and a
verdict with its reason. Exit status 0 only when every requested site is READY, so it works as
the gate before a run (no fault tolerance: every participating site must be ready) and as the
progress list during a migration week.

    fleet_status.py --expect-kit 1.9.0 --expect-image jefftud/odelia:1.9.1
    fleet_status.py --sites CAM_1,UKA_1 --expect-image-id sha256:c527c8c509aa --json
    fleet_status.py --since 2026-10-19 --max-age 30m

Verdicts, first match wins:

  NO HEARTBEAT  no readable heartbeat for the site
  STALE         the newest heartbeat is older than --max-age: it proves nothing about what the
                site runs now (client or live_sync down). With --since, a heartbeat from before
                that date is called out as predating the migration.
  OLD KIT       kit_version is not --expect-kit, or another kit of the site still uploads
                fresh heartbeats (an old kit's live_sync outlived the move, F16)
  OLD IMAGE     the client's image id is not one of the expected image's ids
  STOPPED       fresh heartbeat, but it says the client container is not running (status
                stopped, client_state stopped, no image id) or live_sync has ended (final)
  READY         fresh, expected kit, expected image (or no image expectation given)

What a heartbeat is and when it is written (kit_live_sync/live_sync.sh, build_heartbeat.sh):

* live_sync is a host process that ``docker.sh --start_client`` starts next to the client
  container. Every LOG_SYNC_INTERVAL (30 s) it writes a heartbeat and uploads it to
  ``<live root>/<SITE>/swarm/<newest job folder in the kit>/heartbeat.json``. A kit without
  any job folder (a freshly unpacked 1.9.0 kit whose client idles) uploads to
  ``<SITE>/swarm/_active/heartbeat.json`` with status ``waiting``. So a migrated, idling
  1.9.0 client does heartbeat, before it ever receives a job. Once a job folder exists the
  heartbeat goes there with status ``running`` on every cycle, also after the job ended:
  ``running`` means "the daemon runs and this is the newest job", not "training".
* A 1.9.0 daemon (``live_sync_version`` 3) ends about 10 minutes after its kit's client
  container stopped (60 minutes after start if it never ran) and uploads a last heartbeat with
  status ``stopped``; when docker.sh stops it, it writes ``heartbeat_final.json``. Daemons of
  older kits (``sync_version`` 2) run until killed, whether or not a client runs.
* ``kit_version`` is the kit that started the daemon (passed by a 1.9.0 docker.sh, scraped from
  docker.sh before). ``image_id`` is ``docker inspect {{.Image}}`` of that kit's client
  container, cut to ``sha256:`` + 12 hex: the image CONFIG digest on Docker's classic image
  store (every site so far), the index or manifest digest on the containerd store. Older kits
  took the first running ``odelia_swarm_client`` container of the host. No running client
  container: empty image_id.
* ``timestamp`` is the site's clock. A timestamp in the future (RUMC_1 runs about 2 h ahead)
  is replaced by the upload time on this host (the file's ctime). A site clock that is behind
  makes a fresh heartbeat look old (a false STALE, never a false READY); the note says so.

So a fresh heartbeat proves that the site's live_sync of that kit runs and, with an image id,
which image the client container was created from. It does not prove the client is registered
with the server (admin console ``check_status server``) or that a job will configure (F8).

Image ids: ``--expect-image REF`` resolves REF with local ``docker image inspect`` only (the
local id, repo digests, the linux/amd64 manifest). On a host with the containerd image store
(the coordinator host) that cannot yield the config digest the classic-store sites report; the tool warns and
needs ``--expect-image-id <config digest>`` or ``--registry`` (which asks the registry through
scripts/build/heartbeat_image_id.py, the only network access, opt-in). Read-only: nothing is
written under the live root.

Exit status: 0 every site READY, 1 at least one site not READY, 2 usage error or an image
expectation that cannot be resolved.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_TOOLS = str(REPO_ROOT / "server_tools")
if _SERVER_TOOLS not in sys.path:
    sys.path.insert(0, _SERVER_TOOLS)
from heartbeat_io import HEARTBEAT_FILES, clean_value, parse_timestamp, read_heartbeat_file  # noqa: E402

EXIT_READY = 0
EXIT_NOT_READY = 1
EXIT_USAGE = 2

ODELIA_SITES = ["CAM_1", "MHA_1", "RSH_1", "RUMC_1", "UKA_1", "UMCU_1", "USZ_1", "VHIO_1"]
DEFAULT_LIVE_ROOT = "/srv/mediswarm/live"
DEFAULT_EXPECT_KIT = "1.9.0"
DEFAULT_MAX_AGE = "15m"
# A site timestamp this far ahead of our clock is a site clock error, not a heartbeat from
# the future: the upload time stands in for it.
CLOCK_TOLERANCE_SECONDS = 300

READY = "READY"
OLD_KIT = "OLD KIT"
OLD_IMAGE = "OLD IMAGE"
STOPPED = "STOPPED"
STALE = "STALE"
NO_HEARTBEAT = "NO HEARTBEAT"

FINAL_STATUSES = {"finished", "complete", "completed", "stopped"}
_SITE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")
SHORT_ID = 12


# --------------------------------------------------------------------------- helpers

def parse_duration(text: str) -> int:
    """'900', '30s', '15m', '2h', '1d' -> seconds."""
    m = re.fullmatch(r"\s*(\d+)\s*([smhd]?)\s*", str(text).lower())
    if not m:
        raise ValueError(f"not a duration: {text!r} (use e.g. 900, 15m, 2h, 1d)")
    return int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def parse_since(text: str) -> float:
    """'2026-10-19' or an ISO datetime (UTC unless it says otherwise) -> epoch seconds."""
    ts = parse_timestamp(text)
    if ts is None:
        raise ValueError(f"not a date: {text!r} (use e.g. 2026-10-19 or 2026-10-19T08:00Z)")
    return ts.timestamp()


def fmt_age(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"


def fmt_utc(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def id_hex(value: Any) -> str:
    """'sha256:CBF6ba...' -> 'cbf6ba...'; '' when it is not a hex digest."""
    text = clean_value(value).lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text if _HEX_RE.match(text or "-") else ""


def ids_match(a: str, b: str) -> bool:
    """Two digests (any length >= 12 hex) are the same image when the shorter is a prefix."""
    n = min(len(a), len(b))
    return n >= SHORT_ID and a[:n] == b[:n]


def version_tuple(text: str) -> Optional[Tuple[int, ...]]:
    m = re.match(r"v?(\d+(?:\.\d+)*)", text)
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def _upload_epoch(path: Path) -> Optional[float]:
    """When this host received the file: its ctime (rsync keeps the site's mtime, not the ctime)."""
    try:
        return os.stat(path).st_ctime
    except OSError:
        return None


def _mtime(path: Path) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


# --------------------------------------------------------------------------- heartbeats

@dataclass
class Beat:
    rel: str                      # <SITE>/<mode>/<run>/<file>, relative to the live root
    mode: str
    run: str
    name: str
    data: Dict[str, Any]
    error: str = ""
    when: float = 0.0             # epoch used for age and ordering
    site_time: Optional[float] = None
    clock_note: str = ""
    mtime: float = 0.0

    def get(self, key: str) -> str:
        return clean_value(self.data.get(key))


def _place_in_time(beat: Beat, upload: Optional[float], now: float) -> None:
    ts = parse_timestamp(beat.data.get("timestamp"))
    beat.site_time = ts.timestamp() if ts else None
    if beat.site_time is None:
        beat.when = upload if upload is not None else beat.mtime
        beat.clock_note = "no readable timestamp in the heartbeat: age from the upload time"
    elif beat.site_time - now > CLOCK_TOLERANCE_SECONDS:
        received = min(upload, now) if upload is not None else now
        beat.when = received
        beat.clock_note = (f"site clock about {fmt_age(beat.site_time - received)} ahead: "
                           f"age from the upload time")
    else:
        beat.when = beat.site_time
        if upload is not None and upload - beat.site_time > CLOCK_TOLERANCE_SECONDS and now - upload < 3600:
            beat.clock_note = (f"uploaded {fmt_age(now - upload)} ago but stamped {fmt_age(now - beat.site_time)} "
                               f"ago: site clock behind? (age uses the site's stamp)")


def collect_beats(live_root: Path, site: str, modes: Sequence[str], now: float) -> Tuple[bool, List[Beat]]:
    """(site folder exists, every heartbeat file of the site in these modes)."""
    site_dir = live_root / site
    if not site_dir.is_dir():
        return False, []
    beats: List[Beat] = []
    for mode in modes:
        mode_dir = site_dir / mode
        if not mode_dir.is_dir():
            continue
        for run_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
            for name in HEARTBEAT_FILES:
                path = run_dir / name
                if not path.is_file():
                    continue
                data = read_heartbeat_file(path)
                beat = Beat(rel=f"{site}/{mode}/{run_dir.name}/{name}", mode=mode, run=run_dir.name,
                            name=name, data=data, error=str(data.get("_parse_error", "")),
                            mtime=_mtime(path))
                if not beat.error:
                    _place_in_time(beat, _upload_epoch(path), now)
                beats.append(beat)
    return True, beats


# --------------------------------------------------------------------------- image ids

@dataclass
class ImageExpectation:
    refs: List[str] = field(default_factory=list)
    ids: List[Tuple[str, str]] = field(default_factory=list)     # (hex digest, where it came from)
    warnings: List[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.ids)

    def add(self, digest: str, source: str) -> None:
        h = id_hex(digest)
        if len(h) < SHORT_ID:
            raise ValueError(f"not an image id: {digest!r} (want sha256:<at least {SHORT_ID} hex>)")
        if not any(ids_match(h, known) for known, _ in self.ids):
            self.ids.append((h, source))

    def matches(self, image_id: str) -> bool:
        h = id_hex(image_id)
        return any(ids_match(h, known) for known, _ in self.ids)

    def describe(self) -> str:
        what = " / ".join(self.refs) or "image"
        return what + " = " + ", ".join(f"{h[:SHORT_ID]} ({src})" for h, src in self.ids)


def _docker_json(args: Sequence[str]) -> Any:
    """Parsed JSON of a local `docker ...` command, or None (no docker, unknown image)."""
    try:
        out = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except ValueError:
        return None


def _registry_config_digest(ref: str) -> str:
    """The linux/amd64 config digest from the registry (network), via heartbeat_image_id.py."""
    spec = importlib.util.spec_from_file_location("_heartbeat_image_id",
                                                  REPO_ROOT / "scripts" / "build" / "heartbeat_image_id.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config_digest(ref)


def _id_kind(entry: Dict[str, Any]) -> str:
    media = str((entry.get("Descriptor") or {}).get("mediaType", ""))
    if not entry.get("Descriptor"):
        return "config"               # classic image store: the image id IS the config digest
    if "index" in media or "manifest.list" in media:
        return "index"
    return "manifest"


def resolve_image(ref: str, expectation: ImageExpectation, use_registry: bool) -> None:
    """Add the ids under which a site's heartbeat can report image `ref`."""
    expectation.refs.append(ref)
    found = _docker_json(["image", "inspect", ref])
    have_config = False
    if isinstance(found, list) and found:
        entry = found[0]
        kind = _id_kind(entry)
        expectation.add(entry.get("Id", ""), f"{kind}, local docker")
        have_config = kind == "config"
        for repo_digest in entry.get("RepoDigests") or []:
            expectation.add(repo_digest.rpartition("@")[2], "repo digest, local docker")
        platform = _docker_json(["image", "inspect", "--platform", "linux/amd64", ref])
        if isinstance(platform, list) and platform:
            pkind = _id_kind(platform[0])
            expectation.add(platform[0].get("Id", ""), f"linux/amd64 {pkind}, local docker")
            have_config = have_config or pkind == "config"
    elif not use_registry:
        raise LookupError(f"{ref}: not in the local Docker image store (or docker is not available). "
                          f"Pull it, pass --registry, or give the ids with --expect-image-id")
    if not have_config:
        if use_registry:
            expectation.add(_registry_config_digest(ref), "config, registry")
        elif not any(src == "--expect-image-id" for _, src in expectation.ids):
            expectation.warnings.append(
                f"{ref}: this host's Docker uses the containerd image store, so the image's config "
                f"digest, which sites on the classic store report (every site so far), is not known "
                f"locally. Add --expect-image-id <config digest> (scripts/build/heartbeat_image_id.py "
                f"{ref} prints it) or --registry, or those sites show OLD IMAGE")


# --------------------------------------------------------------------------- verdict

@dataclass
class SiteResult:
    site: str
    verdict: str
    reason: str
    notes: List[str] = field(default_factory=list)
    beat: Optional[Beat] = None
    age_seconds: Optional[int] = None
    image_expected: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        b = self.beat
        d: Dict[str, Any] = {"site": self.site, "verdict": self.verdict, "ready": self.verdict == READY,
                             "reason": self.reason, "notes": self.notes,
                             "age_seconds": self.age_seconds, "image_expected": self.image_expected}
        if b is not None:
            d.update({
                "heartbeat": b.rel,
                "mode": b.mode,
                "job_id": b.get("job_id"),
                "kit_version": b.get("kit_version"),
                "image_ref": b.get("image_ref"),
                "image_id": b.get("image_id"),
                "status": b.get("status"),
                "sync_status": b.get("sync_status"),
                "client_state": b.get("client_state"),
                "live_sync_version": b.get("live_sync_version") or b.get("sync_version"),
                "timestamp": b.get("timestamp"),
                "heartbeat_utc": fmt_utc(b.when),   # the time its age is measured from
            })
        return d


def _kit_reason(kit: str, expect_kit: str) -> str:
    have, want = version_tuple(kit), version_tuple(expect_kit)
    if not kit:
        return f"no kit_version in the heartbeat, expected {expect_kit}"
    if have is not None and want is not None and have < want:
        return f"kit {kit}, expected {expect_kit}: not migrated"
    return f"kit {kit}, expected {expect_kit}"


def _client_down(beat: Beat) -> str:
    status = beat.get("status").lower()
    if beat.name == "heartbeat_final.json" or status in FINAL_STATUSES:
        if status == "stopped":
            return "live_sync stopped because the client container is gone (status stopped)"
        return f"live_sync ended ({beat.name}, status {status or '?'}): nothing reports for this site"
    state = beat.get("client_state").lower()
    if state == "stopped":
        return "the client container was not running when the heartbeat was written (client_state stopped)"
    if not beat.get("image_id"):
        if state == "unknown":
            return "live_sync could not ask Docker at the site (client_state unknown): client and image unproven"
        return "no client container found when the heartbeat was written (no image id)"
    return ""


def judge_site(site: str, folder_exists: bool, beats: List[Beat], *, now: float, expect_kit: str,
               image: ImageExpectation, max_age: int, since: Optional[float],
               include_local: bool) -> SiteResult:
    where = "swarm or local" if include_local else "swarm"
    if not folder_exists:
        return SiteResult(site, NO_HEARTBEAT, f"no folder {site} under the live root: this site never uploaded")
    good = sorted((b for b in beats if not b.error), key=lambda b: b.when, reverse=True)
    bad = sorted((b for b in beats if b.error), key=lambda b: b.mtime, reverse=True)
    if not good:
        if bad:
            return SiteResult(site, NO_HEARTBEAT, f"only unreadable heartbeats ({len(bad)}), newest "
                                                  f"{bad[0].rel}: {bad[0].error}")
        return SiteResult(site, NO_HEARTBEAT, f"no heartbeat under {site}/{where}")

    newest = good[0]
    age = int(now - newest.when)
    res = SiteResult(site, READY, "", beat=newest, age_seconds=age)
    if newest.clock_note:
        res.notes.append(newest.clock_note)
    if bad and bad[0].mtime > newest.mtime:
        res.notes.append(f"a newer heartbeat file is unreadable and was skipped: {bad[0].rel} ({bad[0].error})")

    kit = newest.get("kit_version")
    image_id = newest.get("image_id")
    if image.active and image_id:
        res.image_expected = image.matches(image_id)

    if age > max_age:
        res.verdict = STALE
        when = f"{fmt_age(age)} old ({fmt_utc(newest.when)})"
        if since is not None and newest.when < since:
            res.reason = (f"newest heartbeat {when} predates {fmt_utc(since)}: it proves nothing about the "
                          f"migration (kit {kit or '?'} then)")
        elif kit != expect_kit:
            res.reason = (f"newest heartbeat {when} is from kit {kit or '?'}: it proves nothing about the "
                          f"migration (a running {expect_kit} client would upload every 30 s)")
        else:
            res.reason = (f"newest heartbeat {when}, older than {fmt_age(max_age)}: the {kit} client or its "
                          f"live_sync is not running")
        return res

    # Another kit of this site still uploading (an old kit's live_sync outlived the move, F16).
    # A final heartbeat (heartbeat_final.json, status finished/stopped) is that daemon's last word.
    others = [b for b in good[1:] if now - b.when <= max_age and b.get("kit_version") != kit
              and b.name != "heartbeat_final.json" and b.get("status").lower() not in FINAL_STATUSES]
    for other in others:
        res.notes.append(f"kit {other.get('kit_version') or '?'} uploads too: {other.rel}, "
                         f"{fmt_age(now - other.when)} ago")

    if kit != expect_kit:
        res.verdict, res.reason = OLD_KIT, _kit_reason(kit, expect_kit)
        return res
    stray = [b for b in others if b.get("kit_version") != expect_kit]
    if stray:
        res.verdict = OLD_KIT
        res.reason = (f"kit {kit} reports, but kit {stray[0].get('kit_version') or '?'} still uploads for "
                      f"this site ({stray[0].mode}/{stray[0].run}): its live_sync (maybe its client) still "
                      f"runs; retire it with ./docker.sh --migrate_from <old kit> (F16/F17)")
        return res
    if res.image_expected is False:
        ref = newest.get("image_ref") or "?"
        hint = ("restart the client so docker.sh pulls :current" if ref.endswith(":current")
                else "pinned by image.conf or --image?")
        res.verdict = OLD_IMAGE
        res.reason = f"runs {ref} = {image_id}, not {' / '.join(image.refs) or 'the expected image'}: {hint}"
        return res
    down = _client_down(newest)
    if down:
        res.verdict, res.reason = STOPPED, down
        return res
    if newest.get("sync_status").lower() == "error" and newest.get("live_sync_version"):
        res.notes.append("uploads failing in the last cycle (sync_status error; a 1.9.0 daemon clears it "
                         "after a clean cycle)")
    res.reason = f"{newest.get('status') or 'heartbeat'}, kit {kit}" + (
        ", expected image" if image.active else "; image not checked")
    return res


# --------------------------------------------------------------------------- output

COLUMNS = ["SITE", "KIT", "IMAGE", "ID", "MATCH", "AGE", "STATUS", "JOB", "VERDICT", "REASON"]


def _row(r: SiteResult) -> List[str]:
    b = r.beat
    if b is None:
        return [r.site, "-", "-", "-", "-", "-", "-", "-", r.verdict, r.reason]
    image_id = id_hex(b.get("image_id"))[:SHORT_ID] or "-"
    match = {True: "yes", False: "no", None: "-"}[r.image_expected]
    status = "/".join(x for x in (b.get("status"), b.get("sync_status")) if x) or "-"
    job = b.get("job_id")[:8] if b.get("job_id") else ("waiting" if b.run == "_active" else "-")
    if b.mode != "swarm":
        job = f"{b.mode}:{b.run[:16]}"
    age = fmt_age(r.age_seconds or 0) + ("*" if b.clock_note else "")
    return [r.site, b.get("kit_version") or "-", b.get("image_ref") or "-", image_id, match, age, status,
            job, r.verdict, r.reason]


def render_table(results: List[SiteResult]) -> str:
    rows = [COLUMNS] + [_row(r) for r in results]
    widths = [max(len(row[i]) for row in rows) for i in range(len(COLUMNS) - 1)]
    lines = []
    for row in rows:
        cells = [cell.ljust(widths[i]) for i, cell in enumerate(row[:-1])]
        lines.append("  ".join(cells + [row[-1]]).rstrip())
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-root", default=DEFAULT_LIVE_ROOT, help=f"upload tree (default {DEFAULT_LIVE_ROOT})")
    ap.add_argument("--sites", action="append", default=None,
                    help="sites to check, comma-separated or repeated (default: the eight ODELIA sites)")
    ap.add_argument("--expect-kit", default=DEFAULT_EXPECT_KIT, help=f"kit version (default {DEFAULT_EXPECT_KIT})")
    ap.add_argument("--expect-image", action="append", default=[], metavar="REF",
                    help="image every site must run, e.g. jefftud/odelia:1.9.1; resolved with local docker inspect")
    ap.add_argument("--expect-image-id", action="append", default=[], metavar="ID",
                    help="an id the expected image may report (sha256:<12+ hex>); repeat for config/index/manifest")
    ap.add_argument("--registry", action="store_true",
                    help="resolve the config digest of --expect-image from the registry (network) when the "
                         "local image store cannot")
    ap.add_argument("--max-age", default=DEFAULT_MAX_AGE,
                    help=f"a heartbeat older than this is STALE (default {DEFAULT_MAX_AGE}; 900, 30m, 2h, 1d)")
    ap.add_argument("--since", default=None, metavar="DATE",
                    help="start of the migration (e.g. 2026-10-19): an older newest heartbeat is called out as "
                         "predating it")
    ap.add_argument("--include-local", action="store_true",
                    help="also count local-training heartbeats (default: swarm only, the client a run needs)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the table")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        max_age = parse_duration(args.max_age)
        since = parse_since(args.since) if args.since else None
    except ValueError as exc:
        print(f"fleet_status: {exc}", file=sys.stderr)
        return EXIT_USAGE

    sites: List[str] = []
    for chunk in args.sites or [",".join(ODELIA_SITES)]:
        sites += [s.strip() for s in chunk.split(",") if s.strip()]
    bad_sites = [s for s in sites if not _SITE_RE.match(s) or ".." in s]
    if bad_sites or not sites:
        print(f"fleet_status: invalid --sites {bad_sites or sites}", file=sys.stderr)
        return EXIT_USAGE
    live_root = Path(args.live_root)
    if not live_root.is_dir():
        print(f"fleet_status: live root {live_root} is not a folder", file=sys.stderr)
        return EXIT_USAGE

    image = ImageExpectation()
    try:
        for value in args.expect_image_id:
            image.add(value, "--expect-image-id")
        for ref in args.expect_image:
            resolve_image(ref, image, args.registry)
    except (ValueError, LookupError, OSError, subprocess.SubprocessError, SystemExit) as exc:
        print(f"fleet_status: {exc}", file=sys.stderr)
        return EXIT_USAGE

    now = time.time()
    modes = ["swarm", "local"] if args.include_local else ["swarm"]
    results = []
    for site in dict.fromkeys(sites):
        exists, beats = collect_beats(live_root, site, modes, now)
        results.append(judge_site(site, exists, beats, now=now, expect_kit=clean_value(args.expect_kit),
                                  image=image, max_age=max_age, since=since,
                                  include_local=args.include_local))
    ready = sum(r.verdict == READY for r in results)
    all_ready = ready == len(results)

    if args.json:
        print(json.dumps({
            "generated_utc": fmt_utc(now),
            "live_root": str(live_root),
            "expect_kit": args.expect_kit,
            "expect_image": {"refs": image.refs, "ids": [{"id": "sha256:" + h, "source": s} for h, s in image.ids],
                             "checked": image.active, "warnings": image.warnings},
            "max_age_seconds": max_age,
            "since_utc": fmt_utc(since) if since is not None else None,
            "ready": all_ready,
            "ready_sites": ready,
            "sites": [r.as_dict() for r in results],
        }, indent=2))
        return EXIT_READY if all_ready else EXIT_NOT_READY

    print(f"Fleet status {fmt_utc(now)}  ({live_root}, {'swarm + local' if args.include_local else 'swarm'} heartbeats, "
          f"stale after {fmt_age(max_age)}" + (f", migration since {fmt_utc(since)}" if since is not None else "") + ")")
    print(f"expected kit {args.expect_kit}; image "
          + (image.describe() if image.active else "NOT CHECKED (no --expect-image / --expect-image-id)"))
    for warning in image.warnings:
        print(f"[WARN] {warning}")
    print()
    print(render_table(results))
    notes = [(r.site, n) for r in results for n in r.notes]
    if notes or any(r.beat is not None and r.beat.clock_note for r in results):
        print()
        for site, note in notes:
            print(f"  {site}: {note}")
        if any(r.beat is not None and r.beat.clock_note for r in results):
            print("  (* age not from the site's own timestamp, see the note)")
    print()
    print(f"{ready}/{len(results)} READY" + ("" if all_ready else
          ": not every site is ready (exit 1)"))
    return EXIT_READY if all_ready else EXIT_NOT_READY


if __name__ == "__main__":
    sys.exit(main())
