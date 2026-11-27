#!/usr/bin/env python3
"""Non-interactive admin commands for a MediSwarm swarm, without the interactive console.

The interactive `fl_admin.sh` console is awkward to script and to use over ssh. This tool
runs one admin command through NVFlare's FLARE API (`new_secure_session`) and prints the
result, so an operator (or a script) can check the swarm, submit or abort a job, fetch a
finished job's server workspace, or read a site's job log.

It needs nvflare, so run it with the swarm's own Docker image and the admin startup kit
mounted. Example for a DECADE admin kit on the server host (the server's name is mapped to
the loopback address inside the container only, as the server kit does):

    KIT=/path/to/admin_kit             # the folder with startup/fed_admin.json
    JOBS=/path/to/jobs                 # must be WRITABLE: NVFlare signs a job folder in place
    docker run --rm --net=host --add-host dl3.tud.de:127.0.0.1 \\
        -u "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \\
        -v "$KIT":/fl_admin -v "$JOBS":/jobs -v "$PWD/scripts/admin/flare_admin.py":/flare_admin.py:ro \\
        -w /fl_admin jefftud/decade:1.7.0 python3 /flare_admin.py status

Commands:

    status                         server status and every client's job status
    list                           all jobs with their status (e.g. RUNNING, FINISHED:COMPLETED)
    meta JOB_ID                    one job's metadata
    submit JOB_DIR                 submit a job folder (path inside the container, e.g. /jobs/NAME)
    abort JOB_ID --yes             abort a running job (refuses without --yes)
    download JOB_ID                fetch a finished job's server workspace into the kit's transfer/
    grep SITE PATTERN FILE [-o O]  grep a file in a site's workspace, e.g. JOB_ID/log.txt
    tail SITE FILE [-n N]          last N lines of a file in a site's workspace
    ls SITE [PATH]                 list a folder in a site's workspace

`grep`, `tail` and `ls` are NVFlare shell commands. They work only where the site's
authorization policy allows shell commands for the admin's role (the default policy gives a
project_admin "any"), and only inside that site's client workspace (its kit folder): job
logs and `startup/.mediswarm_sync/live_sync_daemon.log`, never the site's data. Read only
what you need to diagnose a failure, and tell the site what you read.

Exit status: 0 on success, 1 when the server reports an error, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Optional, Sequence

DEFAULT_KIT = "/fl_admin"
READ_ONLY_HINT = (
    "The job folder is read-only, but NVFlare signs a job folder in place before it uploads it. "
    "Mount the jobs folder writable (no ':ro')."
)


def admin_username(kit_dir: str) -> str:
    """The admin's user name from the kit's startup/fed_admin.json."""
    path = os.path.join(kit_dir, "startup", "fed_admin.json")
    try:
        with open(path, encoding="utf-8") as f:
            name = json.load(f)["admin"]["username"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"cannot read the admin user name from {path}: {e}; pass --username") from None
    if not name:
        raise SystemExit(f"{path} has no admin username; pass --username")
    return name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run one NVFlare admin command non-interactively (status, list, submit, abort, logs).",
    )
    p.add_argument("--kit", default=DEFAULT_KIT, help=f"admin startup kit folder (default {DEFAULT_KIT})")
    p.add_argument("--username", help="admin user name (default: from the kit's startup/fed_admin.json)")
    p.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for the server (default 60)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="server status and every client's job status")
    sub.add_parser("list", help="all jobs with their status")
    s = sub.add_parser("meta", help="one job's metadata")
    s.add_argument("job_id")
    s = sub.add_parser("submit", help="submit a job folder (path inside the container)")
    s.add_argument("job_dir")
    s = sub.add_parser("abort", help="abort a running job")
    s.add_argument("job_id")
    s.add_argument("--yes", action="store_true", help="really abort (required)")
    s = sub.add_parser("download", help="fetch a finished job's server workspace into the kit's transfer/")
    s.add_argument("job_id")
    s = sub.add_parser("grep", help="grep a file in a site's workspace")
    s.add_argument("site")
    s.add_argument("pattern")
    s.add_argument("file", help="path relative to the site's workspace, e.g. JOB_ID/log.txt")
    s.add_argument("-o", "--options", default="-n", help="grep options (default -n)")
    s = sub.add_parser("tail", help="last lines of a file in a site's workspace")
    s.add_argument("site")
    s.add_argument("file", help="path relative to the site's workspace")
    s.add_argument("-n", "--lines", type=int, default=40)
    s = sub.add_parser("ls", help="list a folder in a site's workspace")
    s.add_argument("site")
    s.add_argument("path", nargs="?")
    return p


def run(args: argparse.Namespace, sess) -> str:
    """Run the parsed command on an open FLARE API session and return what to print."""
    cmd = args.command
    if cmd == "status":
        return f"{sess.check_status('server')}\n{sess.check_status('client')}"
    if cmd == "list":
        return "\n".join(str(j) for j in sess.list_jobs(detailed=False))
    if cmd == "meta":
        return str(sess.get_job_meta(args.job_id))
    if cmd == "submit":
        try:
            return f"JOB_ID {sess.submit_job(args.job_dir)}"
        except Exception as e:  # noqa: BLE001 — the FLARE API raises plain RuntimeError
            if "Read-only file system" in str(e):
                raise RuntimeError(f"{e}\n{READ_ONLY_HINT}") from None
            raise
    if cmd == "abort":
        if not args.yes:
            raise SystemExit(f"abort {args.job_id}: add --yes to really abort the job")
        return str(sess.abort_job(args.job_id))
    if cmd == "download":
        return str(sess.download_job_result(args.job_id))
    if cmd == "grep":
        return sess.grep_target(args.site, options=args.options or None, pattern=args.pattern, file=args.file)
    if cmd == "tail":
        return sess.tail_target(args.site, options=f"-n {args.lines}", file=args.file)
    if cmd == "ls":
        return sess.ls_target(args.site, options=None, path=args.path)
    raise SystemExit(f"unknown command {cmd}")  # argparse prevents this


def main(argv: Optional[Sequence[str]] = None, session_factory: Optional[Callable[[argparse.Namespace], object]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "abort" and not args.yes:
        print(f"abort {args.job_id}: add --yes to really abort the job", file=sys.stderr)
        return 2
    if session_factory is None:
        from nvflare.fuel.flare_api.flare_api import new_secure_session

        username = args.username or admin_username(args.kit)

        def session_factory(a: argparse.Namespace):
            return new_secure_session(username=username, startup_kit_location=a.kit, timeout=a.timeout)

    sess = session_factory(args)
    try:
        print(run(args, sess))
        return 0
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        sess.close()


if __name__ == "__main__":
    sys.exit(main())
