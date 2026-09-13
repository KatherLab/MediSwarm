#!/usr/bin/env python3
"""Freeze the ODELIA run history into a committed dataset.

Why this exists
---------------
The quantitative record of the consortium is decaying and partly untrustworthy:

* The coordinator's job store retains only ~22 jobs. The remaining ~150 exist
  solely under /srv/mediswarm/live/<SITE>/swarm/<job_id>/, a directory that keeps
  changing as runs come and go.
* **The site label is not the site.** Internal TU Dresden machines (dl0, dl2, dl3)
  run under real hospital site names during deploy tests, so records from
  `RUMC_1` may have been produced by dl0 and records from `MHA_1` by dl2. Reading
  the site directory name alone silently mixes test infrastructure into hospital
  data -- it produced four wrong "join dates" in an earlier analysis, including a
  claim that Utrecht joined in April with 122 volumes when it joined in July with
  6,133.

Each job directory carries a `heartbeat.json` recording the **hostname** and
**ip_address** that actually produced it. That is the field to trust, and this
script filters on it.

Output
------
`run_history.csv`  one row per (job_id, site) with what that site did in that run
`run_summary.json` one entry per job_id, aggregated across sites

Usage
    python3 scripts/analysis/extract_run_history.py [--live-root DIR] [--out DIR]
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

# Machines that belong to the coordinating team, not to a hospital. A record
# from one of these is test infrastructure regardless of the site name it wears.
# Deploy tests run on our own machines. Records from them must not be counted as
# hospital runs -- doing so put four site join dates ~6 weeks early in a published
# chart, because April records from dl0/dl3 carried hospital site names.
#
# From 1.8.0 the deploy-test projects use TEST_A_1..TEST_D_1 instead of real site
# names (see application/provision/project_deploy_test_*site.yml), so new records
# are unambiguous by name alone. This hostname heuristic stays because the
# historical records it was written for do not change.
TEST_HOSTS = {"dl0.tud.de", "dl2.tud.de", "dl3.tud.de", "cosmos", "agh1"}
TEST_SITE_NAMES = {"TEST_A_1", "TEST_B_1", "TEST_C_1", "TEST_D_1"}

SITES = ["CAM_1", "MHA_1", "RSH_1", "RUMC_1", "UKA_1", "UMCU_1", "USZ_1", "VHIO_1"]

# Failure signatures, most specific first. Order matters: a run can show several,
# and we want the one that names the cause rather than a downstream symptom.
#
# ADVISORY vs DISCRIMINATIVE. Validated against the 22 runs whose terminal status
# the coordinator still records:
#
#   liveness_timeout      7 failures, 0 successes   discriminative
#   client_disconnected   3 failures, 0 successes   discriminative
#   posix_spawn           2 failures, 1 SUCCESS     advisory
#   vpn_or_peer_transfer  1 failure,  1 SUCCESS     advisory
#
# posix_spawn fires in 70 of 120 jobs including ones that completed cleanly
# (fd8ce441), so treating it as a cause would have dominated a failure Pareto with
# a warning. A log signature is not a failure cause until it has been checked
# against runs that succeeded. One failure (831d16b4) matches no signature at all,
# so coverage is incomplete and the taxonomy must say so rather than imply that
# every failure is classified.
ADVISORY = {"posix_spawn", "vpn_or_peer_transfer"}

SIGNATURES = [
    ("vpn_or_peer_transfer", r"failed to download from|Stream error|ConnectionError: Connection lost"),
    ("client_disconnected",  r"is deemed disconnected"),
    ("liveness_timeout",     r"didn't report status|FATAL_SYSTEM_ERROR"),
    ("posix_spawn",          r"posix_spawn failed"),
    ("stale_lock",           r"There seems to be one instance|daemon_pid\.fl"),
    ("cert_mismatch",        r"ClientConnectorCertificateError"),
    ("min_clients",          r"min_clients \(\d+\) exceeds"),
    ("gpu_error",            r"NVML: Unknown Error|CUDA error|no CUDA-capable device"),
    ("oom_or_shm",           r"Bus error|out of memory|shared memory"),
    ("data_missing",         r"No such file or directory: .*annotation\.csv|FileNotFoundError"),
    ("read_only_fs",         r"Read-only file system"),
]


def read_text(path):
    try:
        with open(path, errors="ignore") as fh:
            return re.sub(r"\x1b\[[0-9;]*m", "", fh.read())
    except Exception:
        return ""


# Deploy-test kits install under .../deploy_test/<SITE>; production kits live under
# the site's own MediSwarm workspace. For runs predating heartbeat.json this path,
# which appears in the console log, is the only surviving attribution.
TEST_PATH_MARK = re.compile(r"deploy_test|/fl_admin/local/mediswarm_jobs")
PROD_PATH_MARK = re.compile(r"Documents/MediSwarm/workspace|/startupkit/[0-9a-f]{8}-")


def classify_host(host, text="", site=""):
    """real | test | unknown -- the distinction the site *label* used to be unable to make.

    From 1.8.0 a TEST_* site name is decisive on its own; that is the whole point
    of renaming the deploy-test participants. Everything below is for the archive
    written before the rename, where the label was a real hospital name whatever
    machine produced it.

    heartbeat.json's hostname is authoritative when present. Roughly a third of
    the archive predates it, so fall back to the kit path in the console log:
    a deploy test installs under .../deploy_test/<SITE>, production does not.
    """
    if site in TEST_SITE_NAMES:
        return "test"
    if host and host != "?":
        return "test" if host in TEST_HOSTS else "real"
    if text:
        if TEST_PATH_MARK.search(text):
            return "test"
        if PROD_PATH_MARK.search(text):
            return "real"
    return "unknown"


def parse_job_dir(job_dir, site):
    hb_path = os.path.join(job_dir, "heartbeat.json")
    hb = {}
    if os.path.exists(hb_path):
        try:
            hb = json.load(open(hb_path))
        except Exception:
            hb = {}

    text = read_text(os.path.join(job_dir, "nohup.out")) + \
           read_text(os.path.join(job_dir, "log.txt"))

    rec = {
        "job_id":     os.path.basename(job_dir.rstrip("/")),
        "site":       site,
        "hostname":   hb.get("hostname", "") or "?",
        "ip_address": hb.get("ip_address", "") or "",
        "host_class": "",   # filled below, needs the log text
        "kit_version": hb.get("kit_version", ""),
        "image_ref":  hb.get("image_ref", ""),
        "mode":       hb.get("mode", ""),
        "run_name":   hb.get("run_name", ""),
        "hb_status":  hb.get("status", ""),
    }

    rec["host_class"] = classify_host(hb.get("hostname", ""), text, site)

    dates = sorted(set(re.findall(r"(20\d\d-\d\d-\d\d) \d\d:\d\d:\d\d", text)))
    rec["date_first"] = dates[0] if dates else ""
    rec["date_last"]  = dates[-1] if dates else ""

    ns = [int(n) for n in re.findall(r"train_samples=(\d+)", text)]
    rec["train_n"] = max(ns) if ns else ""

    cc = re.findall(r"Class counts: \{([^}]*)\}", text)
    if cc:
        d = {int(k): int(v) for k, v in re.findall(r"(\d+):\s*(\d+)", cc[-1])}
        # A class absent from the dict has zero examples -- that is a fact about the
        # site, not missing data. VHIO genuinely has no benign cases; writing that as
        # blank would lose the most important thing the row says.
        rec["class_0"], rec["class_1"], rec["class_2"] = d.get(0, 0), d.get(1, 0), d.get(2, 0)
    else:
        rec["class_0"] = rec["class_1"] = rec["class_2"] = ""

    rec["learn_tasks"] = len(re.findall(r"start_learn_task", text))
    rec["epochs_seen"] = len(re.findall(r"Epoch \d+", text))
    rec["configured"]  = 1 if re.search(r"successfully configured client", text) else 0

    hits = [name for name, pat in SIGNATURES if re.search(pat, text)]
    rec["signatures"] = "|".join(hits)
    # Only a discriminative signature may be reported as the cause.
    strong = [h for h in hits if h not in ADVISORY]
    rec["failure_class"] = strong[0] if strong else ""
    rec["advisory_only"] = 1 if (hits and not strong) else 0
    return rec



COORD_CONTAINER = "odelia_swarm_server_flserver_a19be57"


def coordinator_jobs(container=COORD_CONTAINER):
    """Pull the runs the coordinator still holds.

    The two archives are almost disjoint. The live monitor keeps older per-site
    history; the coordinator keeps a short tail of recent runs and prunes the
    rest -- and that tail contains the runs that matter most, including the
    twenty-round eight-site benchmark. Reading only one of them silently omits
    either the history or the headline results.
    """
    import subprocess

    def sh(cmd):
        try:
            r = subprocess.run(["docker", "exec", container, "sh", "-c", cmd],
                               capture_output=True, text=True, timeout=120)
            return r.stdout
        except Exception:
            return ""

    ids = [j for j in sh("ls /tmp/nvflare/jobs-storage 2>/dev/null").split() if len(j) > 8]
    out = []
    for jid in ids:
        meta = sh(f"cat /tmp/nvflare/jobs-storage/{jid}/meta 2>/dev/null")
        try:
            m = json.loads(meta)
        except Exception:
            m = {}
        log = sh(
            "python3 -c \"import zipfile,sys;"
            f"z=zipfile.ZipFile('/tmp/nvflare/jobs-storage/{jid}/workspace');"
            "print(z.read('log.txt').decode('utf8','ignore')[:400000])\" 2>/dev/null"
        )
        sites = sorted(set(re.findall(r"successfully configured client ([A-Za-z0-9_]+)", log)))
        dates = sorted(set(re.findall(r"(20\d\d-\d\d-\d\d) \d\d:\d\d:\d\d", log)))
        hits = [name for name, pat in SIGNATURES if re.search(pat, log)]
        out.append({
            "job_id": jid,
            "date_first": dates[0] if dates else "",
            "date_last": dates[-1] if dates else "",
            "n_sites": len(sites), "sites": sites,
            "status": m.get("status", ""), "duration": m.get("duration", ""),
            "learn_tasks": len(re.findall(r"start_learn_task", log)),
            "failure_classes": sorted(set(hits)),
            "source": "coordinator",
        })
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-root", default="/srv/mediswarm/live")
    ap.add_argument("--out", default="workspace/run_history")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for site in SITES:
        for job_dir in sorted(glob.glob(os.path.join(args.live_root, site, "swarm", "*"))):
            if os.path.isdir(job_dir):
                rows.append(parse_job_dir(job_dir, site))

    cols = ["job_id", "site", "hostname", "ip_address", "host_class", "date_first",
            "date_last", "kit_version", "image_ref", "mode", "run_name", "hb_status",
            "train_n", "class_0", "class_1", "class_2", "learn_tasks", "epochs_seen",
            "configured", "failure_class", "advisory_only", "signatures"]
    csv_path = os.path.join(args.out, "run_history.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    jobs = defaultdict(lambda: {"sites": [], "real_sites": [], "test_sites": []})
    for r in rows:
        j = jobs[r["job_id"]]
        j["sites"].append(r["site"])
        j[("real_sites" if r["host_class"] == "real" else "test_sites")].append(r["site"])
        for k in ("date_first", "date_last"):
            cur = j.get(k, "")
            if r[k] and (not cur or (r[k] < cur if k == "date_first" else r[k] > cur)):
                j[k] = r[k]
        if r["failure_class"]:
            j.setdefault("failure_classes", []).append(r["failure_class"])
        j["learn_tasks"] = j.get("learn_tasks", 0) + r["learn_tasks"]

    summary = []
    for job_id, j in jobs.items():
        summary.append({
            "job_id": job_id,
            "date_first": j.get("date_first", ""), "date_last": j.get("date_last", ""),
            "n_sites": len(set(j["sites"])),
            "n_real_sites": len(set(j["real_sites"])),
            "n_test_sites": len(set(j["test_sites"])),
            "is_production": len(set(j["real_sites"])) > 0 and len(set(j["test_sites"])) == 0,
            "learn_tasks": j.get("learn_tasks", 0),
            "failure_classes": sorted(set(j.get("failure_classes", []))),
        })
    for r in summary:
        r["source"] = "live_monitor"

    # merge the coordinator's tail; it holds runs the monitor never received
    seen = {r["job_id"] for r in summary}
    coord_added = 0
    for c in coordinator_jobs():
        if c["job_id"] in seen:
            for r in summary:
                if r["job_id"] == c["job_id"]:
                    r["status"], r["duration"] = c["status"], c["duration"]
                    r["source"] = "both"
        else:
            summary.append({
                "job_id": c["job_id"], "date_first": c["date_first"],
                "date_last": c["date_last"], "n_sites": c["n_sites"],
                "n_real_sites": c["n_sites"], "n_test_sites": 0,
                "is_production": c["n_sites"] > 0,
                "learn_tasks": c["learn_tasks"],
                "failure_classes": c["failure_classes"],
                "status": c["status"], "duration": c["duration"],
                "source": "coordinator",
            })
            coord_added += 1

    summary.sort(key=lambda x: x["date_first"] or "")
    json.dump(summary, open(os.path.join(args.out, "run_summary.json"), "w"), indent=1)

    real = [r for r in rows if r["host_class"] == "real"]
    test = [r for r in rows if r["host_class"] == "test"]
    print(f"  site-run rows : {len(rows)}   ({len(real)} real host, {len(test)} test host, "
          f"{len(rows)-len(real)-len(test)} unknown)")
    print(f"  unique jobs   : {len(summary)}  "
          f"({len(jobs)} from the live monitor, {coord_added} only in the coordinator)")
    print(f"  production    : {sum(1 for s in summary if s['is_production'])}")
    print(f"  wrote {csv_path}")
    print(f"  wrote {os.path.join(args.out, 'run_summary.json')}")


if __name__ == "__main__":
    main()
