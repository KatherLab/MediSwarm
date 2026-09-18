#!/usr/bin/env python3
"""Validate the lessons-learned evidence matrix and emit the publishable view.

The matrix (``docs/manuscript/evidence_matrix.csv``) is both the analysis
instrument and the paper's appendix. This script enforces the two rules the
manuscript plan commits to, mechanically rather than by good intentions:

**1. Every claim carries a pointer.** A row with no ``evidence``, ``issue``,
``pr`` or ``job_id`` is reported as ``[UNSOURCED]`` rather than being left to
look established.

**2. The anonymisation rule.** Sites are named where the fact is neutral and
verifiable, and anonymised wherever the variable describes *participation* --
response latency, unavailable periods, follow-ups required, support burden.
Rows carry ``attribution_kind``: ``technical`` may name the site,
``participation`` must not. The publishable view rewrites the latter to
``Site A``..``Site H`` using a fixed mapping.

The mapping is deliberately **not** ordered by dataset size. Per-site sizes are
published under real names elsewhere, so a size-ordered mapping would
re-identify every anonymised row on sight.

Usage
    python3 scripts/manuscript/build_evidence_matrix.py            # check only
    python3 scripts/manuscript/build_evidence_matrix.py --publish  # + write view
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MATRIX = os.path.join(ROOT, "docs", "manuscript", "evidence_matrix.csv")
OUT_MD = os.path.join(ROOT, "docs", "manuscript", "evidence_matrix_public.md")

# Fixed, non-size-ordered. Changing this invalidates every published reference.
PSEUDONYM = {
    "USZ": "Site A", "RSH": "Site B", "UKA": "Site C", "VHIO": "Site D",
    "MHA": "Site E", "UMCU": "Site F", "CAM": "Site G", "RUMC": "Site H",
}

LESSONS = {
    1: "Consortium governance is part of the technical system",
    2: "Site readiness is staged, not binary",
    3: "Infrastructure heterogeneity dominates real distributed training",
    4: "Detect before training, contain locally",
    5: "Observability and provenance are prerequisites",
    6: "Documentation, testing and automation encode consortium knowledge",
    7: "Successful distributed computation does not imply valid evaluation",
    8: "Prototype to consortium requires organisational and architectural redesign",
}

REQUIRED = ["id", "date", "phase", "lesson", "site_real", "attribution_kind",
            "symptom", "root_cause", "impact", "evidence", "issue", "pr",
            "job_id", "preventive_change", "general_lesson"]


def load():
    with open(MATRIX, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("evidence matrix is empty")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        sys.exit(f"matrix is missing columns: {missing}")
    return rows


def sites_of(row):
    return [s.strip() for s in row["site_real"].split(",") if s.strip()]


def check(rows):
    """Returns (errors, warnings). Errors block publication."""
    errors, warns = [], []
    seen = set()

    for r in rows:
        rid = r["id"]
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)

        if not (r["evidence"] or r["issue"] or r["pr"] or r["job_id"]):
            warns.append(f"{rid}: [UNSOURCED] — no evidence, issue, PR or job id")

        kind = r["attribution_kind"]
        if kind not in ("technical", "participation"):
            errors.append(f"{rid}: attribution_kind must be technical|participation, got {kind!r}")

        for s in sites_of(r):
            if s not in PSEUDONYM:
                errors.append(f"{rid}: unknown site {s!r}")

        try:
            n = int(r["lesson"])
            if n not in LESSONS:
                errors.append(f"{rid}: lesson {n} is not one of 1-8")
        except ValueError:
            errors.append(f"{rid}: lesson must be an integer, got {r['lesson']!r}")

        # The rule that actually matters: a participation row must not carry a
        # real site name anywhere a reader could see it.
        if kind == "participation":
            for s in sites_of(r):
                for col in ("symptom", "root_cause", "impact", "general_lesson"):
                    if s in r[col]:
                        errors.append(
                            f"{rid}: participation row names {s} in '{col}' — "
                            f"anonymise the prose, not just the site column")
    return errors, warns


def publish_row(r):
    out = dict(r)
    if r["attribution_kind"] == "participation":
        out["site"] = ", ".join(PSEUDONYM[s] for s in sites_of(r)) or "—"
    else:
        out["site"] = ", ".join(sites_of(r)) or "—"
    del out["site_real"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true",
                    help="write the anonymised Markdown appendix")
    args = ap.parse_args()

    rows = load()
    errors, warns = check(rows)

    print(f"\nevidence matrix: {len(rows)} rows, {len(REQUIRED)} columns\n")

    by_lesson = defaultdict(list)
    for r in rows:
        by_lesson[r["lesson"]].append(r["id"])
    print("Coverage by lesson")
    gaps = []
    for n, title in LESSONS.items():
        ids = by_lesson.get(str(n), [])
        mark = "  " if ids else "!!"
        print(f" {mark} {n}. {title[:58]:<58} {len(ids):>2}  {' '.join(ids)}")
        if not ids:
            gaps.append(n)
    if gaps:
        print(f"\n    Lessons {', '.join(map(str, gaps))} have no incident rows. That is expected"
              f"\n    for arguments carried by correspondence and architecture rather than"
              f"\n    incidents — but the manuscript must not imply the matrix supports them.")

    print(f"\nPhases: {dict(Counter(r['phase'] for r in rows))}")
    print(f"Attribution: {dict(Counter(r['attribution_kind'] for r in rows))}")

    if warns:
        print(f"\n{len(warns)} warning(s):")
        for w in warns:
            print(f"   {w}")
    if errors:
        print(f"\n{len(errors)} ERROR(s):")
        for e in errors:
            print(f"   {e}")
        sys.exit(1)
    print("\nAnonymisation rule: OK — no participation row names a site in its prose.")

    if args.publish:
        pub = [publish_row(r) for r in rows]
        cols = [("id", "ID"), ("date", "Date"), ("phase", "Phase"), ("site", "Site"),
                ("symptom", "Symptom"), ("root_cause", "Root cause"),
                ("impact", "Impact"), ("preventive_change", "Preventive change"),
                ("general_lesson", "Lesson")]
        with open(OUT_MD, "w") as fh:
            fh.write("# Evidence matrix (appendix)\n\n")
            fh.write("Generated by `scripts/manuscript/build_evidence_matrix.py --publish`. "
                     "Do not edit by hand — edit `evidence_matrix.csv`.\n\n")
            fh.write("Sites are named where the fact is neutral and verifiable, and "
                     "pseudonymised as Site A–H wherever the variable describes "
                     "participation.\n\n")
            for n, title in LESSONS.items():
                sel = [r for r in pub if r["lesson"] == str(n)]
                if not sel:
                    continue
                fh.write(f"## Lesson {n} — {title}\n\n")
                fh.write("| " + " | ".join(c[1] for c in cols) + " |\n")
                fh.write("|" + "|".join("---" for _ in cols) + "|\n")
                for r in sel:
                    cells = [r[c[0]].replace("|", "\\|") for c in cols]
                    fh.write("| " + " | ".join(cells) + " |\n")
                fh.write("\n")
        print(f"wrote {os.path.relpath(OUT_MD, ROOT)}")


if __name__ == "__main__":
    main()
