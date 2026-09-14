# ODELIA presentations and reports

Newest first. Every figure in these is read from the live coordination server, the
sites' own run logs, or CI — not from earlier notes. Where a number could not be
verified, the page says so rather than omitting the gap.

## Current

### `presentation_consortium_45min.html` — consortium briefing, 45-minute version
**21 slides · 14 September 2026 · for the 45-minute consortium slot · ODELIA deck style**

Rebuilt on 14 September in the visual identity of the ODELIA EAB deck
(`odelia_EAB_meeting_opensource_jeff.pptx`): white slides, magenta titles, Open Sans, the
pink-to-coral gradient title slide, the numbered agenda blocks, and the TUD / UKD / EKFZ /
Kather Lab / ODELIA logo strip in header and footer. Logos are in `docs/assets/logos/` and
inlined by the build. The deck is deliberately light-only, like its source.

Slides 3–16 carry the 10 September content unchanged (data, results, external validation,
the four deliverables). New on 14 September:

| Slide | What it does |
|---|---|
| 2 | Agenda — the five numbered blocks of the source deck |
| 17 | **Milestone timeline** — one bar per deliverable from its grant-month window to its due date (M48 / M54 / M60), filled to the share of its "done when" checklist that is complete; today (M45) and 31 Dec 2026 (M48) as vertical lines |
| 18 | **What concludes by 31 December 2026** — the checklists behind every percentage on slide 17, split into "due M48, on track" (D2.5, D3.2, D3.4) and "continues into 2027" (T3.3, MS6, D3.3, D3.5), plus the month-by-month plan to year end |
| 19 | Status table, updated for 1.8.1 |
| 20 | **To-do list by site** — coordinator registration after the 1.8.1 restart, kit version each node reports, last log upload from the hospital's own machine, and the concrete actions per site |

The site table is read from the coordinator log (`Re-activate the client` after the
13 Sep 23:08 UTC restart), the live-sync heartbeats under `/srv/mediswarm/live/<SITE>/`
(records whose hostname is one of our test machines are excluded — Nijmegen's latest real
upload is 30 Aug, not the 10 Sep record from dl0), and the board's site checklist.

Rebuild with `scripts/presentation/build_consortium_45min.py`; the carried-over slide
content lives in `scripts/presentation/consortium_45min_slides.html` next to it. This is
the file behind the published "ODELIA Consortium Briefing" artifact.

### `presentation_consortium_briefing.html` — consortium briefing
**12 slides · 10 September 2026 · built for a meeting, prints one slide per page**

The deck to show partners, rewritten for a non-technical audience: no job IDs, no
config paths, no code identifiers. It explains the *design* first and the
*achievement* second, and carries four hand-drawn diagrams.

| Slide | What it does |
|---|---|
| 2 | **Infrastructure diagram** — the eight sites as a ring, the model travelling between them, the coordinator drawn deliberately small because it holds no data |
| 3 | **Flow diagram** — one training round in four steps, with the return loop that makes eight hops into a round |
| 4 | **Boundary diagram** — what stays inside a hospital against what crosses the network, including the opt-in per-case predictions |
| 5 | Per-site training volumes at true scale, stacked by class |
| 6–7 | The model comparison, and why the smallest sites gain most |
| 8 | **Prevention-loop diagram** — how one site's incident becomes a permanent fix for all eight, beside the recent failure timeline |
| 9 | Radboud's 99.1 % accuracy against an AUROC of 0.417, as the plain-language case for reporting support counts |
| 10 | The active-learning result — entropy sampling beats random at small budgets (corrected 10 Sep) — and the privacy budget |
| 12 | The ask: per-case predictions, and a slot with Cambridge and Nijmegen |

The diagrams are **inline SVG, not mermaid**. Mermaid only auto-renders inside the
artifact viewer; this file is also opened directly from the repo, where a mermaid
fence would show as raw text. The SVG picks up the deck's own theme tokens and works
in both light and dark.

Rebuild with `scripts/presentation/build_consortium_briefing.py` rather than editing
the HTML by hand — the diagram coordinates are computed, and the data assertions in
it are what keep the class counts summing.

### `presentation_data_by_site.html` — training data by site
**5 charts · 10 September 2026**

How much data each hospital brought when it joined, how the consortium total
accumulated to 34,462 volumes across eight sites, and the class distribution shown
twice: at true scale and as shares. Reading them together is the point — RSH looks
malignant-rich at 63 % until you see that is 221 volumes against UKA's 1,251.

Every count was emitted by that site's own trainer as it loaded its dataset, so each
site's three classes sum exactly to its training size. The captions say this in plain
language rather than by naming log lines — the deck is shown to partners.

The last chart carries the same model comparison as slide 6 of the briefing — the
eight-site swarm against the six-site swarm and ten single-site models — so the data
and what it produced sit on one page.

### `report_wp2_wp3_status_M45.html` — WP2/WP3 status at M45
**10 September 2026 · for the technical committee · stays technical on purpose**

Where the four outstanding grant tasks stand with three deliverables due at M48, what
is verified and what is not, and the recommended next task.

Updated 10 September with the section **"Two tasks produced their first result"**:

- **D3.2** — entropy sampling *beats* random at 5 of 8 labelling budgets and loses at
  none: 5 vs 2.4 ± 1.3 malignant in the first 10 cases, 9 vs 4.6 ± 1.8 at 20, 14 vs
  8.9 ± 2.3 at 40; the methods converge by 100 of 165. A first version said the
  opposite — computed on wrong-architecture predictions (pitfall E2) and withdrawn in
  #568. Acquisition quality only; the retraining curve is still open.
- **T3.3 Phase 2** — the privacy guarantee is now a number. Over 20 rounds at
  δ = 10⁻⁵: ε ≤ 10 needs σ ≥ 2.4, ε ≤ 3 needs σ ≥ 6.7, ε ≤ 1 needs σ ≥ 18.1. The
  guarantee is client-level, not record-level, and the report says so.

The board also reflects the nine PRs merged on 9–10 September and the recommended next
task has changed accordingly: D2.5 no longer needs anything built, it needs a two-site
run with per-case return enabled.

A `.docx` of the earlier material for committee submission is at
`workspace/reports/ODELIA_WP2_WP3_progress_2026-09-08.docx` — it predates the
10 September update.

## Earlier

### `presentation_swarm_results.html` — swarm learning results
**18 slides · June 2026 · speaker notes in `presentation_swarm_results_SPEAKER_NOTES.md`**

The experiments deck: Study A (mechanism proof on Duke, 3 clients), Study B (the
6 × 6 model-by-site matrix, where the swarm global beat the best single-site model
externally on 4 of 6 models), Study C (single-site checkpoint transfer), and USZ
onboarding. Regenerate with `scripts/presentation/build_swarm_results_deck.py`
rather than editing by hand.

## Related material

| Path | What it holds |
|---|---|
| `ODELIA_SINGLE_SITE_CKPT_CHALLENGE_EVAL_CONDENSED_REPORT.md` | Per-checkpoint external results behind Study C |
| `EVALUATION_PITFALLS.md` | E1–E3 — the three ways our evaluation has misled us |
| `SWARM_FAILURE_MODES.md` | F1–F10 operational failure catalogue |
| `TIMEOUTS.md` | Per-site training sizes, and why to read them from the logs |

## A note on the numbers

The model comparison on slide 6 draws on three separate evaluations against
overlapping but not identical subsets of the challenge set — 165 cases for the
eight-site run, 300 for the June matrix. The ordering is consistent and repeated
across them; the exact gaps are not a controlled comparison, and the deck says so on
the slide.
