# ODELIA presentations and reports

Newest first. Every figure in these is read from the live coordination server, the
sites' own run logs, or CI — not from earlier notes. Where a number could not be
verified, the page says so rather than omitting the gap.

## Current

### `presentation_consortium_briefing.html` — consortium technical meeting
**8 slides · 9 September 2026 · built for a meeting, prints one slide per page**

The deck to show partners. Opens with per-site results now reaching the coordinator,
uses Radboud's 99.1 % accuracy against an AUROC of 0.417 to explain why support
counts travel with every metric, states plainly that five consecutive runs failed on
infrastructure, and closes on one request: permission in principle to return
per-case predictions.

Slide 6 is the model comparison — the eight-site swarm against the previous six-site
swarm and against ten single-site models.

### `presentation_data_by_site.html` — training data by site
**5 charts · 9 September 2026**

How much data each hospital brought when it joined, how the consortium total
accumulated to 34,462 volumes across eight sites, and the class distribution shown
twice: at true scale and as shares. Reading them together is the point — RSH looks
malignant-rich at 63 % until you see that is 221 volumes against UKA's 1,251.

Sources are each site's own `Class counts` and `Weighted epochs` log lines, so every
site's three classes sum exactly to its training size.

The last chart carries the same model comparison as slide 6 of the briefing — the
eight-site swarm against the six-site swarm and ten single-site models — so the data
and what it produced sit on one page.

### `report_wp2_wp3_status_M45.html` — WP2/WP3 status at M45
**9 September 2026 · for the technical committee**

Where the four outstanding grant tasks stand with three deliverables due at M48, what
is verified and what is not, and the recommended next task. A `.docx` of the same
material for committee submission is at
`workspace/reports/ODELIA_WP2_WP3_progress_2026-09-08.docx`.

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
