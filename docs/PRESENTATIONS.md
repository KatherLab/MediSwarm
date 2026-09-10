# ODELIA presentations and reports

Newest first. Every figure in these is read from the live coordination server, the
sites' own run logs, or CI — not from earlier notes. Where a number could not be
verified, the page says so rather than omitting the gap.

## Current

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
| 10 | The active-learning negative result and the privacy budget |
| 12 | The ask: per-case predictions, and a slot with Cambridge and Nijmegen |

The diagrams are **inline SVG, not mermaid**. Mermaid only auto-renders inside the
artifact viewer; this file is also opened directly from the repo, where a mermaid
fence would show as raw text. The SVG picks up the deck's own theme tokens and works
in both light and dark.

Rebuild with `scripts/presentation/build_consortium_briefing.py` rather than editing
the HTML by hand — the diagram coordinates are computed, and the data assertions in
it are what keep the class counts summing.

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
**10 September 2026 · for the technical committee · stays technical on purpose**

Where the four outstanding grant tasks stand with three deliverables due at M48, what
is verified and what is not, and the recommended next task.

Updated 10 September with the section **"Two tasks produced their first result"**:

- **D3.2** — uncertainty sampling *lost* to random at every labelling budget
  (19 / 21 / 22.2 ± 2.8 malignant at a 100-label budget). The calibration diagnostic
  explains it: the model is most confident on benign cases, the rarest class, so an
  uncertainty rule deprioritises exactly what you want it to find. Reported as a
  finding about calibration, not a verdict on active learning.
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
