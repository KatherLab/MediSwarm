---
name: odelia-slides
description: Build or revise an ODELIA/MediSwarm slide deck in the consortium's house style (ODELIA EAB visual identity, Jeff's text rules, speaker notes). Use for any partner-facing deck, status slides, or when the user says "slides", "deck", "presentation".
---

# ODELIA slides

The deck is generated, not hand-edited: `scripts/presentation/build_consortium_45min.py`
writes `docs/presentation_consortium_45min.html`, which is published as the artifact
"ODELIA Consortium Briefing". Edit the Python (slide strings, diagrams, notes), rebuild,
screenshot, publish to the same artifact URL, open a docs PR.

## Visual identity (from the ODELIA EAB deck, April 2024)

- White slides, magenta `#E5316B` titles, coral `#E64F39` gradient on the title slide,
  light blue `#82C3EC` and indigo `#4B56D2` as the two accents, grey `#888888` for secondary
  text, `#F2F2F2` panels. Open Sans throughout. Light theme only.
- Logo strip in the header of the title slide and in every footer: TU Dresden, UKD, EKFZ,
  Kather Lab, NCT Heidelberg, ODELIA. Logos live in `docs/assets/logos/` and are inlined by
  the build; `nct.png` is optional (a text mark stands in).
- Numbered agenda blocks (01 to 05), quarter-by-quarter timelines, a Gantt for milestones.
- Section eyebrow on every content slide: `01 · Data`, `03 · Other deliverables · D3.2 · M48`.

## Text rules (Jeff, 15 September 2026)

1. **Descriptive titles.** Say what the slide shows: "Site data detailed view",
   "Active learning: how it works", "Operating point: recall and specificity". No
   headline-style claims ("A number without its denominator is not a result").
2. **No dashes in visible text.** Neither em dash nor en dash. Use a comma, a colon, a
   full stop, or parentheses. The build fails if one slips in.
3. **No aphorisms or clever phrasing.** Delete sentences like "a site name in a config is
   a label, not an identity" or "shares tell you what a site sees". State the fact.
4. **Concise.** One idea per callout; cut any sentence that only comments on the previous
   one ("nothing on this slide was typed by hand", "we started without waiting").
5. **Site abbreviations only:** UKA, UMCU, RUMC, USZ, CAM, MHA, RSH, VHIO. In tables the
   site ID form (`RUMC_1`) as on the board. Never city names, and no country names or
   nationalities either ("Dutch cohort" is "the RUMC and UMCU cohort", "Greek" is MHA), in
   the speaker notes as much as on the slides. If a cohort has no site pair (Duke is a
   public US data set, not a site), say so to Jeff instead of inventing one.
6. **Title slide:** presenter only (JieFu Zhu, M.Sc.), "Else Kröner Fresenius Center for
   Digital Health, TU Dresden · NCT Heidelberg", "Work Package 2 / 3", title "Month N",
   venue and date as `ODELIA · Bremen · 17.09.2026`. No subtitle paragraph.
7. **Panel headings** are plain labels ("Challenge set", "Duke", "Exists", "Needed",
   "Proposal text"), never rhetorical ("Why the intervals matter", "What is solid").
8. **Speech, not prose:** the speaker notes under each slide are spoken language, informal,
   first person plural, one to two minutes per slide.

## Structure rules

- Sections: Data, Model results, Other deliverables, Milestones and timeline, Sites.
- Before a block about other tasks or deliverables, one **divider/overview slide** that
  lists them with: what the proposal asks, due month, which sites are needed.
- **Each deliverable gets two to three slides:** first a concept slide with an inline-SVG
  diagram (flow or infrastructure) that explains the mechanism to someone who has never
  heard of it, then the numbers, then (if needed) what is needed to conclude it. Quote the
  proposal text in a panel. Name which sites must do what.
- Status/site slides are read from live data at build time or the day before: coordinator
  client registry (`Re-activate the client`, `New client`), heartbeats under
  `/srv/mediswarm/live/<SITE>/` (hospital hosts only), the board's checklist tab.
- Milestone slide: one bar per deliverable over its grant-month window, filled to the share
  of its "done when" checklist; today and 31 December as vertical lines; the checklists on
  the following slide so every percentage is traceable.
- The last slide is the requests to the consortium, at most three, each naming the sites.

## Diagrams

Inline SVG, `viewBox` sized to content, boxes on a grid, labelled arrows, one accent
(magenta) for the element the slide is about, indigo for the recommended option. Short
labels in the drawing; the explanation in the `<figcaption>`. Follow the
`artifact-diagramming` skill.

## Workflow

1. Edit `scripts/presentation/build_consortium_45min.py` (slide strings, `FIG_*` diagrams,
   `NOTES`). Charts are in `consortium_45min_charts.js`.
2. `python3 scripts/presentation/build_consortium_45min.py` (fails on a dash).
3. `scripts/presentation/render_slides.py --out <scratch dir>` renders every slide as the web
   page lays it out and writes the PDF and the PowerPoint (notes in the notes pane); look at
   every PNG, check diagram text does not overflow its box and nothing touches the footer.
   The page's own *Present* button is the way to present the HTML (arrows, N notes, Esc).
4. Publish with the Artifact tool to the existing URL (read it first if the session did not
   publish it). Update `docs/PRESENTATIONS.md`. Open a docs PR; Jeff merges.
5. Speaker notes: `docs/presentation_consortium_45min_notes.md` is generated from the same
   `NOTES` and is what Jeff reads while presenting.
