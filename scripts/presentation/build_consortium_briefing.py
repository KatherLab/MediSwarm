#!/usr/bin/env python3
"""Build docs/presentation_consortium_briefing.html.

Regenerate rather than editing the HTML by hand: the diagram coordinates are
computed here, and the assertions in `bars()` are what keep the per-site class
counts summing to each site's training total.

Diagrams are **inline SVG, not mermaid**. Mermaid auto-renders inside the Claude
artifact viewer but this file is also opened straight from the repo, where a
mermaid fence would show as raw text. Inline SVG renders everywhere and inherits
the deck's own theme tokens, so it works in light and dark alike.

    python3 scripts/presentation/build_consortium_briefing.py \
        docs/presentation_consortium_briefing.html
"""

import math
import sys

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------------------------------------------------------- 1. the ring
SITES = [
    ("UKA",  "Aachen"),
    ("RSH",  "Guildford"),
    ("CAM",  "Cambridge"),
    ("MHA",  "Athens"),
    ("VHIO", "Barcelona"),
    ("USZ",  "Zurich"),
    ("UMCU", "Utrecht"),
    ("RUMC", "Nijmegen"),
]

def ring():
    cx, cy, r = 255, 262, 155
    bw, bh = 100, 38
    out = []
    out.append('<svg viewBox="0 0 500 486" role="img" aria-label="Eight hospitals arranged in a ring. The model is passed clockwise from hospital to hospital: Aachen, Guildford, Cambridge, Athens, Barcelona, Zurich, Utrecht, Nijmegen. Patient data stays inside each hospital. A coordination server in Dresden only schedules whose turn it is.">')
    # travel path
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="var(--accent)" stroke-width="1.6" stroke-dasharray="7 6" opacity=".55"/>')
    # direction arrowheads at the midpoints between nodes
    for i in range(8):
        th = math.radians(-90 + 45 * i + 22.5)
        x, y = cx + r * math.cos(th), cy + r * math.sin(th)
        rot = math.degrees(th) + 90
        out.append(f'<polygon points="-5.5,-4.5 6,0 -5.5,4.5" fill="var(--accent)" '
                   f'transform="translate({x:.1f} {y:.1f}) rotate({rot:.1f})"/>')
    # centre
    out.append(f'<circle cx="{cx}" cy="{cy}" r="66" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="1.4"/>')
    out.append(f'<text class="d-ctr-h" x="{cx}" y="{cy-14}" text-anchor="middle">THE MODEL</text>')
    out.append(f'<text class="d-ctr" x="{cx}" y="{cy+6}" text-anchor="middle">a file of numbers</text>')
    out.append(f'<text class="d-ctr" x="{cx}" y="{cy+22}" text-anchor="middle">no images inside</text>')
    out.append(f'<text class="d-ctr-g" x="{cx}" y="{cy+42}" text-anchor="middle">better at every stop</text>')
    # nodes
    for i, (code, city) in enumerate(SITES):
        th = math.radians(-90 + 45 * i)
        x, y = cx + r * math.cos(th), cy + r * math.sin(th)
        out.append(f'<g><rect x="{x-bw/2:.1f}" y="{y-bh/2:.1f}" width="{bw}" height="{bh}" rx="6" '
                   f'fill="var(--slide)" stroke="var(--rule)" stroke-width="1.2"/>')
        out.append(f'<text class="d-node" x="{x:.1f}" y="{y-2:.1f}" text-anchor="middle">{code}</text>')
        out.append(f'<text class="d-node-sub" x="{x:.1f}" y="{y+11:.1f}" text-anchor="middle">{esc(city)}</text></g>')
    # coordinator, deliberately small
    out.append('<g><rect x="6" y="10" width="196" height="36" rx="6" fill="none" stroke="var(--muted)" '
               'stroke-width="1.1" stroke-dasharray="4 4"/>')
    out.append('<text class="d-note" x="104" y="26" text-anchor="middle">Coordination server · Dresden</text>')
    out.append('<text class="d-note-sm" x="104" y="39" text-anchor="middle">schedules turns · holds no data</text></g>')
    out.append('<path d="M104 46 L104 70 Q104 82 116 88 L140 108" fill="none" stroke="var(--muted)" '
               'stroke-width="1.1" stroke-dasharray="4 4"/>')
    out.append('</svg>')
    return "\n".join(out)

# --------------------------------------------------------------- 2. the round
def round_flow():
    steps = [
        ("1", "The coordinator", "sets the order", "for this round"),
        ("2", "A hospital receives", "the current model", "over the network"),
        ("3", "It trains", "on its own patients,", "on its own hardware"),
        ("4", "It hands the", "improved model", "to the next hospital"),
    ]
    bw, bh, gap, y = 190, 84, 34, 18
    out = ['<svg viewBox="0 0 880 200" role="img" aria-label="A training round in four steps: the coordinator sets the order; a hospital receives the current model; it trains on its own patients on its own hardware; it hands the improved model to the next hospital. The last three steps repeat once per hospital, and eight hops complete one round.">']
    xs = []
    for i, (n, l1, l2, l3) in enumerate(steps):
        x = 12 + i * (bw + gap)
        xs.append(x)
        fill = "var(--accent-soft)" if i == 2 else "var(--slide-alt)"
        stroke = "var(--accent)" if i == 2 else "var(--rule)"
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        out.append(f'<text class="d-step-n" x="{x+13}" y="{y+21}">{n}</text>')
        out.append(f'<text class="d-step" x="{x+13}" y="{y+43}">{esc(l1)}</text>')
        out.append(f'<text class="d-step" x="{x+13}" y="{y+59}">{esc(l2)}</text>')
        out.append(f'<text class="d-step" x="{x+13}" y="{y+75}">{esc(l3)}</text>')
        if i < 3:
            ax = x + bw + 6
            out.append(f'<path d="M{ax} {y+bh/2} L{ax+gap-12} {y+bh/2}" stroke="var(--muted)" stroke-width="1.4"/>')
            out.append(f'<polygon points="-5,-4 5,0 -5,4" fill="var(--muted)" transform="translate({ax+gap-11} {y+bh/2})"/>')
    # return loop: step 4 -> step 2
    x4c = xs[3] + bw / 2
    x2c = xs[1] + bw / 2
    ly = y + bh + 36
    out.append(f'<path d="M{x4c} {y+bh} L{x4c} {ly} L{x2c} {ly} L{x2c} {y+bh+9}" fill="none" '
               f'stroke="var(--accent)" stroke-width="1.4" stroke-dasharray="6 5"/>')
    out.append(f'<polygon points="-4.5,4.5 0,-5 4.5,4.5" fill="var(--accent)" transform="translate({x2c} {y+bh+7})"/>')
    out.append(f'<rect x="{(x2c+x4c)/2-142}" y="{ly-13}" width="284" height="26" rx="5" fill="var(--slide)"/>')
    out.append(f'<text class="d-loop" x="{(x2c+x4c)/2}" y="{ly+4}" text-anchor="middle">eight hops — one per hospital — complete one round</text>')
    out.append('</svg>')
    return "\n".join(out)

# ------------------------------------------------------------ 3. the boundary
def boundary():
    stay = [
        ("MRI images and the raw DICOM series", None),
        ("Patient names, dates of birth, record numbers", None),
        ("Radiology reports and clinical notes", None),
        ("The hospital's own database and file store", None),
    ]
    cross = [
        ("The model itself — weights, which are numbers", "no images and no text are stored inside it"),
        ("How many cases the site trained on", "and how many of each of the three classes"),
        ("How well the shared model scored at that site", "accuracy and AUROC, with the counts behind them"),
        ("Per-case predictions — opt-in, off by default", "a row number, the label, the probabilities. No identifier."),
    ]
    out = ['<svg viewBox="0 0 880 312" role="img" aria-label="Two columns divided by the hospital boundary. Staying inside: MRI images and raw DICOM, patient identifiers, radiology reports, the hospital database. Crossing the boundary: the model weights, case counts per class, accuracy scores, and optionally per-case predictions carrying a row number but no identifier.">']
    # left
    out.append('<rect x="6" y="34" width="394" height="272" rx="8" fill="var(--slide-alt)" stroke="var(--rule)" stroke-width="1.2"/>')
    out.append('<text class="d-panel-h" x="24" y="22">STAYS INSIDE THE HOSPITAL — ALWAYS</text>')
    for i, (t, _) in enumerate(stay):
        yy = 74 + i * 58
        out.append(f'<circle cx="34" cy="{yy-5}" r="9" fill="var(--bad-bg)"/>')
        out.append(f'<path d="M30 {yy-9} L38 {yy-1} M38 {yy-9} L30 {yy-1}" stroke="var(--bad)" stroke-width="1.8" stroke-linecap="round"/>')
        out.append(f'<text class="d-item" x="54" y="{yy}">{esc(t)}</text>')
    # boundary
    out.append('<path d="M436 8 L436 306" stroke="var(--rule)" stroke-width="2" stroke-dasharray="6 6"/>')
    out.append('<rect x="404" y="140" width="64" height="34" rx="6" fill="var(--slide)" stroke="var(--rule)" stroke-width="1.2"/>')
    out.append('<text class="d-note-sm" x="436" y="154" text-anchor="middle">HOSPITAL</text>')
    out.append('<text class="d-note-sm" x="436" y="166" text-anchor="middle">BOUNDARY</text>')
    # right
    out.append('<rect x="472" y="34" width="402" height="272" rx="8" fill="var(--slide)" stroke="var(--accent)" stroke-width="1.2"/>')
    out.append('<text class="d-panel-h" x="490" y="22">CROSSES THE BOUNDARY</text>')
    for i, (t, sub) in enumerate(cross):
        yy = 70 + i * 58
        col = "var(--warn)" if i == 3 else "var(--good)"
        bg = "var(--warn-bg)" if i == 3 else "var(--good-bg)"
        out.append(f'<circle cx="500" cy="{yy-5}" r="9" fill="{bg}"/>')
        out.append(f'<path d="M496 {yy-5} L505 {yy-5} M501.5 {yy-9} L505.5 {yy-5} L501.5 {yy-1}" fill="none" stroke="{col}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>')
        out.append(f'<text class="d-item" x="520" y="{yy}">{esc(t)}</text>')
        if sub:
            out.append(f'<text class="d-item-sub" x="520" y="{yy+16}">{esc(sub)}</text>')
    out.append('</svg>')
    return "\n".join(out)

# ------------------------------------------------------- 4. prevention loop
def loop():
    nodes = [
        ("A run fails", "at one hospital"),
        ("We reproduce it", "on our own machines"),
        ("The cause is fixed", "in the software"),
        ("An automatic check", "is added"),
        ("The next run", "cannot hit it again"),
    ]
    cx, cy, r = 250, 208, 138
    bw, bh = 138, 46
    out = ['<svg viewBox="0 0 500 376" role="img" aria-label="A five-step loop: a run fails at one hospital, we reproduce it on our own machines, the cause is fixed in the software, an automatic check is added, and the next run cannot hit it again. At the centre: ten named failure modes, all written up for partners.">']
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="var(--rule)" stroke-width="1.4" stroke-dasharray="6 6"/>')
    for i in range(5):
        th = math.radians(-90 + 72 * i + 36)
        x, y = cx + r * math.cos(th), cy + r * math.sin(th)
        rot = math.degrees(th) + 90
        out.append(f'<polygon points="-5,-4 6,0 -5,4" fill="var(--accent)" transform="translate({x:.1f} {y:.1f}) rotate({rot:.1f})"/>')
    out.append(f'<text class="d-ctr-h" x="{cx}" y="{cy-6}" text-anchor="middle">10 NAMED FAILURE MODES</text>')
    out.append(f'<text class="d-ctr" x="{cx}" y="{cy+14}" text-anchor="middle">each one written up for partners</text>')
    for i, (l1, l2) in enumerate(nodes):
        th = math.radians(-90 + 72 * i)
        x, y = cx + r * math.cos(th), cy + r * math.sin(th)
        out.append(f'<rect x="{x-bw/2:.1f}" y="{y-bh/2:.1f}" width="{bw}" height="{bh}" rx="6" '
                   f'fill="var(--slide-alt)" stroke="var(--rule)" stroke-width="1.2"/>')
        out.append(f'<text class="d-node" x="{x:.1f}" y="{y-3:.1f}" text-anchor="middle">{esc(l1)}</text>')
        out.append(f'<text class="d-node-sub" x="{x:.1f}" y="{y+11:.1f}" text-anchor="middle">{esc(l2)}</text>')
    out.append('</svg>')
    return "\n".join(out)

# ------------------------------------------------------------- 5. data bars
DATA = [
    ("UKA",  "Aachen",     17834, 4747, 11836, 1251),
    ("UMCU", "Utrecht",     6133, 5337,   740,   56),
    ("RUMC", "Nijmegen",    3608, 3571,     2,   35),
    ("USZ",  "Zurich",      3448, 1911,  1244,  293),
    ("CAM",  "Cambridge",   1778, 1678,    42,   58),
    ("MHA",  "Athens",      1120,  904,   120,   96),
    ("RSH",  "Guildford",    351,    4,   126,  221),
    ("VHIO", "Barcelona",    190,  159,     0,   31),
]

def bars():
    assert all(n == a + b + c for _, _, n, a, b, c in DATA), "class counts must sum to n"
    total = sum(d[2] for d in DATA)
    assert total == 34462, total
    x0, maxw = 96, 500
    rh, gapy, y0 = 26, 8, 14
    sc = maxw / DATA[0][2]
    out = [f'<svg viewBox="0 0 720 {y0 + len(DATA)*(rh+gapy) + 6}" role="img" '
           f'aria-label="Training volumes per hospital at true scale: Aachen 17834, Utrecht 6133, Nijmegen 3608, Zurich 3448, Cambridge 1778, Athens 1120, Guildford 351, Barcelona 190. Total 34462.">']
    for i, (code, city, n, c0, c1, c2) in enumerate(DATA):
        y = y0 + i * (rh + gapy)
        out.append(f'<text class="d-bar-lbl" x="{x0-12}" y="{y+17}" text-anchor="end">{code}</text>')
        xx = x0
        for val, col in ((c0, "var(--s1)"), (c1, "var(--s2)"), (c2, "var(--s3)")):
            w = val * sc
            if w > 0:
                out.append(f'<rect x="{xx:.1f}" y="{y}" width="{max(w,0.8):.1f}" height="{rh}" fill="{col}"/>')
            xx += w
        out.append(f'<text class="d-bar-val" x="{x0+maxw+18}" y="{y+17}" text-anchor="end">{n:,}</text>')
    out.append('</svg>')
    return "\n".join(out)


CSS = """<title>ODELIA Consortium Briefing</title>
<style>
  :root {
    --ground:      #FAFBFC;
    --slide:       #FFFFFF;
    --slide-alt:   #F1F5F8;
    --ink:         #131C24;
    --ink-soft:    #35485A;
    --muted:       #5B6B79;
    --rule:        #D9E1E7;
    --rule-soft:   #E9EFF3;
    --accent:      #0F6E8C;
    --accent-soft: #E0EEF3;
    --good:        #2E7250;
    --good-bg:     #E3F0E8;
    --warn:        #96662A;
    --warn-bg:     #F6EDDF;
    --bad:         #9E3339;
    --bad-bg:      #F7E5E6;
    --s1:          #2a78d6;
    --s2:          #eb6834;
    --s3:          #1baf7a;
    --serif: "Iowan Old Style", "Charter", "Bitstream Charter", "Palatino Linotype", Palatino, Georgia, serif;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", "Roboto Mono", Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #0B1116; --slide: #131C23; --slide-alt: #19242C;
      --ink: #E3EBF0; --ink-soft: #BDCBD5; --muted: #92A4B1;
      --rule: #253440; --rule-soft: #1B262D;
      --accent: #57B7D2; --accent-soft: #11303B;
      --good: #6FBE92; --good-bg: #15301F;
      --warn: #D6A45C; --warn-bg: #33260F;
      --bad: #E08088;  --bad-bg: #351A1D;
      --s1: #3987e5;   --s2: #d95926;  --s3: #199e70;
    }
  }
  :root[data-theme="dark"] {
    --ground: #0B1116; --slide: #131C23; --slide-alt: #19242C;
    --ink: #E3EBF0; --ink-soft: #BDCBD5; --muted: #92A4B1;
    --rule: #253440; --rule-soft: #1B262D;
    --accent: #57B7D2; --accent-soft: #11303B;
    --good: #6FBE92; --good-bg: #15301F;
    --warn: #D6A45C; --warn-bg: #33260F;
    --bad: #E08088;  --bad-bg: #351A1D;
    --s1: #3987e5;   --s2: #d95926;  --s3: #199e70;
  }

  body { background: var(--ground); color: var(--ink); font-family: var(--sans); line-height: 1.5; }
  .deck { max-width: 68rem; margin: 0 auto; padding: 1.5rem 1rem 4rem; display: flex; flex-direction: column; gap: 1.5rem; }

  .slide {
    background: var(--slide);
    border: 1px solid var(--rule);
    border-radius: 8px;
    padding: clamp(1.4rem, 3.2vw, 2.5rem);
    /* Slide-shaped when the content fits, taller when it does not. A hard
       aspect-ratio plus overflow:hidden silently clips the diagram slides. */
    min-height: min(56.25vw, 37rem);
    display: flex; flex-direction: column;
    position: relative;
  }
  @media (max-width: 820px) { .slide { min-height: 0; } }

  .num { position: absolute; top: 1rem; right: 1.25rem; font-family: var(--mono); font-size: .7rem; color: var(--muted); font-variant-numeric: tabular-nums; }
  .eyebrow { font-family: var(--mono); font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
  h1 { font-family: var(--serif); font-size: clamp(1.9rem, 4.4vw, 2.9rem); line-height: 1.08; margin: .5rem 0 0; font-weight: 600; letter-spacing: -.015em; text-wrap: balance; }
  h2 { font-family: var(--serif); font-size: clamp(1.3rem, 2.7vw, 1.85rem); line-height: 1.15; margin: .35rem 0 0; font-weight: 600; letter-spacing: -.01em; text-wrap: balance; }
  .sub { color: var(--ink-soft); font-size: clamp(.94rem, 1.55vw, 1.06rem); margin-top: .7rem; max-width: 46rem; text-wrap: pretty; }
  .body { margin-top: 1.2rem; display: flex; flex-direction: column; gap: .85rem; flex: 1; min-height: 0; }
  p { margin: 0; }
  strong { font-weight: 650; }

  ul { margin: 0; padding-left: 1.15rem; display: flex; flex-direction: column; gap: .5rem; }
  li { color: var(--ink-soft); font-size: clamp(.88rem, 1.45vw, .98rem); }
  li::marker { color: var(--accent); }

  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
  .stats > div { background: var(--slide); padding: .9rem 1rem; }
  .stats dt { font-family: var(--mono); font-size: .64rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  .stats dd { margin: .2rem 0 0; font-family: var(--serif); font-size: 1.55rem; font-weight: 600; font-variant-numeric: tabular-nums; }
  .stats .cap { display: block; font-family: var(--sans); font-size: .74rem; font-weight: 400; color: var(--muted); margin-top: .1rem; line-height: 1.35; }

  .scroll { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; }
  table { border-collapse: collapse; width: 100%; font-size: .87rem; min-width: 32rem; }
  th, td { text-align: left; padding: .5rem .8rem; border-bottom: 1px solid var(--rule-soft); }
  thead th { font-family: var(--mono); font-size: .64rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); font-weight: 500; background: var(--slide-alt); border-bottom: 1px solid var(--rule); white-space: nowrap; }
  tbody tr:last-child td { border-bottom: none; }
  td.n { font-variant-numeric: tabular-nums; white-space: nowrap; }
  td.id { white-space: nowrap; }

  .chip { display: inline-block; font-family: var(--mono); font-size: .64rem; letter-spacing: .06em; text-transform: uppercase; padding: .16em .5em; border-radius: 3px; font-weight: 600; white-space: nowrap; }
  .chip.good { background: var(--good-bg); color: var(--good); }
  .chip.warn { background: var(--warn-bg); color: var(--warn); }
  .chip.bad  { background: var(--bad-bg);  color: var(--bad); }
  .chip.info { background: var(--accent-soft); color: var(--accent); }

  .callout { background: var(--accent-soft); border-left: 3px solid var(--accent); border-radius: 4px; padding: .8rem 1rem; font-size: .92rem; color: var(--ink-soft); }
  .callout strong { color: var(--ink); }
  .callout.plain { background: var(--slide-alt); border-left-color: var(--muted); }

  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .split { display: grid; grid-template-columns: minmax(0,0.82fr) minmax(0,1fr); gap: 1.4rem; align-items: center; flex: 1; min-height: 0; }
  @media (max-width: 820px) { .two, .split { grid-template-columns: 1fr; } }
  .panel { background: var(--slide-alt); border: 1px solid var(--rule-soft); border-radius: 6px; padding: .9rem 1rem; }
  .panel h3 { margin: 0 0 .35rem; font-size: .8rem; font-family: var(--mono); letter-spacing: .07em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  .panel p, .panel li { font-size: .88rem; }

  .ask { border: 1px solid var(--accent); border-radius: 6px; overflow: hidden; }
  .ask-h { background: var(--accent-soft); padding: .65rem 1.1rem; font-family: var(--mono); font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
  .ask-b { padding: .9rem 1.1rem; font-size: .93rem; color: var(--ink-soft); }

  /* ---- diagrams ---------------------------------------------------- */
  .fig { display: flex; flex-direction: column; gap: .5rem; min-width: 0; }
  .fig svg { display: block; width: 100%; height: auto; max-height: 100%; overflow: visible; }
  .fig figcaption { font-size: .78rem; color: var(--muted); }
  .d-node      { font-family: var(--sans); font-size: 12.5px; font-weight: 650; fill: var(--ink); }
  .d-node-sub  { font-family: var(--sans); font-size: 10px; fill: var(--muted); }
  .d-ctr-h     { font-family: var(--mono); font-size: 9.5px; letter-spacing: .1em; fill: var(--accent); font-weight: 700; }
  .d-ctr       { font-family: var(--sans); font-size: 11px; fill: var(--ink-soft); }
  .d-ctr-g     { font-family: var(--sans); font-size: 10px; fill: var(--muted); font-style: italic; }
  .d-note      { font-family: var(--sans); font-size: 11px; fill: var(--ink-soft); font-weight: 600; }
  .d-note-sm   { font-family: var(--mono); font-size: 8.5px; letter-spacing: .06em; fill: var(--muted); }
  .d-step-n    { font-family: var(--mono); font-size: 11px; fill: var(--accent); font-weight: 700; }
  .d-step      { font-family: var(--sans); font-size: 12px; fill: var(--ink-soft); }
  .d-loop      { font-family: var(--sans); font-size: 11.5px; fill: var(--accent); font-weight: 600; }
  .d-panel-h   { font-family: var(--mono); font-size: 9.5px; letter-spacing: .11em; fill: var(--muted); font-weight: 600; }
  .d-item      { font-family: var(--sans); font-size: 12.5px; fill: var(--ink); }
  .d-item-sub  { font-family: var(--sans); font-size: 10.5px; fill: var(--muted); }
  .d-bar-lbl   { font-family: var(--sans); font-size: 12px; font-weight: 650; fill: var(--ink); }
  .d-bar-val   { font-family: var(--mono); font-size: 11.5px; fill: var(--ink-soft); font-variant-numeric: tabular-nums; }

  .legend { display: flex; flex-wrap: wrap; gap: .9rem; font-size: .8rem; color: var(--ink-soft); }
  .legend span { display: inline-flex; align-items: center; gap: .35rem; }
  .sw { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }

  footer { text-align: center; color: var(--muted); font-size: .8rem; padding-top: 1rem; }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  @media print {
    body { background: #fff; }
    .deck { gap: 0; padding: 0; max-width: none; }
    .slide { page-break-after: always; border: none; border-radius: 0; min-height: 100vh; }
  }
</style>
"""

SLIDES = []

def slide(n, eyebrow, head, body, h="h2"):
    SLIDES.append(f'''
  <!-- {n} -->
  <section class="slide">
    <span class="num">{n}</span>
    <div class="eyebrow">{eyebrow}</div>
    <{h}>{head}</{h}>
{body}
  </section>''')

# ---- 1 -------------------------------------------------------------------
slide(1, "ODELIA consortium · September 2026",
      "Eight hospitals, one shared model, no patient data moved",
      '''    <p class="sub">
      How the system works, what it has achieved, and what we need from partners next.
      This briefing avoids technical detail on purpose — everything here is the plain version.
    </p>
    <div class="body" style="justify-content:flex-end">
      <div class="stats">
        <div><dt>Hospitals</dt><dd>8<span class="cap">six countries, all training together</span></dd></div>
        <div><dt>Scans trained on</dt><dd>34,462<span class="cap">none of them left its hospital</span></dd></div>
        <div><dt>Best shared model</dt><dd>0.887<span class="cap">detecting malignancy — better than any single hospital achieved alone</span></dd></div>
        <div><dt>Project month</dt><dd>45<span class="cap">of 60</span></dd></div>
      </div>
    </div>''', h="h1")

# ---- 2 -------------------------------------------------------------------
slide(2, "The design",
      "The model travels. The data never does.",
      f'''    <div class="split">
      <div style="display:flex;flex-direction:column;gap:.8rem">
        <ul>
          <li>Every hospital keeps its images on its own machines, behind its own firewall. Nothing is copied to a central store, because there is no central store.</li>
          <li>What moves is <strong>the model</strong> — a file of numbers. It contains no images, no reports and no patient details.</li>
          <li>Each hospital improves the model on its own patients, then passes it to the next hospital in the ring.</li>
          <li>A coordination server in Dresden decides <em>whose turn it is</em>. It never receives data and never does the learning itself.</li>
        </ul>
        <div class="callout">
          <strong>That last point is what makes this a swarm.</strong> In most collaborative
          AI there is a central server that collects everyone's contributions. Here there
          is not — the hospitals hand the model directly to one another.
        </div>
      </div>
      <figure class="fig">
        {ring()}
        <figcaption>The eight ODELIA sites and the path the model takes.</figcaption>
      </figure>
    </div>''')

# ---- 3 -------------------------------------------------------------------
slide(3, "The design",
      "What actually happens during a training round",
      f'''    <div class="body">
      <figure class="fig">
        {round_flow()}
      </figure>
      <div class="two">
        <div class="panel">
          <h3>The unit of work</h3>
          <p>One hospital, one turn with the model. Eight turns make a round. The August
          benchmark ran <strong>twenty rounds</strong> — 160 turns in all, roughly two
          hours each, spread over two days.</p>
        </div>
        <div class="panel">
          <h3>Why it takes so long</h3>
          <p>The hospitals are very different sizes, and a round is only finished when
          the slowest one has finished. A site with eighteen thousand scans and a site
          with a hundred and ninety are both taking a full turn.</p>
        </div>
      </div>
    </div>''')

# ---- 4 -------------------------------------------------------------------
slide(4, "The design · what partners are usually asked",
      "What leaves a hospital, and what never does",
      f'''    <div class="body">
      <figure class="fig">
        {boundary()}
      </figure>
      <div class="callout">
        <strong>The last row is the one we would like to discuss.</strong> Returning
        per-case predictions is <em>off by default</em> and each site decides for itself.
        It is what makes proper statistical comparison between hospitals possible, and it
        is required for the regulatory testing node due next year.
      </div>
    </div>''')

# ---- 5 -------------------------------------------------------------------
slide(5, "The achievement · the data",
      "What the consortium can now train on",
      f'''    <div class="body">
      <figure class="fig">
        {bars()}
        <figcaption>Training volumes per hospital, drawn at true scale. Read from each
        site's own training logs; the three classes sum exactly to each site's total.</figcaption>
      </figure>
      <div class="legend">
        <span><i class="sw" style="background:var(--s1)"></i> No lesion — 18,311</span>
        <span><i class="sw" style="background:var(--s2)"></i> Benign — 14,110</span>
        <span><i class="sw" style="background:var(--s3)"></i> Malignant — 2,041</span>
        <span style="margin-left:auto;font-variant-numeric:tabular-nums"><strong>34,462 total</strong></span>
      </div>
      <div class="callout">
        <strong>The hospitals are not comparable in size, and that shapes everything.</strong>
        Aachen alone holds 51.7&#8239;% of the consortium's training data — more than the other
        seven put together — and the range from largest to smallest is 94-fold. Timeouts,
        round budgets and the way results are weighted all had to be designed around that.
      </div>
    </div>''')

# ---- 6 -------------------------------------------------------------------
slide(6, "The achievement · the result",
      "The shared model beats what any hospital could train alone",
      '''    <div class="body">
      <div class="scroll">
        <table>
          <thead>
            <tr><th>Model</th><th>Hospitals</th><th>When</th><th>Malignancy detection</th><th>All three classes</th></tr>
          </thead>
          <tbody>
            <tr style="background:var(--good-bg)">
              <td class="id"><strong>Shared model, full consortium</strong></td><td class="n">8</td><td class="n">Aug 2026</td>
              <td class="n"><strong>0.887</strong></td><td class="n"><strong>0.820</strong></td></tr>
            <tr><td class="id">Shared model, previous run</td><td class="n">6</td><td class="n">Jun 2026</td>
                <td class="n">0.847</td><td class="n">0.764</td></tr>
            <tr><td class="id">Best hospital training on its own — Aachen</td><td class="n">1</td><td class="n">Jun 2026</td>
                <td class="n">0.858</td><td class="n">0.709</td></tr>
            <tr><td class="id">A typical hospital training on its own</td><td class="n">1</td><td class="n">Jun 2026</td>
                <td class="n">0.658</td><td class="n">0.601</td></tr>
            <tr><td class="id">Weakest hospital training on its own</td><td class="n">1</td><td class="n">Jun 2026</td>
                <td class="n">0.408</td><td class="n">0.483</td></tr>
          </tbody>
        </table>
      </div>
      <div class="two">
        <div class="panel">
          <h3>How to read these numbers</h3>
          <p>They run from 0.5 (no better than guessing) to 1.0 (perfect). The
          malignancy column is the clinically decisive one: how reliably the model
          separates a malignant finding from everything else.</p>
        </div>
        <div class="panel">
          <h3>What the table says</h3>
          <ul>
            <li>The shared model is ahead of <strong>every</strong> hospital's own model, including Aachen's — and Aachen holds half the data</li>
            <li>It is ahead of the previous six-hospital run on both measures</li>
            <li>A typical hospital gains <strong>0.229</strong> by joining, which is a large clinical difference</li>
          </ul>
        </div>
      </div>
      <div class="callout plain">
        <strong>An honest caveat.</strong> These come from three separate evaluations on
        overlapping but not identical sets of test cases. The ordering is consistent and
        has repeated; the exact size of each gap is not a controlled comparison.
      </div>
    </div>''')

# ---- 7 -------------------------------------------------------------------
slide(7, "The achievement · why it matters",
      "The smaller the hospital, the more it gains",
      '''    <div class="body">
      <p class="sub" style="margin-top:0">
        This is the case for the consortium existing, stated in one line: a hospital that
        trains alone is limited by the patients it happens to have seen.
      </p>
      <div class="two">
        <div class="panel">
          <h3>What a small site sees on its own</h3>
          <p>Barcelona has 190 scans and <strong>no benign cases at all</strong>.
          Guildford has 351. Nijmegen has 3,608 scans but only 2 benign among them. A
          model trained on any of these learns whatever that hospital's referral pattern
          happens to look like, and mistakes it for how breast MRI works.</p>
        </div>
        <div class="panel">
          <h3>What the same site sees in the swarm</h3>
          <p>All 34,462 scans, from six countries, six scanner manufacturers and eight
          different ways of running a breast-MRI service — without a single image being
          copied anywhere. Every site takes the same finished model home.</p>
        </div>
      </div>
      <div class="callout">
        <strong>And it is not only the small sites.</strong> Aachen holds more than half
        the data and still ends up with a better model by taking part. Being the largest
        contributor is not the same as having seen enough.
      </div>
    </div>''')

# ---- 8 -------------------------------------------------------------------
slide(8, "What the work actually was",
      "Most of the effort went into making it survive the real world",
      f'''    <div class="split" style="align-items:start">
      <div style="display:flex;flex-direction:column;gap:.8rem">
        <p class="sub" style="margin-top:0">
          The learning itself was rarely the hard part. Hospital IT is eight different
          networks, eight security policies and eight sets of hardware, and any one of
          them can end a two-day run.
        </p>
        <ul>
          <li><strong>101 runs since March</strong>, 59 of them involving hospital sites.</li>
          <li>Every distinct way a run can fail has been named, written up and given an
          automatic check, so it is caught before the next run rather than during it.</li>
          <li>Development moved onto our own machines, so partner hardware is used only
          for real benchmark runs.</li>
        </ul>
        <div class="scroll">
          <table>
            <thead><tr><th>Recent run</th><th>What ended it</th><th>Outcome</th></tr></thead>
            <tbody>
              <tr><td class="id">4 Sep</td><td>Aachen's machine rebooted mid-run — twice, at the same point</td><td><span class="chip bad">Lost</span></td></tr>
              <tr><td class="id">7 Sep</td><td>Stopped by us, deliberately — it was about to overwrite the shared model</td><td><span class="chip warn">Stopped</span></td></tr>
              <tr><td class="id">7 Sep</td><td>Athens could not download the model from Guildford; Cambridge dropped out</td><td><span class="chip bad">Lost</span></td></tr>
              <tr><td class="id">7–8 Sep</td><td>Dutch run finished all its rounds, then took five hours to publish</td><td><span class="chip good">Recovered</span></td></tr>
              <tr><td class="id">8 Sep</td><td>Our own test runs — every model transfer was being routed via Frankfurt</td><td><span class="chip info">Cause found</span></td></tr>
            </tbody>
          </table>
        </div>
        <div class="callout">
          <strong>Not one of these was a defect in the training software.</strong> A crashing
          server, dropped VPN links, and a network route that sent a 700&#8239;MB model across
          the country twice per round. That is the honest state of the field.
        </div>
      </div>
      <figure class="fig">
        {loop()}
        <figcaption>How an incident at one hospital becomes a permanent fix for all eight.</figcaption>
      </figure>
    </div>''')

# ---- 9 : merged from the previously published deck (its slides 2-4) ------
slide(9, "How we read the results",
      "A number without its denominator is not a result",
      '''    <div class="body">
      <p class="sub" style="margin-top:0">
        Until recently, each hospital's results had to be collected by logging into that
        hospital by hand. They now come back to the coordinator automatically — and the
        first thing they showed was a trap worth explaining to everyone.
      </p>
      <div class="scroll">
        <table>
          <thead>
            <tr><th>Hospital</th><th>Cases</th><th>No lesion</th><th>Benign</th><th>Malignant</th><th>Accuracy</th><th>Actually useful?</th></tr>
          </thead>
          <tbody>
            <tr style="background:var(--bad-bg)"><td class="id">Nijmegen</td><td class="n">896</td><td class="n">888</td><td class="n">0</td><td class="n">8</td><td class="n">99.1&#8239;%</td><td class="n">0.417 — worse than guessing</td></tr>
            <tr><td class="id">Utrecht</td><td class="n">1,557</td><td class="n">1,361</td><td class="n">186</td><td class="n">10</td><td class="n">87.4&#8239;%</td><td class="n">0.698</td></tr>
          </tbody>
        </table>
      </div>
      <div class="two">
        <div class="panel">
          <h3>Why 99&#8239;% means nothing here</h3>
          <p>Of Nijmegen's 896 cases, 888 have no lesion. A model that simply answers
          "no lesion" every single time scores 99&#8239;% — and finds no cancer at all.
          Reported on its own, that figure would make Nijmegen look like the strongest
          centre in the consortium.</p>
        </div>
        <div class="panel">
          <h3>This has already misled us once</h3>
          <p>Utrecht was described as performing materially worse than other centres. That
          came from an average across three classes in which one class had just
          <strong>two</strong> cases. Set that class aside and Utrecht moves
          0.602&nbsp;→&nbsp;0.717, Cambridge 0.710&nbsp;→&nbsp;0.908.</p>
        </div>
      </div>
      <div class="callout">
        <strong>None of this reflects on any hospital's data.</strong> It reflects on how we
        were summarising it. Every figure now travels with the counts behind it, and we no
        longer rank centres on an average where a class has only a handful of cases.
      </div>
    </div>''')

# ---- 10 ------------------------------------------------------------------
slide(10, "New this quarter",
      "Two new capabilities — and one useful negative result",
      '''    <div class="body">
      <div class="two">
        <div class="panel">
          <h3>Choosing what to label next</h3>
          <p>Labelling breast MRI is expensive, so we tested whether the model can point
          at the cases worth a radiologist's time.</p>
          <p style="margin-top:.5rem"><strong>It cannot — yet.</strong> Asking the model
          for the cases it is least sure about found <em>fewer</em> malignant cases than
          picking at random: 19 against 22 out of a 100-case budget.</p>
          <p style="margin-top:.5rem">The diagnostic explains why, and it is worth
          knowing: the model is <strong>most confident on benign cases</strong>, which are
          the rarest. Its confidence does not track whether it is right, so asking it what
          it is unsure about points in the wrong direction. Confidence has to be fixed
          before this technique can help.</p>
        </div>
        <div class="panel">
          <h3>Putting a number on privacy</h3>
          <p>Until now we could say the design protects data. We can now say <em>by how
          much</em>, on the standard scale used by regulators.</p>
          <p style="margin-top:.5rem">Adding measured statistical noise to each
          hospital's contribution buys a formal guarantee. Over a twenty-round run:</p>
          <ul style="margin-top:.5rem">
            <li>a moderate guarantee needs a noise level of 2.4</li>
            <li>a strong one needs 6.7</li>
            <li>the strictest commonly cited standard needs 18.1</li>
          </ul>
          <p style="margin-top:.5rem">More noise costs accuracy, so this is now a
          decision the consortium can take deliberately instead of by accident.</p>
        </div>
      </div>
      <div class="callout plain">
        <strong>What the privacy guarantee covers.</strong> It hides whether a
        <em>hospital</em> took part in a round. Hiding whether an individual
        <em>patient</em> was in a hospital's data is a further step, and we are not
        claiming it yet.
      </div>
    </div>''')

# ---- 10 ------------------------------------------------------------------
slide(11, "Where we stand",
      "Three deliverables due in three months",
      '''    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Deliverable</th><th>Due</th><th>Status</th><th>What it still needs</th></tr></thead>
          <tbody>
            <tr><td class="id">Regional fine-tuning</td><td class="n">M48</td><td><span class="chip warn">In progress</span></td><td>Per-case predictions — slide 4</td></tr>
            <tr><td class="id">Active learning</td><td class="n">M48</td><td><span class="chip good">First result in</span></td><td>Fix model confidence, then re-test</td></tr>
            <tr><td class="id">Adversarial robustness</td><td class="n">M48</td><td><span class="chip warn">Starting</span></td><td>—</td></tr>
            <tr><td class="id">Regulatory testing node</td><td class="n">M54</td><td><span class="chip good">Unblocked</span></td><td>Builds on the per-site reporting now working</td></tr>
            <tr><td class="id">Differential privacy</td><td class="n">M54</td><td><span class="chip good">Measurable</span></td><td>Consortium decision on the noise level</td></tr>
            <tr><td class="id">White-hat attack exercise</td><td class="n">M54</td><td><span class="chip bad">Needs partners</span></td><td><strong>Cambridge and Nijmegen</strong> — scheduling now</td></tr>
            <tr><td class="id">Full demonstration</td><td class="n">M60</td><td><span class="chip info">Part-evidenced</span></td><td>Re-run on the released version</td></tr>
          </tbody>
        </table>
      </div>
    </div>''')

# ---- 12 : the ask, restored in full from the previously published deck ----
slide(12, "What we need from partners",
      "Two things, and neither is technical",
      '''    <div class="body">
      <div class="ask">
        <div class="ask-h">1 · Let sites return per-case predictions, not just summaries</div>
        <div class="ask-b">
          What would be sent back: a <strong>row number</strong>, the correct answer, and
          the model's three probabilities. <strong>No case identifier and no patient
          identifier</strong> — the row number is a position within your own validation
          list and means nothing outside it. No images. Nothing new is computed: this is
          what your site already writes to its own disk during a normal training run.
        </div>
      </div>
      <div class="two">
        <div class="panel">
          <h3>Why we need it</h3>
          <ul>
            <li>Regional fine-tuning is currently blocked on physically moving a 722&#8239;MB model out of two hospitals. This removes that need entirely.</li>
            <li>A fair statistical comparison between hospitals needs case-by-case pairing. Summary figures cannot provide it.</li>
            <li>The regulatory testing node due next year requires it regardless.</li>
          </ul>
        </div>
        <div class="panel">
          <h3>What we are asking of you</h3>
          <ul>
            <li>Confirmation that this is acceptable under your local data agreement</li>
            <li>It stays off unless your site switches it on — the decision is yours, per site</li>
            <li>Nothing technical: no new software, no reconfiguration</li>
            <li>Please keep your client online so active-learning tests can run</li>
          </ul>
        </div>
      </div>
      <div class="ask">
        <div class="ask-h">2 · A slot with Cambridge and Nijmegen</div>
        <div class="ask-b">
          For the security testing exercise due at M54, where we deliberately attack our
          own system to show what it withstands. We are scheduling this now.
        </div>
      </div>
    </div>''')

html = CSS + '\n<div class="deck">\n' + "\n".join(SLIDES) + '''

  <footer>ODELIA WP2 / WP3 · 10 September 2026 · every figure read from the coordination server, the sites' own training logs, or CI</footer>
</div>
'''
out_path = sys.argv[1] if len(sys.argv) > 1 else "docs/presentation_consortium_briefing.html"
open(out_path, "w").write(html)
print(f"wrote {out_path} — {len(html):,} bytes, {html.count(chr(60) + chr(115) + chr(101) + chr(99))} slides")
