#!/usr/bin/env python3
"""Build docs/presentation_consortium_45min.html, the consortium briefing in the ODELIA
deck style (white slides, magenta titles, Open Sans, gradient title slide, logo strip).

Usage:  python3 scripts/presentation/build_consortium_45min.py

Visual identity: the ODELIA EAB deck (odelia_EAB_meeting_opensource_jeff.pptx): magenta
E5316B titles, coral E64F39 gradient, light-blue 82C3EC / indigo 4B56D2 accents, 888888
muted text, F2F2F2 panels, Open Sans. Logos live in docs/assets/logos and are inlined once
as CSS classes. An NCT Heidelberg logo is picked up from docs/assets/logos/nct.png when
present; otherwise a text mark stands in.

House style for the text (owner, 15 Sep 2026): descriptive titles, no dashes in prose, no
aphorisms, site abbreviations (UKA, UMCU, RUMC, USZ, CAM, MHA, RSH, VHIO), concise.
Charts come from consortium_45min_charts.js next to this file; the concept diagrams for the
deliverables are inline SVG defined here.
"""
from __future__ import annotations

import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "presentation_consortium_45min.html"
LOGOS = ROOT / "docs" / "assets" / "logos"
CHARTS_JS = pathlib.Path(__file__).with_name("consortium_45min_charts.js")

DATE = "17.09.2026"
VENUE = "Bremen"
STATUS_DATE = "17 September 2026"


def data_uri(name: str) -> str:
    p = LOGOS / name
    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


# --------------------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------------------
STYLE = """
<title>ODELIA Consortium Briefing</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --magenta:#E5316B; --coral:#E64F39; --sky:#82C3EC; --indigo:#4B56D2; --purple:#472183; --navy:#002445;
    --ground:#F4F4F4; --slide:#FFFFFF; --slide-alt:#F2F2F2;
    --ink:#1A1A1A; --ink-soft:#3A3A3A; --muted:#888888;
    --rule:#DDDDDD; --rule-soft:#EBEBEB;
    --accent:var(--magenta); --accent-soft:#FCE6ED;
    --good:#1F7A4D; --good-bg:#E4F3EA;
    --warn:#A2661D; --warn-bg:#FBEEDC;
    --bad:#B3262E;  --bad-bg:#FBE3E5;
    --s1:var(--indigo); --s2:var(--sky); --s3:var(--magenta);
    --sans:"Open Sans","Segoe UI","Helvetica Neue",Arial,sans-serif;
    --mono:ui-monospace,"SF Mono","Cascadia Mono","Roboto Mono",Menlo,Consolas,monospace;
  }
  html { color-scheme: light; }
  body { background:var(--ground); color:var(--ink); font-family:var(--sans); line-height:1.5; }
  .deck { max-width:68rem; margin:0 auto; padding:1.5rem 1rem 4rem; display:flex; flex-direction:column; gap:1.5rem; }
  .slide { background:var(--slide); border:1px solid var(--rule); border-radius:4px;
           padding:clamp(1.2rem,3vw,2.2rem) clamp(1.2rem,3vw,2.2rem) 3.4rem; min-height:min(56.25vw,37rem);
           display:flex; flex-direction:column; position:relative; }
  @media (max-width:820px){ .slide{ min-height:0; } }
  .eyebrow { font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:600; }
  h1 { font-size:clamp(2rem,4.6vw,3rem); line-height:1.1; margin:.4rem 0 0; font-weight:700; letter-spacing:-.01em; color:#fff; }
  h2 { font-size:clamp(1.25rem,2.5vw,1.7rem); line-height:1.2; margin:.3rem 0 0; font-weight:700; color:var(--magenta); text-wrap:balance; }
  .sub { color:var(--ink-soft); font-size:clamp(.92rem,1.5vw,1.02rem); margin-top:.6rem; max-width:46rem; text-wrap:pretty; }
  .body { margin-top:1rem; display:flex; flex-direction:column; gap:.8rem; flex:1; min-height:0; }
  p { margin:0; }
  strong { font-weight:700; }
  .cap { font-size:.78rem; color:var(--muted); }
  code { font-family:var(--mono); font-size:.9em; }
  /* logo strip + footer */
  .logos { display:flex; align-items:center; gap:1.4rem; flex-wrap:wrap; }
  .lg { display:inline-block; height:26px; background:center/contain no-repeat; }
  .lg-odelia { aspect-ratio:613/119; height:24px; background-image:url("{odelia}"); }
  .lg-tud    { aspect-ratio:269/78;  background-image:url("{tud}"); }
  .lg-ekfz   { aspect-ratio:213/77;  background-image:url("{ekfz}"); }
  .lg-kather { aspect-ratio:372/123; background-image:url("{kather}"); }
  .lg-ukd    { aspect-ratio:337/156; background-image:url("{ukd}"); }
  .lg-nct    { aspect-ratio:{nct_ratio}; background-image:url("{nct}"); }
  .lg-nct-text { display:inline-flex; align-items:center; gap:.4rem; height:26px; color:var(--navy); font-weight:700; font-size:1.05rem; letter-spacing:.02em; }
  .lg-nct-text small { font-weight:600; font-size:.52rem; line-height:1.1; letter-spacing:.04em; text-transform:uppercase; color:var(--navy); }
  .lg-nct-text small b { display:block; color:#3C7DD9; }
  .foot { position:absolute; left:0; right:0; bottom:0; padding:.55rem clamp(1.2rem,3vw,2.2rem);
          display:flex; align-items:center; justify-content:space-between; gap:1rem; border-top:1px solid var(--rule-soft); }
  .foot .lg { height:18px; } .foot .lg-odelia { height:17px; } .foot .lg-nct-text { height:18px; font-size:.8rem; } .foot .lg-nct-text small { font-size:.4rem; }
  .foot .pg { font-size:.66rem; color:var(--muted); letter-spacing:.06em; }
  /* title slide */
  .title { background:linear-gradient(90deg,var(--magenta) 0%,var(--coral) 100%); color:#fff; border-radius:4px;
           padding:clamp(1.4rem,3.4vw,2.4rem); margin-top:1rem; flex:1; display:flex; flex-direction:column; justify-content:center; gap:.6rem; }
  .title .who { font-size:.95rem; font-weight:600; }
  .title .who span { font-weight:400; opacity:.92; }
  .title .wp { font-size:clamp(1.1rem,2.2vw,1.4rem); font-weight:700; letter-spacing:.02em; margin-top:1rem; }
  .title .date { font-size:.92rem; opacity:.95; margin-top:1rem; }
  .title .stats { margin-top:1.2rem; background:transparent; border:none; gap:.5rem; }
  .title .stats > div { background:rgba(255,255,255,.14); }
  .title .stats dt { color:rgba(255,255,255,.85); } .title .stats dd { color:#fff; } .title .stats .cap { color:rgba(255,255,255,.85); }
  /* agenda blocks */
  .agenda { display:grid; grid-template-columns:repeat(5,1fr); gap:.6rem; margin-top:1.2rem; }
  @media (max-width:820px){ .agenda{ grid-template-columns:repeat(2,1fr);} }
  .agenda > div { background:var(--magenta); color:#fff; padding:1.1rem .9rem 1.2rem; min-height:11rem; display:flex; flex-direction:column; }
  .agenda > div:nth-child(2n) { background:var(--coral); }
  .agenda .n { font-size:2.3rem; font-weight:400; line-height:1; }
  .agenda .t { margin-top:auto; font-size:.8rem; line-height:1.3; font-weight:600; }
  /* divider slide */
  .divider { background:linear-gradient(90deg,var(--indigo) 0%,var(--sky) 100%); color:#fff; border-radius:4px; padding:clamp(1.2rem,3vw,2rem); margin-top:.8rem; }
  .divider h2 { color:#fff; }
  .divider p { color:#fff; opacity:.95; font-size:.95rem; margin-top:.4rem; max-width:46rem; }
  /* stats, tables, panels */
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(9rem,1fr)); gap:1px; background:var(--rule); border:1px solid var(--rule); border-radius:4px; overflow:hidden; }
  .stats > div { background:var(--slide); padding:.9rem 1rem; }
  .stats dt { font-size:.64rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:600; }
  .stats dd { margin:.2rem 0 0; font-size:1.55rem; font-weight:700; font-variant-numeric:tabular-nums; color:var(--magenta); }
  .stats .cap { display:block; font-size:.74rem; font-weight:400; color:var(--muted); margin-top:.1rem; line-height:1.35; }
  .scroll { overflow-x:auto; border:1px solid var(--rule); border-radius:4px; }
  table { border-collapse:collapse; width:100%; font-size:.86rem; min-width:32rem; }
  th, td { text-align:left; padding:.48rem .75rem; border-bottom:1px solid var(--rule-soft); vertical-align:top; }
  thead th { font-size:.64rem; letter-spacing:.1em; text-transform:uppercase; color:#fff; font-weight:600; background:var(--magenta); border-bottom:none; white-space:nowrap; }
  tbody tr:nth-child(even) td { background:var(--slide-alt); }
  tbody tr:last-child td { border-bottom:none; }
  tfoot td { font-weight:700; border-top:2px solid var(--rule); }
  td.n { font-variant-numeric:tabular-nums; white-space:nowrap; }
  td.id { white-space:nowrap; }
  .chip { display:inline-block; font-size:.64rem; letter-spacing:.06em; text-transform:uppercase; padding:.18em .55em; border-radius:3px; font-weight:700; white-space:nowrap; }
  .chip.good{background:var(--good-bg);color:var(--good);} .chip.warn{background:var(--warn-bg);color:var(--warn);}
  .chip.bad{background:var(--bad-bg);color:var(--bad);} .chip.info{background:var(--accent-soft);color:var(--magenta);}
  .callout { background:var(--accent-soft); border-left:4px solid var(--magenta); border-radius:3px; padding:.8rem 1rem; font-size:.9rem; color:var(--ink-soft); }
  .callout strong { color:var(--ink); }
  .callout.plain { background:var(--slide-alt); border-left-color:var(--muted); }
  .two { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  .two.wide { grid-template-columns:3fr 2fr; }
  @media (max-width:820px){ .two, .two.wide { grid-template-columns:1fr; } }
  .panel { background:var(--slide-alt); border-radius:4px; padding:.9rem 1rem; }
  .panel h3 { margin:0 0 .35rem; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:var(--magenta); font-weight:700; }
  .panel p { font-size:.87rem; }
  .panel.alt h3 { color:var(--indigo); }
  .ask { border:1px solid var(--magenta); border-radius:4px; overflow:hidden; }
  .ask-h { background:var(--magenta); color:#fff; padding:.55rem 1rem; font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700; }
  .ask-b { padding:.85rem 1rem; font-size:.92rem; color:var(--ink-soft); }
  .legend { display:flex; flex-wrap:wrap; gap:.9rem; font-size:.8rem; color:var(--ink-soft); }
  .legend span { display:inline-flex; align-items:center; gap:.35rem; }
  .sw { width:11px; height:11px; border-radius:2px; display:inline-block; }
  .chartwrap { padding:0; margin:0; max-width:none; display:block; background:transparent; }
  .chartwrap svg { display:block; width:100%; height:auto; overflow:visible; }
  figure { margin:0; }
  figure svg { display:block; width:100%; max-width:100%; height:auto; color:var(--ink); }
  figcaption { font-size:.78rem; color:var(--muted); margin-top:.35rem; }
  .fig-box { fill:#fff; stroke:currentColor; stroke-width:1.2; }
  .fig-box-alt { fill:var(--slide-alt); stroke:currentColor; stroke-width:1.2; }
  .fig-box-hi { fill:var(--accent-soft); stroke:var(--magenta); stroke-width:1.5; }
  .fig-box-ind { fill:#EEF0FB; stroke:var(--indigo); stroke-width:1.5; }
  .fig-t { font-family:var(--sans); font-size:12px; fill:currentColor; }
  .fig-t-b { font-family:var(--sans); font-size:12px; font-weight:700; fill:currentColor; }
  .fig-t-s { font-family:var(--sans); font-size:10.5px; fill:var(--ink-soft); }
  .fig-t-hi { font-family:var(--sans); font-size:11px; font-weight:700; fill:var(--magenta); }
  .fig-t-ind { font-family:var(--sans); font-size:11px; font-weight:700; fill:var(--indigo); }
  .fig-a { stroke:currentColor; stroke-width:1.3; fill:none; }
  .fig-a-hi { stroke:var(--magenta); stroke-width:1.8; fill:none; }
  .fig-a-dash { stroke:currentColor; stroke-width:1.3; fill:none; stroke-dasharray:4 3; }
  ul.todo { margin:0; padding-left:1.1rem; font-size:.84rem; }
  ul.todo li { margin:.12rem 0; }
  ul.plain { margin:0; padding-left:1.1rem; font-size:.9rem; }
  ul.plain li { margin:.2rem 0; }
  footer { text-align:center; color:var(--muted); font-size:.78rem; padding-top:1rem; }
  .notes { margin-top:auto; padding-top:.6rem; font-size:.86rem; color:var(--ink-soft); }
  .notes summary { cursor:pointer; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:var(--indigo); font-weight:700; list-style:none; }
  .notes summary::-webkit-details-marker { display:none; }
  .notes summary::before { content:"▸ "; }
  .notes[open] summary::before { content:"▾ "; }
  .notes p { margin:.4rem 0 0; max-width:60rem; line-height:1.55; }
  .notes-bar { display:flex; justify-content:flex-end; gap:.6rem; }
  .notes-bar button { font:inherit; font-size:.78rem; padding:.35rem .8rem; border:1px solid var(--indigo); color:var(--indigo); background:#fff; border-radius:3px; cursor:pointer; }
  .notes-bar button:hover { background:#EEF0FB; }
  @media print { .notes, .notes-bar { display:none; } }
  /* present mode: one slide at a time, 16:9, scaled to the window */
  .present-hint { font-size:.74rem; color:var(--muted); align-self:center; margin-right:auto; }
  .pres-inner { display:flex; flex-direction:column; flex:1 1 auto; min-height:0; transform-origin:top left; }
  body.presenting { overflow:hidden; background:#111; }
  body.presenting .deck { max-width:none; margin:0; padding:0; gap:0; display:block; }
  body.presenting .notes-bar, body.presenting footer, body.presenting .slide, body.presenting .notes { display:none; }
  body.presenting .slide.current { display:flex; position:fixed; left:50%; top:50%; width:1280px; height:720px; min-height:0; margin:0; border:0; border-radius:0;
                                   overflow:hidden; box-sizing:border-box; transform:translate(-50%,-50%) scale(var(--pk,1)); transform-origin:center center; cursor:pointer; }
  body.presenting .slide.current.measuring { height:auto; }
  .pres-hud, .pres-notes { display:none; }
  body.presenting .pres-hud { display:block; position:fixed; right:1rem; top:.6rem; z-index:5; color:#9a9a9a; font-size:.74rem; letter-spacing:.02em; }
  body.presenting.with-notes .pres-notes { display:block; position:fixed; left:0; right:0; bottom:0; max-height:34vh; overflow:auto; z-index:4;
                                           background:#1c1c1c; color:#eee; border-top:1px solid #444; padding:.9rem 1.4rem 1.2rem; font-size:1.02rem; line-height:1.5; }
  .pres-notes b { color:var(--sky); }
  @media (prefers-reduced-motion: reduce){ *{ animation:none!important; transition:none!important; } }
  @media print {
    @page { size: 13.33in 7.5in; margin: 0; }
    html { font-size: 12.5px; }
    body{background:#fff;} .deck{gap:0;padding:0;max-width:none;}
    .slide{break-before:page; break-inside:avoid; border:none; border-radius:0; min-height:0; height:7.2in; box-sizing:border-box; overflow:hidden; padding-bottom:2.6rem;}
    .slide:first-of-type{break-before:auto;}
    .foot{padding-top:.3rem;padding-bottom:.3rem;}
    .title{flex:0 0 auto; padding:1.6rem 2rem;}
    footer, .notes-bar{display:none;}
  }
</style>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1:      #ffffff;
    --text-primary:   #1A1A1A;
    --text-secondary: #3A3A3A;
    --text-muted:     #888888;
    --rule:           #DDDDDD;
    --rule-soft:      #EBEBEB;
    --series-1:       #4B56D2;   /* no lesion  */
    --series-2:       #82C3EC;   /* benign     */
    --series-3:       #E5316B;   /* malignant  */
    --bar:            #E5316B;
    --sans:"Open Sans","Segoe UI","Helvetica Neue",Arial,sans-serif;
    --mono:ui-monospace,"SF Mono","Cascadia Mono","Roboto Mono",Menlo,Consolas,monospace;
  }
  .viz-root.chartwrap { max-width:none; margin:0; padding:0; display:block; background:transparent; }
  .viz-root .lbl      { font-family: var(--sans); font-size: 12px; fill: var(--text-secondary); }
  .viz-root .lbl-site { font-family: var(--sans); font-size: 12px; font-weight:600; fill: var(--text-primary); }
  .viz-root .val      { font-family: var(--sans); font-size: 12px; fill: var(--text-primary); font-weight: 700; }
  .viz-root .val-sm   { font-family: var(--sans); font-size: 10.5px; fill: var(--text-secondary); }
  .viz-root .axis     { stroke: var(--rule); stroke-width: 1; }
  .viz-root .grid     { stroke: var(--rule-soft); stroke-width: 1; }
  .viz-root .bar rect, .viz-root .stack rect { transition: opacity .12s; }
  .viz-root .bar:hover rect, .viz-root .stack:hover rect { opacity: .82; }
  .viz-root svg { display:block; width:100%; height:auto; overflow:visible; }
</style>
"""

ARROW_DEFS = """<defs><marker id="{id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{fill}"/></marker></defs>"""


def defs(prefix: str) -> str:
    return (ARROW_DEFS.format(id=f"{prefix}-a", fill="currentColor")
            + ARROW_DEFS.format(id=f"{prefix}-h", fill="#E5316B")
            + ARROW_DEFS.format(id=f"{prefix}-i", fill="#4B56D2"))


# --------------------------------------------------------------------------------------
# concept diagrams (inline SVG)
# --------------------------------------------------------------------------------------
FIG_ACTIVE_LEARNING = f"""
<figure>
<svg viewBox="0 0 720 250" role="img" aria-label="Active learning loop: the unlabelled pool is scored by the model, the most uncertain cases are selected, a radiologist labels them, they join the training set, the model is retrained and scores the pool again; a dashed path marks the random-selection control arm.">
{defs("al")}
<rect x="10" y="60" width="130" height="56" rx="4" class="fig-box"/>
<text x="75" y="83" text-anchor="middle" class="fig-t-b">Unlabelled pool</text>
<text x="75" y="101" text-anchor="middle" class="fig-t-s">165 cases at the sites</text>
<line x1="140" y1="88" x2="188" y2="88" class="fig-a" marker-end="url(#al-a)"/>
<text x="164" y="80" text-anchor="middle" class="fig-t-s">score</text>
<rect x="190" y="60" width="140" height="56" rx="4" class="fig-box-hi"/>
<text x="260" y="83" text-anchor="middle" class="fig-t-hi">Model uncertainty</text>
<text x="260" y="101" text-anchor="middle" class="fig-t-s">entropy of the 3 outputs</text>
<line x1="330" y1="88" x2="378" y2="88" class="fig-a" marker-end="url(#al-a)"/>
<text x="354" y="80" text-anchor="middle" class="fig-t-s">rank</text>
<rect x="380" y="60" width="120" height="56" rx="4" class="fig-box"/>
<text x="440" y="83" text-anchor="middle" class="fig-t-b">Select top k</text>
<text x="440" y="101" text-anchor="middle" class="fig-t-s">k = labelling budget</text>
<line x1="500" y1="88" x2="548" y2="88" class="fig-a" marker-end="url(#al-a)"/>
<text x="524" y="80" text-anchor="middle" class="fig-t-s">label</text>
<rect x="550" y="60" width="160" height="56" rx="4" class="fig-box"/>
<text x="630" y="83" text-anchor="middle" class="fig-t-b">Radiologist labels k cases</text>
<text x="630" y="101" text-anchor="middle" class="fig-t-s">the scarce resource</text>
<path d="M630,116 L630,170 L260,170 L260,118" class="fig-a" marker-end="url(#al-a)"/>
<text x="445" y="163" text-anchor="middle" class="fig-t-s">add to the training set, retrain, score again</text>
<rect x="380" y="195" width="120" height="40" rx="4" class="fig-box-alt"/>
<text x="440" y="219" text-anchor="middle" class="fig-t">Random k</text>
<path d="M75,116 L75,215 L378,215" class="fig-a-dash" marker-end="url(#al-a)"/>
<text x="220" y="207" text-anchor="middle" class="fig-t-s">control arm: same budget, no scoring</text>
<line x1="500" y1="215" x2="630" y2="215" class="fig-a-dash"/>
<line x1="630" y1="215" x2="630" y2="118" class="fig-a-dash" marker-end="url(#al-a)"/>
</svg>
<figcaption>One acquisition round. The result on the next slide compares the two arms at fixed budgets, before any retraining.</figcaption>
</figure>
"""

FIG_DP = f"""
<figure>
<svg viewBox="0 0 720 240" role="img" aria-label="Per site and per round: local training produces a model update, the update is clipped to a norm bound, calibrated Gaussian noise is added, and only then is it sent to the aggregating site; a privacy accountant adds up epsilon per round against a budget.">
{defs("dp")}
<rect x="10" y="40" width="120" height="56" rx="4" class="fig-box"/>
<text x="70" y="63" text-anchor="middle" class="fig-t-b">Local training</text>
<text x="70" y="81" text-anchor="middle" class="fig-t-s">at one site, one round</text>
<line x1="130" y1="68" x2="178" y2="68" class="fig-a" marker-end="url(#dp-a)"/>
<text x="154" y="60" text-anchor="middle" class="fig-t-s">update</text>
<rect x="180" y="40" width="130" height="56" rx="4" class="fig-box-hi"/>
<text x="245" y="63" text-anchor="middle" class="fig-t-hi">Clip to norm C</text>
<text x="245" y="81" text-anchor="middle" class="fig-t-s">bounds one site's share</text>
<line x1="310" y1="68" x2="358" y2="68" class="fig-a-hi" marker-end="url(#dp-h)"/>
<rect x="360" y="40" width="150" height="56" rx="4" class="fig-box-hi"/>
<text x="435" y="63" text-anchor="middle" class="fig-t-hi">Add Gaussian noise</text>
<text x="435" y="81" text-anchor="middle" class="fig-t-s">scale σ · C</text>
<line x1="510" y1="68" x2="558" y2="68" class="fig-a" marker-end="url(#dp-a)"/>
<text x="534" y="60" text-anchor="middle" class="fig-t-s">send</text>
<rect x="560" y="40" width="150" height="56" rx="4" class="fig-box"/>
<text x="635" y="63" text-anchor="middle" class="fig-t-b">Aggregating site</text>
<text x="635" y="81" text-anchor="middle" class="fig-t-s">noisy weighted mean</text>
<rect x="180" y="150" width="330" height="56" rx="4" class="fig-box-ind"/>
<text x="345" y="173" text-anchor="middle" class="fig-t-ind">Privacy accountant (Rényi DP)</text>
<text x="345" y="191" text-anchor="middle" class="fig-t-s">adds ε per round; halts training when the budget is spent</text>
<line x1="435" y1="96" x2="435" y2="148" class="fig-a-dash" marker-end="url(#dp-i)"/>
<text x="470" y="126" text-anchor="start" class="fig-t-s">σ, rounds, δ</text>
<text x="10" y="135" class="fig-t-s">What leaves the site is the clipped, noised update. Training data never leaves.</text>
<text x="10" y="225" class="fig-t-s">Client-level guarantee: an observer of the shared model cannot tell whether a given hospital took part in a round.</text>
</svg>
<figcaption>Where the noise enters the swarm round. The two magenta steps are the calibrated-noise filter shipped in 1.8.x; the accountant turns σ and the number of rounds into an ε.</figcaption>
</figure>
"""

FIG_ROBUST = f"""
<figure>
<svg viewBox="0 0 720 260" role="img" aria-label="Eight sites send updates to the aggregating site. One update is poisoned. Under a weighted mean it moves the shared model by 5.5 times its own share; under a norm-bounded mean each contribution is capped and the shared model moves by 0.04.">
{defs("rb")}
<text x="10" y="18" class="fig-t-b">Sites send updates</text>
<rect x="10" y="30" width="92" height="26" rx="3" class="fig-box"/><text x="56" y="47" text-anchor="middle" class="fig-t-s">UKA 52%</text>
<rect x="10" y="62" width="92" height="26" rx="3" class="fig-box"/><text x="56" y="79" text-anchor="middle" class="fig-t-s">UMCU 18%</text>
<rect x="10" y="94" width="92" height="26" rx="3" class="fig-box"/><text x="56" y="111" text-anchor="middle" class="fig-t-s">RUMC 10%</text>
<rect x="10" y="126" width="92" height="26" rx="3" class="fig-box"/><text x="56" y="143" text-anchor="middle" class="fig-t-s">USZ 10%</text>
<rect x="10" y="158" width="92" height="26" rx="3" class="fig-box"/><text x="56" y="175" text-anchor="middle" class="fig-t-s">CAM MHA RSH 9%</text>
<rect x="10" y="190" width="92" height="26" rx="3" class="fig-box-hi"/><text x="56" y="207" text-anchor="middle" class="fig-t-hi">VHIO 0.6%</text>
<text x="10" y="236" class="fig-t-s">poisoned update (the attacker)</text>
<line x1="102" y1="43" x2="176" y2="118" class="fig-a"/>
<line x1="102" y1="75" x2="176" y2="118" class="fig-a"/>
<line x1="102" y1="107" x2="176" y2="118" class="fig-a"/>
<line x1="102" y1="139" x2="176" y2="118" class="fig-a"/>
<line x1="102" y1="171" x2="176" y2="118" class="fig-a"/>
<line x1="102" y1="203" x2="176" y2="118" class="fig-a-hi"/>
<rect x="178" y="94" width="126" height="48" rx="4" class="fig-box-alt"/>
<text x="241" y="114" text-anchor="middle" class="fig-t-b">Aggregation rule</text>
<text x="241" y="131" text-anchor="middle" class="fig-t-s">one of the two below</text>
<line x1="304" y1="118" x2="333" y2="66" class="fig-a" marker-end="url(#rb-a)"/>
<line x1="304" y1="118" x2="333" y2="182" class="fig-a" marker-end="url(#rb-a)"/>
<rect x="335" y="30" width="375" height="70" rx="4" class="fig-box"/>
<text x="350" y="52" class="fig-t-b">Weighted mean (current)</text>
<text x="350" y="70" class="fig-t-s">no bound on any contribution</text>
<text x="350" y="88" class="fig-t-hi">shared model moves 5.5 × the attacker's share</text>
<rect x="335" y="150" width="375" height="70" rx="4" class="fig-box-ind"/>
<text x="350" y="172" class="fig-t-ind">Norm-bounded mean (recommended)</text>
<text x="350" y="190" class="fig-t-s">each contribution capped, data-set weights kept, every site stays</text>
<text x="350" y="208" class="fig-t-ind">same attack: error 0.04, no-attack error 0.032</text>
</svg>
<figcaption>The exposure and the defence. Trimmed mean, coordinate median and Krum discard contributions and drop the data-set weights (next slide). The bound is proportional to a site's weight, so it does not limit the largest contributor.</figcaption>
</figure>
"""

FIG_WHITEHAT = f"""
<figure>
<svg viewBox="0 0 720 250" role="img" aria-label="Milestone MS6 as the proposal describes it: the swarm-trained model weights go to the attacking parties CAM and RUMC, who attempt to reconstruct or infer training data; findings feed a fix and a preprint. Below, the preliminary in-house probe that uses only the model's outputs.">
{defs("wh")}
<text x="10" y="18" class="fig-t-b">MS6 as written in the proposal (due M54, June 2027)</text>
<rect x="10" y="30" width="150" height="56" rx="4" class="fig-box"/>
<text x="85" y="53" text-anchor="middle" class="fig-t-b">Swarm-trained model</text>
<text x="85" y="71" text-anchor="middle" class="fig-t-s">full weights, from TUD</text>
<line x1="160" y1="58" x2="208" y2="58" class="fig-a" marker-end="url(#wh-a)"/>
<rect x="210" y="30" width="200" height="56" rx="4" class="fig-box-hi"/>
<text x="310" y="53" text-anchor="middle" class="fig-t-hi">CAM and RUMC attack</text>
<text x="310" y="71" text-anchor="middle" class="fig-t-s">reconstruction, membership inference</text>
<line x1="410" y1="58" x2="448" y2="58" class="fig-a" marker-end="url(#wh-a)"/>
<rect x="450" y="30" width="100" height="56" rx="4" class="fig-box"/>
<text x="500" y="53" text-anchor="middle" class="fig-t-b">Findings</text>
<text x="500" y="71" text-anchor="middle" class="fig-t-s">what leaks, how</text>
<line x1="550" y1="58" x2="588" y2="58" class="fig-a" marker-end="url(#wh-a)"/>
<rect x="590" y="30" width="120" height="56" rx="4" class="fig-box"/>
<text x="650" y="53" text-anchor="middle" class="fig-t-b">Fix + preprint</text>
<text x="650" y="71" text-anchor="middle" class="fig-t-s">verification of MS6</text>
<text x="10" y="128" class="fig-t-b">Done at TUD in the meantime: outputs-only probe</text>
<rect x="10" y="140" width="150" height="56" rx="4" class="fig-box-alt"/>
<text x="85" y="163" text-anchor="middle" class="fig-t">Model outputs only</text>
<text x="85" y="181" text-anchor="middle" class="fig-t-s">3 probabilities per case</text>
<line x1="160" y1="168" x2="208" y2="168" class="fig-a-dash" marker-end="url(#wh-a)"/>
<rect x="210" y="140" width="150" height="56" rx="4" class="fig-box-alt"/>
<text x="285" y="163" text-anchor="middle" class="fig-t">Which hospital?</text>
<text x="285" y="181" text-anchor="middle" class="fig-t-s">classifier on the outputs</text>
<line x1="360" y1="168" x2="408" y2="168" class="fig-a-dash" marker-end="url(#wh-a)"/>
<rect x="410" y="140" width="300" height="56" rx="4" class="fig-box-alt"/>
<text x="560" y="163" text-anchor="middle" class="fig-t">60 % correct on malignant cases (chance 33 %)</text>
<text x="560" y="181" text-anchor="middle" class="fig-t-s">a lower bound; the MS6 attacker has the weights</text>
<text x="10" y="235" class="fig-t-s">The probe needs no partner action and told us what to look for. The milestone itself is the top row and needs CAM and RUMC.</text>
</svg>
<figcaption>Plan and interim work for the white-hat milestone.</figcaption>
</figure>
"""

FIG_PERCASE = f"""
<figure>
<svg viewBox="0 0 720 150" role="img" aria-label="What per-case return sends from a site to the coordinator: a row number, the label and three probabilities per validation case; no identifiers and no images.">
{defs("pc")}
<rect x="10" y="30" width="200" height="80" rx="4" class="fig-box"/>
<text x="110" y="55" text-anchor="middle" class="fig-t-b">Site (RUMC, UMCU, MHA)</text>
<text x="110" y="75" text-anchor="middle" class="fig-t-s">images and labels stay here</text>
<text x="110" y="93" text-anchor="middle" class="fig-t-s">opt-in per site, off by default</text>
<line x1="210" y1="70" x2="268" y2="70" class="fig-a-hi" marker-end="url(#pc-h)"/>
<text x="239" y="60" text-anchor="middle" class="fig-t-s">per case</text>
<rect x="270" y="30" width="340" height="80" rx="4" class="fig-box-hi"/>
<text x="440" y="55" text-anchor="middle" class="fig-t-hi">row · label · p(no lesion) · p(benign) · p(malignant)</text>
<text x="440" y="75" text-anchor="middle" class="fig-t-s">no case or patient identifier</text>
<text x="440" y="93" text-anchor="middle" class="fig-t-s">no image data</text>
<line x1="610" y1="70" x2="648" y2="70" class="fig-a" marker-end="url(#pc-a)"/>
<rect x="650" y="30" width="60" height="80" rx="4" class="fig-box"/>
<text x="680" y="65" text-anchor="middle" class="fig-t-b">TUD</text>
<text x="680" y="83" text-anchor="middle" class="fig-t-s">D2.5</text>
</svg>
<figcaption>Per-case return, shipped in 1.8.0 and verified on real kits on 14 September. It closes D2.5 without moving a model out of a hospital.</figcaption>
</figure>
"""

# --------------------------------------------------------------------------------------
# slide fragments
# --------------------------------------------------------------------------------------
HEADER_LOGOS = """<div class="logos"><span class="lg lg-tud" role="img" aria-label="TU Dresden"></span><span class="lg lg-ukd" role="img" aria-label="Universitätsklinikum Carl Gustav Carus Dresden"></span><span class="lg lg-ekfz" role="img" aria-label="EKFZ for Digital Health"></span><span class="lg lg-kather" role="img" aria-label="Kather Lab"></span>{nct}<span class="lg lg-odelia" role="img" aria-label="ODELIA"></span></div>"""

FOOTER = """<div class="foot"><div class="logos"><span class="lg lg-odelia" role="img" aria-label="ODELIA"></span></div><div class="pg">Slide {n}</div><div class="logos"><span class="lg lg-ukd"></span><span class="lg lg-ekfz"></span>{nct}<span class="lg lg-tud"></span></div></div>"""

NCT_IMG = '<span class="lg lg-nct" role="img" aria-label="NCT Heidelberg"></span>'
NCT_TEXT = '<span class="lg-nct-text" role="img" aria-label="NCT Heidelberg">NCT<small>Nationales Centrum<br>für Tumorerkrankungen<b>Heidelberg</b></small></span>'


def slide(body: str, cls: str = "slide") -> str:
    return f'\n  <section class="{cls}">\n{body}\n  </section>\n'


def eyebrow_h2(eyebrow: str, title: str) -> str:
    return f'    <div class="eyebrow">{eyebrow}</div>\n    <h2>{title}</h2>'


S_TITLE = """    {header}
    <div class="title">
      <div class="who">JieFu Zhu, M.Sc. <span>· Else Kröner Fresenius Center for Digital Health, TU Dresden · NCT Heidelberg</span></div>
      <div class="wp">Work Package 2 / 3</div>
      <h1>Month 45</h1>
      <div class="stats">
        <div><dt>Training volumes</dt><dd>34,462<span class="cap">8 hospitals, 6 countries, none moved</span></dd></div>
        <div><dt>Shared model</dt><dd>0.887<span class="cap">malignant AUROC, external challenge set</span></dd></div>
        <div><dt>Independent US cohort</dt><dd>0.887<span class="cap">Duke, 260 volumes, never seen in training</span></dd></div>
        <div><dt>Software</dt><dd>1.8.1<span class="cap">released 13 Sep 2026, kits at every site</span></dd></div>
      </div>
      <div class="date">ODELIA · {venue} · {date}</div>
    </div>"""

S_AGENDA = """    <h2>Agenda</h2>
    <div class="agenda">
      <div><span class="n">01</span><span class="t">Data<br>what eight hospitals bring</span></div>
      <div><span class="n">02</span><span class="t">Model results<br>and external validation</span></div>
      <div><span class="n">03</span><span class="t">Other deliverables<br>D2.5 · D3.2 · T3.3 · D3.4 · MS6</span></div>
      <div><span class="n">04</span><span class="t">Milestones and timeline<br>what concludes in 2026</span></div>
      <div><span class="n">05</span><span class="t">Sites<br>status, to-do list, requests</span></div>
    </div>
    <p class="cap" style="margin-top:1rem">Grant month M45 = September 2026. M48 = December 2026, M54 = June 2027, M60 = December 2027.</p>"""

S_DATA_BARS = eyebrow_h2("01 · Data", "Training data per site") + """
    <div class="body">
    <div class="viz-root chartwrap">
      <svg viewBox="0 0 720 300" role="img" aria-label="Training samples per site: UKA 17834, UMCU 6133, RUMC 3608, USZ 3448, CAM 1778, MHA 1120, RSH 351, VHIO 190"><g id="sizeBars"></g></svg>
    </div>
    <p class="cap">Training volumes per site, counted by each site's trainer when it loads the data set.</p>
      <div class="callout">
        <strong>UKA holds 51.7&#8239;% of all training data</strong>, more than the other seven sites together.
        The range is 94-fold, from 17,834 volumes (UKA) to 190 (VHIO).
      </div>
    </div>"""

S_DATA_CUM = eyebrow_h2("01 · Data", "How the data set grew") + """
    <div class="body">
    <div class="viz-root chartwrap">
      <svg viewBox="0 0 720 330" role="img" aria-label="Cumulative consortium training data growing from 4418 volumes across two sites on 30 April 2026 to 34462 across eight sites by 31 July 2026"><g id="cumChart"></g></svg>
    </div>
    <p class="cap">Running total of training data available to the swarm as sites came online.</p>
      <div class="callout">
        <strong>UKA and CAM on 11 June added 18,804 volumes</strong>, UMCU on 3 July another 6,877.
        The eighth site, VHIO, joined on 31 July.
      </div>
    </div>"""

S_DATA_ABS = eyebrow_h2("01 · Data", "Class counts per site") + """
    <div class="body">
    <div class="viz-root chartwrap">
      <svg viewBox="0 0 720 300" role="img" aria-label="Class counts per site at true scale: UKA's 17834 volumes dwarf the rest; VHIO's 190 and RSH's 351 are barely visible"><g id="absBars"></g></svg>
    </div>
    <div class="legend">
      <span><i class="sw" style="background:var(--s1)"></i> No lesion</span>
      <span><i class="sw" style="background:var(--s2)"></i> Benign</span>
      <span><i class="sw" style="background:var(--s3)"></i> Malignant</span>
    </div>
    <p class="cap">The eight sites stacked by diagnostic class, drawn at true scale.</p>
      <div class="callout">
        <strong>UKA holds 83.9&#8239;% of all benign cases</strong> (11,836 of 14,110). VHIO has no benign cases.
        This imbalance shapes the aggregation and robustness results later in the deck.
      </div>
    </div>"""

S_DATA_SHARE = eyebrow_h2("01 · Data", "Class distribution per site") + """
    <div class="body">
    <div class="viz-root chartwrap">
      <svg viewBox="0 0 720 300" role="img" aria-label="Training-set class distribution per site: UKA is 66 percent benign, RSH is 63 percent malignant, RUMC and VHIO have almost no benign cases"><g id="classBars"></g></svg>
    </div>
    <div class="legend">
      <span><i class="sw" style="background:var(--s1)"></i> No lesion</span>
      <span><i class="sw" style="background:var(--s2)"></i> Benign</span>
      <span><i class="sw" style="background:var(--s3)"></i> Malignant</span>
    </div>
    <p class="cap">Each site's class mix, normalised to 100&#8239;%.</p>
      <div class="callout">
        <strong>RSH is 63&#8239;% malignant, which is 221 volumes.</strong> UKA is 7&#8239;% malignant, which is 1,251 volumes.
        Read this chart together with the previous one.
      </div>
    </div>"""

S_DATA_TABLE = eyebrow_h2("01 · Data", "Site data detailed view") + """
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr>
            <th>Site</th><th>Joined</th><th>At joining</th><th>Now</th><th>Share</th>
            <th>No lesion</th><th>Benign</th><th>Malignant</th><th>Benign&#8239;%</th>
          </tr></thead>
          <tbody id="tableBody"></tbody>
        </table>
      </div>
      <p class="cap">Counts from each site's trainer at data-set load. The three classes sum to each site's total; the eight sites sum to 34,462.</p>
      <div class="callout plain">
        Join dates are taken from the reporting host. An earlier version listed four join dates about six weeks early because
        our test machines had reported under hospital site names; test sites now have their own names.
      </div>
    </div>"""

S_RES_MODEL = eyebrow_h2("02 · Model results", "Swarm model versus single-site models") + """
    <div class="body">
    <div class="viz-root chartwrap">
      <svg viewBox="0 0 720 300" role="img" aria-label="External challenge AUROC: the eight-site swarm reaches 0.887 malignant and 0.820 macro, above the six-site swarm at 0.847 and 0.764, the best single site at 0.858 and 0.709, and the median single site at 0.658 and 0.601"><g id="modelBars"></g></svg>
    </div>
    <div class="legend">
      <span><i class="sw" style="background:var(--s1)"></i> Malignant vs rest AUROC</span>
      <span><i class="sw" style="background:var(--s2)"></i> Macro AUROC (three classes)</span>
    </div>
    <p class="cap">External challenge set. The eight-site swarm compared with the previous six-site swarm and with single-site models.</p>
      <div class="callout">
        <strong>The swarm model scores higher than every single-site model</strong>, including UKA's, which trains on half the data.
        A typical site gains +0.229 malignant AUROC by joining.
      </div>
    </div>"""

S_RES_CI = eyebrow_h2("02 · Model results", "Results with confidence intervals") + """
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Evaluation</th><th>n</th><th>Malignant AUROC</th><th>95&#8239;% CI</th><th>Recall</th><th>Specificity</th></tr></thead>
          <tbody>
            <tr><td class="id">Challenge set, malignant vs rest</td><td class="n">165</td><td class="n"><strong>0.887</strong></td><td class="n">0.818 to 0.945</td><td class="n">0.486</td><td class="n">0.977</td></tr>
            <tr><td class="id">Challenge set, malignant vs no lesion</td><td class="n">148</td><td class="n"><strong>0.903</strong></td><td class="n">0.837 to 0.956</td><td class="n">0.486</td><td class="n">0.991</td></tr>
            <tr style="background:var(--good-bg)"><td class="id"><strong>Duke, independent US cohort</strong></td><td class="n">260</td><td class="n"><strong>0.887</strong></td><td class="n">0.845 to 0.926</td><td class="n">0.662</td><td class="n">0.976</td></tr>
          </tbody>
        </table>
      </div>
      <p class="cap">Bootstrap intervals computed from the stored per-case predictions.</p>
      <div class="callout">
        The challenge set has 37 malignant cases, so an AUROC on it is known to about &plusmn;0.06.
        0.887 and 0.903 lie within one interval of each other.
      </div>
    </div>"""

S_RES_DUKE = eyebrow_h2("02 · Model results", "External validation on the Duke cohort") + """
    <div class="body">
      <p class="sub" style="margin-top:0">
        Duke Breast MRI is a public US data set of 260 unilateral volumes, none seen in training.
        Same task on both sides: malignant against no lesion.
      </p>
      <div class="scroll">
        <table>
          <thead><tr><th>Cohort</th><th>n</th><th>Malignant</th><th>AUROC</th><th>95&#8239;% CI</th></tr></thead>
          <tbody>
            <tr><td class="id">Europe, ODELIA challenge set, 5 sites</td><td class="n">148</td><td class="n">37</td><td class="n">0.903</td><td class="n">0.837 to 0.956</td></tr>
            <tr><td class="id">United States, Duke</td><td class="n">260</td><td class="n">136</td><td class="n">0.887</td><td class="n">0.845 to 0.926</td></tr>
          </tbody>
          <tfoot><tr><td class="id"><strong>Difference</strong></td><td class="n"></td><td class="n"></td><td class="n"><strong>0.016</strong></td><td class="n">intervals overlap</td></tr></tfoot>
        </table>
      </div>
      <div class="callout">
        <strong>AUROC drops by 0.016 with overlapping intervals.</strong> The model trained on eight European hospitals
        transfers to a US cohort without a measurable loss.
      </div>
    </div>"""

S_RES_THRESH = eyebrow_h2("02 · Model results", "Operating point: recall and specificity") + """
    <div class="body">
      <p class="sub" style="margin-top:0">
        AUROC measures ranking. The decision threshold is a separate choice, and the current one is conservative.
      </p>
      <div class="two">
        <div class="panel">
          <h3>Challenge set</h3>
          <p><strong>Recall 0.486</strong> at specificity <strong>0.991</strong>. About half the cancers are missed; false alarms are rare.</p>
        </div>
        <div class="panel">
          <h3>Duke</h3>
          <p><strong>Recall 0.662</strong> at specificity <strong>0.976</strong>. The same pattern on the independent cohort.</p>
        </div>
      </div>
      <div class="callout">
        <strong>Moving the threshold trades specificity for recall without retraining.</strong> The acceptable
        false-positive rate is a clinical decision for the consortium (slide 27).
      </div>
    </div>"""

S_RES_SITES = eyebrow_h2("02 · Model results", "Accuracy versus AUROC per site") + """
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Site</th><th>Cases</th><th>No lesion</th><th>Benign</th><th>Malignant</th><th>Accuracy</th><th>AUROC</th></tr></thead>
          <tbody>
            <tr style="background:var(--bad-bg)"><td class="id">RUMC_1</td><td class="n">896</td><td class="n">888</td><td class="n">0</td><td class="n">8</td><td class="n">99.1&#8239;%</td><td class="n">0.417</td></tr>
            <tr><td class="id">UMCU_1</td><td class="n">1,557</td><td class="n">1,361</td><td class="n">186</td><td class="n">10</td><td class="n">87.4&#8239;%</td><td class="n">0.698</td></tr>
          </tbody>
        </table>
      </div>
      <p class="cap">Per-site validation metrics as returned to the coordinator by the 1.8.x software, job 513c8988.</p>
      <div class="two">
        <div class="panel">
          <h3>RUMC_1</h3>
          <p>888 of 896 validation cases have no lesion. Predicting no lesion for every case gives 99&#8239;% accuracy and finds no cancer.
          The AUROC of 0.417 shows that the model does not separate the classes there.</p>
        </div>
        <div class="panel">
          <h3>Macro averages</h3>
          <p>UMCU was reported as weak on a three-class macro average where one class had two cases. Without that class,
          UMCU moves from 0.602 to 0.717 and CAM from 0.710 to 0.908.</p>
        </div>
      </div>
      <div class="callout">
        Every per-site figure now carries its class counts. Sites are not ranked on averages over classes with a handful of cases.
      </div>
    </div>"""

S_DELIV_OVERVIEW = """    <div class="eyebrow">03 · Other deliverables</div>
    <div class="divider">
      <h2>The remaining WP2 / WP3 deliverables</h2>
      <p>What the proposal asks for, when it is due, and which sites are needed to conclude it. The next slides explain each one and show what exists today.</p>
    </div>
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Deliverable</th><th>Task</th><th>Due</th><th>What the proposal asks</th><th>Sites needed</th></tr></thead>
          <tbody>
            <tr><td class="id"><strong>D2.5</strong> Regional fine-tuning</td><td>T2.4</td><td class="n">M48 · Dec 2026</td><td>Compare the consortium model with models fine-tuned to the RUMC and UMCU cohort and to the MHA cohort</td><td>RUMC, UMCU, MHA with per-case return on</td></tr>
            <tr><td class="id"><strong>D3.2</strong> Active learning</td><td>T3.2</td><td class="n">M48 · Dec 2026</td><td>Show that the model can request the cases worth labelling next</td><td>The fleet, for one reduced retraining run</td></tr>
            <tr><td class="id"><strong>D3.4</strong> Adversarial robustness</td><td>WP3</td><td class="n">M48 · Dec 2026</td><td>Report on attacks against a swarm and the measures against them</td><td>None (simulation at TUD)</td></tr>
            <tr><td class="id"><strong>T3.3</strong> Differential privacy</td><td>T3.3</td><td class="n">M54 · Jun 2027</td><td>Optional noise in training with a privacy budget by design</td><td>One 20-round run to measure the accuracy cost</td></tr>
            <tr><td class="id"><strong>MS6</strong> White-hat attack</td><td>T3.3</td><td class="n">M54 · Jun 2027</td><td>CAM and RUMC attack the trained model to expose and fix vulnerabilities</td><td>CAM, RUMC</td></tr>
            <tr><td class="id"><strong>D3.3</strong> Software v2.0, testing node</td><td>T3.2</td><td class="n">M54 · Jun 2027</td><td>Active learning as a feature; a testing node that receives per-site performance</td><td>All sites, after D3.2</td></tr>
            <tr><td class="id"><strong>D3.5</strong> Operational demonstration</td><td>T3.4</td><td class="n">M60 · Dec 2027</td><td>The prototype performing in an operational environment (TRL7)</td><td>All eight sites: the October benchmark</td></tr>
          </tbody>
        </table>
      </div>
    </div>"""

S_D25 = eyebrow_h2("03 · Other deliverables · D2.5 · M48", "Regional fine-tuning: what is needed and what exists") + """
    <div class="body">
      """ + FIG_PERCASE + """
      <div class="two">
        <div class="panel">
          <h3>Exists</h3>
          <p>Baselines of the consortium model on the challenge split for the RUMC and UMCU cohort and for the MHA cohort. Fine-tuning code and the paired analysis.
          Per-case return in the software, verified on real kits.</p>
        </div>
        <div class="panel alt">
          <h3>Needed</h3>
          <p>One consortium run with per-case return switched on at RUMC, UMCU and MHA (the October benchmark), then the comparison report.
          A null result is a valid D2.5.</p>
        </div>
      </div>
    </div>"""

S_AL_CONCEPT = eyebrow_h2("03 · Other deliverables · D3.2 · M48", "Active learning: how it works") + """
    <div class="body">
      """ + FIG_ACTIVE_LEARNING + """
      <div class="two">
        <div class="panel">
          <h3>Proposal text</h3>
          <p>"The image types and label categories in which the model has a low performance will be actively requested by the model
          and can then be supplied for the next training iteration."</p>
        </div>
        <div class="panel alt">
          <h3>In this project</h3>
          <p>Uncertainty is the entropy of the model's three class probabilities per case. The control arm draws cases at random with the same budget.
          The signal to request against: the model under-calls malignancy (21 predicted, 37 true, on the challenge set).</p>
        </div>
      </div>
    </div>"""

S_AL_RESULT = eyebrow_h2("03 · Other deliverables · D3.2 · M48", "Active learning: acquisition result") + """
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Labelling budget</th><th>Random selection</th><th>Entropy selection</th><th>Ratio</th></tr></thead>
          <tbody>
            <tr><td class="id">10 cases</td><td class="n">2.4 &plusmn; 1.3</td><td class="n"><strong>5</strong></td><td class="n">2.1&times;</td></tr>
            <tr><td class="id">20 cases</td><td class="n">4.6 &plusmn; 1.8</td><td class="n"><strong>9</strong></td><td class="n">2.0&times;</td></tr>
            <tr><td class="id">30 cases</td><td class="n">6.8 &plusmn; 2.1</td><td class="n"><strong>11</strong></td><td class="n">1.6&times;</td></tr>
            <tr><td class="id">40 cases</td><td class="n">8.9 &plusmn; 2.3</td><td class="n"><strong>14</strong></td><td class="n">1.6&times;</td></tr>
            <tr><td class="id">100 cases</td><td class="n">22.2 &plusmn; 2.8</td><td class="n">21</td><td class="n">1.0&times;</td></tr>
          </tbody>
        </table>
      </div>
      <p class="cap">Malignant cases found within each budget, from a pool of 165 cases containing 37. Entropy selection beats random at 5 of 8 budgets and loses at none.</p>
      <div class="callout">
        <strong>At small budgets entropy selection finds about twice as many malignant cases as random selection.</strong>
        By 100 of 165 cases both methods have taken most of the pool and converge.
      </div>
      <div class="callout plain">
        Not yet shown: a model retrained on the selected cases. That is the reduced retraining curve (2 arms, 2 budgets, about 45 h of consortium GPU)
        planned for October and November.
      </div>
    </div>"""

S_DP_CONCEPT = eyebrow_h2("03 · Other deliverables · T3.3 · M54", "Differential privacy: where the noise goes") + """
    <div class="body">
      """ + FIG_DP + """
      <div class="two">
        <div class="panel">
          <h3>Proposal text</h3>
          <p>"Implement the technology to optionally include artificial noise in the training process and limit the privacy budget by design."</p>
        </div>
        <div class="panel alt">
          <h3>Status</h3>
          <p>Calibrated-noise filter and the Rényi accountant exist (phase 2, September). Still open: per-round ε in the metrics, the budget halt,
          and the accuracy cost measured on the benchmark (phase 3).</p>
        </div>
      </div>
    </div>"""

S_DP_RESULT = eyebrow_h2("03 · Other deliverables · T3.3 · M54", "Differential privacy: guarantee and cost") + """
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Privacy guarantee (&epsilon;)</th><th>Noise required (&sigma;)</th><th>Reading</th></tr></thead>
          <tbody>
            <tr><td class="n">&epsilon; &le; 10</td><td class="n"><strong>2.4</strong></td><td>Widely accepted in practice</td></tr>
            <tr><td class="n">&epsilon; &le; 3</td><td class="n"><strong>6.7</strong></td><td>Strong</td></tr>
            <tr><td class="n">&epsilon; &le; 1</td><td class="n"><strong>18.1</strong></td><td>Strictest commonly cited; likely destroys utility at this scale</td></tr>
          </tbody>
        </table>
      </div>
      <p class="cap">Over a 20-round run at &delta; = 10<sup>&minus;5</sup>. Rényi-DP accounting, implemented in the ODELIA software.</p>
      <div class="two">
        <div class="panel">
          <h3>Covered</h3>
          <p>Client level: an observer of the shared model cannot tell whether a given hospital took part in a round.</p>
        </div>
        <div class="panel">
          <h3>Not covered</h3>
          <p>Whether a given patient was in a hospital's data. That needs noise inside local training and is not claimed.</p>
        </div>
      </div>
      <div class="callout">
        <strong>The noise level is a consortium decision.</strong> One 20-round run at &sigma; = 2.4 measures the accuracy cost of &epsilon; &le; 10 and turns this table into a trade-off.
      </div>
    </div>"""

S_ROB_CONCEPT = eyebrow_h2("03 · Other deliverables · D3.4 · M48", "Adversarial robustness: attack and defence") + """
    <div class="body">
      """ + FIG_ROBUST + """
      <div class="two">
        <div class="panel">
          <h3>Proposal text</h3>
          <p>"Report describing potential adversarial attacks and the subsequent measures needed to make SL-based AI fully robust against such attacks."
          In a swarm every site is a potential adversary, because every site sends weights.</p>
        </div>
        <div class="panel alt">
          <h3>Measured exposure</h3>
          <p>Every job aggregates by a weighted mean with no bound. VHIO, with 190 volumes (0.6&#8239;% of the data), can move the shared model
          by 5.5 times its own share.</p>
        </div>
      </div>
    </div>"""

S_ROB_RESULT = eyebrow_h2("03 · Other deliverables · D3.4 · M48", "Adversarial robustness: measured effect of five rules") + """
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Aggregation rule</th><th>Error with no attacker</th><th>Error under the attack</th><th>Keeps data-set weights</th></tr></thead>
          <tbody>
            <tr><td class="id">Weighted mean (current)</td><td class="n">0.023</td><td class="n">5.5&times; the attacker's share</td><td>Yes, no protection</td></tr>
            <tr style="background:var(--good-bg)"><td class="id"><strong>Norm-bounded mean</strong></td><td class="n"><strong>0.032</strong></td><td class="n"><strong>0.04</strong></td><td><strong>Yes</strong></td></tr>
            <tr><td class="id">Trimmed mean</td><td class="n">0.466</td><td class="n">bounded</td><td>No</td></tr>
            <tr><td class="id">Coordinate median</td><td class="n">0.600</td><td class="n">bounded</td><td>No</td></tr>
            <tr><td class="id">Krum</td><td class="n">0.639</td><td class="n">keeps one site's update</td><td>No</td></tr>
          </tbody>
        </table>
      </div>
      <p class="cap">Poisoning simulation on the consortium's data-set weights, September 2026.</p>
      <div class="two">
        <div class="panel">
          <h3>Why the standard rules cost so much here</h3>
          <p>Trimmed mean, coordinate median and Krum discard unusual contributions and cannot use data-set weights. With UKA holding 84&#8239;% of benign cases,
          they aim at the average site instead of the pooled data.</p>
        </div>
        <div class="panel alt">
          <h3>Recommendation</h3>
          <p>Bound each contribution and keep every site. The bound is proportional to weight, so it does not limit the largest contributor;
          capping any site's share is a governance decision with an accuracy cost.</p>
        </div>
      </div>
    </div>"""

S_WH_PLAN = eyebrow_h2("03 · Other deliverables · MS6 · M54", "White-hat attack: plan and interim work") + """
    <div class="body">
      """ + FIG_WHITEHAT + """
      <div class="two">
        <div class="panel">
          <h3>Proposal text</h3>
          <p>"We will also set up a white hat hacker attack to expose and subsequently fix any vulnerabilities of our code." Part A names CAM and RUMC as the attacking parties.
          Verification: a preprint plus a journal submission.</p>
        </div>
        <div class="panel alt">
          <h3>Needed from CAM and RUMC</h3>
          <p>A slot in Q1 2027, agreed this autumn. TUD provides the model weights and an attack protocol; the attack runs at the two sites; findings and the fix are written up together.</p>
        </div>
      </div>
    </div>"""

S_WH_PROBE = eyebrow_h2("03 · Other deliverables · MS6 · M54", "White-hat attack: result of the interim probe") + """
    <div class="body">
      <div class="ask">
        <div class="ask-h">Question tested at TUD</div>
        <div class="ask-b">From a case's three output probabilities alone, can an observer tell which hospital it came from?
        This is exactly the data that per-case return moves between sites.</div>
      </div>
      <div class="scroll">
        <table>
          <thead><tr><th>Cases tested</th><th>n</th><th>Attacker accuracy</th><th>Chance</th><th>Verdict</th></tr></thead>
          <tbody>
            <tr><td class="id">No-lesion cases only</td><td class="n">111</td><td class="n">0.185</td><td class="n">0.192</td><td><span class="chip good">No leak</span></td></tr>
            <tr style="background:var(--warn-bg)"><td class="id"><strong>Malignant cases only</strong></td><td class="n">33</td><td class="n"><strong>0.604</strong></td><td class="n">0.317</td><td><span class="chip bad">Leaks</span></td></tr>
          </tbody>
        </table>
      </div>
      <p class="cap">Balanced accuracy against a permutation null, p = 0.007. Restricted to hospitals with at least 5 malignant cases.</p>
      <div class="callout">
        <strong>On malignant cases the model's outputs identify the source hospital 60&#8239;% of the time against 33&#8239;% chance.</strong>
        On no-lesion cases they do not. The model behaves differently per site on the clinically relevant cases.
      </div>
      <div class="callout plain">
        This is a lower bound. The MS6 attacker has the model weights, not only the outputs. The result tells CAM and RUMC where to look first.
      </div>
    </div>"""

S_GANTT = eyebrow_h2("04 · Milestones and timeline", "Grant tasks against the calendar") + """
    <div class="body">
      <div class="viz-root chartwrap">
        <svg viewBox="0 0 720 340" role="img" aria-label="Timeline from April 2025 to December 2027 with one bar per deliverable, filled to its completion share, and markers at the due dates M48, M54 and M60; a vertical line marks today, September 2026"><g id="gantt"></g></svg>
      </div>
      <div class="legend">
        <span><i class="sw" style="background:var(--magenta)"></i> completed share of the deliverable's checklist</span>
        <span><i class="sw" style="background:#F2D2DD"></i> remaining</span>
        <span><i class="sw" style="background:var(--indigo);border-radius:50%"></i> due date</span>
        <span><i class="sw" style="background:var(--coral);width:3px;border-radius:0"></i> today, M45</span>
      </div>
      <p class="cap">Task windows are the grant's Part A months (M1 = January 2023). Each bar's fill is the share of that deliverable's checklist that is complete; the checklists are on the next slide.</p>
    </div>"""

S_EOY = eyebrow_h2("04 · Milestones and timeline", "What concludes by 31 December 2026") + """
    <div class="body">
      <div class="two">
        <div class="panel">
          <h3>Due M48, on track for this year</h3>
          <p><strong>D2.5 regional fine-tuning</strong> · 3 of 5 steps<br>
          <span class="cap">✓ baselines · ✓ fine-tuning and paired analysis · ✓ per-case return shipped and verified · ☐ October run with per-case return at RUMC, UMCU, MHA · ☐ report</span></p>
          <p style="margin-top:.5rem"><strong>D3.2 active learning</strong> · 2 of 4<br>
          <span class="cap">✓ strategies merged · ✓ acquisition experiment, positive · ☐ reduced retraining curve, 2 arms, 2 budgets, about 45 h · ☐ report</span></p>
          <p style="margin-top:.5rem"><strong>D3.4 adversarial robustness</strong> · 2 of 4<br>
          <span class="cap">✓ attack surface for a swarm · ✓ five aggregation rules measured · ☐ malicious-aggregator case (simulation) · ☐ report</span></p>
        </div>
        <div class="panel alt">
          <h3>Due M54 and M60, continues into 2027</h3>
          <p><strong>T3.3 differential privacy</strong> · 2 of 6 (M54)<br>
          <span class="cap">✓ calibrated-noise filter · ✓ RDP accountant · ☐ per-round ε logging · ☐ budget halt · ☐ accuracy cost on ODELIA · ☐ accuracy cost on STAMP</span></p>
          <p style="margin-top:.5rem"><strong>MS6 white-hat attack</strong> · 1 of 4 (M54)<br>
          <span class="cap">✓ interim probe · ☐ slot with CAM and RUMC · ☐ attack run and findings · ☐ fix and preprint</span></p>
          <p style="margin-top:.5rem"><strong>D3.3 software v2.0, testing node</strong> · 1 of 3 (M54)<br>
          <span class="cap">✓ per-site metrics transport · ☐ testing-node participant · ☐ active learning as a feature</span></p>
          <p style="margin-top:.5rem"><strong>D3.5 operational demonstration</strong> · 2 of 4 (M60)<br>
          <span class="cap">✓ 8-site 20-round run · ✓ difficulties account · ☐ re-run on the released version · ☐ comparison with WP1 and WP2</span></p>
        </div>
      </div>
      <div class="scroll">
        <table>
          <thead><tr><th>When</th><th>What happens</th><th>Serves</th></tr></thead>
          <tbody>
            <tr><td class="id">Sep 2026</td><td>1.8.1 kit at all eight sites (MHA, CAM and USZ done); one-round metrics check across the fleet</td><td class="n">everything below</td></tr>
            <tr><td class="id">Oct 2026</td><td><strong>20-round consortium benchmark on the released version</strong>, per-case return on at RUMC, UMCU, MHA (about two days)</td><td class="n">D2.5 · D3.5</td></tr>
            <tr><td class="id">Oct to Nov</td><td>Reduced retraining curve for active learning; malicious-aggregator simulation; noise-cost run if the fleet has slack</td><td class="n">D3.2 · D3.4 · T3.3</td></tr>
            <tr><td class="id">Nov 2026</td><td>Three report drafts to the consortium for review</td><td class="n">D2.5 · D3.2 · D3.4</td></tr>
            <tr><td class="id">Dec 2026</td><td>Reports submitted by 31 December; MS6 slot with CAM and RUMC fixed for Q1 2027</td><td class="n">M48 cluster · MS6</td></tr>
          </tbody>
        </table>
      </div>
      <div class="callout"><strong>The October benchmark needs all eight sites online for two days.</strong> The site list on slide 26 exists to make that run possible.</div>
    </div>"""

S_STATUS = eyebrow_h2("04 · Milestones and timeline", "Deliverable status, " + STATUS_DATE) + """
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Deliverable</th><th>Due</th><th>Status</th><th>What remains</th></tr></thead>
          <tbody>
            <tr><td class="id">D2.5 · regional fine-tuning</td><td class="n">M48</td><td><span class="chip warn">Needs a run</span></td><td>Per-case return verified on 1.8.1 (14 Sep). RUMC, UMCU and MHA enable it for the October run.</td></tr>
            <tr><td class="id">D3.2 · active learning</td><td class="n">M48</td><td><span class="chip good">Positive result</span></td><td>Reduced retraining curve, then the report.</td></tr>
            <tr><td class="id">D3.4 · adversarial robustness</td><td class="n">M48</td><td><span class="chip good">Evidence in</span></td><td>Malicious-aggregator simulation, then the report.</td></tr>
            <tr><td class="id">D3.3 · software v2.0, testing node</td><td class="n">M54</td><td><span class="chip good">Unblocked</span></td><td>Builds on per-site reporting, now working.</td></tr>
            <tr><td class="id">T3.3 · differential privacy</td><td class="n">M54</td><td><span class="chip good">Measurable</span></td><td>Consortium decision on &sigma;; one run to measure the cost.</td></tr>
            <tr><td class="id">MS6 · white-hat attack</td><td class="n">M54</td><td><span class="chip warn">Needs partners</span></td><td>CAM and RUMC for the exercise itself.</td></tr>
            <tr><td class="id">D3.5 · operational demonstration</td><td class="n">M60</td><td><span class="chip info">Part-evidenced</span></td><td>The October benchmark is the re-run on the released version.</td></tr>
          </tbody>
        </table>
      </div>
      <div class="callout">
        <strong>Software since 13 September.</strong> 1.8.0 and 1.8.1 released; per-site metrics, per-case return and the warm-start guard verified on real kits on 14 September.
        1.8.1 kits distributed to all sites on 14 September; MHA installed the same day, CAM and USZ reported their kits ready on 17 September. CAM's SAM-Med2D model is merged as <code>MST_SAMMed2D</code>.
        Four-client fault-injection tests on 14 to 16 September found three defects: a site whose worker crashed at launch (a duplicate OpenMP runtime in the image, seen once at CAM in April; fixed and merged),
        the tolerant mode not telling surviving sites about a pruned one (fixed; the injection test passed on 16 September with the stopped site as the round's aggregator; in review),
        and a configure quorum that proceeds the instant the minimum is reached (filed). Production benchmarks run strict and are affected only by the first.
      </div>
    </div>"""

S_SITES = eyebrow_h2("05 · Sites", "Site status and to-do list, " + STATUS_DATE) + """
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Site</th><th>Coordinator</th><th>Kit</th><th>Log feed</th><th>To do</th></tr></thead>
          <tbody>
            <tr><td class="id"><strong>CAM_1</strong></td><td><span class="chip good">connected</span></td><td class="n"><span class="chip good">1.8.1</span></td><td class="n">today</td><td><ul class="todo"><li>Done: 1.8.1 kit ready (email of 17 Sep); the log feed already comes from the new kit</li><li>Keep only the mediswarm-vpn tunnel; two OpenVPN processes were seen (F10)</li><li>Agree a slot for the MS6 exercise</li></ul></td></tr>
            <tr><td class="id"><strong>MHA_1</strong></td><td><span class="chip good">connected</span><br><span class="cap">re-registered 14 Sep 20:50 UTC</span></td><td class="n"><span class="chip good">1.8.1</span></td><td class="n">today</td><td><ul class="todo"><li>Done: 1.8.1 installed on 14 Sep, first site; kit confirmed ready by email on 17 Sep</li><li>Turn on per-case return for the October run (D2.5)</li></ul></td></tr>
            <tr><td class="id"><strong>RSH_1</strong></td><td><span class="chip good">connected</span></td><td class="n"><span class="chip warn">3 old kits alive</span></td><td class="n">today</td><td><ul class="todo"><li>Asked for help with the 1.8.1 setup (email of 17 Sep): TUD and RSH do it together in a shared session this week</li><li>Three old kits are still sending logs (one 1.6.0, two 1.5.0); stop the leftover containers and daemons in that session, keep one kit</li><li>Complete the site checklist on the board</li></ul></td></tr>
            <tr><td class="id"><strong>RUMC_1</strong></td><td><span class="chip good">connected</span></td><td class="n">1.6.0</td><td class="n"><span class="chip warn">30 Aug</span></td><td><ul class="todo"><li>Install the 1.8.1 kit</li><li>Log upload silent since 30 Aug: check the upload key</li><li>VPN as a service; a hand-started tunnel is in use (F11); TUD owes a debug session</li><li>Turn on per-case return for the October run (D2.5)</li><li>Agree a slot for the MS6 exercise</li></ul></td></tr>
            <tr><td class="id"><strong>UKA_1</strong></td><td><span class="chip good">connected</span><br><span class="cap">back 14 Sep 07:57 UTC</span></td><td class="n">1.6.0</td><td class="n">4 Sep</td><td><ul class="todo"><li>Install the 1.8.1 kit</li><li>Confirm stability after the 4 Sep kernel crashes; stay online through October</li></ul></td></tr>
            <tr><td class="id"><strong>UMCU_1</strong></td><td><span class="chip warn">two clients</span><br><span class="cap">a second client re-registers every few seconds and is rejected</span></td><td class="n"><span class="chip warn">unconfirmed</span></td><td class="n">today</td><td><ul class="todo"><li>Two clients are running under the same identity; stop the old one and keep the 1.8.1 kit</li><li>Confirm the kit version once a single client is left</li><li>Complete the site checklist on the board</li><li>Turn on per-case return for the October run (D2.5)</li></ul></td></tr>
            <tr><td class="id"><strong>USZ_1</strong></td><td><span class="chip good">connected</span><br><span class="cap">restarted 16 Sep 08:26 UTC</span></td><td class="n"><span class="chip good">1.8.1</span><br><span class="cap">by email, 17 Sep</span></td><td class="n"><span class="chip warn">22 Jul</span></td><td><ul class="todo"><li>Done: 1.8.1 kit prepared (email of 17 Sep)</li><li>Log upload silent since 22 Jul: set up the upload key so the kit shows up at the coordinator</li><li>VPN as a service instead of the hand-started tunnel (F11)</li></ul></td></tr>
            <tr><td class="id"><strong>VHIO_1</strong></td><td><span class="chip good">connected</span></td><td class="n">1.6.0</td><td class="n">today</td><td><ul class="todo"><li>Install the 1.8.1 kit</li></ul></td></tr>
          </tbody>
        </table>
      </div>
      <p class="cap">Coordinator: the coordination server's client registry, evening of 16 September. Kit: the version each site's node reports about itself, or the site's email of 17 September where marked. Log feed: the last upload received from the hospital's own machine. F10 and F11 are entries in the board's Known issues tab.</p>
    </div>"""

S_NEEDS = eyebrow_h2("05 · Sites", "Three requests to the consortium") + """
    <div class="body">
      <div class="ask">
        <div class="ask-h">1 · Per-case return at RUMC, UMCU and MHA before the October run</div>
        <div class="ask-b">
          A row number, the label and three probabilities per validation case. No case or patient identifier, no image data.
          It closes D2.5 without moving a model out of a hospital and is what the M54 testing node needs. Off by default; each site decides.
          The interim probe (slide 22) shows these rows are not fully free of site information; that should be known before agreeing.
        </div>
      </div>
      <div class="ask">
        <div class="ask-h">2 · A slot with CAM and RUMC for the MS6 exercise</div>
        <div class="ask-b">Q1 2027, agreed this autumn. TUD provides the weights and the attack protocol.</div>
      </div>
      <div class="ask">
        <div class="ask-h">3 · A clinical view on the operating point</div>
        <div class="ask-b">The model misses about half the cancers at a false-alarm rate near 1&#8239;%. Moving the threshold is free. Where it should sit is a clinical judgement.</div>
      </div>
    </div>"""


# --------------------------------------------------------------------------------------
# speaker notes, one per slide, spoken language
# --------------------------------------------------------------------------------------
NOTES = [
    # 1 title
    "Hi everyone. I'm Jeff, I run the swarm platform for the consortium at TUD, and this is where we stand at month 45. "
    "Four numbers to start: 34 thousand training volumes across eight hospitals, none of which ever left its site. "
    "A shared model at 0.887 malignant AUROC on the external challenge set, and the same number on an American cohort it has never seen. "
    "And software version 1.8.1, which went out to all of you on Monday.",
    # 2 agenda
    "Five parts. First the data, what each of you brought. Then what the model does with it, including the external validation. "
    "Then the other deliverables, and I'll explain each one before I show numbers, because some of them are new to most of you. "
    "Then the milestone timeline and what we can close this year. And at the end, where each site stands and three things I need from you.",
    # 3 data bars
    "This is the training data per site, counted by each site's own trainer when it loads the data set, so these are the real numbers, not what was promised. "
    "UKA alone is half of the consortium. The smallest site has 190 volumes. That 94-fold range shapes almost every technical decision I'll show you today.",
    # 4 growth
    "How we got there. Two sites in April, and then two big jumps: UKA and CAM together on the eleventh of June, UMCU in July. "
    "The eighth site came online at the end of July, and since then the data set has been stable.",
    # 5 class counts
    "Same sites, now split by class at true scale. The thing to notice is the light blue: the benign cases. "
    "UKA holds 84 percent of all benign cases in the consortium. VHIO has none. Keep that in mind for the robustness slides later, it comes back.",
    # 6 class shares
    "And the same thing normalised, so you can see each site's own mix. RSH looks very malignant-heavy at 63 percent, but that is 221 volumes. "
    "UKA's 7 percent malignant is 1,251 volumes. So please read this slide together with the previous one, not on its own.",
    # 7 table
    "All the numbers in one place, for reference. The classes add up per site and the sites add up to 34,462. "
    "One correction from an earlier version: four join dates were about six weeks too early, because our own test machines had reported under hospital names. Test sites have their own names now.",
    # 8 model comparison
    "Now the result. The eight-site swarm model on the external challenge set, against the previous six-site swarm and against single-site models. "
    "The swarm beats every single-site model, including UKA's, and UKA trains on half the data. For a typical site, joining is worth about 0.23 AUROC.",
    # 9 CIs
    "Same numbers with their uncertainty. The challenge set has only 37 malignant cases, so an AUROC on it is known to about plus or minus 0.06. "
    "That means 0.887 and 0.903 are within one interval of each other, and I would not argue about the third decimal.",
    # 10 Duke
    "This is the slide I'm most pleased with. Duke is a public American data set, 260 volumes, none of them anywhere near our training. "
    "Same task, malignant against no lesion. We lose 0.016 and the intervals overlap. A model trained across eight European hospitals transfers to a US cohort without a measurable loss.",
    # 11 operating point
    "One caveat before anyone takes the model into a clinic. AUROC measures ranking. The threshold is a separate choice, and ours is conservative: "
    "on the challenge set we catch about half the cancers and almost never raise a false alarm. On Duke the same picture. Moving that threshold costs nothing, "
    "but where it should sit is a clinical question, and I'll come back to it at the end.",
    # 12 accuracy vs AUROC
    "Since 1.8 every site's metrics come back to the coordinator with their class counts, and this is why that matters. "
    "RUMC reports 99 percent accuracy. 888 of its 896 validation cases have no lesion, so a model that always says no lesion scores 99 percent and finds no cancer. The AUROC of 0.42 tells the truth. "
    "And UMCU was once called weak on a macro average where one class had two cases. Drop that class and it moves from 0.60 to 0.72. So: no ranking of sites on averages over tiny classes.",
    # 13 overview
    "Now the other deliverables. This table is the map: what the proposal asks for, when it is due, and which of you I need for it. "
    "Three are due in December, three in June next year, one at the end of the project. I'll take them one by one, and for each I first explain what it is, then show what exists.",
    # 14 D2.5
    "Regional fine-tuning. The proposal wants the consortium model compared with versions fine-tuned to the RUMC and UMCU cohort and to the MHA cohort. "
    "What I need for that is per-case predictions coming back from RUMC, UMCU and MHA: a row number, the label and three probabilities. No identifiers, no images. "
    "That feature is in 1.8, verified on real kits, and it is off by default: each site switches it on. The comparison itself happens in the October run.",
    # 15 AL concept
    "Active learning. The idea: labelling breast MRI is the expensive part, so let the model tell the radiologist which cases are worth labelling next. "
    "The model scores every unlabelled case by how uncertain it is, we take the top k, a radiologist labels them, they go into the training set, retrain, repeat. "
    "The control is the dashed path: pick k cases at random with the same budget. If the model can't beat random, it isn't useful.",
    # 16 AL result
    "And it does beat random. With a budget of 10 cases, entropy selection finds 5 malignant cases where random finds 2 or 3. At 20, 9 against 4 or 5. "
    "So roughly double, where the budget is small, which is exactly where it matters. By 100 cases everyone has taken most of the pool and the methods converge, as they must. "
    "What this doesn't show yet is a model retrained on those cases. That's the reduced retraining run in October and November.",
    # 17 DP concept
    "Differential privacy. What the proposal asks is optional noise in training with a privacy budget. Here is where the noise goes. "
    "A site trains, produces a model update. We clip that update to a fixed norm so no single site can push too hard, then add Gaussian noise, and only then send it. "
    "An accountant adds up the privacy cost, epsilon, over the rounds and stops training when the budget is spent. "
    "The guarantee is at the hospital level: from the shared model you cannot tell whether a given hospital took part in a round.",
    # 18 DP numbers
    "So what does a guarantee cost? This table is for a 20-round run. Epsilon of 10, which is the commonly accepted level, needs noise scale 2.4. Epsilon 3 needs 6.7. Epsilon 1 would need 18 and probably destroys the model. "
    "The honest gap: we have not yet measured what 2.4 costs in accuracy. One 20-round run does that, and the noise level is then a decision for this room, not a default I set.",
    # 19 robustness concept
    "Adversarial robustness. In a swarm every site sends weights, so every site is a potential attacker. Here eight sites send updates, one of them poisoned. "
    "With the aggregation rule we use today, a plain weighted mean, there is no bound: VHIO with 0.6 percent of the data can move the shared model by five and a half times its share. "
    "With a norm-bounded mean, each contribution is capped, every site stays in, and the same attack moves the model by 0.04.",
    # 20 robustness numbers
    "We tried five rules. The textbook robust ones, trimmed mean, coordinate median, Krum, cost a lot even with no attacker at all, 0.47 to 0.64 error. "
    "That's because they throw away contributions and ignore data-set weights, so with UKA holding 84 percent of benign cases they aim at the average site instead of the pooled data. "
    "The norm-bounded mean costs almost nothing and holds the attack. That's the recommendation. It does not limit the largest contributor, and capping any site's share is a governance decision, not mine.",
    # 21 white-hat plan
    "The white-hat attack, milestone 6. In the proposal, CAM and RUMC attack the trained model to expose vulnerabilities, and the deliverable is a preprint. "
    "Top row is that plan: we give you the model weights and a protocol, you try to reconstruct data or infer membership, we write up the findings and the fix together. "
    "What I need from CAM and RUMC is a slot in the first quarter of next year, and I'd like to fix the date this autumn.",
    # 22 white-hat probe
    "In the meantime we ran the part that needs nobody else. The question: from a case's three output probabilities alone, can you tell which hospital it came from? "
    "On no-lesion cases, no. On malignant cases, yes: 60 percent correct against 33 percent chance. So the model behaves differently per site exactly on the clinically relevant cases. "
    "This is a lower bound, the real attacker has the weights. But it tells CAM and RUMC where to look first, and it is relevant for the per-case predictions I'm asking for.",
    # 23 gantt
    "The timeline. One bar per deliverable, from the grant's task window to the due date, filled to how much of its checklist is done. The orange line is today, the dashed one is the end of the year. "
    "The three December items are between 50 and 60 percent. The June items are earlier, which is fine, they have nine months.",
    # 24 EOY
    "The checklists behind those percentages, so you can hold me to them. On the left, the three that must close this year: regional fine-tuning, active learning, robustness. "
    "On the right, what continues into 2027. And the plan: kits this month, the big benchmark run in October, the extra experiments in October and November, drafts in November, submission in December. "
    "The one thing that decides the year is the October run: all eight sites online for two days.",
    # 25 status
    "Status in one table. And a short note on what happened since the software went out: 1.8.1 is verified on real kits, MHA, CAM and USZ have their kits ready, "
    "CAM's SAM-Med2D model is merged into the platform, and fault-injection tests over the last three days found three defects. "
    "One, a worker crash at start-up, had actually hit CAM once in April; that fix is merged and matters for the October run. "
    "The second, the tolerant mode not telling the other sites when one is dropped, is fixed and passed its test yesterday. The third is a small timing race that I've filed.",
    # 26 sites
    "Where each site stands as of this morning. Everyone is connected to the coordinator. MHA, CAM and USZ, thank you, your emails came in this morning: the 1.8.1 kits are ready at all three, and CAM's log feed already comes from the new kit. "
    "RSH asked for a hand with the setup, so we'll do it together in a shared session this week and stop the three old kits at the same time. UKA, RUMC and VHIO still need to install the kit; it's ten minutes and the certificates don't change. "
    "One thing I saw on Tuesday evening: UMCU has two clients running under the same identity, so the second one keeps re-registering every few seconds and gets rejected; stop the old one and keep the 1.8.1 kit. "
    "RUMC and USZ, your log feed has been silent for a while, please check the upload key. USZ, once the feed is back I can confirm the kit from here.",
    # 27 requests
    "Three requests. One: RUMC, UMCU and MHA, switch on per-case return before the October run. You saw on slide 22 that those rows carry some site information, so decide with that in mind. "
    "Two: CAM and RUMC, a slot for the white-hat exercise in the first quarter. "
    "Three: a clinical view on the operating point. The model misses half the cancers at a 1 percent false-alarm rate. Moving the threshold is free; deciding where it goes is yours. Thank you.",
]

NOTES_MD = ROOT / "docs" / "presentation_consortium_45min_notes.md"

# Present mode. The Present button shows one slide at a time in a 1280 x 720 box scaled to the
# window; a slide taller than the box is scaled down uniformly (same rule as render_slides.py).
# Keys: arrows, space, PageUp/PageDown move; Home/End; N toggles the speaker notes; F full
# screen; Esc leaves. The URL hash (#s8) remembers the slide, so a reload resumes there.
PRESENT_JS = r"""<script>
(function(){
  const slides = Array.from(document.querySelectorAll('.slide'));
  const bar = document.querySelector('.notes-bar');
  if (!slides.length || !bar) return;
  const W = 1280, H = 720;
  const hint = document.createElement('span'); hint.className = 'present-hint';
  hint.textContent = 'Present shows one slide at a time: arrow keys move, N shows the speaker notes, F is full screen, Esc leaves.';
  const btn = document.createElement('button'); btn.type = 'button'; btn.id = 'present-btn'; btn.textContent = 'Present';
  bar.insertBefore(hint, bar.firstChild); bar.appendChild(btn);
  const hud = document.createElement('div'); hud.className = 'pres-hud'; document.body.appendChild(hud);
  const notes = document.createElement('div'); notes.className = 'pres-notes'; document.body.appendChild(notes);
  let cur = 0, on = false;
  function prep(s){
    if (s.querySelector(':scope > .pres-inner')) return;
    const inner = document.createElement('div'); inner.className = 'pres-inner';
    Array.from(s.children).forEach(c => { if (!c.classList.contains('foot') && !c.classList.contains('notes')) inner.appendChild(c); });
    s.insertBefore(inner, s.firstChild);
  }
  function fit(){
    const s = slides[cur]; prep(s);
    const nh = document.body.classList.contains('with-notes') ? notes.offsetHeight : 0;
    const avH = innerHeight - nh;
    const k = Math.min(innerWidth / W, avH / H);
    s.style.setProperty('--pk', k); s.style.top = (avH / 2) + 'px';
    const inner = s.querySelector(':scope > .pres-inner'); inner.style.transform = ''; inner.style.minHeight = '';
    s.classList.add('measuring'); const need = inner.offsetHeight; s.classList.remove('measuring');
    const foot = s.querySelector('.foot');
    const avail = H - parseFloat(getComputedStyle(s).paddingTop) - (foot ? foot.offsetHeight : 0) - 8;
    if (need > avail) { inner.style.minHeight = need + 'px'; inner.style.transform = 'scale(' + (avail / need) + ')'; }
  }
  function show(i){
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, j) => s.classList.toggle('current', j === cur));
    const p = slides[cur].querySelector('.notes p');
    notes.innerHTML = '<b>Slide ' + (cur + 1) + '</b> ' + (p ? p.innerHTML : '');
    hud.textContent = (cur + 1) + ' / ' + slides.length + '   arrows: next and back   N: notes   F: full screen   Esc: leave';
    history.replaceState(null, '', '#s' + (cur + 1));
    fit();
  }
  function enter(){
    on = true; document.body.classList.add('presenting');
    let start = 0; const m = /^#s(\d+)$/.exec(location.hash);
    if (m) start = parseInt(m[1], 10) - 1;
    else { const y = scrollY + 40; start = slides.findIndex(s => s.offsetTop + s.offsetHeight > y); if (start < 0) start = 0; }
    show(start);
  }
  function leave(){
    on = false; document.body.classList.remove('presenting', 'with-notes');
    slides.forEach(s => { s.classList.remove('current'); s.style.top = ''; const i = s.querySelector(':scope > .pres-inner'); if (i) { i.style.transform = ''; i.style.minHeight = ''; } });
    if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {});
    history.replaceState(null, '', location.pathname + location.search);
    slides[cur].scrollIntoView({ block: 'start' });
  }
  btn.addEventListener('click', enter);
  addEventListener('resize', () => { if (on) fit(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { if (on) fit(); });   /* re-measure once the web font is in */
  addEventListener('keydown', e => {
    if (!on) return;
    const k = e.key;
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'Enter'].includes(k)) { e.preventDefault(); show(cur + 1); }
    else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'].includes(k)) { e.preventDefault(); show(cur - 1); }
    else if (k === 'Home') { e.preventDefault(); show(0); }
    else if (k === 'End') { e.preventDefault(); show(slides.length - 1); }
    else if (k === 'n' || k === 'N') { document.body.classList.toggle('with-notes'); fit(); }
    else if (k === 'f' || k === 'F') {
      if (document.fullscreenElement) { document.exitFullscreen().catch(() => {}); }
      else if (document.documentElement.requestFullscreen) { document.documentElement.requestFullscreen().catch(() => {}); }
    }
    else if (k === 'Escape') { e.preventDefault(); leave(); }
  });
  document.addEventListener('click', e => {
    if (on && e.target.closest('.slide.current') && !e.target.closest('a, button, summary, details')) show(cur + 1);
  });
  if (/^#s\d+$/.test(location.hash)) enter();
})();
</script>"""

# --------------------------------------------------------------------------------------
# milestone timeline (Gantt) JS
# --------------------------------------------------------------------------------------
GANTT_JS = r"""
  /* ---------- 5. milestone timeline ---------- */
  (function () {
    const g = document.getElementById("gantt");
    if (!g) return;
    const md = m => new Date(Date.UTC(2023 + Math.floor((m - 1) / 12), (m - 1) % 12, 1));
    const ROWS = [
      { id:"D2.5", name:"Regional fine-tuning · T2.4", m0:28, m1:48, done:3, of:5 },
      { id:"D3.2", name:"Active learning report · T3.2", m0:31, m1:48, done:2, of:4 },
      { id:"D3.4", name:"Adversarial robustness · WP3", m0:40, m1:48, done:2, of:4, est:true },
      { id:"T3.3", name:"Differential privacy", m0:37, m1:54, done:2, of:6 },
      { id:"MS6",  name:"White-hat attack · CAM + RUMC", m0:45, m1:54, done:1, of:4, est:true },
      { id:"D3.3", name:"Software v2.0, testing node · T3.2", m0:31, m1:54, done:1, of:3 },
      { id:"D3.5", name:"Operational demonstration · T3.4", m0:40, m1:60, done:2, of:4 }
    ];
    const L = 232, R = 20, T = 30, rowH = 36, barH = 16;
    const W = 720 - L - R;
    const t0 = md(28), t1 = md(61);
    const x = d => L + (d - t0) / (t1 - t0) * W;
    for (let m = 28; m <= 60; m += 3) {
      const px = x(md(m));
      g.appendChild(el("line", { x1: px, y1: T - 8, x2: px, y2: T + ROWS.length * rowH - 10, class: (m - 1) % 12 === 0 ? "axis" : "grid" }));
      const q = Math.floor(((m - 1) % 12) / 3) + 1;
      const t = el("text", { x: px + 2, y: T - 12, class: "val-sm" });
      t.textContent = "Q" + q; g.appendChild(t);
      if ((m - 1) % 12 === 0) {
        const y = el("text", { x: px + 2, y: T - 24, class: "val" });
        y.textContent = String(2023 + Math.floor((m - 1) / 12)); g.appendChild(y);
      }
    }
    ROWS.forEach((r, i) => {
      const y = T + i * rowH;
      const x0 = x(md(r.m0)), x1 = x(md(r.m1));
      const grp = el("g", { class: "bar" });
      grp.appendChild(el("rect", { x: x0, y, width: x1 - x0, height: barH, rx: 3, fill: "#F2D2DD",
                                   "stroke-dasharray": r.est ? "3 3" : "", stroke: r.est ? "var(--series-3)" : "none", "stroke-width": r.est ? 1 : 0 }));
      const frac = r.done / r.of;
      grp.appendChild(el("rect", { x: x0, y, width: (x1 - x0) * frac, height: barH, rx: 3, fill: "var(--series-3)" }));
      const pct = el("text", { x: x0 + 6, y: y + barH - 4, class: "val-sm" });
      pct.setAttribute("style", "fill:" + (frac > .2 ? "#fff" : "#1A1A1A"));
      pct.textContent = Math.round(frac * 100) + " %"; grp.appendChild(pct);
      grp.appendChild(el("circle", { cx: x1, cy: y + barH / 2, r: 6, fill: "var(--series-1)", stroke: "#fff", "stroke-width": 2 }));
      const due = el("text", { x: x1 + 10, y: y + barH - 3, class: "val-sm" });
      due.textContent = "M" + r.m1; grp.appendChild(due);
      const ti = el("title");
      ti.textContent = `${r.id}, ${r.name}: window M${r.m0} to M${r.m1}, ${r.done} of ${r.of} checklist items done`;
      grp.appendChild(ti);
      g.appendChild(grp);
      const idt = el("text", { x: L - 12, y: y + barH - 3, "text-anchor": "end", class: "lbl-site" });
      idt.textContent = r.id + "  " + r.name; g.appendChild(idt);
    });
    const now = x(new Date(Date.UTC(2026, 8, 15)));
    g.appendChild(el("line", { x1: now, y1: T - 8, x2: now, y2: T + ROWS.length * rowH - 10, stroke: "#E64F39", "stroke-width": 2 }));
    const nt = el("text", { x: now - 5, y: T + ROWS.length * rowH + 2, "text-anchor": "end", class: "val", style: "fill:#E64F39" });
    nt.textContent = "today · M45"; g.appendChild(nt);
    const eoy = x(md(49));
    g.appendChild(el("line", { x1: eoy, y1: T - 8, x2: eoy, y2: T + ROWS.length * rowH - 10, stroke: "var(--series-1)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
    const et = el("text", { x: eoy + 4, y: T + ROWS.length * rowH + 2, class: "val-sm", style: "fill:#4B56D2" });
    et.textContent = "31 Dec 2026 · M48"; g.appendChild(et);
    const note = el("text", { x: L, y: T + ROWS.length * rowH + 22, class: "val-sm" });
    note.textContent = "Dashed outline: no task window in Part A for this item; drawn from the start of the work to its due date."; g.appendChild(note);
  })();
"""


def build() -> str:
    logos = {k: data_uri(v) for k, v in {
        "odelia": "odelia.png", "tud": "tud.png", "ekfz": "ekfz.png", "kather": "katherlab.png", "ukd": "ukd.jpg"}.items()}
    nct_file = LOGOS / "nct.png"
    if nct_file.is_file():
        try:
            from PIL import Image  # type: ignore
            w, h = Image.open(nct_file).size
            ratio = f"{w}/{h}"
        except Exception:
            ratio = "3/1"
        logos["nct"] = data_uri("nct.png")
        nct_mark = NCT_IMG
    else:
        logos["nct"] = ""
        ratio = "3/1"
        nct_mark = NCT_TEXT
    style = STYLE.replace("{nct_ratio}", ratio)
    for k, v in logos.items():
        style = style.replace("{" + k + "}", v)

    header = HEADER_LOGOS.format(nct=nct_mark)
    slides = [
        S_TITLE.format(header=header, venue=VENUE, date=DATE),
        S_AGENDA,
        S_DATA_BARS, S_DATA_CUM, S_DATA_ABS, S_DATA_SHARE, S_DATA_TABLE,
        S_RES_MODEL, S_RES_CI, S_RES_DUKE, S_RES_THRESH, S_RES_SITES,
        S_DELIV_OVERVIEW, S_D25,
        S_AL_CONCEPT, S_AL_RESULT,
        S_DP_CONCEPT, S_DP_RESULT,
        S_ROB_CONCEPT, S_ROB_RESULT,
        S_WH_PLAN, S_WH_PROBE,
        S_GANTT, S_EOY, S_STATUS, S_SITES, S_NEEDS,
    ]
    assert len(NOTES) == len(slides), f"{len(NOTES)} notes for {len(slides)} slides"
    out = [style, '<div class="deck">',
           '<div class="notes-bar"><button type="button" onclick="document.querySelectorAll(\'details.notes\').forEach(d=>d.open=true)">Show all speaker notes</button>'
           '<button type="button" onclick="document.querySelectorAll(\'details.notes\').forEach(d=>d.open=false)">Hide notes</button></div>']
    import re as _re
    titles = []
    for n, (body, note) in enumerate(zip(slides, NOTES), 1):
        m = _re.search(r"<h[12]>(.*?)</h[12]>", body)
        titles.append(_re.sub(r"<[^>]+>", "", m.group(1)) if m else f"Slide {n}")
        notes_html = f'    <details class="notes"><summary>Speaker notes</summary><p>{note}</p></details>'
        out.append(slide(body + "\n" + notes_html + "\n" + FOOTER.format(n=n, nct=nct_mark)))
    md = ["# Speaker notes: ODELIA consortium briefing, " + VENUE + ", " + DATE, "",
          "Spoken, informal. About one to two minutes per slide, 27 slides in 45 minutes.", ""]
    for n, (t, note) in enumerate(zip(titles, NOTES), 1):
        md += [f"## {n}. {t}", "", note, ""]
    NOTES_MD.write_text("\n".join(md))
    out.append(f'  <footer>ODELIA WP2 / WP3 · {VENUE}, {DATE} · figures regenerated from stored per-case predictions, the run-history dataset, the coordinator log and CI</footer>\n</div>')
    charts = CHARTS_JS.read_text().rstrip()
    assert charts.endswith("})();"), "unexpected chart script tail"
    out.append("<script>\n" + charts[:-len("})();")] + GANTT_JS + "})();\n</script>")
    out.append(PRESENT_JS)
    html = "\n".join(out)
    # House style: no dashes in visible prose (the owner's rule). Checked on the slide
    # bodies and the notes, which is all the visible text; styles and scripts never enter.
    visible = " ".join(_re.sub(r"<[^>]*>", " ", body) for body in slides) + " " + " ".join(NOTES)
    bad = [m.group(0) for m in _re.finditer(r".{0,30}[\u2014\u2013].{0,30}", visible)]
    if bad:
        raise SystemExit("dash in visible text:\n  " + "\n  ".join(bad[:10]))
    return html


if __name__ == "__main__":
    html = build()
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html)} bytes)")
