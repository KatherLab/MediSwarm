#!/usr/bin/env python3
"""Build docs/presentation_consortium_45min.html — the consortium briefing in the ODELIA
deck style (white slides, magenta titles, Open Sans, gradient title slide, logo strip).

Usage:  python3 scripts/presentation/build_consortium_45min.py

The visual identity follows the ODELIA EAB deck (odelia_EAB_meeting_opensource_jeff.pptx):
magenta E5316B titles, coral E64F39 gradient, light-blue 82C3EC / indigo 4B56D2 accents,
888888 muted text, F2F2F2 panels, Open Sans throughout. Logos live in docs/assets/logos and
are inlined as data URIs so the page is self-contained.

Content slides (data, results, deliverables) are the same text and charts as the 10 Sep
version; the 14 Sep build adds the agenda, the milestone timeline, the end-of-2026 plan and
the per-site to-do list, and updates the status slide for 1.8.1.
"""
from __future__ import annotations

import base64
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "presentation_consortium_45min.html"
LOGOS = ROOT / "docs" / "assets" / "logos"
SLIDES_SRC = pathlib.Path(__file__).with_name("consortium_45min_slides.html")


def data_uri(name: str) -> str:
    p = LOGOS / name
    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


STYLE = """
<title>ODELIA Consortium Briefing</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    /* ODELIA deck identity (EAB deck, April 2024) */
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
  .num { display:none; }
  .eyebrow { font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:600; }
  h1 { font-size:clamp(1.8rem,4.2vw,2.7rem); line-height:1.12; margin:.4rem 0 0; font-weight:700; letter-spacing:-.01em; text-wrap:balance; color:#fff; }
  h2 { font-size:clamp(1.25rem,2.5vw,1.7rem); line-height:1.2; margin:.3rem 0 0; font-weight:700; color:var(--magenta); text-wrap:balance; }
  .sub { color:var(--ink-soft); font-size:clamp(.92rem,1.5vw,1.02rem); margin-top:.6rem; max-width:46rem; text-wrap:pretty; }
  .body { margin-top:1rem; display:flex; flex-direction:column; gap:.8rem; flex:1; min-height:0; }
  p { margin:0; }
  strong { font-weight:700; }
  .cap { font-size:.78rem; color:var(--muted); }
  code { font-family:var(--mono); font-size:.9em; }
  /* logo strip + footer, as on the source deck */
  .logos { display:flex; align-items:center; gap:1.4rem; flex-wrap:wrap; }
  .lg { display:inline-block; height:26px; background:center/contain no-repeat; }
  .lg-odelia { aspect-ratio:613/119; height:24px; background-image:url("{odelia}"); }
  .lg-tud    { aspect-ratio:269/78;  background-image:url("{tud}"); }
  .lg-ekfz   { aspect-ratio:213/77;  background-image:url("{ekfz}"); }
  .lg-kather { aspect-ratio:372/123; background-image:url("{kather}"); }
  .lg-ukd    { aspect-ratio:337/156; background-image:url("{ukd}"); }
  .foot { position:absolute; left:0; right:0; bottom:0; padding:.55rem clamp(1.2rem,3vw,2.2rem);
          display:flex; align-items:center; justify-content:space-between; gap:1rem; border-top:1px solid var(--rule-soft); }
  .foot .lg { height:18px; } .foot .lg-odelia { height:17px; }
  .foot .pg { font-size:.66rem; color:var(--muted); letter-spacing:.06em; }
  /* title slide */
  .title { background:linear-gradient(90deg,var(--magenta) 0%,var(--coral) 100%); color:#fff; border-radius:4px;
           padding:clamp(1.4rem,3.4vw,2.4rem); margin-top:1rem; flex:1; display:flex; flex-direction:column; justify-content:center; gap:.6rem; }
  .title .who { font-size:.92rem; font-weight:600; }
  .title .who span { font-weight:400; opacity:.9; }
  .title .wp { font-size:clamp(1.1rem,2.2vw,1.4rem); font-weight:700; letter-spacing:.02em; margin-top:1rem; }
  .title .date { font-size:.9rem; opacity:.95; margin-top:1rem; }
  .title .stats { margin-top:1.2rem; }
  .title .stats > div { background:rgba(255,255,255,.14); }
  .title .stats dt { color:rgba(255,255,255,.85); } .title .stats dd { color:#fff; } .title .stats .cap { color:rgba(255,255,255,.85); }
  .title .stats { background:transparent; border:none; gap:.5rem; }
  /* agenda blocks */
  .agenda { display:grid; grid-template-columns:repeat(5,1fr); gap:.6rem; margin-top:1.2rem; }
  @media (max-width:820px){ .agenda{ grid-template-columns:repeat(2,1fr);} }
  .agenda > div { background:var(--magenta); color:#fff; padding:1.1rem .9rem 1.2rem; min-height:11rem; display:flex; flex-direction:column; }
  .agenda > div:nth-child(2n) { background:var(--coral); }
  .agenda .n { font-size:2.3rem; font-weight:400; line-height:1; }
  .agenda .t { margin-top:auto; font-size:.8rem; line-height:1.3; font-weight:600; }
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
  @media (max-width:820px){ .two{ grid-template-columns:1fr; } }
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
  ul.todo { margin:0; padding-left:1.1rem; font-size:.84rem; }
  ul.todo li { margin:.12rem 0; }
  ul.todo li.done { color:var(--muted); text-decoration:line-through; }
  footer { text-align:center; color:var(--muted); font-size:.78rem; padding-top:1rem; }
  @media (prefers-reduced-motion: reduce){ *{ animation:none!important; transition:none!important; } }
  @media print {
    body{background:#fff;} .deck{gap:0;padding:0;max-width:none;}
    .slide{page-break-after:always;border:none;border-radius:0;min-height:100vh;}
  }
</style>
<style>
  /* chart marks — light theme only, on purpose (the deck is white like the source deck) */
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

HEADER_LOGOS = """<div class="logos"><span class="lg lg-tud" role="img" aria-label="TU Dresden"></span><span class="lg lg-ukd" role="img" aria-label="Universitätsklinikum Carl Gustav Carus Dresden"></span><span class="lg lg-ekfz" role="img" aria-label="EKFZ for Digital Health"></span><span class="lg lg-kather" role="img" aria-label="Kather Lab"></span><span class="lg lg-odelia" role="img" aria-label="ODELIA"></span></div>"""

FOOTER = """<div class="foot"><div class="logos"><span class="lg lg-odelia" role="img" aria-label="ODELIA"></span></div><div class="pg">Slide {n}</div><div class="logos"><span class="lg lg-ukd"></span><span class="lg lg-ekfz"></span><span class="lg lg-tud"></span></div></div>"""

TITLE_SLIDE = """
  <section class="slide">
    {header}
    <div class="title">
      <div class="who">JieFu Zhu, M.Sc. | Prof. Dr. Jakob Nikolas Kather <span>· Else Kröner Fresenius Center for Digital Health, TU Dresden</span></div>
      <div class="wp">Work Package 2 / 3</div>
      <h1>Where the swarm stands at month 45</h1>
      <p class="sub" style="color:#fff;opacity:.95">What the consortium's data has produced, how it holds up outside Europe, where the
        remaining deliverables stand against the grant timeline, and what each site still has to do.</p>
      <div class="stats">
        <div><dt>Training volumes</dt><dd>34,462<span class="cap">8 hospitals, 6 countries, none moved</span></dd></div>
        <div><dt>Shared model</dt><dd>0.887<span class="cap">malignant AUROC, external challenge set</span></dd></div>
        <div><dt>Independent US cohort</dt><dd>0.887<span class="cap">Duke, 260 volumes, never seen in training</span></dd></div>
        <div><dt>Software</dt><dd>1.8.1<span class="cap">released 13 Sep 2026, kits at every site</span></dd></div>
      </div>
      <div class="date">ODELIA · 14.09.2026</div>
    </div>
  </section>
"""

AGENDA_SLIDE = """
  <section class="slide">
    <h2>Agenda</h2>
    <div class="agenda">
      <div><span class="n">01</span><span class="t">The data<br>what eight hospitals bring</span></div>
      <div><span class="n">02</span><span class="t">The result<br>and its external validation</span></div>
      <div><span class="n">03</span><span class="t">Deliverables<br>D3.2 · T3.3 · D3.4 · MS6</span></div>
      <div><span class="n">04</span><span class="t">Milestones and timeline<br>what concludes in 2026</span></div>
      <div><span class="n">05</span><span class="t">Sites<br>to-do list and what we need</span></div>
    </div>
    <p class="cap" style="margin-top:1rem">Grant month M45 = September 2026. M48 = December 2026, M54 = June 2027, M60 = December 2027.</p>
  </section>
"""

MILESTONE_SLIDE = """
  <section class="slide">
    <div class="eyebrow">04 · Milestones and timeline</div>
    <h2>Grant tasks against the calendar: where each one stands today</h2>
    <div class="body">
      <div class="viz-root chartwrap">
        <svg viewBox="0 0 720 340" role="img" aria-label="Timeline from April 2025 to December 2027 with one bar per deliverable, filled to its completion share, and diamonds at the due dates M48, M54 and M60; a vertical line marks today, September 2026"><g id="gantt"></g></svg>
      </div>
      <div class="legend">
        <span><i class="sw" style="background:var(--magenta)"></i> done share of the deliverable's checklist</span>
        <span><i class="sw" style="background:#F2D2DD"></i> remaining</span>
        <span><i class="sw" style="background:var(--indigo);border-radius:50%"></i> due date</span>
        <span><i class="sw" style="background:var(--coral);width:3px;border-radius:0"></i> today, M45</span>
      </div>
      <p class="cap">Task windows are the grant's Part A months (M1 = January 2023). Each bar's fill is the share of that deliverable's "done when" checklist that is complete — the checklists are on the next slide, so every percentage can be checked line by line.</p>
    </div>
  </section>
"""

EOY_SLIDE = """
  <section class="slide">
    <div class="eyebrow">04 · Milestones and timeline</div>
    <h2>What concludes by 31 December 2026, and what does not</h2>
    <div class="body">
      <div class="two">
        <div class="panel">
          <h3>Due M48 — on track to conclude this year</h3>
          <p><strong>D2.5 regional fine-tuning</strong> · 3 of 5 steps done<br>
          <span class="cap">✓ baseline measured · ✓ fine-tuning + paired analysis · ✓ per-case return shipped (1.8.0/1.8.1) · ☐ one consortium run with per-case return at Nijmegen, Utrecht and Athens · ☐ report</span></p>
          <p style="margin-top:.5rem"><strong>D3.2 active learning</strong> · 2 of 4<br>
          <span class="cap">✓ strategies merged · ✓ acquisition experiment, positive (corrected 10 Sep) · ☐ reduced retraining curve, 2 arms × 2 budgets ≈ 45 h of consortium GPU · ☐ report</span></p>
          <p style="margin-top:.5rem"><strong>D3.4 adversarial robustness</strong> · 2 of 4<br>
          <span class="cap">✓ attack surface for a swarm · ✓ five aggregation rules measured · ☐ malicious-aggregator case (simulation, no sites needed) · ☐ report</span></p>
        </div>
        <div class="panel alt">
          <h3>Due M54 / M60 — continues into 2027</h3>
          <p><strong>T3.3 differential privacy</strong> · 2 of 6 (M54)<br>
          <span class="cap">✓ calibrated-noise filter · ✓ RDP accountant · ☐ per-round ε logging · ☐ budget-exhausted halt · ☐ utility cost on ODELIA · ☐ utility cost on STAMP</span></p>
          <p style="margin-top:.5rem"><strong>MS6 white-hat attack</strong> · 1 of 4 (M54)<br>
          <span class="cap">✓ in-house site-inference probe · ☐ slot agreed with Cambridge and Nijmegen · ☐ attack run and findings · ☐ fix + preprint</span></p>
          <p style="margin-top:.5rem"><strong>D3.3 software v2.0 + testing node</strong> · 1 of 3 (M54)<br>
          <span class="cap">✓ per-site metrics transport · ☐ testing-node participant · ☐ active learning as a supported feature</span></p>
          <p style="margin-top:.5rem"><strong>D3.5 operational demonstration</strong> · 2 of 4 (M60)<br>
          <span class="cap">✓ 8-site 20-round run · ✓ difficulties account · ☐ re-run on the released version · ☐ comparison with WP1/WP2</span></p>
        </div>
      </div>
      <div class="scroll">
        <table>
          <thead><tr><th>When</th><th>What happens</th><th>Serves</th></tr></thead>
          <tbody>
            <tr><td class="id">Sep 2026</td><td>1.8.1 kit at all eight sites; Athens reconnects; one-round metrics check across the fleet</td><td class="n">everything below</td></tr>
            <tr><td class="id">Oct 2026</td><td><strong>20-round consortium benchmark on the released version</strong>, per-case return on at Nijmegen, Utrecht, Athens (about two days, plus retries)</td><td class="n">D2.5 · D3.5</td></tr>
            <tr><td class="id">Oct – Nov</td><td>Reduced retraining curve for active learning; malicious-aggregator measurement in simulation; DP utility-cost run if the fleet has slack</td><td class="n">D3.2 · D3.4 · T3.3</td></tr>
            <tr><td class="id">Nov 2026</td><td>Three report drafts to the consortium for review</td><td class="n">D2.5 · D3.2 · D3.4</td></tr>
            <tr><td class="id">Dec 2026</td><td>Reports submitted by 31 December; MS6 slot with Cambridge and Nijmegen fixed for Q1 2027</td><td class="n">M48 cluster · MS6</td></tr>
          </tbody>
        </table>
      </div>
      <div class="callout"><strong>The one dependency that decides the year:</strong> the October benchmark needs all eight sites online for two days. Every to-do on the next slide exists to make that run possible.</div>
    </div>
  </section>
"""

SITES_SLIDE = """
  <section class="slide">
    <div class="eyebrow">05 · Sites</div>
    <h2>To-do list by site, as of 14 September 2026</h2>
    <div class="body">
      <div class="scroll">
        <table>
          <thead><tr><th>Site</th><th>Coordinator</th><th>Kit</th><th>Log feed</th><th>To do</th></tr></thead>
          <tbody>
            <tr><td class="id"><strong>Aachen</strong> UKA_1</td><td><span class="chip good">connected</span><br><span class="cap">back today 07:57 UTC after the 7 Sep drop</span></td><td class="n">1.6.0</td><td class="n">4 Sep</td><td><ul class="todo"><li>Install the 1.8.1 kit</li><li>Confirm the node is stable after the 4 Sep kernel crashes (two unclean reboots mid-run); keep it online through October</li></ul></td></tr>
            <tr><td class="id"><strong>Athens</strong> MHA_1</td><td><span class="chip bad">not connected</span><br><span class="cap">left 02:21 UTC today; machine still uploading logs</span></td><td class="n">1.6.0</td><td class="n">today</td><td><ul class="todo"><li>Restart the client / check the VPN service (F6)</li><li>Install the 1.8.1 kit</li><li>Turn on per-case return for the October run (Greek cohort, D2.5)</li></ul></td></tr>
            <tr><td class="id"><strong>Barcelona</strong> VHIO_1</td><td><span class="chip good">connected</span></td><td class="n">1.6.0</td><td class="n">today</td><td><ul class="todo"><li>Install the 1.8.1 kit — nothing else</li></ul></td></tr>
            <tr><td class="id"><strong>Cambridge</strong> CAM_1</td><td><span class="chip good">connected</span></td><td class="n">1.6.0</td><td class="n">today</td><td><ul class="todo"><li>Install the 1.8.1 kit</li><li>Keep only the mediswarm-vpn tunnel — two OpenVPN processes were seen (F10)</li><li>Agree a slot for the MS6 white-hat exercise (attacking party)</li></ul></td></tr>
            <tr><td class="id"><strong>Guildford</strong> RSH_1</td><td><span class="chip good">connected</span></td><td class="n"><span class="chip warn">1.5.0</span></td><td class="n">today</td><td><ul class="todo"><li>Install the 1.8.1 kit — the 1.5.0 kit is pinned to an old image and cannot follow releases</li><li>Complete the site checklist on the board (all ten rows still unconfirmed)</li></ul></td></tr>
            <tr><td class="id"><strong>Nijmegen</strong> RUMC_1</td><td><span class="chip good">connected</span></td><td class="n">1.6.0</td><td class="n"><span class="chip warn">30 Aug</span></td><td><ul class="todo"><li>Install the 1.8.1 kit</li><li>Log upload has been silent since 30 Aug — check the upload key</li><li>VPN as a service: the setup scripts fail there and a hand-started tunnel is in use (F11); we owe a debug session with logs</li><li>Turn on per-case return for the October run (Dutch cohort, D2.5)</li><li>Agree a slot for the MS6 white-hat exercise (attacking party)</li></ul></td></tr>
            <tr><td class="id"><strong>Utrecht</strong> UMCU_1</td><td><span class="chip good">connected</span></td><td class="n">1.6.0</td><td class="n">today</td><td><ul class="todo"><li>Install the 1.8.1 kit</li><li>Complete the site checklist on the board (unconfirmed)</li><li>Turn on per-case return for the October run (Dutch cohort, D2.5)</li></ul></td></tr>
            <tr><td class="id"><strong>Zurich</strong> USZ_1</td><td><span class="chip good">connected</span></td><td class="n"><span class="chip warn">1.5.0</span></td><td class="n"><span class="chip warn">22 Jul</span></td><td><ul class="todo"><li>Install the 1.8.1 kit — the 1.5.0 kit is pinned to an old image</li><li>Log upload has been silent since 22 Jul — set up the upload key</li><li>VPN as a service instead of the hand-started USZ.conf tunnel (F11)</li></ul></td></tr>
          </tbody>
        </table>
      </div>
      <p class="cap">"Coordinator" is the coordination server's own client registry after the 1.8.1 restart (13 Sep 23:08 UTC). "Kit" is what each site's node reports about itself; the 1.6.0 kits still run the image pulled before 13 Sep until the site installs 1.8.1. "Log feed" is the last upload received from that hospital's own machine (records from our test machines under hospital names are excluded). Fn refer to the board's Known issues tab.</p>
    </div>
  </section>
"""

# ---- deliverable timeline data (grant months; M1 = 2023-01) -------------------------------
GANTT_JS = r"""
  /* ---------- 5. milestone timeline ---------- */
  (function () {
    const g = document.getElementById("gantt");
    if (!g) return;
    // grant month -> Date (M1 = Jan 2023)
    const md = m => new Date(Date.UTC(2023 + Math.floor((m - 1) / 12), (m - 1) % 12, 1));
    const ROWS = [
      { id:"D2.5", name:"Regional fine-tuning · T2.4", m0:28, m1:48, done:3, of:5 },
      { id:"D3.2", name:"Active learning report · T3.2", m0:31, m1:48, done:2, of:4 },
      { id:"D3.4", name:"Adversarial robustness · WP3", m0:40, m1:48, done:2, of:4, est:true },
      { id:"T3.3", name:"Differential privacy", m0:37, m1:54, done:2, of:6 },
      { id:"MS6",  name:"White-hat attack · CAM + RUMC", m0:45, m1:54, done:1, of:4, est:true },
      { id:"D3.3", name:"SL software v2.0 + testing node · T3.2", m0:31, m1:54, done:1, of:3 },
      { id:"D3.5", name:"Operational demonstration · T3.4", m0:40, m1:60, done:2, of:4 }
    ];
    const L = 232, R = 20, T = 30, rowH = 36, barH = 16;
    const W = 720 - L - R;
    const t0 = md(28), t1 = md(61);
    const x = d => L + (d - t0) / (t1 - t0) * W;
    // quarter grid + year labels
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
    // bars
    ROWS.forEach((r, i) => {
      const y = T + i * rowH;
      const x0 = x(md(r.m0)), x1 = x(md(r.m1));
      const grp = el("g", { class: "bar" });
      grp.appendChild(el("rect", { x: x0, y, width: x1 - x0, height: barH, rx: 3, fill: "#F2D2DD",
                                   "stroke-dasharray": r.est ? "3 3" : "", stroke: r.est ? "var(--series-3)" : "none", "stroke-width": r.est ? 1 : 0 }));
      const frac = r.done / r.of;
      grp.appendChild(el("rect", { x: x0, y, width: (x1 - x0) * frac, height: barH, rx: 3, fill: "var(--series-3)" }));
      const pct = el("text", { x: x0 + 6, y: y + barH - 4, class: "val-sm", fill: frac > .2 ? "#fff" : "var(--text-primary)" });
      pct.setAttribute("style", "fill:" + (frac > .2 ? "#fff" : "#1A1A1A"));
      pct.textContent = Math.round(frac * 100) + " %"; grp.appendChild(pct);
      // due-date marker
      grp.appendChild(el("circle", { cx: x1, cy: y + barH / 2, r: 6, fill: "var(--series-1)", stroke: "#fff", "stroke-width": 2 }));
      const due = el("text", { x: x1 + 10, y: y + barH - 3, class: "val-sm" });
      due.textContent = "M" + r.m1; grp.appendChild(due);
      const ti = el("title");
      ti.textContent = `${r.id} — ${r.name}: window M${r.m0}–M${r.m1}, ${r.done} of ${r.of} checklist items done`;
      grp.appendChild(ti);
      g.appendChild(grp);
      const idt = el("text", { x: L - 12, y: y + barH - 3, "text-anchor": "end", class: "lbl-site" });
      idt.textContent = r.id + "  " + r.name; g.appendChild(idt);
    });
    // today
    const now = x(new Date(Date.UTC(2026, 8, 14)));
    g.appendChild(el("line", { x1: now, y1: T - 8, x2: now, y2: T + ROWS.length * rowH - 10, stroke: "#E64F39", "stroke-width": 2 }));
    const nt = el("text", { x: now - 5, y: T + ROWS.length * rowH + 2, "text-anchor": "end", class: "val", style: "fill:#E64F39" });
    nt.textContent = "today · M45"; g.appendChild(nt);
    // end of 2026
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
    src = SLIDES_SRC.read_text()
    old = re.findall(r'<section class="slide ">.*?</section>', src, flags=re.S)
    if len(old) != 16:
        raise SystemExit(f"expected 16 carried-over slides, found {len(old)}")
    # map the carried-over slides to agenda sections
    eyebrow_map = {
        "The data": "01 · The data", "The result": "02 · The result", "External validation": "02 · The result",
        "The honest caveat": "02 · The result", "How we read results": "02 · The result",
        "Deliverable D3.2 · M48": "03 · Deliverables · D3.2 · M48", "Deliverable T3.3 · M54": "03 · Deliverables · T3.3 · M54",
        "Deliverable D3.4 · M48": "03 · Deliverables · D3.4 · M48", "Milestone MS6 · M54": "03 · Deliverables · MS6 · M54",
        "Where we stand": "04 · Milestones and timeline", "What we need": "05 · Sites",
    }
    carried = []
    for s in old:
        for k, v in eyebrow_map.items():
            s = s.replace(f'<div class="eyebrow">{k}</div>', f'<div class="eyebrow">{v}</div>')
        s = s.replace('<section class="slide ">', '<section class="slide">')
        carried.append(s)
    data, result, deliv, status, need = carried[0:5], carried[5:10], carried[10:14], carried[14], carried[15]
    # status slide: 1.8.1 wording
    status = status.replace(
        "Shipped in 1.8.0 (13 Sep) and verified at two sites; needs Nijmegen and Utrecht to enable per-case return.",
        "Shipped in 1.8.0 and re-verified end to end on 1.8.1 (14 Sep); needs Nijmegen, Utrecht and Athens to enable per-case return for the October run.")
    status = status.replace("1.8.0 released 13 Sep; the re-run is now possible", "1.8.1 released 13 Sep; the October benchmark is the re-run")
    status = status.replace(
        "<strong>Since 13 September:</strong> version 1.8.0 is released — per-site metrics, per-case\n        predictions and the warm-start guard were each verified on real startup kits before\n        publishing, and every site picks it up on its next client restart.",
        "<strong>Since 13 September:</strong> versions 1.8.0 and 1.8.1 are released — per-site metrics, per-case\n        predictions and the warm-start guard were each verified on real startup kits before\n        publishing — and every site has received a 1.8.1 kit to install.")
    js = src[src.index("<script>"):src.index("</script>")]
    js = js.rstrip()
    assert js.endswith("})();"), "unexpected script tail"
    js = js[:-len("})();")] + GANTT_JS + "})();\n</script>"

    slides = [TITLE_SLIDE.format(header=HEADER_LOGOS), AGENDA_SLIDE] + data + result + deliv + \
             [MILESTONE_SLIDE, EOY_SLIDE, status, SITES_SLIDE, need]
    style = STYLE
    for k, v in logos.items():
        style = style.replace("{" + k + "}", v)
    out = [style, '<div class="deck">']
    for n, s in enumerate(slides, 1):
        out.append(s.replace("</section>", FOOTER.format(n=n) + "\n  </section>"))
    out.append('  <footer>ODELIA WP2 / WP3 · 14 September 2026 · every figure regenerated from stored per-case predictions, the run-history dataset, the coordinator log or CI</footer>\n</div>')
    out.append(js)
    return "\n".join(out)


if __name__ == "__main__":
    html = build()
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html)} bytes)")
