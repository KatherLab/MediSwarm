(function () {
  const SITES = [
    { s:"UKA",  joined:"2026-06-11", atJoin:17834, now:17834, c:[4747,11836,1251], d:"2026-09-04" },
    { s:"UMCU", joined:"2026-07-03", atJoin:6133,  now:6133,  c:[5337,740,56],     d:"2026-09-07" },
    { s:"RUMC", joined:"2026-04-30", atJoin:3608,  now:3608,  c:[3571,2,35],       d:"2026-06-21" },
    { s:"USZ",  joined:"2026-06-05", atJoin:3448,  now:3448,  c:[1911,1244,293],   d:"2026-07-06" },
    { s:"CAM",  joined:"2026-06-11", atJoin:970,   now:1778,  c:[1678,42,58],      d:"2026-09-07" },
    { s:"MHA",  joined:"2026-04-30", atJoin:810,   now:1120,  c:[904,120,96],      d:"2026-09-07" },
    { s:"RSH",  joined:"2026-06-29", atJoin:351,   now:351,   c:[4,126,221],       d:"2026-09-07" },
    { s:"VHIO", joined:"2026-07-31", atJoin:190,   now:190,   c:[159,0,31],        d:"2026-09-07" }
  ];
  const CUM = [
    { d:"2026-04-30", t:4418,  n:2, ev:"MHA and RUMC join" },
    { d:"2026-06-05", t:7866,  n:3, ev:"USZ joins" },
    { d:"2026-06-11", t:26670, n:5, ev:"CAM and UKA join" },
    { d:"2026-06-29", t:27021, n:6, ev:"RSH joins" },
    { d:"2026-07-03", t:33898, n:7, ev:"UMCU joins" },
    { d:"2026-07-31", t:34462, n:8, ev:"VHIO joins" }
  ];
  const TOTAL = 34462;
  const NS = "http://www.w3.org/2000/svg";
  const el = (n, a = {}) => { const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]); return e; };
  const fmt = v => v.toLocaleString("en-GB");

  /* ---------- 1. per-site bars ---------- */
  (function () {
    const g = document.getElementById("sizeBars");
    const L = 58, R = 96, T = 10, rowH = 34, barH = 17;
    const W = 720 - L - R;
    const max = Math.max(...SITES.map(d => d.now));
    SITES.forEach((d, i) => {
      const y = T + i * rowH;
      const w = Math.max(2, (d.now / max) * W);
      const grp = el("g", { class: "bar" });
      grp.appendChild(el("text", { x: L - 10, y: y + barH - 3, "text-anchor": "end", class: "lbl-site" }))
         .textContent = d.s;
      const r = el("rect", { x: L, y, width: w, height: barH, rx: 4, fill: "var(--bar)" });
      grp.appendChild(r);
      const t = el("text", { x: L + w + 8, y: y + barH - 3, class: "val" });
      t.textContent = fmt(d.now);
      grp.appendChild(t);
      const p = el("text", { x: L + w + 8, y: y + barH + 10, class: "val-sm" });
      p.textContent = (d.now / TOTAL * 100).toFixed(1) + " %";
      grp.appendChild(p);
      const title = el("title");
      title.textContent = `${d.s}: ${fmt(d.now)} training samples (${(d.now/TOTAL*100).toFixed(1)}% of consortium), joined ${d.joined} with ${fmt(d.atJoin)}`;
      grp.appendChild(title);
      g.appendChild(grp);
    });
    g.appendChild(el("line", { x1: L, y1: T - 4, x2: L, y2: T + SITES.length * rowH - 12, class: "axis" }));
  })();

  /* ---------- 2. cumulative step ---------- */
  (function () {
    const g = document.getElementById("cumChart");
    const L = 56, R = 22, T = 18, B = 62, H = 330;
    const W = 720 - L - R, PH = H - T - B;
    const t0 = Date.parse("2026-04-15"), t1 = Date.parse("2026-08-20");
    const x = d => L + (Date.parse(d) - t0) / (t1 - t0) * W;
    const maxY = 36000;
    const y = v => T + PH - (v / maxY) * PH;

    [0, 9000, 18000, 27000, 36000].forEach(v => {
      g.appendChild(el("line", { x1: L, y1: y(v), x2: L + W, y2: y(v), class: "grid" }));
      const t = el("text", { x: L - 9, y: y(v) + 4, "text-anchor": "end", class: "lbl" });
      t.textContent = v === 0 ? "0" : (v / 1000) + "k";
      g.appendChild(t);
    });

    let dPath = `M ${x(CUM[0].d)} ${y(0)}`, aPath = dPath;
    CUM.forEach((p, i) => {
      const px = x(p.d), py = y(p.t);
      dPath += ` L ${px} ${i ? y(CUM[i-1].t) : y(0)} L ${px} ${py}`;
      aPath += ` L ${px} ${i ? y(CUM[i-1].t) : y(0)} L ${px} ${py}`;
    });
    const endX = L + W;
    dPath += ` L ${endX} ${y(TOTAL)}`;
    aPath += ` L ${endX} ${y(TOTAL)} L ${endX} ${y(0)} Z`;
    g.appendChild(el("path", { d: aPath, fill: "var(--bar)", "fill-opacity": ".12" }));
    g.appendChild(el("path", { d: dPath, fill: "none", stroke: "var(--bar)", "stroke-width": "2",
                               "stroke-linejoin": "round" }));

    CUM.forEach((p, i) => {
      const px = x(p.d), py = y(p.t);
      const grp = el("g", { class: "bar" });
      grp.appendChild(el("circle", { cx: px, cy: py, r: 4.5, fill: "var(--bar)",
                                     stroke: "var(--surface-1)", "stroke-width": "2" }));
      const ti = el("title");
      ti.textContent = `${p.d} — ${fmt(p.t)} samples across ${p.n} site${p.n>1?"s":""} (${p.ev})`;
      grp.appendChild(ti);
      g.appendChild(grp);
      if (["2026-06-11", "2026-07-31", "2026-04-30"].includes(p.d)) {
        const lab = el("text", { x: px, y: py - 12, "text-anchor": i === 0 ? "start" : "middle", class: "val" });
        lab.textContent = fmt(p.t);
        g.appendChild(lab);
      }
    });

    g.appendChild(el("line", { x1: L, y1: T + PH, x2: L + W, y2: T + PH, class: "axis" }));
    [["2026-05-01","May"],["2026-06-01","Jun"],["2026-07-01","Jul"],["2026-08-01","Aug"]]
      .forEach(([d, lbl]) => {
        const px = x(d);
        g.appendChild(el("line", { x1: px, y1: T + PH, x2: px, y2: T + PH + 5, class: "axis" }));
        const t = el("text", { x: px, y: T + PH + 19, "text-anchor": "middle", class: "lbl" });
        t.textContent = lbl; g.appendChild(t);
      });

    [["2026-06-11","UKA + CAM  +18,804", -4],["2026-07-03","UMCU  +6,877", 4]].forEach(([d, txt, dx]) => {
      const px = x(d);
      const t = el("text", { x: px + dx, y: T + PH + 40, "text-anchor": dx < 0 ? "end" : "start", class: "val-sm" });
      t.textContent = txt; g.appendChild(t);
      g.appendChild(el("line", { x1: px, y1: T + PH + 26, x2: px, y2: T + PH + 32, class: "axis" }));
    });
  })();

  /* ---------- 3a. class stacks at TRUE SCALE ---------- */
  (function () {
    const g = document.getElementById("absBars");
    const rows = SITES.slice().sort((a, b) => b.now - a.now);
    const L = 62, R = 130, T = 12, rowH = 30, barH = 16, GAP = 2;
    const W = 720 - L - R;
    const max = Math.max(...rows.map(d => d.now));
    const COL = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
    const NAME = ["No lesion", "Benign", "Malignant"];

    // gridlines so the scale is readable, not just implied
    [0, 5000, 10000, 15000].forEach(v => {
      const gx = L + (v / max) * W;
      g.appendChild(el("line", { x1: gx, y1: T - 6, x2: gx, y2: T + rows.length * rowH - 8, class: "grid" }));
      const t = el("text", { x: gx, y: T + rows.length * rowH + 6, "text-anchor": "middle", class: "val-sm" });
      t.textContent = v === 0 ? "0" : (v / 1000) + "k";
      g.appendChild(t);
    });

    rows.forEach((d, i) => {
      const y = T + i * rowH;
      let xoff = L;
      const full = (d.now / max) * W;
      d.c.forEach((v, k) => {
        if (v === 0) return;
        const w = Math.max(1, (v / d.now) * full - (full > 12 ? GAP : 0));
        const grp = el("g", { class: "stack" });
        grp.appendChild(el("rect", { x: xoff, y, width: w, height: barH, rx: w > 6 ? 3 : 1, fill: COL[k] }));
        const ti = el("title");
        ti.textContent = `${d.s} — ${NAME[k]}: ${fmt(v)} of ${fmt(d.now)} (${(v/d.now*100).toFixed(1)}%)`;
        grp.appendChild(ti);
        g.appendChild(grp);
        xoff += w + (full > 12 ? GAP : 0);
      });
      const sl = el("text", { x: L - 10, y: y + barH - 3, "text-anchor": "end", class: "lbl-site" });
      sl.textContent = d.s; g.appendChild(sl);
      const tl = el("text", { x: L + full + 10, y: y + barH - 3, class: "val-sm" });
      tl.textContent = `${fmt(d.now)}  ·  ${fmt(d.c[1])} benign`;
      g.appendChild(tl);
    });
    g.appendChild(el("line", { x1: L, y1: T - 6, x2: L, y2: T + rows.length * rowH - 8, class: "axis" }));
  })();

  /* ---------- 3. class stacks (share of each site's training set) ---------- */
  (function () {
    const g = document.getElementById("classBars");
    const rows = SITES.slice().sort((a, b) => b.now - a.now)
      .concat([{ s: "CONSORTIUM", now: 34462, c: [18311, 14110, 2041], pooled: true }]);
    const L = 70, R = 150, T = 8, rowH = 30, barH = 16, GAP = 2;
    const W = 720 - L - R;
    const COL = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
    const NAME = ["No lesion", "Benign", "Malignant"];
    rows.forEach((d, i) => {
      const y = T + i * rowH + (d.pooled ? 10 : 0);
      let xoff = L;
      d.c.forEach((v, k) => {
        if (v === 0) return;
        const w = Math.max(1.5, (v / d.now) * W - GAP);
        const grp = el("g", { class: "stack" });
        grp.appendChild(el("rect", { x: xoff, y, width: w, height: barH, rx: 3, fill: COL[k] }));
        const ti = el("title");
        ti.textContent = `${d.s} — ${NAME[k]}: ${fmt(v)} of ${fmt(d.now)} (${(v/d.now*100).toFixed(1)}%)`;
        grp.appendChild(ti);
        g.appendChild(grp);
        if (w > 34) {
          const t = el("text", { x: xoff + w / 2, y: y + barH - 4, "text-anchor": "middle",
                                 class: "val-sm", fill: "#fff" });
          t.textContent = (v / d.now * 100).toFixed(0) + "%";
          g.appendChild(t);
        }
        xoff += w + GAP;
      });
      const sl = el("text", { x: L - 10, y: y + barH - 3, "text-anchor": "end",
                              class: d.pooled ? "val" : "lbl-site" });
      sl.textContent = d.s; g.appendChild(sl);
      const tl = el("text", { x: L + W + 10, y: y + barH - 3, class: "val-sm" });
      tl.textContent = `n=${fmt(d.now)}  ·  benign ${fmt(d.c[1])}`;
      g.appendChild(tl);
    });
  })();

  /* ---------- 4. model comparison ---------- */
  (function () {
    const g = document.getElementById("modelBars");
    const M = [
      { m:"Swarm · 8 sites · 1DC",      sub:"Aug 2026 · current",  mal:0.887, mac:0.820, hi:true },
      { m:"Best single site · UKA 1DC", sub:"trained alone",       mal:0.858, mac:0.709 },
      { m:"Swarm · 6 sites · 4LME",     sub:"Jun 2026",            mal:0.847, mac:0.764 },
      { m:"Median single-site model",   sub:"of 10",               mal:0.658, mac:0.601 },
      { m:"Weakest single site · RUMC", sub:"trained alone",       mal:0.408, mac:0.483 }
    ];
    const L = 176, R = 58, T = 14, rowH = 52, barH = 15, GAP = 4;
    const W = 720 - L - R;
    const x = v => L + v * W;                       // AUROC 0..1 on one axis

    [0, .25, .5, .75, 1].forEach(v => {
      g.appendChild(el("line", { x1: x(v), y1: T - 6, x2: x(v), y2: T + M.length * rowH - 14, class: "grid" }));
      const t = el("text", { x: x(v), y: T + M.length * rowH + 2, "text-anchor": "middle", class: "val-sm" });
      t.textContent = v.toFixed(2); g.appendChild(t);
    });
    // chance reference
    const ch = el("line", { x1: x(.5), y1: T - 6, x2: x(.5), y2: T + M.length * rowH - 14,
                            stroke: "var(--text-muted)", "stroke-width": "1", "stroke-dasharray": "3 3" });
    g.appendChild(ch);
    const cl = el("text", { x: x(.5), y: T - 10, "text-anchor": "middle", class: "val-sm" });
    cl.textContent = "chance"; g.appendChild(cl);

    M.forEach((d, i) => {
      const y = T + i * rowH;
      [["mal", "var(--series-1)", 0], ["mac", "var(--series-2)", barH + GAP]].forEach(([k, col, dy]) => {
        const w = Math.max(2, d[k] * W);
        const grp = el("g", { class: "bar" });
        grp.appendChild(el("rect", { x: L, y: y + dy, width: w, height: barH, rx: 4, fill: col }));
        const ti = el("title");
        ti.textContent = `${d.m} — ${k === "mal" ? "malignant vs rest" : "macro"} AUROC ${d[k].toFixed(3)}`;
        grp.appendChild(ti);
        g.appendChild(grp);
        const t = el("text", { x: L + w + 7, y: y + dy + barH - 3,
                               class: d.hi ? "val" : "val-sm" });
        t.textContent = d[k].toFixed(3); g.appendChild(t);
      });
      const nm = el("text", { x: L - 12, y: y + barH - 2, "text-anchor": "end",
                              class: d.hi ? "val" : "lbl" });
      nm.textContent = d.m; g.appendChild(nm);
      const sb = el("text", { x: L - 12, y: y + barH + 12, "text-anchor": "end", class: "val-sm" });
      sb.textContent = d.sub; g.appendChild(sb);
    });
    g.appendChild(el("line", { x1: L, y1: T - 6, x2: L, y2: T + M.length * rowH - 14, class: "axis" }));
  })();

  /* ---------- table ---------- */
  (function () {
    const tb = document.getElementById("tableBody");
    if (!tb) return;                       // slide may omit the table
    SITES.slice().sort((a, b) => b.now - a.now).forEach(d => {
      const tr = document.createElement("tr");
      [ d.s, d.joined, fmt(d.atJoin), fmt(d.now),
        (d.now / TOTAL * 100).toFixed(1) + " %",
        fmt(d.c[0]), fmt(d.c[1]), fmt(d.c[2]),
        (d.c[1] / d.now * 100).toFixed(1) + " %"
      ].forEach((c, i) => {
        const td = document.createElement("td");
        if (i) td.className = "n";
        td.textContent = c;
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  })();
})();
