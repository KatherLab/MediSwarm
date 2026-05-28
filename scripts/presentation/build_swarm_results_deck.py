#!/usr/bin/env python3
"""Build the ODELIA swarm-results presentation.

Single source of truth for the ~12-minute results talk. Emits two artifacts:

  docs/presentation_swarm_results.html              self-contained slide deck
                                                    (SVG figures inlined, arrow-key
                                                    navigation, press "S" for notes)
  docs/presentation_swarm_results_SPEAKER_NOTES.md  per-slide figure + speech script

All numbers come from the committed evaluation reports under docs/:
  - OLE_SWARM_EVALUATION_ARTIFACT_REPORT.md            (Study A, Duke 3-client swarm)
  - CHALLENGE_SWARM_LOCAL_TEST_REPORT_20260513.md      (Study B, 6 models x 6 sites)
  - ODELIA_SINGLE_SITE_CKPT_CHALLENGE_EVAL_*.md        (Study C, single-site transfer)

Run:  python scripts/presentation/build_swarm_results_deck.py
"""
from __future__ import annotations

import html
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
FIG = DOCS / "figures"

# --------------------------------------------------------------------------- #
# Figure helpers                                                              #
# --------------------------------------------------------------------------- #


def load_svg(rel: str) -> str:
    """Return the inline <svg>...</svg> markup for a figure under docs/figures."""
    text = (FIG / rel).read_text(encoding="utf-8")
    idx = text.find("<svg")
    return text[idx:] if idx >= 0 else text


def grouped_bar_svg(title: str, groups, series, *, ymax=1.0, width=1120, height=560):
    """Minimal grouped-bar chart matching the repo figure aesthetic.

    series: list of (label, color, values) where values aligns with groups.
    """
    left, right, top, bottom = 70, 30, 70, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(groups)
    slot = plot_w / n
    bars = len(series)
    bw = slot * 0.72 / bars
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#18212f}"
        ".title{font-size:24px;font-weight:700}.axis{font-size:13px;fill:#465161}"
        ".lab{font-size:14px;fill:#263241}.val{font-size:12px;fill:#263241}"
        ".grid{stroke:#d8dee8;stroke-width:1}.leg{font-size:14px;fill:#263241}</style>",
        f'<text class="title" x="{left}" y="38">{html.escape(title)}</text>',
    ]
    # gridlines + y labels
    for i in range(6):
        frac = i / 5
        y = top + plot_h * (1 - frac)
        out.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{frac*ymax:.1f}</text>')
    # bars
    for gi, g in enumerate(groups):
        gx = left + slot * gi
        for si, (slabel, color, values) in enumerate(series):
            v = values[gi]
            bx = gx + slot * 0.14 + si * bw
            bh = plot_h * (v / ymax) if v is not None else 0
            by = top + plot_h - bh
            if v is None:
                out.append(f'<text class="val" x="{bx+bw/2:.1f}" y="{top+plot_h-6:.1f}" text-anchor="middle">NA</text>')
                continue
            out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}"/>')
            out.append(f'<text class="val" x="{bx+bw/2:.1f}" y="{by-5:.1f}" text-anchor="middle">{v:.3f}</text>')
        out.append(
            f'<text class="lab" x="{gx+slot/2:.1f}" y="{top+plot_h+24:.1f}" '
            f'text-anchor="middle">{html.escape(g)}</text>'
        )
    # legend
    lx = left
    ly = height - 24
    for slabel, color, _ in series:
        out.append(f'<rect x="{lx}" y="{ly-12}" width="16" height="16" fill="{color}"/>')
        out.append(f'<text class="leg" x="{lx+22}" y="{ly}">{html.escape(slabel)}</text>')
        lx += 30 + 9 * len(slabel)
    out.append("</svg>")
    return "\n".join(out)


# Headline figure: swarm global vs best single-site, EXTERNAL macro AUROC.
SWARM = "#1f77b4"
LOCAL = "#d6a419"
MODELS6 = ["MST", "1DC", "2BCN_AIM", "3agaldran", "4LME_ABMIL", "5Pimed"]
EXT_MACRO_SWARM = [0.645, 0.708, 0.627, 0.692, 0.764, 0.387]
EXT_MACRO_LOCAL = [0.635, 0.609, 0.572, 0.698, 0.734, 0.609]
HEADLINE_SVG = grouped_bar_svg(
    "External challenge macro AUROC: swarm global vs best single-site",
    MODELS6,
    [
        ("Swarm global final", SWARM, EXT_MACRO_SWARM),
        ("Best single-site model", LOCAL, EXT_MACRO_LOCAL),
    ],
)

# Internal-vs-external contrast figure (best site-local checkpoints).
INTEXT_MODELS = ["MST", "1DC", "2BCN_AIM", "3agaldran", "4LME_ABMIL"]
INT_VAL = [0.801, 0.870, 0.775, 0.902, 0.910]   # best site-local INTERNAL val AUROC
EXT_VAL = [0.635, 0.609, 0.572, 0.698, 0.734]   # same checkpoints EXTERNAL macro AUROC
INTEXT_SVG = grouped_bar_svg(
    "Best single-site model: internal validation vs external challenge",
    INTEXT_MODELS,
    [
        ("Internal validation AUROC", "#5b6b7f", INT_VAL),
        ("External challenge macro AUROC", "#c0504d", EXT_VAL),
    ],
)

# --------------------------------------------------------------------------- #
# Slide content                                                               #
# --------------------------------------------------------------------------- #
# Each slide: title, optional subtitle, bullets (list), figure (inline svg str
# or None), caption, notes (speaker speech), seconds (target pacing).

SLIDES = [
    dict(
        kind="title",
        title="Swarm Learning for Breast-MRI Lesion Classification",
        subtitle="Multi-site evaluation results &mdash; ODELIA",
        bullets=[
            "Does a privacy-preserving swarm model generalize better than any single hospital&rsquo;s model?",
            "Three studies &middot; 6 sites &middot; 6 model families",
        ],
        figure=None,
        caption="",
        notes=(
            "Good morning. Today I'm presenting our evaluation results for swarm learning applied to "
            "breast-MRI lesion classification, in the ODELIA project. The whole talk is built around one "
            "question: when several hospitals train a shared model together without ever sharing their "
            "patient data, does that shared swarm model actually generalize better than a model trained at "
            "any single hospital? I'll answer that with three studies, spanning six clinical sites and six "
            "different model architectures. Let me start with why this question matters."
        ),
        seconds=45,
    ),
    dict(
        kind="content",
        title="The problem and the question",
        subtitle="",
        bullets=[
            "<b>Medical imaging is siloed.</b> Breast-MRI lives behind hospital governance &mdash; data cannot be pooled centrally.",
            "<b>Swarm / federated learning</b> trains one shared model by exchanging <i>model weights</i>, never images.",
            "<b>The catch:</b> each site sees a different population &mdash; scanners, prevalence, demographics all shift.",
            "<b>Central question:</b> does aggregating across sites beat the best single-site model on <i>unseen</i> data?",
        ],
        figure=None,
        caption="",
        notes=(
            "The starting point is that medical imaging data is siloed. Breast-MRI studies sit behind each "
            "hospital's data governance, and for good privacy reasons you usually cannot copy them into one "
            "central training set. Swarm learning, and federated learning more broadly, gets around this: the "
            "sites exchange only model weights, never the images themselves, and converge on one shared model. "
            "The catch is distribution shift. Every site has different scanners, a different mix of benign and "
            "malignant cases, different demographics. So a model that looks excellent at its home site may "
            "collapse elsewhere. That is exactly what we want to measure: does aggregating across sites beat the "
            "single best site model when you test on data neither has seen during selection?"
        ),
        seconds=55,
    ),
    dict(
        kind="content",
        title="What we evaluated",
        subtitle="One task, two endpoints, three studies",
        bullets=[
            "<b>Task:</b> 3-class lesion grade per breast &mdash; 0 none, 1 benign, 2 malignant (unilateral MRI).",
            "<b>Sites:</b> CAM, MHA, RSH, RUMC, UKA, UMCU (+ USZ onboarding).",
            "<b>Two endpoints:</b> <i>internal</i> validation (own site) vs <i>external</i> ODELIA challenge test (held-out).",
            "<b>Study A</b> &mdash; swarm mechanism proof (Duke, 3 clients).",
            "<b>Study B</b> &mdash; 6 models &times; 6 sites swarm, swarm-vs-single-site on external data.",
            "<b>Study C</b> &mdash; single-site checkpoint transfer to the external challenge.",
        ],
        figure=None,
        caption="",
        notes=(
            "Here's the structure. The clinical task is a three-class grade for each breast from the MRI: class "
            "zero is no lesion, class one benign, class two malignant. Class two, malignant, is the one we care "
            "most about. We have six participating sites, with a seventh, USZ in Zurich, currently being "
            "onboarded. The single most important idea on this slide is the two endpoints. Internal validation "
            "is a model scored on its own site's held-out split, which is what you use during training to pick "
            "checkpoints. External validation is the held-out ODELIA challenge test set, pooled across "
            "institutions, and that is the honest measure of generalization. Keep that distinction in mind, "
            "because the gap between the two is really the story of this talk. I'll walk through three studies "
            "that each look at this from a different angle."
        ),
        seconds=60,
    ),
    dict(
        kind="figure",
        title="Study A &mdash; the swarm mechanism works end-to-end",
        subtitle="Duke data, 3 clients (node A/B/C), MST model, 20 swarm rounds completed",
        bullets=[
            "Swarm-aggregated AUROC matches or beats each node&rsquo;s own site model:",
            "node&nbsp;A 0.948 &middot; node&nbsp;B 0.872 &middot; node&nbsp;C 0.965 (classes 0 vs 2).",
            "Full three-client run completed; global models written on every node.",
        ],
        figure=load_svg("ole_swarm/validation_auroc_summary.svg"),
        caption="Extracted Duke 3-client swarm: validation AUROC, site model vs swarm-aggregated.",
        notes=(
            "Study A is the proof that the plumbing works. This was a three-client swarm on Duke data, nodes A, "
            "B and C, training the MST model for twenty rounds, and it ran to completion with global model files "
            "written on every node. Look at the bars: for each node, the swarm-aggregated model, in the right-"
            "hand bars, matches or slightly beats that node's own local site model. Node A reaches 0.948, node B "
            "0.872, node C 0.965. So aggregation is not hurting any individual site, which is the first thing you "
            "want to confirm. One honest caveat, which becomes a theme: this run only contained classes zero and "
            "two, no benign cases, so these AUROCs are really binary, no-lesion versus malignant. That limitation "
            "is what set up the next slide."
        ),
        seconds=70,
    ),
    dict(
        kind="content",
        title="Study A &mdash; what the first swarm taught us",
        subtitle="Three issues that became guardrails for Study B",
        bullets=[
            "<b>Label coverage:</b> class&nbsp;1 (benign) absent &rArr; 3-class macro AUROC undefined &mdash; coverage must be checked first.",
            "<b>Best-model selection broke:</b> aggregator looked for <code>accuracy</code> but clients logged <code>val/ACC</code> &rArr; run emitted the <i>last</i> model, not the global-best.",
            "<b>Split hygiene:</b> logs flagged UID overlap across train / val / test &rArr; needs split auditing before numbers are trusted.",
            "All three are <i>data / plumbing</i> issues &mdash; not modeling issues.",
        ],
        figure=None,
        caption="",
        notes=(
            "Running that first swarm taught us three things, and they shaped everything after. First, label "
            "coverage: because benign cases were missing, the three-class macro AUROC simply isn't defined, so "
            "we now check class coverage before trusting any aggregate metric. Second, and this one is subtle but "
            "important, the best-model selection silently broke. The aggregator was configured to track a metric "
            "called 'accuracy', but the clients were logging it under the name 'val/ACC'. The names didn't match, "
            "so no global-best was ever selected, and the run published its last-round model instead of its best "
            "one. Third, the logs flagged the same patient UIDs appearing across train, validation and test, "
            "which inflates scores. The key point: every single problem here was a data or plumbing issue, not a "
            "modeling one. So for the big study we hardened all three."
        ),
        seconds=60,
    ),
    dict(
        kind="figure",
        title="Study B &mdash; a complete 6 &times; 6 swarm matrix",
        subtitle="6 model families trained across 6 sites &mdash; complete artifacts everywhere",
        bullets=[
            "Every model reached <b>6/6</b> global models, run dirs and validation CSVs across all sites.",
            "5Pimed and MST each had one failed retry, then recovered &mdash; tracked separately.",
            "Runtimes ~5&ndash;18&nbsp;h per model (RTX-class GPUs).",
        ],
        figure=load_svg("challenge_swarm_local_tests_20260513/run_status_heatmap.svg"),
        caption="Run-status matrix: all six models complete; failed retries kept separate from the final matrix.",
        notes=(
            "Study B is the main event. We trained all six ODELIA model families &mdash; MST, "
            "1DivideAndConquer, 2BCN_AIM, 3agaldran, 4LME_ABMIL and 5Pimed &mdash; in a swarm across all six "
            "sites. This heatmap is the completeness check: green means that model produced complete artifacts. "
            "Every model reached six-of-six on global models, run directories and validation CSVs. Two models, "
            "5Pimed and MST, needed one retry after a failed job, and we keep those failures separate so they "
            "don't contaminate the quality numbers. The runtimes ranged from about five hours to about eighteen "
            "hours per model. So before we even talk about accuracy: we have a full, reproducible federated "
            "matrix with complete artifacts, which is itself a real engineering result."
        ),
        seconds=55,
    ),
    dict(
        kind="figure",
        title="Study B &mdash; internal validation by model",
        subtitle="Best aggregated validation AUROC during swarm training",
        bullets=[
            "4LME_ABMIL <b>0.900</b> &middot; 3agaldran 0.830 &middot; 1DC 0.811",
            "5Pimed 0.789 &middot; MST 0.775 &middot; 2BCN_AIM 0.760",
            "<b>But internal validation is necessary, not sufficient</b> &mdash; the real test is the next slide.",
        ],
        figure=load_svg("challenge_swarm_local_tests_20260513/validation_auroc_summary.svg"),
        caption="Per-model, per-site validation AUROC (best/last aggregated and best site model).",
        notes=(
            "If we rank the models purely on internal validation AUROC during swarm training, 4LME_ABMIL leads "
            "at 0.900, then 3agaldran at 0.830, 1DivideAndConquer at 0.811, and the rest between 0.76 and 0.79. "
            "These are healthy numbers and they tell us training converged. But I want to flag, before anyone "
            "writes these down as the result: internal validation is necessary but not sufficient. It's measured "
            "on each site's own distribution, and as we'll see, it systematically overstates how the model does "
            "on outside data. The honest comparison is on the external challenge set, and that's the next slide "
            "&mdash; the one I'd ask you to remember."
        ),
        seconds=55,
    ),
    dict(
        kind="figure",
        title="Study B &mdash; HEADLINE: swarm beats the best single site externally",
        subtitle="External ODELIA challenge, 300 cases, sample-weighted",
        bullets=[
            "Swarm global &ge; best single-site model on <b>external macro AUROC for 4 of 6</b> models; tied on 3agaldran.",
            "Biggest gains: 1DC +0.10, 2BCN_AIM +0.06, 4LME_ABMIL +0.03 macro AUROC.",
            "Best external model overall: <b>4LME_ABMIL swarm</b> &mdash; macro 0.764, Class-2 AUROC 0.847.",
            "Only the degenerate 5Pimed swarm regressed (0.39 vs 0.61).",
        ],
        figure=HEADLINE_SVG,
        caption="External challenge macro AUROC: swarm global (blue) vs the strongest single-site model (gold).",
        notes=(
            "This is the headline. For each model we take the swarm's global model, the blue bars, and compare it "
            "against the single best site-local model for that family, the gold bars, on the external ODELIA "
            "challenge test &mdash; three hundred cases, sample-weighted across institutions, that neither model "
            "tuned on. For four of the six models the swarm global wins on external macro AUROC, and it ties on "
            "3agaldran. The gains are real: 1DivideAndConquer jumps about ten points, 2BCN_AIM six, 4LME_ABMIL "
            "three. The strongest model overall is 4LME_ABMIL from the swarm, at 0.764 macro AUROC and 0.847 "
            "AUROC on the malignant class specifically. The one regression is 5Pimed, where the swarm model was "
            "degenerate &mdash; I'll come back to that as a fixable plumbing failure, not evidence against swarm "
            "learning. So the central result: training together generalizes better than the best site alone."
        ),
        seconds=90,
    ),
    dict(
        kind="figure",
        title="Study B &mdash; why: aggregation regularizes against site shift",
        subtitle="The best single-site model overfits its own population",
        bullets=[
            "Best site-local models score <b>0.78&ndash;0.91 internally</b> &hellip; but drop to <b>0.57&ndash;0.73 externally</b>.",
            "Swarm models score <i>lower internally</i> yet <i>higher externally</i> &mdash; classic generalization trade.",
            "Aggregating across heterogeneous sites acts as regularization against distribution shift.",
        ],
        figure=INTEXT_SVG,
        caption="Same best single-site checkpoints: high internal validation collapses on the external challenge.",
        notes=(
            "Why does the swarm win? This chart shows the best single-site model for each family scored two ways: "
            "grey is its internal validation, where it looks superb, 0.78 up to 0.91; red is the very same "
            "checkpoint on the external challenge, where it falls to between 0.57 and 0.73. That collapse is "
            "overfitting to the home population. The swarm models, by contrast, look a little weaker internally "
            "but hold up better externally &mdash; which is exactly the trade you want. Mechanistically, "
            "averaging weights across sites with different scanners and case mixes acts like a regularizer: it "
            "throws away the idiosyncrasies that only help at one site and keeps what transfers. That's the "
            "argument for swarm learning in one picture."
        ),
        seconds=60,
    ),
    dict(
        kind="figure",
        title="Study C &mdash; single-site checkpoint transfer",
        subtitle="Each site's own checkpoints, scored on the external challenge (Class-2 AUROC)",
        bullets=[
            "Top transfer: <b>UKA 1DC 0.858</b> &middot; USZ 1DC 0.810 &middot; MHA 1DC 0.795.",
            "The <b>1DivideAndConquer</b> family dominates the top of the leaderboard.",
            "Best single checkpoint (UKA 1DC, 0.858) edges the best swarm model on Class-2 AUROC &mdash; but see next slide.",
        ],
        figure=load_svg("odelia_single_site_eval/challenge_aggregate_class2_auroc.svg"),
        caption="External challenge weighted Class-2 (malignant) AUROC, ranked by checkpoint.",
        notes=(
            "Study C zooms in on individual checkpoints. Here we took single-site-trained checkpoints and ran "
            "each one on the external challenge, ranked by AUROC on the malignant class. The best transfer comes "
            "from UKA's 1DivideAndConquer checkpoint at 0.858, then USZ's 1DC at 0.810, then MHA's at 0.795. The "
            "clear pattern is that the 1DivideAndConquer architecture owns the top of this leaderboard, which is "
            "useful for deciding where to invest. You'll notice the single best checkpoint, UKA at 0.858, is "
            "actually a touch higher than the best swarm model on this one malignant-class metric. That's real, "
            "but it's cherry-picked with hindsight on the test set &mdash; and the next slide shows why you can't "
            "rely on picking it."
        ),
        seconds=60,
    ),
    dict(
        kind="figure",
        title="Study C &mdash; internal scores do not predict transfer",
        subtitle="High internal validation &ne; good external generalization",
        bullets=[
            "CAM 1DC: internal Class-2 AUROC <b>0.965</b> &rarr; external <b>0.682</b>.",
            "RUMC MST: internal accuracy <b>0.995</b> &rarr; external macro <b>0.495</b> (val split was ~199:0:1 &mdash; degenerate).",
            "RSH 5Pimed: external Class-2 AUROC 0.456 but recall 0.879 &mdash; predicts &lsquo;malignant&rsquo; indiscriminately.",
            "<b>Lesson:</b> select on representative/external validation; under imbalance, trust AUROC + recall, not accuracy.",
        ],
        figure=load_svg("odelia_single_site_eval/challenge_aggregate_macro_auroc.svg"),
        caption="External challenge macro AUROC by checkpoint &mdash; ranking differs sharply from internal scores.",
        notes=(
            "And this is the cautionary half of Study C. Internal scores do not predict transfer. CAM's 1DC "
            "checkpoint looked almost perfect internally, 0.965 on the malignant class, but lands at 0.682 "
            "externally. RUMC's MST hit ninety-nine-point-five percent internal accuracy &mdash; but only because "
            "its validation split was essentially one class, about 199 negatives to a single positive, so the "
            "model just predicted 'negative' every time; externally its macro AUROC is 0.495, no better than "
            "chance. RSH's 5Pimed shows the opposite failure: high malignant recall, 0.88, but terrible AUROC, "
            "because it cries 'malignant' on almost everyone. The lesson is twofold: choose checkpoints on "
            "representative or external validation, not on the home split; and under this kind of class "
            "imbalance, accuracy is a trap &mdash; you have to read AUROC together with recall."
        ),
        seconds=70,
    ),
    dict(
        kind="figure",
        title="Onboarding a new site &mdash; USZ (Zurich)",
        subtitle="Data audit + local training before joining the live swarm",
        bullets=[
            "5,312 cases &mdash; train 64.9% / val 15.3% / test 19.8%; <b>0 images missing</b>.",
            "Heavy imbalance: class&nbsp;0 56% / class&nbsp;1 35% / <b>class&nbsp;2 only 9%</b> (malignant).",
            "Local MST and 1DC trained cleanly; 1DC reached external Class-2 AUROC 0.810.",
            "Same audit gate every new partner passes before contributing to the swarm.",
        ],
        figure=load_svg("usz_partner_eval/usz_split_distribution.svg"),
        caption="USZ split sizes and annotation class distribution from the onboarding audit.",
        notes=(
            "Beyond the headline science, a practical question for a project like this is: how do you safely add "
            "a new hospital? This slide is our onboarding workflow, using USZ in Zurich as the live example. "
            "Before a new site contributes to the swarm we run a data audit and a local-only training pass. USZ "
            "brings 5,312 cases, cleanly split about 65/15/20 percent, with zero missing images, which is a good "
            "sign. The important caveat is the class balance: malignant cases are only nine percent of the data, "
            "so it's heavily imbalanced, and that conditions how much we weight USZ for the malignant class. Its "
            "local 1DC model already transfers well, 0.810 on the external malignant AUROC. Every new partner "
            "passes this same audit gate before joining the live swarm."
        ),
        seconds=60,
    ),
    dict(
        kind="content",
        title="Cross-cutting: data quality is the real bottleneck",
        subtitle="Every study surfaced a data/plumbing issue before a modeling one",
        bullets=[
            "Label <b>coverage</b> &mdash; missing benign class breaks 3-class metrics (Study A).",
            "Split <b>hygiene</b> &mdash; UID overlap across train/val/test inflates scores (Study A).",
            "Metric-name <b>mismatch</b> silently disabled global-best selection (Study A &amp; 5Pimed in B).",
            "Class <b>imbalance</b> &mdash; accuracy is misleading; degenerate single-class val splits (RUMC, USZ 9% malignant).",
            "&rArr; Plausibility auditing is now a gate, not an afterthought.",
        ],
        figure=None,
        caption="",
        notes=(
            "Stepping back, the strongest cross-cutting lesson is that data quality, not model architecture, was "
            "our real bottleneck. Across all three studies, the first thing that broke was always a data or "
            "plumbing issue. Label coverage gaps broke our three-class metrics. Patient overlap across splits "
            "inflated scores. A metric-name mismatch quietly disabled best-model selection, twice. And class "
            "imbalance repeatedly made accuracy look great while the model was effectively guessing, including "
            "the degenerate single-class validation split at RUMC. The upshot is that we now treat plausibility "
            "auditing &mdash; checking coverage, splits, label balance, and metric wiring &mdash; as a gate that "
            "runs before training, not as a post-hoc cleanup. That single change would have caught most of what "
            "I've shown you."
        ),
        seconds=55,
    ),
    dict(
        kind="content",
        title="Takeaways",
        subtitle="",
        bullets=[
            "<b>1.</b> The swarm pipeline runs end-to-end: 6 models &times; 6 sites, complete artifacts.",
            "<b>2.</b> Swarm global <b>generalizes better than the best single-site model</b> externally (4/6 win, 1 tie). <i>Core result.</i>",
            "<b>3.</b> Internal validation overstates performance &mdash; the external challenge is the honest endpoint.",
            "<b>4.</b> 1DivideAndConquer and 4LME_ABMIL are the strongest families (ext. AUROC up to 0.86).",
            "<b>5.</b> Data audits catch most failures before modeling.",
        ],
        figure=None,
        caption="",
        notes=(
            "So, five takeaways. One: the swarm pipeline works end-to-end &mdash; six models across six sites "
            "with complete artifacts, which was not trivial to achieve. Two, and this is the core scientific "
            "result: the swarm-aggregated global model generalizes better than the best single-site model on "
            "external data, winning four of six head-to-heads and tying a fifth. Three: internal validation "
            "consistently overstates performance, so we report the external challenge as the honest endpoint. "
            "Four: on architecture, 1DivideAndConquer and 4LME_ABMIL are the standouts, reaching external AUROCs "
            "up to 0.86 on the malignant class. And five: most of our failures were caught &mdash; or would have "
            "been caught &mdash; by data audits, not by better models."
        ),
        seconds=60,
    ),
    dict(
        kind="content",
        title="Next steps",
        subtitle="",
        bullets=[
            "Fix global-best selection (align metric names) so the swarm publishes its best model, not its last.",
            "Re-run the 5Pimed swarm &mdash; current global model is degenerate externally.",
            "Onboard USZ into the live swarm; standardize external-challenge checkpoint selection.",
            "Make class-1 coverage + split-overlap auditing CI gates before every run.",
            "Report Class-2 (malignant) AUROC + recall as the primary clinical endpoint.",
        ],
        figure=None,
        caption="",
        notes=(
            "Finally, next steps. First, fix the global-best selection by aligning those metric names, so the "
            "swarm publishes its best model rather than its last &mdash; that alone should lift several numbers. "
            "Second, re-run the 5Pimed swarm, since its current global model is degenerate externally. Third, "
            "bring USZ into the live swarm now that it's passed the audit, and standardize how we pick "
            "checkpoints against the external challenge. Fourth, promote the coverage and split-overlap audits "
            "into automated gates that run before every training job. And fifth, going forward we'll report "
            "malignant-class AUROC and recall as the primary clinical endpoint, because that's what matters for "
            "patients. Thank you &mdash; I'm happy to take questions."
        ),
        seconds=55,
    ),
    # ---- backup slides -----------------------------------------------------
    dict(
        kind="figure",
        title="Backup &mdash; runtime by model",
        subtitle="Approximate swarm wall-clock per model",
        bullets=["Spans first-to-last log timestamp; relative comparison only."],
        figure=load_svg("challenge_swarm_local_tests_20260513/duration_by_model.svg"),
        caption="2BCN_AIM was slowest (~18 h); MST/5Pimed fastest (~5 h).",
        notes=(
            "Backup slide on runtimes if it comes up: 2BCN_AIM was the slowest at roughly eighteen hours, while "
            "MST and 5Pimed finished in about five. These are log-timestamp spans, so treat them as relative, "
            "not billing-grade."
        ),
        seconds=20,
    ),
    dict(
        kind="figure",
        title="Backup &mdash; artifact coverage",
        subtitle="Per-model presence of every expected artifact type",
        bullets=["Global models, run dirs, CSVs, TFEvents, local checkpoints &mdash; all 6/6."],
        figure=load_svg("challenge_swarm_local_tests_20260513/artifact_coverage_heatmap.svg"),
        caption="Complete artifact coverage across all six models and sites.",
        notes=(
            "Backup detail on artifact coverage: every model produced every expected artifact type &mdash; "
            "global models, run directories, validation and training CSVs, TensorBoard events and local "
            "checkpoints &mdash; at six-of-six. No gaps."
        ),
        seconds=20,
    ),
    dict(
        kind="figure",
        title="Backup &mdash; example training curve (UKA 1DC)",
        subtitle="Best external single-site checkpoint",
        bullets=["Internal validation AUROC/accuracy over epochs for the top-transfer checkpoint."],
        figure=load_svg("odelia_single_site_eval/UKA_1DC_training_curves.svg"),
        caption="UKA 1DivideAndConquer training curves; epoch-25 checkpoint transferred best (Class-2 AUROC 0.858).",
        notes=(
            "And a representative training curve, for the UKA 1DC run that gave us the best external transfer. "
            "The epoch-25 checkpoint, not the last one, was the strongest externally &mdash; another reminder "
            "that last is not best."
        ),
        seconds=20,
    ),
]

# --------------------------------------------------------------------------- #
# HTML emitter                                                                #
# --------------------------------------------------------------------------- #

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#0d1117;font-family:'Segoe UI',Arial,Helvetica,sans-serif;color:#18212f}
#deck{position:fixed;inset:0}
.slide{position:absolute;inset:0;display:none;flex-direction:column;
  background:#ffffff;padding:54px 72px 64px}
.slide.active{display:flex}
.slide.title-slide{background:linear-gradient(135deg,#0f2c4d 0%,#1f5f8b 100%);color:#fff;justify-content:center}
.slide.title-slide h1{font-size:52px;line-height:1.1;max-width:18ch}
.slide.title-slide .subtitle{font-size:26px;color:#bcd6ec;margin-top:18px}
.slide.title-slide ul{margin-top:40px;list-style:none}
.slide.title-slide li{font-size:21px;color:#e8f1fa;margin:10px 0;padding-left:0}
.kicker{font-size:15px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#1f5f8b}
h1{font-size:34px;line-height:1.15;color:#0f2c4d;margin-bottom:4px}
.subtitle{font-size:19px;color:#5b6b7f;margin-bottom:18px}
.body{flex:1;display:flex;gap:38px;min-height:0}
.body.has-fig .text{flex:0 0 42%}
.text{display:flex;flex-direction:column;justify-content:center}
ul{list-style:none}
.text li{font-size:21px;line-height:1.5;color:#222b38;margin:13px 0;padding-left:26px;position:relative}
.text li:before{content:'';position:absolute;left:2px;top:11px;width:10px;height:10px;
  background:#1f5f8b;border-radius:2px}
.figwrap{flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;min-width:0}
.figwrap svg{width:100%;height:auto;max-height:72vh;border:1px solid #e2e8f0;border-radius:8px;background:#fff}
.caption{font-size:14px;color:#6b7787;margin-top:12px;text-align:center;font-style:italic;max-width:60ch}
code{font-family:'SF Mono',Consolas,monospace;background:#eef2f7;padding:1px 6px;border-radius:4px;font-size:.92em}
.footer{position:absolute;left:72px;right:72px;bottom:20px;display:flex;justify-content:space-between;
  align-items:center;font-size:13px;color:#9aa6b4}
.title-slide .footer{color:#9fc0dc}
.progress{position:fixed;left:0;bottom:0;height:4px;background:#1f5f8b;transition:width .2s;z-index:5}
#notes{position:fixed;left:0;right:0;bottom:0;max-height:42vh;overflow:auto;background:rgba(13,17,23,.96);
  color:#e8eef5;padding:22px 72px;font-size:18px;line-height:1.6;display:none;z-index:10;
  border-top:3px solid #1f5f8b}
#notes.show{display:block}
#notes b{color:#7fc1f0}
.hint{position:fixed;right:14px;top:12px;font-size:12px;color:#6b7787;z-index:6}
@media print{.slide{display:flex !important;position:relative;page-break-after:always;height:100vh}
  .progress,.hint,#notes{display:none !important}}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];
const notesData=window.__NOTES__;
let i=0;
const prog=document.getElementById('prog');
const notes=document.getElementById('notes');
function show(n){
  i=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((s,k)=>s.classList.toggle('active',k===i));
  prog.style.width=((i+1)/slides.length*100)+'%';
  if(notes.classList.contains('show'))renderNotes();
}
function renderNotes(){notes.innerHTML='<b>Slide '+(i+1)+'/'+slides.length+'</b><br>'+notesData[i];}
document.addEventListener('keydown',e=>{
  if(['ArrowRight','PageDown',' '].includes(e.key)){show(i+1);e.preventDefault();}
  else if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);
  else if(e.key==='Home')show(0);
  else if(e.key==='End')show(slides.length-1);
  else if(e.key.toLowerCase()==='s'){notes.classList.toggle('show');renderNotes();}
  else if(e.key.toLowerCase()==='f'){if(!document.fullscreenElement)document.documentElement.requestFullscreen();else document.exitFullscreen();}
});
document.addEventListener('click',e=>{if(e.target.closest('#notes'))return;
  const x=e.clientX/window.innerWidth;show(x<0.25?i-1:i+1);});
show(0);
"""


def render_html() -> str:
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>ODELIA Swarm Learning &mdash; Results</title>",
        f"<style>{CSS}</style></head><body><div id='deck'>",
    ]
    notes_js = []
    total = len(SLIDES)
    for n, s in enumerate(SLIDES):
        notes_js.append(s["notes"].replace("\\", "\\\\").replace("`", "\\`"))
        is_title = s["kind"] == "title"
        cls = "slide title-slide" if is_title else "slide"
        parts.append(f"<section class='{cls}'>")
        if is_title:
            parts.append(f"<h1>{s['title']}</h1>")
            if s.get("subtitle"):
                parts.append(f"<div class='subtitle'>{s['subtitle']}</div>")
            if s.get("bullets"):
                parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in s["bullets"]) + "</ul>")
        else:
            parts.append("<div class='kicker'>ODELIA &middot; Swarm Learning Results</div>")
            parts.append(f"<h1>{s['title']}</h1>")
            if s.get("subtitle"):
                parts.append(f"<div class='subtitle'>{s['subtitle']}</div>")
            has_fig = bool(s.get("figure"))
            parts.append(f"<div class='body{' has-fig' if has_fig else ''}'>")
            if s.get("bullets"):
                parts.append("<div class='text'><ul>" + "".join(f"<li>{b}</li>" for b in s["bullets"]) + "</ul></div>")
            if has_fig:
                parts.append("<div class='figwrap'>" + s["figure"])
                if s.get("caption"):
                    parts.append(f"<div class='caption'>{s['caption']}</div>")
                parts.append("</div>")
            parts.append("</div>")
        parts.append(
            f"<div class='footer'><span>ODELIA breast-MRI swarm learning</span>"
            f"<span>{n+1} / {total}</span></div>"
        )
        parts.append("</section>")
    parts.append("</div>")
    parts.append("<div class='progress' id='prog'></div>")
    parts.append("<div class='hint'>&larr;/&rarr; navigate &middot; S notes &middot; F fullscreen</div>")
    parts.append("<div id='notes'></div>")
    notes_arr = "[" + ",".join("`" + t + "`" for t in notes_js) + "]"
    parts.append(f"<script>window.__NOTES__={notes_arr};{JS}</script>")
    parts.append("</body></html>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Markdown speaker-notes emitter                                              #
# --------------------------------------------------------------------------- #

FIG_REL = {
    id(s["figure"]): None for s in SLIDES
}  # placeholder; we track figure path per slide instead


SLIDE_FIG_PATH = [
    None,                                                              # 1 title
    None,                                                              # 2 problem
    None,                                                              # 3 setup
    "figures/ole_swarm/validation_auroc_summary.svg",                 # 4
    None,                                                              # 5 study A lessons
    "figures/challenge_swarm_local_tests_20260513/run_status_heatmap.svg",       # 6
    "figures/challenge_swarm_local_tests_20260513/validation_auroc_summary.svg", # 7
    "(generated) external macro AUROC: swarm vs single-site bar chart",          # 8 headline
    "(generated) internal vs external AUROC bar chart",                          # 9
    "figures/odelia_single_site_eval/challenge_aggregate_class2_auroc.svg",      # 10
    "figures/odelia_single_site_eval/challenge_aggregate_macro_auroc.svg",       # 11
    "figures/usz_partner_eval/usz_split_distribution.svg",            # 12
    None,                                                              # 13 data quality
    None,                                                              # 14 takeaways
    None,                                                              # 15 next steps
    "figures/challenge_swarm_local_tests_20260513/duration_by_model.svg",        # B1
    "figures/challenge_swarm_local_tests_20260513/artifact_coverage_heatmap.svg",# B2
    "figures/odelia_single_site_eval/UKA_1DC_training_curves.svg",    # B3
]


def render_markdown() -> str:
    total_core = 15
    secs = sum(s["seconds"] for s in SLIDES[:total_core])
    out = [
        "# ODELIA Swarm Learning &mdash; Results Presentation (Speaker Script)",
        "",
        "> Companion script for `docs/presentation_swarm_results.html`. "
        "Open the HTML in any browser, present full-screen, advance with the arrow keys, "
        "and press **S** to show these notes inline.",
        "",
        f"**Length:** {total_core} core slides + 3 backup &middot; "
        f"core target ~{secs//60} min {secs%60}s at a normal speaking pace. "
        "All figures are committed under `docs/figures/`; the two charts on slides 8 and 9 "
        "are generated by this script.",
        "",
        "| # | Slide | Figure to show |",
        "| --- | --- | --- |",
    ]
    for n, (s, fp) in enumerate(zip(SLIDES, SLIDE_FIG_PATH), start=1):
        tag = "backup" if s["kind"] == "figure" and n > 15 else str(n)
        fig = fp if fp else "&mdash; (text slide)"
        title = s["title"].replace("&mdash;", "—").replace("&times;", "×").replace("&amp;", "&")
        out.append(f"| {tag} | {title} | {fig} |")
    out.append("")
    out.append("---")
    out.append("")
    for n, (s, fp) in enumerate(zip(SLIDES, SLIDE_FIG_PATH), start=1):
        title = (
            s["title"]
            .replace("&mdash;", "—").replace("&times;", "×")
            .replace("&amp;", "&").replace("&ge;", "≥").replace("&ne;", "≠")
        )
        out.append(f"## Slide {n} &mdash; {title}")
        if s.get("subtitle"):
            sub = s["subtitle"].replace("&mdash;", "—").replace("&times;", "×").replace("&middot;", "·")
            out.append(f"*{sub}*")
        out.append("")
        if fp:
            out.append(f"**Figure:** `{fp}`")
        else:
            out.append("**Figure:** none (text slide)")
        out.append(f"  ")
        out.append(f"**Target:** ~{s['seconds']}s")
        out.append("")
        out.append("**Speech:**")
        out.append("")
        out.append("> " + s["notes"])
        out.append("")
    return "\n".join(out)


def main() -> None:
    html_path = DOCS / "presentation_swarm_results.html"
    md_path = DOCS / "presentation_swarm_results_SPEAKER_NOTES.md"
    html_path.write_text(render_html(), encoding="utf-8")
    md_path.write_text(render_markdown(), encoding="utf-8")
    core = sum(s["seconds"] for s in SLIDES[:15])
    print(f"wrote {html_path.relative_to(REPO)}  ({html_path.stat().st_size//1024} KB)")
    print(f"wrote {md_path.relative_to(REPO)}  ({md_path.stat().st_size//1024} KB)")
    print(f"slides: {len(SLIDES)} ({15} core + {len(SLIDES)-15} backup)")
    print(f"core target: ~{core//60} min {core%60}s")


if __name__ == "__main__":
    main()
