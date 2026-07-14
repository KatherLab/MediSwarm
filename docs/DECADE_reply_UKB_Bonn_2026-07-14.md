# Reply to Islem Gammoudi (UKB Bonn / UKB_1) — 2026-07-14

**Subject:** Re: DECADE — thanks; your MSI answer changes the target question

Dear Islem,

Thank you — and a very clean baseline: 32 epochs, val loss 0.350 → 0.031 on 1174
patients. That's exactly what we needed.

## Your MSI answer is the important part

To be explicit, because it changes the plan: **Bonn cannot currently supply MSI
status.** The `MSI-High` and `dMMR` columns exist but every slide reads
*not provided*. So the MSI target we proposed to the consortium is **not achievable
across all four sites** as things stand.

**One thing we should not do**, even though it's tempting: derive MSI from Lynch
status. You're right that Lynch patients are biologically dMMR/MSI-High — but the
converse doesn't hold. **Sporadic MSI-High tumours** (typically MLH1
hypermethylated) are common, and they would be labelled "MSI-negative" simply
because the patient has no germline variant. That would systematically mislabel a
large part of the negative class and quietly corrupt the model. Lynch-vs-sporadic
is a *germline* question; MSI is a *tumour* phenotype. They are genuinely different
labels, not two names for the same thing.

So: **please do check with Dr. Robert and Dr. Jacob** whether any real MSI or MMR
testing exists for your cohort (even for a subset — a partial cohort with a *true*
MSI label is far more useful than a full cohort with a derived one). That answer
now determines the consortium's target.

## Where this leaves us

| target | Bonn | Mainz | verdict |
|---|---|---|---|
| **MSI-High** (binary) | ❌ not provided | ✅ 569 patients (61 pos / 508 neg) | blocked on your answer |
| **Lynch vs. Sporadic** (binary) | ✅ 1174 | ❓ needs a germline label | possible alternative |

If Bonn genuinely has no MSI data, the honest options are (a) Mainz/Düsseldorf/
Heidelberg supply a germline label so we run Lynch-vs-sporadic, or (b) we run MSI
without Bonn — which we'd rather avoid. We'll decide once Düsseldorf (by Wed) and
Heidelberg have reported.

## Server

Not up yet, and won't be until the target is agreed — so **please leave your client
stopped**. I'll write to everyone with a date and a go-ahead; you won't have to
guess.

Your `Affected Gene in the Germline` (MLH1, MSH2, MSH6, PMS2, EPCAM, APC) and
adenoma dysplasia grades are noted — we may come back to those for the follow-up
study.

Thanks again for the speed and the care.

Best regards,
Jeff
