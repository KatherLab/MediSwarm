# Reply to Christina Glasner (UM Mainz / Mainz_1) — 2026-07-13 (follow-up)

**Subject:** Re: DECADE — you were right again: the rsync failures were real

Dear Christina,

You were right to push back on both. One of them I mis-reported, and the other
turned out to be a serious bug that you have now found for us. Thank you — and
sorry for telling you it was fixed when it wasn't.

## 1. The rsync warnings — a real bug, and a bad one

Not expected after all. Uploads were failing for **every site**, and the cause was
on **our** server, not yours.

Our upload account is protected by a restricted-rsync wrapper. It turns out that
wrapper takes an **exclusive lock on the whole upload area**, so only **one site in
the entire consortium can upload at a time**. Bonn's live-sync daemon was holding
that lock, and every upload from Mainz was refused:

```
rrsync error: Another instance of rrsync is already accessing this directory.
```

That is exactly the contradiction you spotted: `SSH OK — live sync enabled` (the
connection works) followed by a stream of rsync errors (every transfer refused).

Had we started the swarm run, we would have been **blind** — no logs, no
heartbeats, no checkpoints from any site but the first. Fixed on our monitoring
host; uploads now succeed, including from several sites at once. **Nothing for you
to do**, and no new kit is needed for this — it was purely server-side.

## 2. The "eval mode" warning — my previous answer was imprecise

There were **two** separate things, and I conflated them:

- **The real bug** — from the second epoch onward the model was training with
  **dropout disabled**. That *is* fixed in your kit; I verified it by measuring the
  module state at every training batch.
- **The warning you still see** — a *different* cause that I had not found: we
  validate the model before training it, which leaves it in eval mode, and Lightning
  then prints the warning at the start of `fit`. Lightning immediately puts the
  model back into training mode, so **your training is correct** — but the warning is
  noise and I should not have claimed it was gone.

Now fixed properly at the source (we restore training mode explicitly). It will
disappear in the next kit. In the meantime: **your current run is fine** — please
let it finish.

## 3. Everything else

Confirmed on your side: **32 epochs** — that fix landed. Together with the
multi-slide fix, your Step 5 is now a genuine baseline.

## 4. Your MSI numbers — thank you, this is exactly what we needed

Recorded:

| | |
|---|---|
| Column | `MSI-High` |
| Values | `yes` / `no` |
| Cohort | 719 slides, 569 patients |
| Split | **61 positive / 508 negative** patients |

So `STAMP_NUM_CLASSES=2`, `STAMP_GROUND_TRUTH_LABEL=MSI-High`.

One thing worth flagging now rather than after a run: at roughly **11 % positives**,
the classes are quite imbalanced. That is normal and workable for MSI, but it means
we should judge the model on **AUROC** rather than accuracy (a model that predicts
"no" for everyone would already be ~89 % "accurate"). We will keep an eye on it.

We are still waiting on Bonn (whose labels are germline syndrome, a different
question) and on Düsseldorf and Heidelberg before we can fix the target for
everyone.

Thanks again — that is now three real bugs from your reports, including one that
would have left us blind during the run.

Best regards,
Jeff
