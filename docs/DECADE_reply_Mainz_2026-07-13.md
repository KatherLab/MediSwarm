# Reply to Christina Glasner (UM Mainz / Mainz_1) — 2026-07-13 (short)

**Subject:** Re: DECADE — you found two real bugs; new kit attached

Dear Christina,

You were right to be suspicious. Two of the three were **genuine bugs**, and you
found them.

**1. `max_epochs = 1` — a real bug.** Not intentional. Local training was using the
preflight's 1-epoch default instead of `STAMP_MAX_EPOCHS` (32), so every site's
"baseline" was effectively untrained — and that baseline is exactly what we compare
the swarm model against. Fixed; local training now runs 32 epochs.

**2. The "modules in eval mode" warning — a worse bug.** Our prediction callback
switched the model to `eval()` and never switched it back, so from the **second
epoch onward** (and in the swarm, from the second round onward) the model trained
with **dropout disabled**. We measured it directly:

| epoch | before | after |
|---|---|---|
| 0 | train | train |
| **1** | **eval** ❌ | train ✅ |
| **2** | **eval** ❌ | train ✅ |

Fixed. This would have quietly degraded every model we trained — thank you.

**3. The rsync warnings — expected.** The live-sync helper pushes only **logs and
metrics** (never data or features) to our monitoring host, so we can see what
happened at your site when comparing baselines. It's non-fatal by design. You saw
warnings because your SSH key wasn't authorized yet — it is now, so you should see
`SSH OK — live sync enabled`. Say the word if you'd rather we disable it.

**4. Target — agreed.** Your discussion with Sebastian is exactly what we needed.
Since binary dMMR/pMMR and MSI-High carry the same information, and you have
**719 slides / 569 patients**, we're proposing **MSI-High (binary)** to the whole
consortium as the shared target, and dropping the mixed six-value `dMMR` column.
The open question is Bonn, whose labels are germline syndrome.

**Please take the new kit** (attached, `Mainz_1_1.5.0-dev.260713.6bc06c4.zip`) and
re-run **Steps 2, 3 and 5**. Besides the two fixes above, it contains one that
matters specifically to you: **multi-slide patients previously loaded 0 patients** —
if you were using a slide table, you were silently training on nothing (verified:
0 → 15 in our test). Keep `sudo` out of it (or use `sudo -E`).

The run is postponed (see the consortium mail), so there's no time pressure.

Best regards,
Jeff
