# DECADE consortium hub — how partner communication works

This mirrors the ODELIA consortium hub (`docs/consortium/`) for the DECADE
computational-pathology (STAMP) project. Weekly emails carrying a fresh kit + fresh
instructions produced too many kit versions and no single place to answer *"which
kit am I on, what do I run, and when is the run?"* This folder is that single place.

## The three surfaces

| Surface | Lives in | Changes |
|---|---|---|
| **Handbook** (Google Doc) | `PARTNER_HANDBOOK.md` here | Rarely. How to prepare and run a DECADE node. |
| **Run Board** (Google Sheet) | the 4 `sheet_*.csv` here | Every run. Date, target, ACTIVE kit, per-site checklist. |
| **Kits** | delivered per site | On each fix, for now (see the cert note below). |

## Publishing

1. **Doc** ← paste `PARTNER_HANDBOOK.md` into a new Google Doc, share view-only.
2. **Sheet** ← import the four CSVs, one per tab:
   - `sheet_1_run_schedule.csv` — set the date, the **shared target**, and the image tag
   - `sheet_2_kit_registry.csv` — one row per site per kit, each with its own `sha256` (per-site certs → per-site hash); mark superseded kits **DEPRECATED**
   - `sheet_3_site_checklist.csv` — 4 sites × 9 steps; **sites tick their own rows** (pre-filled with what we know)
   - `sheet_4_known_issues.csv` — cumulative. **Append, never rewrite** — this is the debugging memory
3. Share the Doc + Sheet once. Then stop emailing instructions: change them here.

## What still needs filling in (the Sheet is a template)

- **Run schedule:** the **run date** (blocked on the target), and the **shared target**
  (`STAMP_NUM_CLASSES` + `STAMP_GROUND_TRUTH_LABEL`). See `docs/DECADE_SITE_REGISTRY.md`
  for who can supply what — MSI-High is the front-runner; Bonn is the only site that
  cannot yet, and Düsseldorf replies by 2026-07-22.

(The kit registry's per-site `sha256` values are now filled in from the built kits.)

## DECADE vs ODELIA — the real difference

Both consortia ship plain `.zip` kits now (ODELIA dropped its per-site encryption on
2026-07-16). The DECADE build does not emit a `kit_manifest.csv`, so compute the
`sha256` yourself with `sha256sum` for the registry.

- **DECADE kits still churn certificates on rebuild.** ODELIA has a cert-stability fix
  (#449) so its kits survive image updates; that fix is **not yet applied to STAMP**.
  Until it is, a new fix means a new ACTIVE kit that everyone must switch to (the old
  one can no longer authenticate). Track that on the Run Board.

## Shipping a software update without a new kit

If a fix lives in the **image** and does not touch the certificates, publish the image
and put its tag in the run-schedule row — sites pick it up on their next
`start_client` (`docker.sh` re-pulls). If a fix requires a rebuilt kit (new certs),
mark the new kit ACTIVE and the old one DEPRECATED.

## ⚠️ Before pointing partners at any of this

- **The live monitor is not partner-safe** — no auth, no per-site scoping. Do not link it.
- The current ACTIVE image is `jefftud/decade:1.5.0-dev.260713.6bc06c4` — validated
  end-to-end (partner steps, `Slide ID`, multi-slide, 32-epoch local, 2-node swarm +
  eval).
