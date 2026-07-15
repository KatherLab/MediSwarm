# Consortium hub — how partner communication works now

Weekly emails carrying a fresh kit + fresh instructions produced too many kit
versions and too many instruction variants, with no single place to answer *"which
kit am I on, what do I run, and when is the run?"*

**Why kits churned:** the build version was embedded in the NVFlare project *name*,
so every image build minted a new root CA and new per-site private keys — silently
invalidating every deployed kit. That is fixed (`test_kit_cert_stability.py` guards
it). A kit now survives our software updates, because the backbone, the training code
and the job configs all live in the **image**, which `docker.sh` re-pulls on every run.

## The three surfaces

| Surface | Lives in | Changes |
|---|---|---|
| **Handbook** (Google Doc) | `PARTNER_HANDBOOK.md` here | Rarely. How to prepare and run a node. |
| **Board** (Google Sheet) | the 4 `sheet_*.csv` here | Every run. Date, ACTIVE kit, per-site checklist. |
| **Kits** | delivered per site | Rarely now. Encrypted; contain a private key. |

## Publishing

1. **Doc** ← paste `PARTNER_HANDBOOK.md`.
2. **Sheet** ← import the four CSVs, one per tab:
   - `sheet_1_run_schedule.csv` — set the date + the exact image tag for the run
   - `sheet_2_kit_registry.csv` — rows come from `kit_manifest.csv` (emitted by the build); mark superseded kits **DEPRECATED**
   - `sheet_3_site_checklist.csv` — 8 sites × 10 steps; **sites tick their own rows**
   - `sheet_4_known_issues.csv` — cumulative. **Append, never rewrite** — this is the debugging memory.
3. Share the Doc + Sheet once. Then stop emailing instructions: change them here.

## Shipping a new software version — no kit needed

Publish the image, then put its tag in the Sheet's run-schedule row. Sites pick it up
on their next run. To move a site explicitly:
```bash
echo 'MEDISWARM_IMAGE=jefftud/odelia:<tag>' > startup/image.conf
```
The monitor flags any site running a different image from the rest of the swarm, so
skew is visible instead of silent.

## Kit delivery

The build emits, per site:
- `<SITE>_<version>.zip.enc` — AES-256, safe to put on a shared link
- `kit_passwords.txt` — **mode 600, gitignored. Deliver each password to its own site only, out of band.** Never share the file.
- `kit_manifest.csv` — sha256 per kit → paste into the kit registry so sites can verify.

## ⚠️ Before pointing partners at any of this

- **The live monitor is not partner-safe.** No auth, no per-site scoping — any VPN-reachable
  viewer can read every site's logs. Do not link it.
- **DECADE/STAMP kits have no pre-run checks** (`master_template_STAMP.yml` has no
  `_preflight_host_checks` block), yet several docs claim they do. This handbook is
  ODELIA-only for that reason; do not hand it to a DECADE site as-is.
