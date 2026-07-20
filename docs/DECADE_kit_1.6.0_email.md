# DECADE consortium email — v1.6.0 kits

**To:** Islem.Gammoudi@ukbonn.de; glasnerc@uni-mainz.de;
TobiasPaul.Seraphin@med.uni-duesseldorf.de; ckobrow@mailbox.org;
nina.nelius@med.uni-heidelberg.de

**Subject:** DECADE — new startup kits (v1.6.0), and the last kit swap you'll need

---

Dear all,

New DECADE startup kits (**v1.6.0**) are ready, along with a Handbook and a Run Board
that replace the instructions we have been sending by email.

**Handbook:** `<DOC_LINK>`

**Run Board:** `<SHEET_LINK>`

**Startup kits download link:** `<DRIVE_FOLDER_LINK>`

## Getting your kit

Open the download link above and take **only your own site's file**, e.g.
`UKB_1_1.6.0.zip`. The folder is members-only and each kit contains that site's private
key, so please don't repost it anywhere.

Check it matches the **Kit registry** tab of the Run Board. Each site's kit has its
**own** hash (it contains your certificates), so compare against the row for your site:

```bash
sha256sum <SITE>_1.6.0.zip
```

Unzip it — that's the whole install:

```bash
unzip <SITE>_1.6.0.zip
```

## Worth knowing

**Software updates now reach you automatically.** Your kit includes
`startup/image.conf` pointing at our release channel. When we publish a new version,
your node picks it up the next time you start it — nothing to install, and we never
pull or restart anything on your machine. The Run Board's **Run schedule** tab always
names the exact image a run uses (currently `jefftud/decade:1.6.0`).

**This is the last kit swap you will need.** Until now every rebuild regenerated your
certificates, which is why you have received a new kit almost weekly. That is fixed
from 1.6.0 onward — but because the fix changes how the certificates are stored,
moving to 1.6.0 requires one final switch. After this, a software fix reaches you
through the channel above, with no new kit.

**`--preflight_check` now actually checks your host.** Three of our documents promised
pre-run checks that the DECADE kit never ran — it verified nothing beyond loading your
data. It now checks that the GPU is usable inside the container, warns about a Docker
cgroup-driver setting that can strip the GPU from a running job, and (for
`--start_client`) checks it can reach the swarm server. **Please re-run
`--preflight_check` on the new kit** even if it passed before, and re-tick your row.

## Before the next run

Please work through your row in the **Site checklist** tab (Handbook §1–2 has the
commands) and tick it off. If something fails, check **Known issues** first — it is
cumulative, and the fix is usually already there.

Two notes on what we are still waiting for:

- **The shared target is not settled yet.** MSI-High is the front-runner (Mainz and
  Heidelberg can supply it). Islem — thank you for the update from Dr. Robert; we have
  recorded that UKB has no cohort-wide MSI and that Heidelberg is checking the adenoma
  subset. Tobias, we look forward to your numbers.
- Because of that, **no run date is set**. We will put it on the Run Board rather than
  send another email.

Thanks,
Jeff

---

## Operator notes (not part of the email)

Fill before sending:
- `<DOC_LINK>` / `<SHEET_LINK>` — the DECADE Handbook + Run Board (already published)
- `<DRIVE_FOLDER_LINK>` — members-only Drive folder holding the four `*_1.6.0.zip` kits

Do **not** attach kits to the email: one kit per site, shared folder only.
Do **not** link the live monitor — it has no auth and no per-site scoping.
