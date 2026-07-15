# Reply to Christina Glasner (UM Mainz / Mainz_1) — 2026-07-10

**Subject:** Re: DECADE — checks + 2 values before Monday 13 July

Dear Christina,

Great report — you found a real trap, and the answer to the bolded paragraph is
short: **it's `sudo`.**

## Why `echo` works but the training doesn't

`sudo` resets the environment. So although `STAMP_CLINI_TABLE` is exported in
*your* shell (which is why `echo` prints it), `sudo ./docker.sh …` starts
`docker.sh` with a **clean environment**, it sees no `STAMP_*` variables, forwards
none into the container, and training dies with `KeyError: 'STAMP_CLINI_TABLE'`.

We measured it on one of our clients:

| command | `STAMP_*` variables visible to `docker.sh` |
|---|---|
| `sudo ./docker.sh …` | **0** |
| `sudo -E ./docker.sh …` | all of them |
| `./docker.sh …` (no sudo) | all of them |

### Fix — pick either

**Preferred** (run Docker without `sudo`; one-time setup, then log out/in):

```bash
sudo usermod -aG docker $USER
newgrp docker                      # or just re-login
./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check
```

**Fallback** (keep `sudo`, preserve the environment):

```bash
sudo -E ./docker.sh --data_dir $DATADIR --scratch_dir $SCRATCHDIR --GPU device=0 --preflight_check
```

Quick check of what `docker.sh` will actually forward: `env | grep '^STAMP_'`.

Thanks to your report, the **new kit now detects this** and prints exactly what to
do instead of a Python traceback.

## Multi-slide patients — yes, use a slide table

Exactly right, you need one. Your `clini_table.csv` has one row per **patient**,
while `features/` has one `.h5` per **slide**. A slide table maps between them so
STAMP pools all of a patient's slides into a single bag:

```bash
export STAMP_SLIDE_TABLE="/data/Mainz_1/slide_table.csv"
export STAMP_PATIENT_LABEL="PATIENT"     # patient-ID column (both tables)
export STAMP_FILENAME_LABEL="FILENAME"   # slide-filename column (slide table)
```

`slide_table.csv` — one row per slide:

| PATIENT | FILENAME |
|---------|----------|
| P_001   | slide_001.h5 |
| P_001   | slide_002.h5 |
| P_002   | slide_003.h5 |

`FILENAME` must match the files in your `features/` directory. (Use whatever your
real column names are and set `STAMP_PATIENT_LABEL` / `STAMP_FILENAME_LABEL`
accordingly — column names with spaces are fine in the new kit.)

## New startup kit

Please use the new **`Mainz_1`** kit attached
(`Mainz_1_1.5.0-dev.260710.a295c6b.zip`) — you are currently on an older one
(`…260706.9b0148b`). Besides the `sudo` diagnostic, it also fixes reading features
extracted with STAMP 2.5.0 and column names containing spaces.

## SSH key

Received and **authorized** — nothing further needed.

## Prediction target — not settled yet

Thank you for the concrete options; this is the piece we still have to align
across all four centres, because swarm training requires **every site to train
the identical model** (same classes, in the same order).

Two observations we'd like your view on:

1. **`MSI-High` (binary, yes/no, 678 slides)** looks like the most harmonizable
   target — it is the standard, widely available CRC label. Bonn has proposed
   germline-syndrome labels (Lynch / familial / sporadic), which are a *different*
   kind of label from a tumour MSI/MMR phenotype, so we cannot simply merge them.
2. Your **`dMMR` column mixes two things**: MMR *status* (`pMMR`, `dMMR`) and the
   *specific protein loss* (`MLH1`, `MSH6`, `MLH1, PMS2`, `MSH2, MSH6`). As six
   classes over 103 slides it would be very sparse. If we go this route we would
   almost certainly collapse it to **binary dMMR vs pMMR**.

So: **do you also have MSI status and/or a clean binary dMMR/pMMR column?** That
would let us line the centres up on one target. We will confirm
`STAMP_GROUND_TRUTH_LABEL` + `STAMP_NUM_CLASSES` to everyone once Düsseldorf and
Heidelberg reply.

## Next steps for you

1. Unpack the new `Mainz_1` kit.
2. Re-run Step 2 **without `sudo`** (or with `sudo -E`).
3. Steps 3 and 5 should then work; hold the final Step 5 until we confirm the
   target.

A call is not necessary unless you'd still like one — the `sudo` change should
unblock you immediately. Sorry for the lost afternoon, and thank you for the
precise output; it let us fix this for every centre.

Best regards,
Jeff
