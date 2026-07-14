# Reply to Christian Zoellner (Heidelberg / UKHD_1) — 2026-07-14

**Subject:** Re: DECADE — `--GPU` is fine; your `$SCRATCHDIR` is empty

Dear Christian,

You're not missing anything — `--GPU device=0` **is** valid. The error message is
lying to you, and that's our bug.

## What actually happened

Your `$SCRATCHDIR` was never exported (or is empty). The shell expands it to
nothing, so this:

```bash
./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training
```

is what *you* typed, but what `docker.sh` actually receives is:

```bash
./docker.sh --scratch_dir --GPU device=0 --dummy_training
```

`--scratch_dir` then swallows `--GPU` as its value, and `device=0` is left over as
an unrecognised argument — hence `Unknown parameter passed: device=0`, blaming the
one flag that was correct.

## Fix on your side (30 seconds)

```bash
export SCRATCHDIR=/path/to/a/writable/scratch/folder
mkdir -p $SCRATCHDIR
echo "SCRATCHDIR=[$SCRATCHDIR]"          # must NOT be empty

./docker.sh --scratch_dir $SCRATCHDIR --GPU device=0 --dummy_training 2>&1 | tee dummy_training_console_output.txt
```

Quick check before any step — if either prints `[]`, that's your problem:

```bash
echo "DATADIR=[$DATADIR] SCRATCHDIR=[$SCRATCHDIR]"
```

Also: please **don't** run `docker.sh` with `sudo` — it discards these variables
entirely (a different site lost an afternoon to that one).

## Fix on our side

The script now refuses an empty value and says so plainly:

```
❗ Option --scratch_dir requires a value, but got '--GPU'.
   This usually means a shell variable is empty or was never exported.
```

That will be in the next kit — your current one is fine, just export the variable.

## While you're here

Whenever you get a chance, we still need from Heidelberg:

1. **MSI status** — can you provide MSI-High vs. MSS/MSI-Low (directly, or derived
   from dMMR/pMMR)? How many slides and patients per class? What's the exact column
   name and its values?
2. Your **SSH public key** (`ssh-keygen -t ed25519 -C "$(hostname)@mediswarm"` →
   send `~/.ssh/id_ed25519.pub`).
3. Confirmation that **Tailscale** is connected and `ping dl3.tud.de` works.

No rush on the run itself — it's postponed until the target is settled.

Thanks for reporting this; the error message was genuinely misleading and is now
fixed for everyone.

Best regards,
Jeff
