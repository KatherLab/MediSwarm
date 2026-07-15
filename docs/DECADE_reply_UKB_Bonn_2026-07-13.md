# Reply to Islem Gammoudi (UKB Bonn / UKB_1) — 2026-07-13 (short)

**Subject:** Re: DECADE — please stop your client; new kit attached

Dear Islem,

Thank you — UKB_1 was the first site fully ready, and your reports fixed real bugs
for everyone. Two things, and I'm sorry about the first one.

**1. Please stop your client.** The run is postponed (see the consortium mail), so
the server is deliberately not up and your client is retrying into nothing:

```bash
docker ps                      # find the stamp_swarm_client_UKB_1_* container
docker stop <that container>
```

**2. You need the new kit** (attached, `UKB_1_1.5.0-dev.260713.6bc06c4.zip`).
Please discard `…a295c6b`. Two fixes since then affect you — local training only
ran **1 epoch** (so your baseline was effectively untrained), and the model was
training with **dropout disabled** after the first epoch. Also, the new kits carry
fresh certificates, so the old one can no longer connect.

When you have a moment (no rush): unpack it and re-run **Step 5** — it now trains
32 epochs and gives us a real baseline. Please don't start the client again until
we send the go-ahead.

**One question.** Your target is **germline syndrome** (Lynch / FAP / sporadic,
3 classes). Mainz can only provide **MSI status** (binary) — a different biological
question, and swarm learning needs all four sites on the *same* label. We're
therefore proposing MSI as the shared target:

> **Do you have MSI status (MSI-High vs. MSS/MSI-Low), or dMMR/pMMR — and how many
> slides/patients per class?**

Your germline question is a good one and we'd like to return to it as a follow-up
study once the first run has proven the setup.

Apologies for the extra round-trip after you had everything working.

Best regards,
Jeff
