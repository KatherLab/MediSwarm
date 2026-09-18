# Adversarial robustness in ODELIA swarm learning

Working document for **D3.4** (M48, WP3, type R — report, PU): *"Report describing
potential adversarial attacks and the subsequent measures needed to make SL-based AI
fully robust against such attacks."* Tracking issue: #529.

This is the evidence base, not the deliverable prose.

> **Scope.** D3.4 is about attacks on the model's *behaviour*. T3.3 (#530) and MS6
> (#531) are about *privacy* — noise, epsilon budgets, reconstruction of training data.
> Same work package, different deliverables. Do not let one absorb the other.

## 1. What makes the swarm case different

In single-institution training the party holding the data is the party training the
model. In ODELIA neither is true:

- **Every participant is a peer.** There is no privileged party that can inspect a
  contribution against ground truth, because no party holds the pooled data.
- **In CCWF one participant *is* the aggregator** for a round. The role rotates. So an
  adversary is not merely a contributor; periodically it is the thing combining
  everyone else's contributions.
- **Contributions are weights, not data.** A poisoned update is a plausible-looking
  tensor. There is no image to inspect and no label to check.

This is why a generic adversarial-ML review does not answer D3.4. The attack surface is
the aggregation step.

## 2. The current exposure

Every ODELIA and STAMP job aggregates with `InTimeAccumulateWeightedAggregator` — a
size-weighted mean of client updates. A weighted mean has a **breakdown point of zero**:
there is no bound on how far one participant can move it.

The `PercentilePrivacy` filter currently wired into every job does **not** help. It
clips small diffs for privacy reasons; it does not bound the norm of a contribution, so
it neither provides a privacy guarantee (see `DIFFERENTIAL_PRIVACY.md`) nor limits
influence.

Measured, at `scripts/adversarial/attack_simulation.py`:

| Attacker | Attack scale | Weighted mean error | Norm-clipped error |
|---|---|---|---|
| Barcelona (190 volumes, 0.6 % of data) | ×1 | 0.03 | 0.04 |
| Barcelona | ×1000 | **5.52** | **0.04** |
| Aachen (17,834 volumes, 51.7 % of data) | ×1000 | **518.02** | **1.19** |

Error is `‖aggregate − pooled optimum‖ / ‖pooled optimum‖`. **The smallest site in the
consortium can move the global model to five times the target's own magnitude away from
it.** Data share does not protect the aggregate; it only changes how hard an attacker
has to push.

## 3. Why the textbook fix does not fit ODELIA

The standard answer is a Byzantine-robust aggregation rule — trimmed mean,
coordinate-wise median, Krum. All three are implemented in
`application/jobs/_shared/custom/robust_aggregation.py`, and all three are the wrong
choice here.

They share one property that matters more than their breakdown points: **they cannot
honour contribution weights.** Robustness comes from order statistics over
contributions, and re-weighting the survivors by dataset size would hand back exactly
the influence the trim removed. So they target the *unweighted* mean of site optima.

For a consortium of comparable sites that is a mild cost. ODELIA's sites are not
comparable — sizes span 94-fold, and Aachen holds 83.9 % of all benign cases:

| Class distribution | no-lesion | benign | malignant |
|---|---|---|---|
| Pooled (34,462 volumes) | 0.531 | **0.409** | 0.059 |
| Mean of the 8 sites | 0.660 | **0.204** | 0.136 |

An unweighted rule aims at a distribution that under-represents benign by half and
over-represents malignant by more than double.

**Cost with no attacker present**, relative error against the pooled optimum:

| Rule | Error | |
|---|---|---|
| `weighted_mean` (status quo) | **0.023** | |
| `norm_clipped_mean` | **0.032** | keeps weights |
| `unweighted_mean` *(diagnostic, not a robust rule)* | 0.401 | |
| `trimmed_mean` | 0.466 | |
| `coordinate_median` | 0.600 | |
| `krum` | 0.639 | selects one site; picked Utrecht |

Decomposing the trimmed mean's 0.466: **+0.379 from losing the weights, +0.064 from the
trimming itself.** The weights are 86 % of the cost. It is not that these rules discard
Aachen — a per-coordinate trim does not discard whole sites — it is that they cannot
count it for more than Barcelona.

## 4. Recommendation

**Adopt norm-bounding with a published clip, plus per-round monitoring of contribution
norms.** It is the only rule measured here that bounds an adversary's influence while
keeping every participant and their weight. Its no-attacker cost (0.032) is close to the
status quo (0.023), against 0.401+ for every order-statistic rule.

**Do not adopt trimmed mean, median or Krum for ODELIA as it stands.** They are designed
for exchangeable contributions. ODELIA's are not.

### The limitation that must be stated in the deliverable

Norm clipping bounds influence **proportionally to weight**. It does not defend against
the majority contributor: a poisoning Aachen still holds 51.7 % of the weight after
clipping, which takes the attack from 518× down to 1.19 — bounded, but still a corrupted
model.

That residual risk is not solvable by an aggregation rule. It needs a cap on any single
site's weight share, which is a **consortium governance decision with a real accuracy
cost**, not a config change. The trade-off belongs in the report.

## 5. What is not covered yet

- **This is a simulation over modelled updates, not trained models.** It establishes the
  ordering of the rules and the mechanism behind it. It does not give the accuracy a
  real ODELIA run would lose to a given clip — that needs one training run per clip
  value, and should be costed against the measured 2.2 h per round.
- **Colluding adversaries.** Every measurement here is a single adversary. No rule in
  this module detects a colluding majority.
- **The aggregator role.** CCWF rotates aggregation among participants. What a malicious
  *aggregator* can do — as opposed to a malicious contributor — is not measured and is
  the most swarm-specific part of the attack surface.
- **Backdoor and targeted attacks.** Everything here is untargeted model poisoning
  measured by displacement from the pooled optimum. A backdoor that leaves overall
  accuracy intact would not register.
- **Data poisoning at source.** Out of scope for the aggregation layer, but it belongs
  in the report's threat model.

## 6. Reproducing

```bash
python3 scripts/adversarial/attack_simulation.py --trials 200
# writes workspace/adversarial/attack_simulation.json
```

Dependency-free and deterministic under `--seed`. Site sizes and class counts are the
measured values from each site's own trainer; the script asserts they sum to 34,462
before running, so the figures cannot drift from the per-site record in
`presentation_data_by_site.html`.

## Related

| Doc | Holds |
|---|---|
| `AGGREGATION_STRATEGIES.md` | FedAvg/FedProx/FedOpt/Scaffold — convergence, not robustness |
| `DIFFERENTIAL_PRIVACY.md` | T3.3, and why `PercentilePrivacy` carries no guarantee |
| `SWARM_FAILURE_MODES.md` | F1–F10, accidental failures rather than adversarial ones |
