# Differential Privacy Assessment for MediSwarm

## Executive Summary

MediSwarm currently uses `PercentilePrivacy` (Shokri & Shmatikov, CCS 2015)
as its privacy filter. This is **not formal differential privacy** — it is a
gradient sparsification and clipping heuristic that reduces information
leakage but provides no mathematical privacy guarantee (no epsilon/delta
budget). This document assesses the current state, identifies gaps, and
proposes a roadmap toward formal differential privacy.

## Current State

### PercentilePrivacy Filter

Both ODELIA and STAMP jobs apply `PercentilePrivacy` as a `task_result_filter`
on training results before they leave the client:

```hocon
{
  path = "nvflare.app_common.filters.percentile_privacy.PercentilePrivacy"
  args {
    percentile = 10
    gamma = 0.01
  }
}
```

**What it does:**

1. Normalises weight diffs by the number of local training steps
2. Computes the absolute value of all weight diffs
3. Zeroes out all diffs below the `percentile`-th percentile (10th percentile)
4. Clips remaining diffs to `[-gamma, +gamma]` ([-0.01, +0.01])
5. Rescales by the number of steps before transmission

**What it provides:**
- Reduces the volume of information transmitted (sparsification)
- Bounds the magnitude of any single weight update (clipping)
- Makes gradient inversion attacks harder (fewer non-zero values)

**What it does NOT provide:**
- No formal (epsilon, delta)-differential privacy guarantee
- No privacy budget tracking or accounting
- No composition theorem — privacy "cost" is unknown across rounds
- No noise injection — the mechanism is purely deterministic
- An adversary with sufficient auxiliary information could still
  extract training data properties from the transmitted updates

### SVT Privacy Filter (Available but Unused)

NVFlare ships `SVTPrivacy` (Sparse Vector Technique), which is closer to
formal DP:

```python
class SVTPrivacy(DXOFilter):
    def __init__(self, fraction=0.1, epsilon=0.1, noise_var=0.1,
                 gamma=1e-5, tau=1e-6, ...):
```

SVT adds Laplacian noise with scale derived from epsilon, making it a
differentially private mechanism. However:
- It operates on weight diffs (not per-sample gradients)
- The privacy guarantee applies to the **update release mechanism**, not
  the training process itself
- Privacy accounting across federated rounds is not implemented
- It is not currently used in any MediSwarm job

## Gap Analysis

| Aspect | Current State | Formal DP Requirement | Gap |
|--------|--------------|----------------------|-----|
| **Privacy definition** | Heuristic sparsification + clipping | (epsilon, delta)-DP | No formal definition |
| **Noise mechanism** | None (deterministic) | Gaussian or Laplacian noise calibrated to sensitivity | Missing |
| **Per-sample gradient clipping** | Not applied (PercentilePrivacy clips aggregated diffs) | Required for DP-SGD | Missing |
| **Privacy budget tracking** | None | Cumulative epsilon across rounds via composition | Missing |
| **Budget exhaustion** | N/A — no budget | Training stops when budget exhausted | Not implemented |
| **Sensitivity bound** | gamma=0.01 (arbitrary) | Derived from max gradient norm | Ad hoc |
| **Composition** | Unknown | Advanced composition / RDP / zCDP | Not tracked |
| **Audit / reporting** | None | Per-round epsilon reporting | Not implemented |

## Technical Options

### Option 1: Opacus DP-SGD (Client-Side)

**Approach:** Use [Opacus](https://opacus.ai/) to wrap the local optimizer
with DP-SGD (per-sample gradient clipping + calibrated Gaussian noise).

**NVFlare reference:** `examples/hello-world/hello-dp/client.py` shows a
complete Opacus integration:
- `PrivacyEngine.make_private_with_epsilon()` wraps model, optimizer, and
  dataloader
- Privacy budget is tracked across all federated rounds
- Cumulative epsilon is reported via `privacy_engine.get_epsilon(delta)`
- Model state dict needs `_module.` prefix handling for Opacus-wrapped models

**Compatibility analysis:**

| Pipeline | Compatible? | Notes |
|----------|------------|-------|
| ODELIA | Partial | Raw PyTorch training loop wrapped in Lightning — Opacus needs `GradSampleModule` which may conflict with Lightning's hooks. Would need custom training step. |
| STAMP | Difficult | STAMP 2.4.0 uses its own Lightning models with OneCycleLR scheduler. Opacus requires `DPOptimizer` which replaces the optimizer. Scheduler interaction is complex. |

**Challenges:**
1. **Lightning compatibility:** Opacus wraps the model in `GradSampleModule`
   which changes the model class hierarchy. Lightning's `Trainer.fit()` may
   not work seamlessly. Would need `opacus.lightning.DPLightningModule`.
2. **WEIGHT_DIFF transfer:** Opacus adds noise to gradients during training.
   The `WEIGHT_DIFF` filter would then operate on already-noised updates.
   PercentilePrivacy's clipping could destroy the calibrated noise, breaking
   the DP guarantee.
3. **Batch size constraints:** DP-SGD requires knowing the exact number of
   samples and uses Poisson sampling. STAMP's bag-based dataloader groups
   tiles per patient, which doesn't fit Opacus's per-sample paradigm.
4. **Performance:** DP-SGD typically requires 10-100x more epochs to
   converge to similar utility. For medical imaging where datasets are
   small, this utility loss may be unacceptable.

### Option 2: NVFlare SVTPrivacy Filter (Server-Side)

**Approach:** Replace `PercentilePrivacy` with `SVTPrivacy` in job configs.

**Advantages:**
- Drop-in replacement — same DXO filter interface
- Adds Laplacian noise calibrated to epsilon
- No code changes to training pipelines

**Limitations:**
- Provides (epsilon)-DP for the **update release**, not for the training
  process itself (the distinction matters for formal guarantees)
- No per-sample gradient clipping (operates on aggregated weight diffs)
- Privacy accounting across rounds must be implemented separately
- The epsilon parameter controls noise for a single round; composition
  across R rounds gives epsilon_total ≈ R * epsilon (basic composition)
  or sqrt(R) * epsilon (advanced composition)

### Option 3: Hybrid Approach (Recommended for v1.4.0)

**Approach:** Combine gradient clipping (already present via
PercentilePrivacy) with calibrated noise injection.

1. **Keep PercentilePrivacy** for gradient sparsification (reduce
   communication, remove small updates)
2. **Add Gaussian noise filter** after PercentilePrivacy that injects
   noise calibrated to the clipping bound (gamma=0.01)
3. **Implement privacy accountant** that tracks cumulative epsilon
   using Rényi Differential Privacy (RDP) composition

**Implementation sketch:**
```python
class CalibratedNoiseFilter(DXOFilter):
    """Add Gaussian noise calibrated to sensitivity and epsilon."""
    def __init__(self, sigma, sensitivity=0.01):
        # sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        self.sigma = sigma
        self.sensitivity = sensitivity

    def process_dxo(self, dxo, shareable, fl_ctx):
        for name in dxo.data:
            noise = np.random.normal(0, self.sigma, dxo.data[name].shape)
            dxo.data[name] += noise.astype(dxo.data[name].dtype)
        return dxo
```

**Advantages:**
- Minimal code change — add a filter, don't rewrite training
- Compatible with both ODELIA and STAMP pipelines
- Compatible with `WEIGHT_DIFF` transfer and swarm topology
- Privacy guarantee applies to the released update (output privacy)
- Can co-exist with FedProx

**Limitations:**
- Output-level DP, not record-level DP (weaker guarantee)
- Noise calibration depends on knowing the sensitivity (clipping bound)
- Does not protect against a malicious aggregator seeing raw local
  updates before the filter is applied (trusted aggregator assumption)

## Privacy Budget Accounting

For any DP mechanism, tracking cumulative privacy cost is essential.

### Composition Theorems

| Method | Epsilon after R rounds | Tightness |
|--------|----------------------|-----------|
| Basic composition | R * epsilon_per_round | Loose |
| Advanced composition | sqrt(2R * ln(1/delta')) * epsilon + R * epsilon * (e^epsilon - 1) | Better |
| RDP (Rényi DP) | Numerically tight via `opacus.accountants.RDPAccountant` | Best |
| zCDP | R * rho (where rho = epsilon^2 / 2) → convert back to (epsilon, delta) | Good |

### Recommended Accounting

Use RDP composition via Opacus's `RDPAccountant` (available even without
using DP-SGD):

```python
from opacus.accountants import RDPAccountant

accountant = RDPAccountant()
for round in range(num_rounds):
    accountant.step(
        noise_multiplier=sigma / sensitivity,
        sample_rate=1.0,  # entire dataset per round
    )
    epsilon = accountant.get_epsilon(delta=1e-5)
    logger.info(f"Round {round}: cumulative epsilon = {epsilon:.2f}")
```

## Recommendations

### Phase 1 (v1.3.0) — Documentation and Assessment (This PR)

- [x] Document current PercentilePrivacy limitations
- [x] Analyse Opacus DP-SGD compatibility with ODELIA and STAMP
- [x] Evaluate NVFlare's built-in SVTPrivacy filter
- [x] Propose hybrid approach for v1.4.0
- [x] Define privacy accounting requirements

### Phase 2 (v1.4.0) — Calibrated Noise + Accounting

- [ ] Implement `CalibratedNoiseFilter` as NVFlare DXO filter
- [ ] Add RDP privacy accountant to training pipelines
- [ ] Add per-round epsilon logging to metrics CSV
- [ ] Add `PRIVACY_EPSILON_TARGET` env var to halt training when budget exhausted
- [ ] Test utility impact on ODELIA Duke benchmark (with vs without noise)
- [ ] Test utility impact on STAMP classification

### Phase 3 (v1.5.0) — Record-Level DP (Optional)

- [ ] Evaluate Opacus integration with Lightning for ODELIA
- [ ] Evaluate per-bag DP for STAMP (treat each patient as one sample)
- [ ] Implement `DPLightningModule` wrapper
- [ ] Compare utility: output DP (Phase 2) vs record-level DP (Phase 3)

### Phase 4 (v2.0.0) — Formal Verification

- [ ] Third-party privacy audit
- [ ] Formal proof of privacy guarantee for the chosen mechanism
- [ ] Regulatory documentation (GDPR Art. 35 DPIA template)

## Decision Record

**Decision:** For v1.3.0, document the current state and plan. Do not
implement formal DP yet — the utility cost and engineering complexity
require careful evaluation with real clinical data.

**Rationale:**
1. PercentilePrivacy provides meaningful (if unquantified) protection
   against gradient inversion attacks
2. The swarm topology (peer-to-peer, no central server) already limits
   the attack surface compared to centralised FL
3. Formal DP (especially DP-SGD) would significantly degrade model utility
   on small medical imaging datasets
4. The hybrid approach (Phase 2) provides a quantifiable guarantee with
   minimal utility impact
5. Rushing to formal DP without utility evaluation could produce models
   too noisy to be clinically useful

**Stakeholders:** MediSwarm development team, clinical partners, data
protection officers at participating sites.

## References

1. Shokri & Shmatikov, "Privacy-Preserving Deep Learning", CCS 2015
   (PercentilePrivacy basis)
2. Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
   (DP-SGD, Opacus foundation)
3. Mironov, "Rényi Differential Privacy", CSF 2017
   (RDP composition)
4. NVFlare hello-dp example: `examples/hello-world/hello-dp/`
   (Opacus + NVFlare integration reference)
5. Dwork & Roth, "The Algorithmic Foundations of Differential Privacy",
   2014 (composition theorems)
6. Lyu et al., "SVT: The Sparse Vector Technique", 2017
   (SVTPrivacy basis)
