# Aggregation Strategies in MediSwarm

This document evaluates federated aggregation strategies available in NVFlare 2.7.2
and their applicability to MediSwarm's swarm learning pipelines (ODELIA and STAMP).

## Current Setup

Both ODELIA and STAMP jobs use **FedAvg** (Federated Averaging):

- **Aggregator**: `InTimeAccumulateWeightedAggregator` with `WEIGHT_DIFF` transfer
- **Workflow**: `SwarmServerController` / `SwarmClientController` (peer-to-peer swarm)
- **Privacy filter**: `PercentilePrivacy` (percentile=10, gamma=0.01) on client results
- **Topology**: Decentralised — one client acts as aggregator per round (no central server aggregation)

### How FedAvg Works in MediSwarm

1. Each client trains locally for N epochs
2. Weight **diffs** (not full weights) are sent to the aggregation client
3. `PercentilePrivacy` clips small diffs before transmission
4. The aggregator averages diffs weighted by contribution count
5. The averaged diff is applied to produce the new global model
6. The global model is broadcast to all clients for the next round

## Available Strategies

NVFlare 2.7.2 ships four aggregation strategies:

| Strategy | Server Change | Client Change | Complexity | Best For |
|----------|--------------|---------------|------------|----------|
| **FedAvg** | None (current) | None (current) | Baseline | Homogeneous data distributions |
| **FedProx** | None | Loss wrapper | Low | Heterogeneous data (non-IID) |
| **FedOpt** | New workflow | None | Medium | Large-scale, many rounds |
| **Scaffold** | New workflow + state | Gradient correction | High | Severe non-IID |

### 1. FedAvg (Current)

**How it works**: Simple averaging of model updates weighted by dataset size.

**Strengths**:
- Simple, well-understood
- No additional communication overhead
- Works with any model architecture

**Weaknesses**:
- Struggles with non-IID data distributions across sites
- Can diverge when local training runs many epochs
- No correction for client drift

**MediSwarm config** (current):
```hocon
{
  id = "aggregator"
  path = "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator"
  args {
    expected_data_kind = "WEIGHT_DIFF"
  }
}
```

### 2. FedProx (Recommended Next Step)

**How it works**: Adds a proximal term to the local loss function that penalises
deviation from the global model. This regularises local training to stay closer
to the global model, reducing client drift.

**Loss modification**: `L_local = L_original + (mu/2) * ||w_local - w_global||^2`

**Strengths**:
- Minimal code change — client-side loss wrapper only
- No aggregator or workflow changes needed
- Compatible with existing `WEIGHT_DIFF` transfer and swarm topology
- Proven effective for medical imaging with heterogeneous site data
- Controlled via single hyperparameter `mu`

**Weaknesses**:
- Requires tuning `mu` (typical range: 0.001 to 1.0)
- Adds computational overhead for proximal term calculation
- May slow convergence if `mu` is too large

**NVFlare implementation**: `nvflare.app_opt.pt.fedproxloss.PTFedProxLoss`

**How to enable in MediSwarm**: Set environment variable:
- ODELIA: `FEDPROX_MU=0.01` (or desired value; 0 = disabled)
- STAMP: `STAMP_FEDPROX_MU=0.01`

### 3. FedOpt (Server-Side Optimizer)

**How it works**: Applies a server-side optimizer (e.g., SGD with momentum, Adam)
to the aggregated update before applying it to the global model. This treats the
averaged client updates as a "gradient" for server-side optimisation.

**Strengths**:
- Can accelerate convergence with momentum
- Server-side learning rate schedule provides additional control
- Works with standard client training

**Weaknesses**:
- Requires replacing the workflow (`SwarmServerController` -> `FedOpt` controller)
- Incompatible with current swarm (peer-to-peer) topology — needs centralised server
- Additional hyperparameters: server LR, server momentum, LR scheduler
- Not directly compatible with `SwarmClientController`

**Assessment for MediSwarm**: **Not recommended for v1.3.0**. FedOpt requires
a centralised server workflow, which conflicts with MediSwarm's peer-to-peer
swarm topology. Would require fundamental architecture changes.

### 4. Scaffold (Variance Reduction)

**How it works**: Maintains control variates on both server and clients to correct
for client drift. Each client computes a correction term based on the difference
between its local gradient and the global gradient estimate.

**Strengths**:
- Theoretically optimal convergence for non-IID data
- Addresses client drift directly

**Weaknesses**:
- Doubles communication cost (must send control variates with model updates)
- Requires replacing entire workflow (`ScatterAndGatherScaffold`)
- Incompatible with swarm topology — needs centralised server
- Complex implementation with additional state management
- Not compatible with `WEIGHT_DIFF` transfer mode

**Assessment for MediSwarm**: **Not recommended**. Requires centralised server
and custom workflow, incompatible with swarm topology.

## Recommendation

### Short-term (v1.3.0): FedProx

FedProx is the clear choice for MediSwarm v1.3.0:

1. **Zero infrastructure change** — only adds a loss wrapper on the client side
2. **Compatible with swarm topology** — no server/workflow changes
3. **Compatible with WEIGHT_DIFF** — no transfer mode changes
4. **Compatible with PercentilePrivacy** — privacy filter is unaffected
5. **Configurable via environment variable** — can be enabled/disabled per deployment
6. **Medical imaging evidence** — published results show FedProx improves convergence
   for federated medical imaging tasks with heterogeneous site data

### Medium-term (v1.4.0+): Evaluate FedOpt

If MediSwarm adds a centralised server mode (alongside swarm mode), FedOpt
becomes viable and could provide additional convergence benefits.

## FedProx Configuration Guide

### Hyperparameter Selection

| Data Heterogeneity | Recommended `mu` | Notes |
|--------------------|-------------------|-------|
| Low (similar distributions) | 0.001 - 0.01 | Light regularisation |
| Medium (moderate differences) | 0.01 - 0.1 | Default starting point |
| High (very different distributions) | 0.1 - 1.0 | Strong regularisation |

### Guidelines

- **Start with `mu=0.01`** and adjust based on validation loss convergence
- **If training diverges**: Increase `mu` to keep local models closer to global
- **If convergence is slow**: Decrease `mu` to allow more local adaptation
- **mu=0** is equivalent to standard FedAvg (no proximal term)

### Environment Variables

```bash
# ODELIA pipeline
export FEDPROX_MU=0.01

# STAMP pipeline
export STAMP_FEDPROX_MU=0.01
```

## References

1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg), AISTATS 2017
2. Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx), MLSys 2020
3. Reddi et al., "Adaptive Federated Optimization" (FedOpt), ICLR 2021
4. Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020
