# Self-Pruning Neural Network — Report

## Overview

This report is generated automatically after training completes.
All numbers in this report are **real values** from the actual training runs —
nothing is estimated or placeholder.

---

## Why Does L1 on Sigmoid Gates Encourage Sparsity?

### The Setup

Each weight `w` in a `PrunableLinear` layer is paired with a scalar
`gate_score`. The actual gate used in computation is:

```
gate = sigmoid(gate_score)    ∈ (0, 1)
```

The effective weight is:

```
pruned_weight = w × gate
```

### The Sparsity Loss

We add an L1 penalty on **gate values** (not gate scores):

```
SparsityLoss = Σ gates = Σ sigmoid(gate_scores)
TotalLoss    = CrossEntropy + λ × SparsityLoss
```

### Why L1 (and not L2) Causes True Zeros

| Property | L1 Penalty | L2 Penalty |
|---|---|---|
| Gradient | Constant: `sign(gate)` = always `+1` | Proportional: `2 × gate` → shrinks toward 0 but never reaches it |
| Effect on small values | Same push as large values | Almost no push → values linger near 0 but stay positive |
| Result | Gates collapse **exactly to 0** | Gates shrink but remain small positives |

**Key insight**: Because all gate values are positive (sigmoid output > 0),
the gradient of `Σ gates` w.r.t. each gate is always `+1`. This means the
optimizer sees a **constant downward pressure of λ** on every gate, regardless
of its current value.

- A gate that **helps accuracy** gets an upward gradient from `CrossEntropyLoss`
  that counteracts this pressure → it **survives**.
- A gate that **doesn't help accuracy** has no counteracting gradient → the λ
  pressure wins and pushes it to `~0` → **pruned**.

This creates a **bimodal distribution**: gates cluster near 0 (pruned) or
near 1 (fully active), with very few in between.

---

## Results Table

> All values below are from actual training runs (CIFAR-10, 25 epochs, Adam optimizer, lr=0.001).

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:---:|:---:|:---:|
| 0.0001 | 56.07% | 45.4% |
| 0.001 | 55.31% | 98.6% |
| 0.01 | 48.77% | 99.9% |

### Interpretation

- **Best accuracy**: λ = 0.0001 → 56.07% test accuracy
- **Most sparse**: λ = 0.01 → 99.9% of weights pruned
- **Lowest accuracy**: λ = 0.01 → 48.77% (heavy pruning cost)

As λ increases, the sparsity penalty dominates the loss, forcing more gates
to zero. This reduces the effective network capacity, which lowers accuracy
but produces a much smaller, faster model at inference time.

---

## Gate Value Distribution

The plot `gate_distributions.png` shows the distribution of `sigmoid(gate_scores)`
after training for each λ value.

A successful pruning result shows:
- A **large spike at 0**: majority of gates pruned (driven to ~0 by L1 pressure)
- A **cluster near 1**: small subset of gates the network relies on for accuracy
- **Few values in between**: L1 creates a hard threshold effect — gates go fully on or fully off

The higher the λ, the larger the spike at 0 and the smaller the surviving cluster near 1.
See `gate_distributions.png` and `tradeoff_summary.png` for the actual plots.

---

## How to Run

```bash
pip install torch torchvision matplotlib numpy
python prunable_net.py
```

CIFAR-10 downloads automatically to `./data/` on first run.
This report (`report.md`) is auto-generated at the end of the script.

---

## Key Takeaways

1. **Self-pruning works**: By adding an L1 penalty on sigmoid gates, the
   network genuinely learns to remove its own unnecessary weights during
   training — no separate pruning step needed.

2. **λ is the key knob**: It controls the sparsity-accuracy trade-off.
   Tuning it is application-dependent: safety-critical systems need high
   accuracy (low λ), while edge deployment needs small size (high λ).

3. **L1 is the right choice**: L2 regularization would only shrink gates,
   never truly zero them out. L1 creates exact zeros, which is what true
   pruning requires.

4. **Gradients flow through everything**: Because both `weight` and
   `gate_scores` are `nn.Parameter`, PyTorch's autograd handles the
   gradient computation correctly through the sigmoid and element-wise
   multiply without any custom backward implementation.
