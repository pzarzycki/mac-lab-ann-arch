## A shared block in depth
A looped model repeatedly applies the same parameterized block:

$$
H^{(k+1)}=F_\theta(H^{(k)},X,k),\qquad k=0,\ldots,K-1.
$$

$H^{(k)}\in\mathbb R^{n\times d}$ is a sequence of hidden states after loop $k$; $X$ is input conditioning. The dependence on $k$ may be explicit or omitted by a particular architecture. $K$ changes computation depth without multiplying the shared block's parameter count.

## Historical and contemporary anchors
Universal Transformers combined self-attention with depth recurrence and adaptive per-position halting. A later recurrent-depth language-model study investigates scaling test-time latent computation through a repeated block. These share a computational idea but differ in training, architecture and stopping policies. [1](#ref-universal) [2](#ref-recurrent)

## More loops are not a theorem of better reasoning
Convergence to a fixed point requires assumptions. A trained map can oscillate, diverge or reach a stable but unhelpful state. A standard Transformer checkpoint cannot be assumed to support arbitrary repeated layers simply because the tensors have matching shapes.

The lab’s contraction mode displays repeated applications of a small **toy contraction**, $z_{k+1}=\tanh(0.6z_k+0.4u)$. Its convergence illustrates recurrence only; the curve is not a benchmark for an actual looped language model.

```python
h = initialize(tokens)
for loop in range(compute_budget):
    h = shared_block(h, input_condition=tokens, step=loop)
logits = readout(h)
```

## Training and systems questions
Backpropagation through unrolled loops can increase activation memory. Truncation or other training methods change the estimator. KV-cache semantics also require explicit design: caches indexed by depth and caches indexed by emitted token are not interchangeable.

## Evaluation
Plot accuracy versus total FLOPs and latency across loop counts, including counts outside training. Compare with a larger unshared baseline and with extra output-token reasoning. Report stopping overhead. Distinguish looping a hidden computation from [Coconut-style](#/article/soft-latent-reasoning) feedback through a token interface. [3](#ref-coconut)

## What is shared, and when does computation stop?
The upper figure keeps the sequence axis visible: the three rows of $H^{(0)},H^{(1)},H^{(2)}$ represent the **same three tokens**, each with $d$ features. The intervening copies of $F_\theta$ use the same parameters, including attention across token rows. Input conditioning can enter each iteration; readout comes after a chosen number of loops. Only emitting a new token advances sequence position. The architecture drawing is schematic; the scalar plot below it is a separately specified dynamical example.

| Choice | Universal Transformer | Recurrent-depth study |
| --- | --- | --- |
| Shared computation | Self-attention and transition repeated over depth | A four-layer recurrent core between two-layer prelude and coda |
| Initial state / conditioning | Token embeddings; position and depth signals | Sampled latent state; prelude features injected into each core iteration |
| Halting | Adaptive computation can halt separately at each position | Fixed loop budgets, plus an evaluated per-generated-token exit rule |
| Training / evaluation depth | Task-dependent fixed-depth and adaptive variants | Random training depths around a recurrence target of 32; validation includes 1, 4, 8, 16, 32 and 64 loops |

The later model backpropagates through the last eight recurrent iterations. Its demonstrated exit rule stops when successive output distributions have KL divergence below $5\times10^{-4}$, then emits a token. These are specific design choices, not requirements of every looped model. [1](#ref-universal) [2](#ref-recurrent)

## Stability and stopping are different questions
The auxiliary lab retains the original contraction $z_{k+1}=\tanh(0.6z_k+0.4u)$, initialized at $z_0=-1$. Because $|\tanh'|\le1$, its Lipschitz bound is $0.6$. The dashed horizontal line marks a numerically approximated fixed point, not a target answer.

Two additional fully specified maps expose the caveat. $z_{k+1}=-z_k$ gives the bounded two-cycle $-1,1,-1,1,\ldots$; $z_{k+1}=1.15z_k$ grows in magnitude from this initialization. Neither map is claimed to reproduce a trained Transformer. The optional lab stopping rule tests $|z_{k+1}-z_k|<0.01$, with a hard loop budget. The display distinguishes meeting tolerance from exhausting that budget. A scalar tolerance is not the paper's KL rule, and reaching either sort of tolerance is not a proof of task correctness.
