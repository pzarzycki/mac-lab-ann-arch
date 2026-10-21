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

The lab displays repeated applications of a small **toy contraction**, $z_{k+1}=\tanh(0.6z_k+0.4u)$. Its convergence illustrates recurrence only; the curve is not a benchmark for an actual looped language model.

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
