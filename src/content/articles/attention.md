## The operation
For normalized token states $U\in\mathbb R^{n\times d}$, form $Q=UW_Q$, $K=UW_K$, and $V=UW_V$. For one head, $Q,K\in\mathbb R^{n\times d_k}$ and $V\in\mathbb R^{n\times d_v}$. Then

$$
A=\operatorname{softmax}_{\mathrm{row}}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad O=AV.
$$

$M_{ij}=0$ for allowed source positions and $-\infty$ otherwise. A causal decoder allows $j\le i$. Every row of $A$ is a distribution over allowed source positions for one destination. The scale $\sqrt{d_k}$ controls the magnitude of dot products under common initialization assumptions. [1](#ref-transformer)

## Multiple heads
Head $a$ has its own projections. Concatenating the outputs and applying $W_O$ returns the model width. Heads are not token streams, and the output projection can mix head features before the residual addition. Bahdanau attention provided an influential learned alignment mechanism in recurrent translation; the Transformer made attention the central sequence-mixing operation. [2](#ref-bahdanau)

## The interactive mask
Choose causal, sliding-window causal, or full attention. The grid displays **connectivity**, not learned attention weights. In a sliding window of width $w$, destination $i$ can access $\max(0,i-w+1)\le j\le i$. The plotted edge count is computed from this rule. Full attention is useful for encoders and denoisers, but using it unchanged in next-token training would leak future targets.

```python
scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
scores = scores.masked_fill(~allowed, float('-inf'))
A = scores.softmax(dim=-1)
O = A @ V
```

## Evidence and limits
Attention weights depend on the input, projection parameters and head. A large coefficient alone is not a causal attribution of the final prediction: values, output projections, residual pathways and later layers also matter. The lab deliberately avoids fabricated weight heatmaps.

## Experiment
Compare dense and windowed masks on a synthetic task with a dependency outside the window. Hold parameter count, training tokens and evaluation lengths fixed. Report both task accuracy and actual wall-clock time; fewer edges do not automatically imply faster kernels. Continue to [exact and sparse attention](#/article/efficient-attention).
