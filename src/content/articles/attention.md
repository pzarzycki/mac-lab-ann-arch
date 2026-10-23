## The operation
For normalized token states $U\in\mathbb R^{n\times d}$, form $Q=UW_Q$, $K=UW_K$, and $V=UW_V$. For one head, $Q,K\in\mathbb R^{n\times d_k}$ and $V\in\mathbb R^{n\times d_v}$. Then

$$
A=\operatorname{softmax}_{\mathrm{row}}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad O=AV.
$$

$M_{ij}=0$ for allowed source positions and $-\infty$ otherwise. A causal decoder allows $j\le i$. Every row of $A$ is a distribution over allowed source positions for one destination. The scale $\sqrt{d_k}$ controls the magnitude of dot products under common initialization assumptions. [1](#ref-transformer)

With row-vector token states, $W_Q,W_K\in\mathbb R^{d\times d_k}$ and $W_V\in\mathbb R^{d\times d_v}$. The attention coefficients form $A\in\mathbb R^{n\times n}$ and the output is $O\in\mathbb R^{n\times d_v}$. A **query** is a destination's read request; a **key** is a source's matching feature; a **value** carries the content to combine. Explicitly,

$$
O_i=\sum_{j\text{ allowed}} A_{ij}v_j.
$$

The scaling rationale assumes roughly independent, zero-mean, unit-variance query/key coordinates: a sum of $d_k$ coordinate products then has variance about $d_k$. Dividing by $\sqrt{d_k}$ keeps score variance near one. Without this adjustment large score differences can push softmax toward saturation, where its gradients become small. Learned coordinates need not satisfy these assumptions exactly. [1](#ref-transformer)

## Work through one row
The numeric panel starts at destination 3 with synthetic $q_3=[\sqrt2,0]$, keys $[1,0],[0,1],[1,1],[-1,0]$, and values $[1,0],[0,2],[2,1],[-1,1]$. Scaled scores are $[1,0,1,-1]$. The causal mask changes the fourth to $-\infty$, so

$$
A_{3,:}=\frac{[e,1,e,0]}{2e+1}\approx[0.422,0.155,0.422,0],\qquad
O_3=\left[\frac{3e}{2e+1},\frac{2+e}{2e+1}\right]\approx[1.267,0.733].
$$

The output is a mixture of value coordinates, not a selection of the largest score alone. Edit the query to change its match against every key; edit source 3's value to change its contribution without changing any coefficient. The displayed numbers are computed from these toy inputs and are not empirical model attention.

## Multiple heads
Head $a$ has its own projections. Concatenating the outputs and applying $W_O$ returns the model width. Heads are not token streams, and the output projection can mix head features before the residual addition. Bahdanau attention provided an influential learned alignment mechanism in recurrent translation; the Transformer made attention the central sequence-mixing operation. [2](#ref-bahdanau)

For $h$ heads of value width $d_v$, concatenation has shape $n\times(hd_v)$ and $W_O\in\mathbb R^{hd_v\times d}$, yielding an $n\times d$ residual write. Self-attention takes queries, keys and values from the same sequence; cross-attention takes queries from the destination sequence and keys/values from a source sequence, so its coefficient matrix can be rectangular. [1](#ref-transformer)

## The interactive mask
Choose causal, sliding-window causal, or full attention. The grid displays **connectivity**, not learned attention weights. In a sliding window of width $w$, destination $i$ can access $\max(0,i-w+1)\le j\le i$. The plotted edge count is computed from this rule. Full attention is useful for encoders and denoisers, but using it unchanged in next-token training would leak future targets.

```python
scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
scores = scores.masked_fill(~allowed, float('-inf'))
A = scores.softmax(dim=-1)
O = A @ V
```

## Evidence and limits
Attention weights depend on the input, projection parameters and head. A large coefficient alone is not a causal attribution of the final prediction: values, output projections, residual pathways and later layers also matter. The graph panel displays deterministic connectivity; the numeric panel displays explicitly synthetic, calculated coefficients. Neither presents invented measurements of a trained model.

## Experiment
Compare dense and windowed masks on a synthetic task with a dependency outside the window. Hold parameter count, training tokens and evaluation lengths fixed. Report both task accuracy and actual wall-clock time; fewer edges do not automatically imply faster kernels. Continue to [exact and sparse attention](#/article/efficient-attention).
