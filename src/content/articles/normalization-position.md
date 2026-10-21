## Normalization is a feature-wise operation
LayerNorm subtracts a token's feature mean and scales by its standard deviation. RMSNorm removes that centering step:

$$
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_{k=1}^d x_k^2+\epsilon}}.
$$

$x,\gamma\in\mathbb R^d$; $\epsilon$ is positive. RMSNorm and LN are not interchangeable names. Both usually normalize the last feature dimension in a Transformer implementation, independently for each token. [1](#ref-layernorm) [2](#ref-rmsnorm)

## Position changes attention geometry
Without positional information or another source of ordering, self-attention does not intrinsically identify sequence positions. Rotary position embedding (RoPE) rotates pairs of query/key coordinates according to position. With a compatible block-diagonal rotation $R_i$,

$$
q_i'=R_i q_i,\qquad k_j'=R_j k_j,\qquad
(q_i')^\top k_j'=q_i^\top R_{j-i}k_j.
$$

The last identity expresses relative displacement in the dot product. Here $i,j$ are positions, not feature indices. RoPE generally modifies queries and keys, rather than adding a position vector to the residual stream. Its algebra does not guarantee arbitrary-length generalization. [3](#ref-rope)

## The position-wise network
A SwiGLU-style MLP can be written, using column-vector convention, as

$$
\operatorname{MLP}(x)=W_o\bigl[\operatorname{SiLU}(W_gx)\odot W_ux\bigr].
$$

$W_g,W_u\in\mathbb R^{m\times d}$, $W_o\in\mathbb R^{d\times m}$, and $\operatorname{SiLU}(z)=z\operatorname{sigmoid}(z)$. The intermediate width is $m$. This is a gated feature transform applied separately at each token with shared parameters. It does not supply sequence mixing. [4](#ref-swiglu)

## Implementation discipline
```python
# Conceptual block; projection layouts depend on tensor convention.
x = x + attention(rmsnorm_attn(x), rotary_positions)
x = x + down(silu(gate(rmsnorm_ffn(x))) * up(rmsnorm_ffn(x)))
```

Cache the second normalized value in a real implementation. Separate normalization placement, position encoding and MLP gating in ablations: changing all three prevents a clean attribution of improvement. LLaMA is a useful published example of this combination, not its single point of invention. [5](#ref-llama)

## Limits and reading
RMSNorm does not remove the need for stable initialization. A rotary scheme that supports longer arrays may still fail on longer reasoning tasks. And increasing MLP width changes parameter count as well as capacity. Compare one design choice at a time.
