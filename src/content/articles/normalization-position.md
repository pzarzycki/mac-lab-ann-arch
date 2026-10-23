## Normalization is a feature-wise operation
LayerNorm subtracts a token's feature mean and scales by its standard deviation. RMSNorm removes that centering step:

$$
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_{k=1}^d x_k^2+\epsilon}}.
$$

$x,\gamma\in\mathbb R^d$; $\epsilon$ is positive. RMSNorm and LN are not interchangeable names. Both usually normalize the last feature dimension in a Transformer implementation, independently for each token. [1](#ref-layernorm) [2](#ref-rmsnorm)

For comparison, LayerNorm uses the centered feature variance:

$$
\mu=\frac1d\sum_kx_k,\quad \sigma^2=\frac1d\sum_k(x_k-\mu)^2,\qquad
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
$$

$\gamma$ is a learned per-feature gain and $\beta$ a learned offset in the affine LN form; implementations can omit the bias. Before affine parameters, $x=[-1,1,2]$ has mean $2/3$, variance $14/9$, and mean square $2$. Ignoring the tiny $\epsilon$ only for this hand calculation, LN gives approximately $[-1.336,0.267,1.069]$ while RMSNorm gives $[-0.707,0.707,1.414]$. The panel uses $\epsilon=10^{-5}$ exactly. Adding the same scalar to every feature leaves centered LN unchanged, but generally changes RMSNorm. Learned gains and offsets mean the final outputs need not have unit RMS or zero mean. [1](#ref-layernorm) [2](#ref-rmsnorm)

## Position changes attention geometry
Without positional information or another source of ordering, self-attention does not intrinsically identify sequence positions. Rotary position embedding (RoPE) rotates pairs of query/key coordinates according to position. With a compatible block-diagonal rotation $R_i$,

$$
q_i'=R_i q_i,\qquad k_j'=R_j k_j,\qquad
(q_i')^\top k_j'=q_i^\top R_{j-i}k_j.
$$

The last identity expresses relative displacement in the dot product. Here $i,j$ are positions, not feature indices. RoPE generally modifies queries and keys, rather than adding a position vector to the residual stream. Its algebra does not guarantee arbitrary-length generalization. [3](#ref-rope)

For one pair, define the rotation at angular frequency $\omega$:

$$
R_i(\omega)=\begin{bmatrix}\cos(i\omega)&-\sin(i\omega)\\\sin(i\omega)&\cos(i\omega)\end{bmatrix}.
$$

In a head of query/key width $d_k$, apply these $2\times2$ rotations to coordinate pairs in an even-dimensional rotated subspace, with a frequency for each pair; the full $R_i$ is block diagonal. Since $R_i^\top R_j=R_{j-i}$, the intermediate step is $(R_iq_i)^\top(R_jk_j)=q_i^\top R_i^\top R_jk_j=q_i^\top R_{j-i}k_j$. The panel isolates one pair: $q=[1,0]$, $k=[0,1]$, $\omega=\pi/6$. At $i=0,j=1$ both sides equal $-1/2$. Shifting both positions by the same amount preserves this number while rotating both arrows. [3](#ref-rope)

## The position-wise network
A SwiGLU-style MLP can be written, using column-vector convention, as

$$
\operatorname{MLP}(x)=W_o\bigl[\operatorname{SiLU}(W_gx)\odot W_ux\bigr].
$$

$W_g,W_u\in\mathbb R^{m\times d}$, $W_o\in\mathbb R^{d\times m}$, and $\operatorname{SiLU}(z)=z\operatorname{sigmoid}(z)$. The intermediate width is $m$. This is a gated feature transform applied separately at each token with shared parameters. It does not supply sequence mixing. [4](#ref-swiglu)

The gate and up branches read the same $x$ **in parallel**, then multiply coordinate by coordinate before the down projection. In the toy panel, $x=[1,1]$ gives gate preactivations $[1,1,0]$, up values $[2,0,1]$, product approximately $[1.462,0,0]$, and down-projected output $[1.462,0]$ using the displayed matrices. A SiLU gate is not a probability in $[0,1]$: it can be negative or greater than one.

An ordinary bias-free FFN $W_2\phi(W_1x)$ has $2dm$ weights at intermediate width $m$; this bias-free SwiGLU has $3dm$. Thus equal parameter count calls for a gated width about two-thirds of the ordinary width, subject to implementation rounding. Equal width would spend more parameters. Comparisons of quality need a matched budget and controlled training setup. [4](#ref-swiglu)

## Implementation discipline
```python
# Conceptual block; projection layouts depend on tensor convention.
x = x + attention(rmsnorm_attn(x), rotary_positions)
x = x + down(silu(gate(rmsnorm_ffn(x))) * up(rmsnorm_ffn(x)))
```

Cache the second normalized value in a real implementation. Separate normalization placement, position encoding and MLP gating in ablations: changing all three prevents a clean attribution of improvement. LLaMA is a useful published example of this combination, not its single point of invention. [5](#ref-llama)

## Limits and reading
RMSNorm does not remove the need for stable initialization. A rotary scheme that supports longer arrays may still fail on longer reasoning tasks. And increasing MLP width changes parameter count as well as capacity. Compare one design choice at a time.
