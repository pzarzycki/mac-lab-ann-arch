## A state for every token
Let a sequence contain $n$ tokens, with model width $d$. At depth $\ell$, token $i$ has its own hidden state $h_i^{(\ell)}\in\mathbb R^d$. Stacking the vectors gives $H^{(\ell)}\in\mathbb R^{n\times d}$. The rows are distinct token states; the matrix notation is a batched implementation, not a single shared token vector.

The **residual stream** is the evolving state carried along the skip paths. A **residual connection**, or identity skip connection when dimensions match, adds a sublayer's output to that state. The vector is commonly denoted $h$; the vector space is $\mathbb R^d$. Calling it “embedding space” is reasonable for the common coordinate width, but deeper states are contextual representations, rather than the original lookup embeddings. [1](#ref-transformer)

## A modern pre-norm decoder block
For each token separately, define

$$
u_i=\operatorname{LN}(h_i^{(\ell)}),\qquad
\widetilde h_i=h_i^{(\ell)}+\operatorname{Attn}_i(u_1,\ldots,u_i),
$$
$$
h_i^{(\ell+1)}=\widetilde h_i+\operatorname{MLP}(\operatorname{LN}(\widetilde h_i)).
$$

Here $u_i$ is a normalized token vector, $\widetilde h_i$ the state after attention, and $\operatorname{Attn}_i$ the output assigned to destination token $i$. The causal restriction allows source positions $j\le i$. Layer indices on the learned functions are suppressed.

**Attention mixes information across tokens.** The residual addition still happens independently at every token. The MLP uses shared weights across positions, yet applies to each token independently. Attention heads are another axis entirely: they produce features that are concatenated and projected back to width $d$.

The original 2017 Transformer used an encoder–decoder design and post-sublayer LayerNorm. The pre-norm decoder above is explicitly a later architectural pattern; it is not a reproduction of the original figure. [1](#ref-transformer) [4](#ref-llama)

## What LN means
**LN is Layer Normalization**. For one token vector $x\in\mathbb R^d$,

$$
\mu=\frac1d\sum_{k=1}^d x_k,\quad
\sigma^2=\frac1d\sum_{k=1}^d(x_k-\mu)^2,\quad
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
$$

$\gamma,\beta\in\mathbb R^d$ are learned scale and offset; $\epsilon>0$ is a numerical stabilizer; $\odot$ is elementwise multiplication. Statistics are computed across features **within that token**, separately for every batch item and position. LN does not average over other tokens. [3](#ref-layernorm)

## Read the vector figure
Select a destination token to highlight its causal attention inputs. Switch between pre-LN and post-LN to see where normalization moves. Each colored vertical path is one token's residual state; bypass paths preserve the same token identity. The drawing shows one head for legibility and labels the omitted multi-head projection. It is a structural figure, not measured activation data.

The companion inspector isolates the additions using synthetic width-3 vectors. For token 3, start with $h=[1,-1,0.5]$, an attention write $\Delta a=[0.5,0.25,-0.5]$, and an MLP write $\Delta m=[-0.25,0.5,1]$. Coordinate-wise addition gives

$$
h+\Delta a=[1.5,-0.75,0],\qquad
h+\Delta a+\Delta m=[1.25,-0.25,1].
$$

Changing only the first coordinate of $\Delta a$ changes only the first coordinate of these sums. This is an arithmetic intervention with the other writes held fixed. In a learned network, changing an earlier state can also change later attention and MLP writes; the inspector does not claim those computations remain fixed naturally.

```python
# H: [batch, positions, d_model]; normalize the last dimension only
U = norm_attention(H)
H = H + causal_multihead_attention(U)
H = H + mlp(norm_mlp(H))
# Each addition is elementwise at the same batch/position/feature index.
```

## Why the identity path matters
For $y=x+F(x)$, the Jacobian is $I+J_F$. This introduces a direct gradient route, a key contribution of residual learning popularized by ResNet. It does not guarantee stable optimization for arbitrary depth, initialization or normalization. ResNet is a central architectural predecessor, not an incidental analogy. [2](#ref-resnet)

The normalization toggle changes the scope of that identity route. For one sublayer, the two placements are

$$
y_{\mathrm{pre}}=x+F(\operatorname{LN}(x)),\qquad
y_{\mathrm{post}}=\operatorname{LN}(x+F(x)).
$$

Their input Jacobians are $I+J_FJ_{\mathrm{LN}}$ and $J_{\mathrm{LN}}(I+J_F)$, respectively, with each Jacobian evaluated at its own input. Pre-LN leaves a direct identity summand outside the normalization branch; post-LN passes the whole residual sum through normalization before the next block. This local comparison is not a guarantee about gradients through an entire trained model, which may also include a final normalization. Normalization placement has distinct initialization-gradient behavior under the assumptions studied by Xiong and colleagues. [5](#ref-xiong-layernorm) See [normalization, position and gated MLPs](#/article/normalization-position) for the feature transformations themselves.

## Questions worth testing
Remove the residual path while keeping parameter count comparable; inspect gradient norms by layer. Then vary normalization placement and initialization separately. A change in training stability does not by itself explain the semantic content of the residual stream. For an inference optimization with unchanged learned states, continue to [KV-cache memory](#/article/kv-cache).
