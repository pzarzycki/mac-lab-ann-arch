# Artificial Neural Architectures — Atlas

## An interactive guide to architectures and learning systems

Editorial update: 22 October 2026. Original sources reviewed 21 October; embedding and interpretability additions reviewed 22 October.

42 articles · 6 research domains · 78 primary references.

This manuscript is the content source for the interactive site. It is a curated conceptual map, not an exhaustive literature review. Formulas define local conventions; theoretical claims, empirical findings and illustrative calculations are distinguished. Refer to docs/CONSPECT.md for the exploration map and coverage, and docs/PLAN.md for implementation decisions.

# Transformers

Follow token representations from embeddings and core mechanisms to extensions, efficient execution, and interpretability.

**Family:** Transformers → Core mechanisms

<a id="article-token-embeddings"></a>

## Token embeddings & embedding space

*Discrete identities, continuous coordinates, contextual states.*

**Abstract.** Follow token IDs into learned vectors, distinguish input and output embeddings, and inspect a small geometric example.

**At a glance:**

- Token lookup is not a numeric encoding of IDs
- Contextual states share a width, not one fixed meaning
- Weight tying is a choice; unembedding is not an inverse

**Concepts:** Token embeddings; Embedding space; Unembedding; Weight tying; Soft tokens.

### From a token ID to a vector
A tokenizer first maps text to discrete vocabulary IDs. An ID identifies a vocabulary entry, which may represent a word, subword, byte, punctuation mark, or special symbol. It is not a scalar measurement: ID 400 does not mean twice as much as ID 200. Tokenization determines the discrete interface; the embedding table determines how that interface enters the neural computation.

Choose column-vector notation. Let $V$ be vocabulary size and $d$ the residual width. The learned table is $E\in\mathbb R^{V\times d}$. Token ID $x_i$ selects a row, transposed to give $e_i=E_{x_i,:}^{\mathsf T}\in\mathbb R^d$. Equivalently, with a one-hot column $o_i\in\{0,1\}^V$, $e_i=E^{\mathsf T}o_i$. Frameworks implement a gather rather than constructing the one-hot tensor. The Transformer combines learned token representations with positional information; the exact position mechanism depends on the architecture. [1](https://arxiv.org/abs/1706.03762)

### Embedding space and the residual stream
The input lookup vectors occupy $\mathbb R^d$, often called the token embedding space. Later token states $h_i^{(\ell)}$ commonly retain width $d$, but become contextual: the same token ID can have different hidden states in different prompts and positions. Equal dimension does not mean that distances, directions, or semantic interpretations are constant across layers.

For learned absolute positions, an initial state might be $h_i^{(0)}=e_i+p_i$, with $p_i\in\mathbb R^d$. In a RoPE-based decoder, rotations are applied to queries and keys inside attention instead; the token lookup need not receive an additive position vector. Position, lookup identity, and contextual computation should therefore be drawn separately. The atlas's [residual-stream figure](#article-residual-streams) retains one state per token; its [normalization and position article](#article-normalization-position) develops those operations.

### Output embeddings, unembedding, and tying
Let $U\in\mathbb R^{V\times d}$ be an output projection, $b\in\mathbb R^V$ an optional bias, and $N_f$ the model's final normalization. Next-token scores are

$$
z_i=U N_f(h_i^{(L)})+b,\qquad p_i=\operatorname{softmax}(z_i).
$$

Each row of $U$ is an output direction associated with a vocabulary entry. **Weight tying** sets $U=E$ in this convention; untied models learn separate input and output matrices. Tying is an architectural/training choice, not a property of every language model. Press and Wolf analyze this sharing and its effects on language modeling. [2](https://arxiv.org/abs/1608.05859)

The unembedding is not generally an inverse of lookup. Multiple vectors can yield the same top token, softmax discards a common additive logit offset, and a contextual state contains more than the identity of its input token. Calling $U$ a decoder does not establish a one-to-one semantic coordinate system.

### A small geometric example
The figure uses four invented two-dimensional rows: $e_A=(1,0)$, $e_B=(0,1)$, $e_C=(-1,0)$ and $e_D=(0,-1)$. These are explanatory coordinates, not extracted model embeddings. Selecting A gives one table row. A soft mixture with coefficient $\alpha\in[0,1]$ instead gives

$$
v(\alpha)=\alpha e_A+(1-\alpha)e_B=(\alpha,1-\alpha).
$$

At $\alpha=1/2$, a tied, identity-normalization readout has logits $(1/2,1/2,-1/2,-1/2)$. A and B tie; the vector has not become a new vocabulary ID. A learned soft prompt can also lie outside the convex hull of the input table. Compare [soft prompts and continuous thought](#article-soft-latent-reasoning).

### Geometry needs an explicit metric
Cosine similarity measures angle; Euclidean distance also responds to norm. Neither metric guarantees synonymy, truth, or a human-readable feature axis. Shared learned features were already central to neural probabilistic language modeling before Transformers. [3](https://www.jmlr.org/papers/v3/bengio03a.html)

A two-dimensional projection can hide important differences in the remaining coordinates. State the source layer, normalization, centering, projection method, and reference distribution before interpreting a scatterplot. The **logit lens** asks what the output decoder reads from a hidden state, rather than finding its nearest input embedding; [layer readouts](#article-logit-tuned-lenses) explain the distinction.

### Implementation and gradients
```python
# E: [vocabulary, width]; ids: [batch, positions]
h0 = embedding(ids)              # [batch, positions, width]
h = transformer(h0)              # separate contextual state per token
logits = lm_head(final_norm(h))  # [batch, positions, vocabulary]
```

Integer IDs do not receive gradients. Lookup parameters do, and a frozen backbone can still propagate derivatives to trainable continuous inputs. Freezing parameters is different from disabling autograd. Check the actual model's embedding scale, position scheme, normalization, weight tying, and output bias before adapting this schematic to code.

### Primary references

1. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Research paper.
2. Ofir Press & Lior Wolf (2016). [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859). EACL 2017; preprint 2016.
3. Bengio, Ducharme, Vincent & Jauvin (2003). [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html). JMLR.

### Related topics

- [From symbols to distributed representations](#article-distributed-representations) — Trace the historical origin of distributed features.
- [The residual stream](#article-residual-streams) — Follow an embedding into one token’s contextual state.
- [Logit lens & tuned lens](#article-logit-tuned-lenses) — Read contextual states through the output decoder.
- [Soft prompts & continuous thought](#article-soft-latent-reasoning) — Compare vocabulary mixtures with learned continuous inputs.

---

**Family:** Transformers → Core mechanisms

<a id="article-residual-streams"></a>

## The residual stream

*One token. One evolving state.*

**Abstract.** A precise account of the hidden vectors that persist through a decoder, and the operations that read and write them.

**At a glance:**

- One width-d state per token
- Attention mixes tokens; MLPs act by position
- Residual lanes and attention heads are different axes

**Concepts:** Residual paths; Token states; Pre-LN; Post-LN.

### A state for every token
Let a sequence contain $n$ tokens, with model width $d$. At depth $\ell$, token $i$ has its own hidden state $h_i^{(\ell)}\in\mathbb R^d$. Stacking the vectors gives $H^{(\ell)}\in\mathbb R^{n\times d}$. The rows are distinct token states; the matrix notation is a batched implementation, not a single shared token vector.

The **residual stream** is the evolving state carried along the skip paths. A **residual connection**, or identity skip connection when dimensions match, adds a sublayer's output to that state. The vector is commonly denoted $h$; the vector space is $\mathbb R^d$. Calling it “embedding space” is reasonable for the common coordinate width, but deeper states are contextual representations, rather than the original lookup embeddings. [1](https://arxiv.org/abs/1706.03762)

### A modern pre-norm decoder block
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

The original 2017 Transformer used an encoder–decoder design and post-sublayer LayerNorm. The pre-norm decoder above is explicitly a later architectural pattern; it is not a reproduction of the original figure. [1](https://arxiv.org/abs/1706.03762) [4](https://arxiv.org/abs/2302.13971)

### What LN means
**LN is Layer Normalization**. For one token vector $x\in\mathbb R^d$,

$$
\mu=\frac1d\sum_{k=1}^d x_k,\quad
\sigma^2=\frac1d\sum_{k=1}^d(x_k-\mu)^2,\quad
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
$$

$\gamma,\beta\in\mathbb R^d$ are learned scale and offset; $\epsilon>0$ is a numerical stabilizer; $\odot$ is elementwise multiplication. Statistics are computed across features **within that token**, separately for every batch item and position. LN does not average over other tokens. [3](https://arxiv.org/abs/1607.06450)

### Read the vector figure
Select a destination token to highlight its causal attention inputs. Switch between pre-LN and post-LN to see where normalization moves. Each colored vertical path is one token's residual state; bypass paths preserve the same token identity. The drawing shows one head for legibility and labels the omitted multi-head projection. It is a structural figure, not measured activation data.

```python
# H: [batch, positions, d_model]; normalize the last dimension only
U = norm_attention(H)
H = H + causal_multihead_attention(U)
H = H + mlp(norm_mlp(H))
# Each addition is elementwise at the same batch/position/feature index.
```

### Why the identity path matters
For $y=x+F(x)$, the Jacobian is $I+J_F$. This introduces a direct gradient route, a key contribution of residual learning popularized by ResNet. It does not guarantee stable optimization for arbitrary depth, initialization or normalization. ResNet is a central architectural predecessor, not an incidental analogy. [2](https://arxiv.org/abs/1512.03385)

### Questions worth testing
Remove the residual path while keeping parameter count comparable; inspect gradient norms by layer. Then vary normalization placement and initialization separately. A change in training stability does not by itself explain the semantic content of the residual stream. For an inference optimization with unchanged learned states, continue to [KV-cache memory](#article-kv-cache).

### Primary references

1. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Research paper.
2. He, Zhang, Ren & Sun (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Research paper.
3. Ba, Kiros & Hinton (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450). Research paper.
4. Touvron et al. (2023). [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). Research paper.

### Related topics

- [Attention as content-addressed computation](#article-attention) — Cross-token communication.
- [Normalization, position & gated MLPs](#article-normalization-position) — Normalization inside the block.
- [Hyper-connections & mHC](#article-hyper-connections) — Extend the residual topology.
- [ResNet belongs in the main lineage](#article-resnet-lineage) — Trace the architectural influence.
- [Token embeddings & embedding space](#article-token-embeddings) — Inspect token lookup, embedding geometry and the output decoder explicitly.
- [Transformer observability & interpretability](#article-transformer-observability) — Inspect these computations and distinguish readouts from causal tests.

---

**Family:** Transformers → Core mechanisms

<a id="article-attention"></a>

## Attention as content-addressed computation

*Queries, keys, values—and the mask.*

**Abstract.** Derive one attention head, inspect exactly which token can read which other token, and distinguish weights from explanations.

**At a glance:**

- Queries select from keys; values carry content
- A mask specifies permitted reads
- Attention weights are input-dependent

**Concepts:** Self-attention; Cross-attention; Causal masks; QKV.

### The operation
For normalized token states $U\in\mathbb R^{n\times d}$, form $Q=UW_Q$, $K=UW_K$, and $V=UW_V$. For one head, $Q,K\in\mathbb R^{n\times d_k}$ and $V\in\mathbb R^{n\times d_v}$. Then

$$
A=\operatorname{softmax}_{\mathrm{row}}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad O=AV.
$$

$M_{ij}=0$ for allowed source positions and $-\infty$ otherwise. A causal decoder allows $j\le i$. Every row of $A$ is a distribution over allowed source positions for one destination. The scale $\sqrt{d_k}$ controls the magnitude of dot products under common initialization assumptions. [1](https://arxiv.org/abs/1706.03762)

### Multiple heads
Head $a$ has its own projections. Concatenating the outputs and applying $W_O$ returns the model width. Heads are not token streams, and the output projection can mix head features before the residual addition. Bahdanau attention provided an influential learned alignment mechanism in recurrent translation; the Transformer made attention the central sequence-mixing operation. [2](https://arxiv.org/abs/1409.0473)

### The interactive mask
Choose causal, sliding-window causal, or full attention. The grid displays **connectivity**, not learned attention weights. In a sliding window of width $w$, destination $i$ can access $\max(0,i-w+1)\le j\le i$. The plotted edge count is computed from this rule. Full attention is useful for encoders and denoisers, but using it unchanged in next-token training would leak future targets.

```python
scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
scores = scores.masked_fill(~allowed, float('-inf'))
A = scores.softmax(dim=-1)
O = A @ V
```

### Evidence and limits
Attention weights depend on the input, projection parameters and head. A large coefficient alone is not a causal attribution of the final prediction: values, output projections, residual pathways and later layers also matter. The lab deliberately avoids fabricated weight heatmaps.

### Experiment
Compare dense and windowed masks on a synthetic task with a dependency outside the window. Hold parameter count, training tokens and evaluation lengths fixed. Report both task accuracy and actual wall-clock time; fewer edges do not automatically imply faster kernels. Continue to [exact and sparse attention](#article-efficient-attention).

### Primary references

1. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Research paper.
2. Bahdanau, Cho & Bengio (2014). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). Paper · ICLR 2015.

### Related topics

- [The residual stream](#article-residual-streams) — Place attention inside the block.
- [Exact kernels, sparse patterns](#article-efficient-attention) — Compute the same or a sparser graph.
- [How modalities enter a Transformer](#article-multimodal-fusion) — Attend across modality boundaries.
- [Transformer observability & interpretability](#article-transformer-observability) — Inspect these computations and distinguish readouts from causal tests.

---

**Family:** Transformers → Core mechanisms

<a id="article-normalization-position"></a>

## Normalization, position & gated MLPs

*Three distinct jobs inside a block.*

**Abstract.** Separate state conditioning, positional geometry, and feature transformation in a contemporary decoder.

**At a glance:**

- LN and RMSNorm normalize features per token
- Position enters through distinct encodings
- Gated MLPs change the feature transformation

**Concepts:** LayerNorm; RMSNorm; RoPE; SwiGLU; Position encoding.

### Normalization is a feature-wise operation
LayerNorm subtracts a token's feature mean and scales by its standard deviation. RMSNorm removes that centering step:

$$
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_{k=1}^d x_k^2+\epsilon}}.
$$

$x,\gamma\in\mathbb R^d$; $\epsilon$ is positive. RMSNorm and LN are not interchangeable names. Both usually normalize the last feature dimension in a Transformer implementation, independently for each token. [1](https://arxiv.org/abs/1607.06450) [2](https://arxiv.org/abs/1910.07467)

### Position changes attention geometry
Without positional information or another source of ordering, self-attention does not intrinsically identify sequence positions. Rotary position embedding (RoPE) rotates pairs of query/key coordinates according to position. With a compatible block-diagonal rotation $R_i$,

$$
q_i'=R_i q_i,\qquad k_j'=R_j k_j,\qquad
(q_i')^\top k_j'=q_i^\top R_{j-i}k_j.
$$

The last identity expresses relative displacement in the dot product. Here $i,j$ are positions, not feature indices. RoPE generally modifies queries and keys, rather than adding a position vector to the residual stream. Its algebra does not guarantee arbitrary-length generalization. [3](https://arxiv.org/abs/2104.09864)

### The position-wise network
A SwiGLU-style MLP can be written, using column-vector convention, as

$$
\operatorname{MLP}(x)=W_o\bigl[\operatorname{SiLU}(W_gx)\odot W_ux\bigr].
$$

$W_g,W_u\in\mathbb R^{m\times d}$, $W_o\in\mathbb R^{d\times m}$, and $\operatorname{SiLU}(z)=z\operatorname{sigmoid}(z)$. The intermediate width is $m$. This is a gated feature transform applied separately at each token with shared parameters. It does not supply sequence mixing. [4](https://arxiv.org/abs/2002.05202)

### Implementation discipline
```python
# Conceptual block; projection layouts depend on tensor convention.
x = x + attention(rmsnorm_attn(x), rotary_positions)
x = x + down(silu(gate(rmsnorm_ffn(x))) * up(rmsnorm_ffn(x)))
```

Cache the second normalized value in a real implementation. Separate normalization placement, position encoding and MLP gating in ablations: changing all three prevents a clean attribution of improvement. LLaMA is a useful published example of this combination, not its single point of invention. [5](https://arxiv.org/abs/2302.13971)

### Limits and reading
RMSNorm does not remove the need for stable initialization. A rotary scheme that supports longer arrays may still fail on longer reasoning tasks. And increasing MLP width changes parameter count as well as capacity. Compare one design choice at a time.

### Primary references

1. Ba, Kiros & Hinton (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450). Research paper.
2. Zhang & Sennrich (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). Paper · NeurIPS 2019.
3. Su et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). Research paper.
4. Noam Shazeer (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202). Research paper.
5. Touvron et al. (2023). [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). Research paper.

### Related topics

- [The residual stream](#article-residual-streams) — Compare pre-norm and post-norm paths.
- [Attention as content-addressed computation](#article-attention) — Inspect position-sensitive interactions.
- [From Transformer to language-model scaling](#article-gpt-scaling) — See how design choices combine.

---

**Family:** Transformers → Core mechanisms

<a id="article-training-decoding"></a>

## Training, prefill & decoding

*The same weights, different computation schedules.*

**Abstract.** Understand teacher forcing, shifted targets and why inference becomes sequential even when training is parallel.

**At a glance:**

- Teacher forcing scores many causal targets in parallel
- Prefill creates state; decoding extends it
- SFT, RLHF and DPO change training procedures

**Concepts:** Next-token prediction; Teacher forcing; Prefill; SFT; RLHF; DPO.

### The autoregressive objective
For discrete tokens $x_1,\ldots,x_n$, a decoder models

$$
p_\theta(x_{1:n})=\prod_{t=1}^n p_\theta(x_t\mid x_{<t}),\qquad
\mathcal L_{\mathrm{NLL}}=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

$\theta$ denotes trainable parameters. A start token or other boundary convention supplies the first context. In teacher-forced training the full input is known, so causal masking computes predictions for many positions in one pass without exposing a target to its predictor. [1](https://arxiv.org/abs/1706.03762) [2](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

```python
inputs, targets = tokens[:, :-1], tokens[:, 1:]
logits = decoder(inputs, causal=True)
loss = cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
loss.backward()
optimizer.step()
```

Ignore padded positions in a real loss. An optimizer step usually aggregates a batch; gradient accumulation may aggregate several microbatches before that step.

### Prefill versus decode
**Prefill** processes the prompt and constructs cached keys/values. **Decode** adds generated positions, usually one per target-model step. Earlier tokens are available, but the next sampled token is unknown until its distribution has been evaluated. Caching reuses causal computations; it does not remove dependence on previous generated tokens. See [KV cache](#article-kv-cache).

### Instruction tuning and preferences
Supervised fine-tuning changes the data and loss mask, often scoring assistant completions rather than every input token. RLHF adds preference learning and policy optimization. DPO derives a preference objective relative to a reference policy; these are training procedures, not new kinds of attention. [3](https://arxiv.org/abs/2203.02155) [4](https://arxiv.org/abs/2305.18290)

For preferred/rejected responses $y_w,y_l$ to prompt $x$, define $r_\theta(y)=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)$. DPO uses $-\log\operatorname{sigmoid}[\beta(r_\theta(y_w)-r_\theta(y_l))]$, with scale $\beta>0$. This formula assumes a particular preference-model derivation; it is not a universal guarantee of truthfulness.

### What to measure
Distinguish training loss, held-out negative log-likelihood, instruction-following evaluation and generation latency. Perplexity is $\exp(\text{mean NLL})$ under a fixed tokenization and scoring convention. A lower number across incompatible tokenizers is not a valid comparison.

### Primary references

1. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Research paper.
2. Radford, Narasimhan, Salimans & Sutskever (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). Technical report.
3. Ouyang et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). Research paper.
4. Rafailov et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). Research paper.

### Related topics

- [KV-cache memory & grouped heads](#article-kv-cache) — Reuse inference state.
- [Alignment, generation & instruction](#article-multimodal-objectives) — Compare training objectives.
- [From Transformer to language-model scaling](#article-gpt-scaling) — Connect objectives to scaling.

---

**Family:** Transformers → Structural extensions

<a id="article-mixture-of-experts"></a>

## Mixture of experts

*Sparse activation is a routing problem.*

**Abstract.** Understand per-token expert selection, capacity, balancing and why total parameters differ from active compute.

**At a glance:**

- A router activates a subset of experts
- Active compute differs from total parameter count
- Load balance and communication affect execution

**Concepts:** MoE; Top-k routing; Load balancing; Sparse activation.

### Route a token to a subset
A mixture-of-experts (MoE) feed-forward block uses a router to select experts. With token state $h$, logits $z=W_rh$ and selected set $S(h)$,

$$
y(h)=\sum_{e\in S(h)}g_e(h)E_e(h).
$$

$E_e$ is expert $e$, and $g_e$ its routing weight. The lab uses softmax **renormalized over the selected top-$k$ logits**. This is an explicit educational convention; published systems differ in scoring, normalization, shared experts and balancing. Switch Transformers studied a simplified top-one scheme. [1](https://arxiv.org/abs/2101.03961)

### Sparse does not mean cheap in every sense
Experts increase stored parameters while activating a subset for each token. Routing can require all-to-all communication across devices, uneven loads, capacity management and extra buffers. “Active parameters” is not a complete FLOP or latency measure. Total parameter memory still matters unless experts are otherwise offloaded or partitioned.

```python
selected = topk(router_logits, k)
weights = softmax(router_logits[selected])
y = sum(weights[j] * experts[e](h) for j, e in enumerate(selected))
```

### Balance and specialization
If the router sends most tokens to one expert, the system wastes parallel capacity. Balancing losses, capacity policies and other mechanisms address this issue. DeepSeek-V3 provides another concrete design point with routed and shared experts and a distinct load-balancing approach. [2](https://arxiv.org/abs/2412.19437)

### Inspect the router
Move a router logit and change $k$. Only highlighted experts execute in the illustration. The numeric coefficients are computed from your inputs; the picture does not assert that experts have clean semantic labels or stable human-interpretable specializations.

### Evaluate the right quantities
Compare training compute, communication volume, inference latency at representative batch sizes, parameter storage and downstream quality. Report dropped tokens if capacity-limited routing drops them. A single “trillion parameter” label does not establish superiority over a dense baseline at matched cost.

### Primary references

1. Fedus, Zoph & Shazeer (2021). [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961). Paper · JMLR.
2. DeepSeek-AI et al. (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). Technical report.

### Related topics

- [Hyper-connections & mHC](#article-hyper-connections) — Compare two architectural axes.
- [Exact kernels, sparse patterns](#article-efficient-attention) — Contrast parameter and attention sparsity.
- [From Transformer to language-model scaling](#article-gpt-scaling) — Relate active compute to scaling.

---

**Family:** Transformers → Structural extensions

<a id="article-hyper-connections"></a>

## Hyper-connections & mHC

*Multiple residual lanes, inside each token.*

**Abstract.** Explore learned residual mixing and the doubly stochastic constraint used by manifold-constrained hyper-connections.

**At a glance:**

- Multiple residual lanes remain inside each token
- Learned mixing changes residual connectivity
- mHC constrains residual mixing toward doubly stochastic matrices

**Concepts:** Hyper-connections; mHC; Sinkhorn; Residual mixing.

### Widen the residual pathway
Conventional residual learning carries one width-$d$ state per token. Hyper-Connections introduce multiple residual lanes and learned maps that combine lanes into a sublayer input and redistribute its output. The lane axis is **within each token**. It is separate from the sequence axis and from attention heads. [1](https://arxiv.org/abs/1512.03385) [2](https://arxiv.org/abs/2409.19606)

Using a schematic column convention, let $X_i\in\mathbb R^{m\times d}$ collect $m$ lanes at token $i$. A generalized update is

$$
X_i'=R_iX_i+b_iF_i(\{a_j^\top X_j\}_{j\le i}).
$$

$R_i\in\mathbb R^{m\times m}$ mixes lanes, $a_j,b_i\in\mathbb R^m$, and $F_i$ returns a width-$d$ sublayer output. The cross-token dependence belongs to attention inside $F$; an MLP version is token-local. This notation summarizes connectivity rather than reproducing every published parameterization.

### Constrain the mixing
mHC constrains the residual mixing matrix to nonnegative, doubly stochastic matrices:

$$
R\mathbf1=\mathbf1,\qquad\mathbf1^\top R=\mathbf1^\top,\qquad R_{ab}\ge0.
$$

The resulting Birkhoff polytope is a convex polytope, despite the paper's broader “manifold” terminology. Its matrices mix lanes through convex combinations and are closed under composition. The residual mixer is non-expansive in spectral norm; this property is **not** a proof that the full nonlinear network preserves every norm or cannot diverge. [3](https://arxiv.org/html/2512.24880v2)

### A numerical construction
Sinkhorn-style row and column normalization can approximate these constraints from a positive matrix. A finite number of iterations leaves numerical error. Inspect the row and column sums in the lab as the iteration budget changes.

```python
R = exp(logits - logits.max())
for _ in range(iterations):
    R = R / R.sum(axis=0, keepdims=True)
    R = R / R.sum(axis=1, keepdims=True)
```

This example illustrates the constraint and uses a fixed small matrix; it is not the full mHC infrastructure implementation.

### Evidence and limits
The mHC paper combines connectivity constraints with system optimizations. Its empirical gains must be read with the training configuration and overhead. Multiple lanes cost memory traffic; the mathematical constraint alone does not eliminate that cost. Compare against both standard residual and unconstrained HC baselines.

### Primary references

1. He, Zhang, Ren & Sun (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Research paper.
2. Zhu et al. (2024). [Hyper-Connections](https://arxiv.org/abs/2409.19606). Research paper.
3. Xie et al. (2025). [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/html/2512.24880v2). Research paper · v2 Jan 2026.

### Related topics

- [The residual stream](#article-residual-streams) — Contrast a conventional residual stream.
- [Mixture of experts](#article-mixture-of-experts) — Keep routing and residual mixing distinct.
- [Backpropagation & biological plausibility](#article-backprop-biology) — Consider stable credit transmission.

---

**Family:** Transformers → Structural extensions

<a id="article-recurrent-depth"></a>

## Recurrent depth & looped Transformers

*Spend additional compute without adding new weights.*

**Abstract.** Separate recurrence over sequence positions from recurrence over a shared block in depth.

**At a glance:**

- A shared block can be applied repeatedly
- Depth recurrence differs from sequence recurrence
- More iterations need a stopping and stability argument

**Concepts:** Looped Transformers; Recurrent depth; Adaptive compute.

### A shared block in depth
A looped model repeatedly applies the same parameterized block:

$$
H^{(k+1)}=F_\theta(H^{(k)},X,k),\qquad k=0,\ldots,K-1.
$$

$H^{(k)}\in\mathbb R^{n\times d}$ is a sequence of hidden states after loop $k$; $X$ is input conditioning. The dependence on $k$ may be explicit or omitted by a particular architecture. $K$ changes computation depth without multiplying the shared block's parameter count.

### Historical and contemporary anchors
Universal Transformers combined self-attention with depth recurrence and adaptive per-position halting. A later recurrent-depth language-model study investigates scaling test-time latent computation through a repeated block. These share a computational idea but differ in training, architecture and stopping policies. [1](https://arxiv.org/abs/1807.03819) [2](https://arxiv.org/abs/2502.05171)

### More loops are not a theorem of better reasoning
Convergence to a fixed point requires assumptions. A trained map can oscillate, diverge or reach a stable but unhelpful state. A standard Transformer checkpoint cannot be assumed to support arbitrary repeated layers simply because the tensors have matching shapes.

The lab displays repeated applications of a small **toy contraction**, $z_{k+1}=\tanh(0.6z_k+0.4u)$. Its convergence illustrates recurrence only; the curve is not a benchmark for an actual looped language model.

```python
h = initialize(tokens)
for loop in range(compute_budget):
    h = shared_block(h, input_condition=tokens, step=loop)
logits = readout(h)
```

### Training and systems questions
Backpropagation through unrolled loops can increase activation memory. Truncation or other training methods change the estimator. KV-cache semantics also require explicit design: caches indexed by depth and caches indexed by emitted token are not interchangeable.

### Evaluation
Plot accuracy versus total FLOPs and latency across loop counts, including counts outside training. Compare with a larger unshared baseline and with extra output-token reasoning. Report stopping overhead. Distinguish looping a hidden computation from [Coconut-style](#article-soft-latent-reasoning) feedback through a token interface. [3](https://arxiv.org/abs/2412.06769)

### Primary references

1. Dehghani et al. (2018). [Universal Transformers](https://arxiv.org/abs/1807.03819). Paper · ICLR 2019.
2. Geiping et al. (2025). [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171). Research paper.
3. Hao et al. (2024). [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769). Paper · COLM 2025.

### Related topics

- [Selective state spaces & hybrids](#article-state-space) — Compare recurrence over time.
- [Soft prompts & continuous thought](#article-soft-latent-reasoning) — Connect recurrence to hidden computation.
- [Hyper-connections & mHC](#article-hyper-connections) — Compare residual topology.

---

**Family:** Transformers → Structural extensions

<a id="article-state-space"></a>

## Selective state spaces & hybrids

*Compress history into a state—or retain explicit token memory.*

**Abstract.** Understand the recurrence underlying selective state-space models and the trade-off with attention.

**At a glance:**

- A recurrent state compresses the past
- Selective dynamics depend on the input
- Hybrids combine recurrence and attention

**Concepts:** SSM; Mamba; Selective scan; Hybrids.

### A state update
A discrete state-space recurrence can be written

$$
s_t=\overline A_t s_{t-1}+\overline B_t x_t,\qquad y_t=C_t s_t.
$$

$x_t$ is current input, $s_t\in\mathbb R^{d_s}$ the recurrent state and $y_t$ the output. Shapes and channel structure vary by implementation. In a selective model, some effective parameters depend on the input; the model can choose what to retain or discard instead of applying one fixed linear filter. Mamba combines this selection with a hardware-aware scan and a specific architecture. [1](https://arxiv.org/abs/2312.00752)

### Different memory organizations
Attention can keep explicit keys and values for prior positions. A state-space recurrence compresses history into a state whose size is independent of sequence length for a fixed model. This can improve long-sequence scaling, but compression and explicit content-addressable retrieval have different inductive biases. A recurrent state is not a lossless substitute for arbitrary token memory.

### Hybrid designs
Combining attention layers with recurrent or state-space layers is a design space: retain some explicit retrieval while using compressed state elsewhere. This is a general architectural possibility, not a claim that one layout is optimal for all workloads. The number and placement of attention layers affect cache memory and retrieval capacity. [2](https://arxiv.org/abs/1706.03762)

```python
state = initial_state
for token in tokens:
    A, B, C = selective_parameters(token)
    state = A @ state + B @ token
    output = C @ state
```

The code is conceptual. Efficient implementations exploit structured parameterization and scans rather than arbitrary dense matrix products per token.

### Read the evidence carefully
The Mamba paper's task and throughput results belong to its tested models and hardware. Linear scaling in sequence length does not establish a universal latency advantage or unbounded exact recall. Compare parameter count, training data, retrieval tasks, context length and systems implementation together.

### Experiment
Use a long-range associative-recall task with distractors and variable delays. Test lengths beyond training. Measure state size alongside accuracy. Compare against attention at a matched memory or compute budget, rather than only matching model parameters.

### Primary references

1. Gu & Dao (2023). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752). Research paper.
2. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Research paper.
3. Geiping et al. (2025). [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171). Research paper.

### Related topics

- [Exact kernels, sparse patterns](#article-efficient-attention) — Compare memory and connectivity tradeoffs.
- [Liquid time-constant networks](#article-liquid-networks) — Contrast continuous-time dynamics.
- [Sequence models discover attention](#article-sequence-attention-history) — Trace recurrent ancestry.

---

**Family:** Transformers → Efficient execution

<a id="article-kv-cache"></a>

## KV-cache memory & grouped heads

*The persistent cost of an expanding context.*

**Abstract.** Calculate the cache footprint and understand MHA, MQA, GQA, paged storage and latent compression.

**At a glance:**

- Keys and values persist across decoding steps
- KV-head count changes memory scaling
- Cache state differs from parameters and external memory

**Concepts:** KV cache; GQA; MQA; MLA; PagedAttention.

### Why cache keys and values?
In a causal decoder, earlier positions cannot read later positions. With fixed weights and the same positional convention, their layerwise keys and values can therefore be reused when a new token arrives. Queries for the new position still read the retained context. A KV cache is intermediate inference state, not new model parameters. [1](https://arxiv.org/abs/1911.02150)

### The memory calculation
For a conventional full-context cache, assume equal key/value dimensions and one stored representation per KV head:

$$
M_{\mathrm{KV}}=2BLTH_{\mathrm{KV}}d_hs\quad\text{bytes}.
$$

$B$ is batch size, $L$ layer count, $T$ cached positions, $H_{\mathrm{KV}}$ KV heads, $d_h$ head dimension and $s$ bytes per scalar. The factor two counts keys and values. The calculator reports GiB using $2^{30}$ bytes. It excludes model weights, allocator overhead, temporary buffers, padding, quantization metadata and any architecture-specific extra states.

With $B=1,L=32,T=8192,H_{\mathrm{KV}}=8,d_h=128,s=2$, the result is **1 GiB**. Increasing KV heads to 32 makes it 4 GiB under exactly these assumptions.

### Share the KV representation
- **MHA:** separate KV heads for each query head.
- **MQA:** all query heads share one KV head.
- **GQA:** groups of query heads share an intermediate number of KV heads.

GQA changes the architecture's representation, rather than merely changing an allocator. The original work also studied converting multi-head checkpoints through additional training. [2](https://arxiv.org/abs/2305.13245)

### Orthogonal improvements
PagedAttention organizes cached tensors in blocks to reduce fragmentation and permit sharing; it does not replace GQA's mathematical projection. Multi-head latent attention (MLA), used in DeepSeek models, stores a compressed latent representation with an architecture-specific reconstruction and positional treatment. The simple formula above must not be reused as an exact MLA accounting model. [3](https://arxiv.org/abs/2309.06180) [4](https://arxiv.org/abs/2412.19437)

```python
cache_bytes = 2 * batch * layers * context * kv_heads * head_dim * bytes_per_scalar
cache_gib = cache_bytes / 2**30
```

### What a smaller cache does not prove
End-to-end latency also depends on memory bandwidth, kernel layout, scheduling, tensor parallelism and workload. Report time to first token separately from per-token decode latency and throughput. A cache estimate is not a GPU-memory fit guarantee.

### Primary references

1. Noam Shazeer (2019). [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). Research paper.
2. Ainslie et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). Paper · EMNLP 2023.
3. Kwon et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). Paper · SOSP 2023.
4. DeepSeek-AI et al. (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). Technical report.

### Related topics

- [Training, prefill & decoding](#article-training-decoding) — Separate prefill from decode.
- [Exact kernels, sparse patterns](#article-efficient-attention) — Distinguish state from kernel efficiency.
- [Memory, context & test-time adaptation](#article-memory-adaptation) — Compare different kinds of memory.

---

**Family:** Transformers → Efficient execution

<a id="article-efficient-attention"></a>

## Exact kernels, sparse patterns

*Two different routes to efficient attention.*

**Abstract.** Distinguish changing the implementation of dense attention from changing the set of permitted token interactions.

**At a glance:**

- FlashAttention changes IO while preserving exact attention
- Sparse masks change the connectivity graph
- Edge counts alone do not predict wall-clock speed

**Concepts:** FlashAttention; Sparse attention; Sliding windows; Long context.

### Compute versus materialization
Dense attention evaluates pairwise query–key scores. The $n\times n$ score matrix is expensive to store, but an implementation need not materialize the entire matrix in high-bandwidth memory. FlashAttention tiles the computation and uses online softmax bookkeeping to reduce memory traffic while computing the same dense attention function, up to floating-point differences. It is **exact attention**, not a sparse approximation. [1](https://arxiv.org/abs/2205.14135)

### A sparse graph changes the function
Windowed attention restricts each token to a neighborhood. Additional global tokens or other structured connections can restore longer-range paths. Longformer is a primary example of combining local and global connectivity. A decoder adaptation must still enforce causality; the original pattern should not be copied without considering its prediction objective. [2](https://arxiv.org/abs/2004.05150)

For a causal window of width $w$, the number of allowed edges is

$$
E(n,w)=\sum_{i=0}^{n-1}\min(i+1,w).
$$

For dense causal attention it is $n(n+1)/2$. These counts omit head and feature dimensions. They are structural work proxies, not a measured FLOP count or runtime prediction.

### Inspect the mask
The lab shows deterministic connectivity. Narrow the window and look at the last row: older tokens disappear from a single layer's direct receptive field. Multiple layers may propagate information farther, but the reachable graph and effective retrieval performance are separate questions.

### Implementation questions
```python
allowed = (source_position <= destination_position)
if window is not None:
    allowed &= (destination_position - source_position < window)
```

A sparse mask passed to a dense kernel may preserve dense compute. Efficient sparsity requires a layout and kernel that exploit its structure. Conversely, FlashAttention can improve dense execution without changing the permitted edges.

### Evidence and limits
Published speedups are tied to hardware, sequence length, precision and baseline. This atlas does not translate one paper's GPU result into a promise for a different device. Compare kernel time, end-to-end time, memory and task quality together.

### Primary references

1. Dao et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). Research paper.
2. Beltagy, Peters & Cohan (2020). [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150). Research paper.
3. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Research paper.

### Related topics

- [Attention as content-addressed computation](#article-attention) — Inspect the underlying operation.
- [KV-cache memory & grouped heads](#article-kv-cache) — Account for cached state.
- [Selective state spaces & hybrids](#article-state-space) — Compare compressed recurrent state.

---

**Family:** Transformers → Efficient execution

<a id="article-compression-decoding"></a>

## Compression & speculative decoding

*Spend fewer bytes—or verify more tokens at once.*

**Abstract.** Two complementary inference techniques, with different effects on numerical behavior and sampling correctness.

**At a glance:**

- Quantization changes numeric representation
- Speculative decoding uses a verified proposal
- Latency depends on hardware, batches and acceptance

**Concepts:** Quantization; Speculative decoding; Distillation.

### Quantize a representation
A schematic affine quantizer uses integer codes $q$ to approximate a weight $w$:

$$
q=\operatorname{clip}(\operatorname{round}(w/s)+z,q_{\min},q_{\max}),\qquad
\widehat w=s(q-z).
$$

$s>0$ is scale, $z$ zero point and $[q_{\min},q_{\max}]$ the code range. Scale grouping, calibration data, outliers and kernel support matter. GPTQ uses approximate second-order information to choose a low-bit weight representation; it is more specific than naive rounding. [1](https://arxiv.org/abs/2210.17323)

Weight quantization, activation quantization and KV-cache quantization change different memory terms. Compression can perturb output distributions. The bit width alone is not a quality or speed guarantee, and scales and packing add storage overhead.

### Speculate, then verify
A cheap draft model proposes several tokens. The target model evaluates them together and an acceptance/correction procedure preserves the target distribution under the algorithm's assumptions. This differs from accepting the draft whenever two argmax tokens agree. [2](https://arxiv.org/abs/2211.17192)

For a proposed token $x$ from draft distribution $q$, acceptance uses $\min(1,p(x)/q(x))$, where $p$ is the target distribution conditioned on the corresponding accepted prefix. After a rejection, exact correction samples from a distribution proportional to $\max(0,p-q)$. Details such as the extra target token and sequential conditioning are essential.

### Compare costs
Drafting adds work. Speed depends on acceptance rate, proposal length, verification efficiency and hardware utilization. A strong draft can be too expensive; a weak draft can have poor acceptance. Published speedups are measurements of a specific system, not an architecture-wide constant.

### Experiment design
Measure identical prompts, sampling settings, output lengths and target weights with and without speculation. Check distributions statistically rather than expecting identical random samples. For quantization, add held-out quality evaluation and report true packed memory. Both methods can coexist with a [grouped-head KV cache](#article-kv-cache).

### Primary references

1. Frantar et al. (2022). [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323). Research paper.
2. Leviathan, Kalman & Matias (2022). [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). Paper · ICML 2023.
3. Ainslie et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). Paper · EMNLP 2023.

### Related topics

- [KV-cache memory & grouped heads](#article-kv-cache) — Account for state precision.
- [Training, prefill & decoding](#article-training-decoding) — Locate proposal verification in decoding.
- [Self-improvement needs a verifier](#article-self-improvement) — Compare distinct uses of verification.

---

**Family:** Transformers → Observability & interpretability

<a id="article-transformer-observability"></a>

## Transformer observability & interpretability

*Inspect a tensor. Specify a claim. Test a mechanism.*

**Abstract.** A map of hooks, activation views, attention, readouts, attribution and interventions, with the evidence each can support.

**At a glance:**

- Layer, position, head and feature axes stay distinct
- Readable readouts are different from causal evidence
- A useful explanation specifies a falsifiable intervention

**Concepts:** Observability; Interpretability; Hooks; Probes; Direct logit attribution; Attention visualization.

### What does observability mean here?
In this atlas, **Transformer observability** means making internal activations, computations, and interventions inspectable. It includes representation analysis and mechanistic interpretability. It is not a claim that a model satisfies the formal state-reconstruction observability criterion of control theory, nor is it the logging, metrics, and tracing layer of a production serving system.

An analysis starts by specifying a model checkpoint, tokenizer, prompt distribution, and behavior of interest. A question such as “where is the candidate answer available?” differs from “which component is causally necessary for this answer?” The appropriate measurement changes with the question.

### First identify the tensor and its axes
A residual activation cache may have shape $[B,T,d]$ at each layer, where $B$ is batch size, $T$ sequence positions, and $d$ hidden width. Head activations add a head axis; attention patterns use destination and source positions. A layer-by-position heatmap is not a head-by-feature map.

Record whether a hook captures a state before normalization, before attention, after attention, or after the MLP. Saving all layers, tokens, heads, and gradients can be costly. Select positions and hook sites deliberately, and keep the model in a reproducible evaluation configuration. A hook exposes a tensor; its name does not explain that tensor's role.

### Four different kinds of evidence
| View | Operation | Useful question | Main limitation |
| --- | --- | --- | --- |
| Activations and attention | Inspect cached values and routing weights | What is large or attended to? | Magnitude is not causal importance |
| Lenses and probes | Map states to vocabulary scores or supervised labels | What can this readout recover? | Decoder assumptions and probe capacity matter |
| Jacobians and attribution | Differentiate a specified downstream quantity | What changes under an infinitesimal perturbation? | Local derivative need not predict a large intervention |
| Patching and ablation | Replace or remove an internal signal, then rerun | What happens under this intervention? | Controls, interactions, and distribution shifts matter |

High probe accuracy can reflect information accessible to the probe rather than computation used by the model. Control tasks and held-out generalization help separate these possibilities. [1](https://aclanthology.org/D19-1275/)

### Attention patterns are a starting point
For one head, the output at destination $i$ is $a_i=\sum_j A_{ij}v_j$. A large routing weight $A_{ij}$ does not determine the output effect without the corresponding value vector, output projection, other heads, and downstream computation. Two terms with large weights can cancel; a small-weight value can still matter.

Jain and Wallace give concrete counterexamples to treating attention weights as explanations in their studied NLP models. That result should not be stretched into a theorem that no attention-based analysis can ever be informative. The engineering conclusion is narrower: pair an attention pattern with a behavioral metric and an intervention before treating it as a mechanism. [2](https://aclanthology.org/N19-1357/)

### Direct logit attribution and its boundary
If a pre-normalization state decomposes as $h=\sum_c r_c$, a fixed linear readout gives $Uh=\sum_c Ur_c$. This motivates direct logit attribution to residual contributions. With final normalization included, the normalization scale depends on the combined state; LayerNorm additionally centers features. An exact attribution under a frozen scale is a conditional decomposition, not the effect of deleting a component and recomputing the model.

Useful analyses can therefore disagree without either being a software bug: one describes an algebraic contribution under fixed conditions; another measures a rerun after perturbing those conditions. State which quantity is being plotted. [3](https://arxiv.org/abs/2303.08112)

### From measurement to a falsifiable account
Choose a clean prompt, a controlled alternative, and an output metric such as a target-versus-foil logit difference. Inspect readouts to propose a candidate layer, position, or feature. Then test that candidate with patching, matched controls, and additional prompts. Patching methodology can materially change localization conclusions. [4](https://arxiv.org/abs/2309.16042)

The diagram's three token streams make the axes explicit. Its selectable measurement sites route to different analytical operations. It is a workflow schematic, not an automatic explanation generator.

### Follow the question
For vocabulary readouts, compare the [logit and tuned lenses](#article-logit-tuned-lenses). For an averaged sensitivity-derived transport, inspect the [Jacobian lens](#article-jacobian-lens). For interventions, sparse features, and computational graphs, continue to [causal tracing and feature circuits](#article-causal-features-circuits). Each approach exposes a different part of the evidence; a readable label is the beginning of analysis, not its conclusion.

### Primary references

1. John Hewitt & Percy Liang (2019). [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/). EMNLP-IJCNLP 2019.
2. Sarthak Jain & Byron C. Wallace (2019). [Attention is not Explanation](https://aclanthology.org/N19-1357/). NAACL 2019.
3. Nora Belrose et al. (2023). [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112). Research paper; updated 2025.
4. Fred Zhang & Neel Nanda (2023). [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042). ICLR 2024; preprint 2023.

### Related topics

- [The residual stream](#article-residual-streams) — Locate the token-indexed states being inspected.
- [Logit lens & tuned lens](#article-logit-tuned-lenses) — Compare raw and learned vocabulary readouts.
- [Jacobian lens & J-space](#article-jacobian-lens) — Inspect averaged downstream sensitivity.
- [Causal tracing, features & circuits](#article-causal-features-circuits) — Move from a candidate signal to a controlled intervention.

---

**Family:** Transformers → Observability & interpretability

<a id="article-logit-tuned-lenses"></a>

## Logit lens & tuned lens

*One hidden state. Different readout assumptions.*

**Abstract.** Compare final-head decoding and learned affine translators, with explicit dimensions, token alignment and a toy coordinate mismatch.

**At a glance:**

- Logit lens reuses the final decoder without a fitted translator
- Tuned lens fits per-layer prediction translators
- A next-token readout is not a complete causal explanation

**Concepts:** Logit lens; Tuned lens; Layer readouts; Unembedding; Prediction trajectories.

### Reading a hidden state through the output head
A decoder produces one hidden vector $h_i^{(\ell)}\in\mathbb R^d$ for each position $i$ and layer $\ell$. Let $N_f$ be its final LayerNorm or RMSNorm, $U\in\mathbb R^{V\times d}$ its unembedding, and $b\in\mathbb R^V$ an optional output bias. The actual final prediction uses $U N_f(h_i^{(L)})+b$.

The **logit lens**, introduced by nostalgebraist in 2020 and developed in the tuned-lens literature, applies that final readout prematurely:

$$
p_{\ell,i}^{\mathrm{LL}}=\operatorname{softmax}\!\left(U N_f(h_i^{(\ell)})+b\right).
$$

It requires no separately trained translator. Its interpretation depends on intermediate states being compatible with the final decoder's coordinates and normalization. [1](https://arxiv.org/abs/2303.08112)

### What a layer-by-position grid says
Rows can index layers and columns token positions. A cell may display the top decoded token, a chosen token's rank, probability, or a logit difference. These are different views: a top-1 token can change after an arbitrarily small score change when two candidates are close. A rank can improve even while absolute probability falls.

In an autoregressive model, the actual output at input position $i$ predicts position $i+1$. A readout attached to the token “is” is not automatically a reconstruction of “is.” Middle-layer lens distributions are auxiliary measurements, not additional tokens that the model actually generated. Always label the input position and prediction alignment.

### The tuned lens learns a translator
A **tuned lens** adds a learned affine transformation per layer, $T_\ell(h)=A_\ell h+c_\ell$, before the frozen final readout:

$$
p_{\ell,i}^{\mathrm{TL}}=\operatorname{softmax}\!\left(U N_f(T_\ell(h_i^{(\ell)}))+b\right).
$$

The translators are fitted to match the frozen model's final output distribution, commonly using a distribution-matching loss. This differs from training the language model itself. The purpose is a more reliable intermediate prediction readout, not a literal recording of every internal concept. [1](https://arxiv.org/abs/2303.08112)

```python
# Schematic column-vector convention; adapt to the model's actual hook sites.
h = cached_residual[layer, position]  # [width]
p_logit = softmax(U @ final_norm(h) + bias)
p_tuned = softmax(U @ final_norm(A[layer] @ h + c[layer]) + bias)
# Train the translator on held-out activations; keep backbone and decoder fixed.
```

### A worked coordinate-mismatch example
The interactive figure uses a synthetic two-dimensional model, with identity normalization and no bias. Its vocabulary directions are $u_A=(1,0)$, $u_B=(0,1)$ and $u_C=(-1,-1)$. Take the intermediate state $h=(0,1)$. Raw decoding gives logits $(0,1,-1)$, favoring B.

Now define a coordinate translator

$$
T=\begin{bmatrix}0&1\\1&0\end{bmatrix}.
$$

The translated state is $Th=(1,0)$, giving logits $(1,0,-1)$ and favoring A. The same stored activation can produce a different vocabulary readout under a different decoder. The figure does not fit a translator or reproduce the paper's measurements; it exposes these chosen matrices directly.

A learned translator should earn its interpretation through held-out prediction and control experiments. Merely finding one matrix that gives an appealing word proves little about the model.

### Prediction quality and causal fidelity
A translator can predict the eventual output while overlooking a transient representation used to compute it. Conversely, a raw logit lens can fail to decode meaningful information because the geometry is mismatched. Neither failure demonstrates that the underlying state contains no relevant information.

Compare readouts on unseen prompts; inspect more than the winning token; test multiple normalizations and hook sites only when justified. Use interventions to examine whether a decoded direction affects the behavior of interest. Patching evidence depends on its corruption, metric, and granularity. [2](https://arxiv.org/abs/2309.16042)

### Relation to the Jacobian lens
The tuned lens estimates translators through a prediction objective. The Jacobian lens constructs layer maps from averaged derivatives, including a specified distribution of source and downstream positions. These choices answer related but distinct questions. The [Jacobian-lens article](#article-jacobian-lens) gives its exact dimensions and a toy example of cancellation during averaging. Avoid interpreting any lens's softmax as a calibrated probability that a model “has a thought”; it is a vocabulary distribution produced by the selected readout.

### Primary references

1. Nora Belrose et al. (2023). [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112). Research paper; updated 2025.
2. Fred Zhang & Neel Nanda (2023). [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042). ICLR 2024; preprint 2023.

### Related topics

- [Token embeddings & embedding space](#article-token-embeddings) — Define the input lookup and output vocabulary directions.
- [Jacobian lens & J-space](#article-jacobian-lens) — Compare fitted prediction maps with averaged derivatives.
- [Causal tracing, features & circuits](#article-causal-features-circuits) — Test whether decoded directions affect behavior.

---

**Family:** Transformers → Observability & interpretability

<a id="article-jacobian-lens"></a>

## Jacobian lens & J-space

*Average a derivative; keep its assumptions visible.*

**Abstract.** Derive the J-Lens transport and its matrix dimensions, inspect cancellation across contexts, and distinguish sparse J-space from a low-rank projection.

**At a glance:**

- The transport is d × d; vocabulary directions are V × d
- Averaging can cancel strong context-specific sensitivities
- J-space is an overcomplete sparse-frame construction

**Concepts:** J-Lens; Jacobian lens; J-space; Global workspace; Jacobians; Sparse frames.

### A lens derived from downstream sensitivity
The **Jacobian lens (J-Lens)** uses averaged downstream derivatives to transport an intermediate residual vector toward the final-layer representation. It then applies the model's output readout. It is distinct from the training-free raw logit lens and from a translator fitted to predict the final distribution. Gurnee and collaborators introduce the method in their 2026 global-workspace study. [1](https://transformer-circuits.pub/2026/workspace/index.html)

Use column vectors. Let $h_{\ell,t}\in\mathbb R^d$ denote a source state at layer $\ell$ and position $t$. Let $h_{L,t'}\in\mathbb R^d$ denote a final-layer state at a downstream position $t'\ge t$. A local Jacobian is

$$
J_{\ell,t,t'}(x)=\frac{\partial h_{L,t'}}{\partial h_{\ell,t}}\in\mathbb R^{d\times d}.
$$

For a small perturbation $\delta h$, the first-order change is $\delta h_{L,t'}\approx J_{\ell,t,t'}(x)\delta h$. This is a derivative at a particular context, not a globally exact reconstruction of a nonlinear network.

### What gets averaged, and why dimensions matter
The paper's schematic definition averages over prompts $x$, source positions, and current-and-future target positions:

$$
\bar J_\ell=\mathbb E_{x,t,t'\ge t}\left[J_{\ell,t,t'}(x)\right],\qquad
p_\ell(h)=\operatorname{softmax}\!\left(U N_f(\bar J_\ell h)+b\right).
$$

Here $U\in\mathbb R^{V\times d}$ is the unembedding, $N_f$ final normalization, and $b$ an optional model-specific bias. The published formula omits a bias term; include one only when the model's actual readout has it. $\bar J_\ell$ is $d\times d$, whereas the composed direction matrix $U\bar J_\ell$ is $V\times d$. Neither is a vocabulary-by-vocabulary matrix. Normalization remains part of the actual readout, so the composed rows alone are not a universal exact linear decoder. [1](https://transformer-circuits.pub/2026/workspace/index.html)

The reference implementation sums cotangents over eligible target positions and averages over source positions. Sampling, aggregation, padding, and target-position weighting must be recorded to reproduce an estimator; a symbolic expectation does not settle every implementation convention. [2](https://github.com/anthropics/jacobian-lens)

### A worked example of information lost by averaging
Consider the explicitly invented contexts

$$
J_A=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
J_B=\begin{bmatrix}1&0\\0&-1\end{bmatrix}.
$$

With equal weight, $\bar J=(J_A+J_B)/2=\operatorname{diag}(1,0)$. Each individual Jacobian has rank two, but their average has rank one. A perturbation along $(0,1)$ has a strong context-specific effect that cancels in the mean. A weak average direction therefore does not establish causal irrelevance in every prompt.

The interactive figure varies the fraction $q$ of context A. Then $\bar J=\operatorname{diag}(1,2q-1)$. At $q=1/2$ the second direction vanishes; at $q=1$ it survives fully. These are exact calculations for the toy matrices, not extracted LLM measurements.

### Reading is different from writing
Ranking vocabulary directions is a readout. Editing an activation along a direction, ablating it, or swapping coordinates is an intervention requiring a new forward pass. A direction can be useful for both without making a high readout score sufficient evidence of a causal mechanism. Compare a local intervention with a matched random direction, several perturbation sizes, and held-out contexts.

### J-space is not simply a low-rank matrix
The paper treats token-associated directions as a correlated, overcomplete frame and uses sparse decompositions to study active contents. This is different from declaring that the rank or leading singular vectors of $\bar J$ alone define the active workspace. A sparse representation can involve few directions even when their full matrix spans a much larger space.

The named token inventory, context distribution, estimator, and decomposition procedure all affect what becomes visible. Single-token vocabulary labels can also miss multi-token concepts. Global-workspace interpretations are empirical hypotheses about studied models, not a proof of consciousness or a theorem about every Transformer.

### Practical replication boundary
The authors provide [reference code](https://github.com/anthropics/jacobian-lens). Fit or load a lens for the exact checkpoint and hook convention, preserve its tokenizer and normalization, and compare against simple readout baselines. Computing an average transport can be expensive; applying an already computed map is a different workload. The atlas runs only the small matrix example locally, not the reference model-fitting procedure.

### Primary references

1. Wes Gurnee et al. (2026). [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html). Transformer Circuits research publication.
2. Anthropic (2026). [Jacobian lens reference implementation](https://github.com/anthropics/jacobian-lens). Official reference implementation.

### Related topics

- [Logit lens & tuned lens](#article-logit-tuned-lenses) — Compare the meaning and construction of each layer decoder.
- [Transformer observability & interpretability](#article-transformer-observability) — Place sensitivity readouts in a broader evidence map.
- [Causal tracing, features & circuits](#article-causal-features-circuits) — Separate a readout from an intervention test.
- [Token embeddings & embedding space](#article-token-embeddings) — Track hidden width and vocabulary dimensions.

---

**Family:** Transformers → Observability & interpretability

<a id="article-causal-features-circuits"></a>

## Causal tracing, features & circuits

*Replace a signal, rerun the model, measure the difference.*

**Abstract.** Connect activation patching, ablation, sparse dictionaries, transcoders and attribution graphs while retaining their methodological limits.

**At a glance:**

- Patching is specific to a site, control and behavioral metric
- Feature dictionaries and replacement models are approximations
- Recovery scores and graph edges need explicit definitions

**Concepts:** Activation patching; Causal tracing; Path patching; Attribution patching; SAE; Sparse autoencoders; Transcoders; Circuit tracing; Steering.

### Intervene on an explicit computational site
Activation patching replaces an internal signal in one run with a cached signal from another, then measures the downstream output. Suppose clean prompt $x_c$ supports answer A and a controlled alternative $x_a$ changes the relevant information. At a specified layer, position, and component, replace

$$
h_{\ell,t}(x_a)\leftarrow h_{\ell,t}(x_c).
$$

The rest of the alternative run then continues. This can test a localization hypothesis under the chosen intervention. It does not automatically identify a unique or minimal circuit. Noising, denoising, ablation baselines, redundancy, and nonlinear interactions require separate interpretation. [1](https://arxiv.org/abs/2404.15255)

### A metric with a visible denominator
A useful behavioral quantity is the answer-versus-foil logit difference $m=z_A-z_B$. For illustration, define

$$
R=\frac{m_{\mathrm{patched}}-m_{\mathrm{alternative}}}{m_{\mathrm{clean}}-m_{\mathrm{alternative}}}.
$$

The figure uses invented clean and alternative margins 4 and 0. Its selectable patches produce 1, 3, or 5, giving $R=0.25$, $0.75$, or $1.25$. The last case illustrates that normalized recovery can exceed one; it is not intrinsically a percentage of “the explanation.” If clean and alternative margins nearly coincide, this ratio is unstable or undefined. Always report the raw margins too.

Changes to corruption methods and metrics can lead to different localization results. Repeat the analysis over multiple controlled examples and include uncertainty, rather than presenting a single successful patch as a complete account. [2](https://arxiv.org/abs/2309.16042)

### From activations to candidate features
A neuron index is a coordinate, not necessarily one concept. A representation can distribute a feature across coordinates and can superpose multiple features. A sparse-autoencoder-style dictionary describes a state approximately as

$$
f=\operatorname{ReLU}(W_e(h-b_d)+b_e),\qquad
\hat h=W_d f+b_d.
$$

Here $f\in\mathbb R^m$ contains feature coefficients, $m$ can exceed residual width $d$, and $W_d\in\mathbb R^{d\times m}$ contains decoder directions. A generic reconstruction-plus-sparsity objective is $\|h-\hat h\|_2^2+\lambda\|f\|_1$. This is one illustrative objective, not a specification of every SAE training recipe.

When patching feature coefficients, preserve the reconstruction error where the intervention protocol calls for it: $h'=W_d f'+b_d+(h-\hat h)$. Otherwise, differences can reflect deleting unexplained activation content rather than changing the chosen feature alone. Decoder directions can be correlated; a label attached to one feature is an empirical summary, not its exact mathematical meaning.

### Transcoders and attribution graphs
An SAE reconstructs an activation. A transcoder instead approximates a component's input-to-output transformation using sparse features. Cross-layer variants can connect these approximations across depth. The 2025 circuit-tracing work builds prompt-specific attribution graphs using a replacement model and validates proposed mechanisms with interventions. Its graph is an approximate analytical object, with a faithfulness boundary to the original network. [3](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

A graph edge should therefore identify the quantity being approximated: a local derivative, an attributed contribution under frozen attention/normalization conditions, or a measured effect from rerunning a patch. These are not interchangeable merely because all can be drawn as arrows.

### Useful neighboring techniques
**Path patching** constrains which source-to-destination route is tested. **Attribution patching** uses a gradient-based approximation to rank possible interventions before expensive reruns. **Causal tracing** is a family of corruption-and-restoration experiments. **Steering** adds a chosen activation direction and tests a behavior change. Each requires a precise intervention site and an evaluation distribution; none removes the need to validate the hypothesis on the original model.

The diagram deliberately keeps clean and alternative token streams separate and draws the replacement as a distinct cross-run connection. The displayed recovery values come from the toy margins above; no model is loaded in the browser.

### A reproducible investigation
Specify the checkpoint, prompts, token alignment, hook site, raw metric, replacement rule, and any frozen components. Use exploratory prompts to find candidates and separate prompts to test them. Compare the original model with any replacement or reconstructed model. Retain failures and alternative explanations: a mechanism that works only for one hand-selected sentence is a narrower result than a circuit that generalizes across a controlled prompt family.

### Primary references

1. Stefan Heimersheim & Neel Nanda (2024). [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255). Research preprint.
2. Fred Zhang & Neel Nanda (2023). [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042). ICLR 2024; preprint 2023.
3. Emmanuel Ameisen, Jack Lindsey, Adam Pearce et al. (2025). [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). Transformer Circuits research publication.

### Related topics

- [Transformer observability & interpretability](#article-transformer-observability) — Distinguish observational, derivative and intervention evidence.
- [Jacobian lens & J-space](#article-jacobian-lens) — Compare average sensitivity and actual reruns.
- [Logit lens & tuned lens](#article-logit-tuned-lenses) — Turn a readable candidate into a causal hypothesis.

---

# Learning & adaptation

Explore how credit is assigned, networks evolve, and useful changes persist.

**Family:** Learning & adaptation → Credit assignment → Gradients & local objectives

<a id="article-backprop-biology"></a>

## Backpropagation & biological plausibility

*One exact derivative; several implementation questions.*

**Abstract.** Explain reverse-mode credit assignment while treating biological plausibility as a set of testable constraints.

**At a glance:**

- Reverse-mode differentiation assigns parameter credit
- Weight transport, locality and timing are separate constraints
- Task performance does not establish biological plausibility

**Concepts:** Backpropagation; Biological plausibility; Credit assignment; Weight transport.

### The mathematical operation
For $a_\ell=W_\ell h_{\ell-1}$ and $h_\ell=\sigma(a_\ell)$, define $\delta_\ell=\partial\mathcal L/\partial a_\ell$. Backpropagation applies the chain rule:

$$
\delta_\ell=(W_{\ell+1}^\top\delta_{\ell+1})\odot\sigma'(a_\ell),\qquad
\nabla_{W_\ell}\mathcal L=\delta_\ell h_{\ell-1}^\top.
$$

The first equation transports an error signal; the second combines it with presynaptic activity. With a batch, gradients sum or average over samples. Reverse-mode differentiation efficiently computes a scalar objective's derivatives; the 1986 papers are major milestones, not the entire historical origin of the chain rule. [1](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)

### What could be biologically difficult?
Several separate questions are often compressed into “the brain cannot backprop”:

- **Weight transport:** exact transposed forward weights in a separate feedback path.
- **Timing:** availability of forward activities when downstream errors arrive.
- **Signals:** representing signed errors and derivatives in neural circuits.
- **Locality:** which variables a synapse can access.
- **Objective:** where a global supervised target or reward originates.

A method can relax one issue while retaining others. Feedback alignment changes feedback weights; predictive coding changes the organization of state inference; surrogate gradients permit training spikes but can retain global backpropagation through time. [2](https://www.nature.com/articles/ncomms13276) [3](https://pubmed.ncbi.nlm.nih.gov/28333583/) [4](https://arxiv.org/abs/1901.09948)

### Avoid a false binary
Numerically approximating the BP gradient is not the same as implementing textbook forward/backward phases. Conversely, using local updates is not experimental proof of biological realization. Plausibility should specify circuitry, time scales, plasticity and signal representations rather than use one yes/no label.

### A useful evaluation table
For each method, record: derivative target, finite-budget approximation, communication graph, weight symmetry, stored state, temporal coordination, measured compute and biological evidence. A lower training loss cannot settle the neuroscience questions, and a plausible circuit cannot substitute for task performance.

### Experiment
On a fixed small network, compare update direction to exact BP. Then vary depth and inference budget. Report cosine similarity, norm ratio, loss reduction and runtime. Keep this diagnostic distinct from a full training experiment that changes the weights after each batch.

### Primary references

1. Rumelhart, Hinton & Williams (1986). [Learning representations by back-propagating errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf). Research paper.
2. Lillicrap et al. (2016). [Random synaptic feedback weights support error backpropagation for deep learning](https://www.nature.com/articles/ncomms13276). Nature Communications.
3. Whittington & Bogacz (2017). [An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity](https://pubmed.ncbi.nlm.nih.gov/28333583/). Neural Computation.
4. Neftci, Mostafa & Zenke (2019). [Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948). Research paper.

### Related topics

- [Feedback alignment & local objectives](#article-local-learning) — Inspect alternative feedback rules.
- [Augmented Lagrangian predictive coding](#article-pc-alm) — Compare inference-based credit.
- [NEAT: evolve weights and topology](#article-neat) — Contrast derivatives with evolutionary search.

---

**Family:** Learning & adaptation → Credit assignment → Gradients & local objectives

<a id="article-local-learning"></a>

## Feedback alignment & local objectives

*What must a layer know to learn?*

**Abstract.** Compare alternative error transport, independent objectives and iterative constraint enforcement.

**At a glance:**

- Feedback alignment changes the backward signal
- Local objectives change where errors originate
- Equivalence claims depend on specific assumptions

**Concepts:** Feedback alignment; Local losses; Equilibrium propagation.

### Change the feedback path
Feedback alignment studies whether fixed random feedback weights can support useful learning instead of exact transposed forward weights. A schematic update replaces $W_{\ell+1}^\top$ by a fixed compatible matrix $B_\ell$:

$$
\widetilde\delta_\ell=(B_\ell\widetilde\delta_{\ell+1})\odot\sigma'(a_\ell).
$$

This is an approximate credit signal; it is not generally the gradient of the original loss. Forward weights may learn representations that make the feedback useful in the tested setting. The result relaxes exact weight symmetry, not every biological requirement. [1](https://www.nature.com/articles/ncomms13276)

### Give a layer its own objective
Another design attaches an auxiliary predictor to intermediate features. A layer can reduce $\mathcal L_\ell(g_\ell(h_\ell),y)$ without waiting for the final network's exact gradient. That local goal may conflict with downstream needs. “Local loss” specifies where a target acts; it does not guarantee globally optimal representations.

### Enforce neighboring consistency
Predictive coding instead uses hidden-state inference and neighboring prediction errors. PC-ALM adds multipliers to accumulate constraint violations. These methods address the same broad credit-assignment problem through different mathematical constructions. [2](https://pubmed.ncbi.nlm.nih.gov/28333583/) [3](https://arxiv.org/html/2605.31022v1)

| Family | Credit source | Primary caveat |
| --- | --- | --- |
| Feedback alignment | Approximate feedback projection | No general exact-gradient guarantee |
| Layer-local objective | Auxiliary target at each layer | Local goals may conflict |
| Predictive coding | Inferred neighboring consistency | Finite inference and equilibrium assumptions |
| PC-ALM | Primal–dual local dynamics | Proven scope narrower than nonlinear use |

### Implementation discipline
```python
# Fixed random feedback is a parameter of the learning algorithm.
feedback.requires_grad_(False)
local_error = feedback @ downstream_error
```

State precisely which tensors require gradients and which update in each phase. An implementation that silently computes the full global gradient for the “local” update defeats the intended comparison.

### What remains open
Scaling local learning to modern language tasks requires evidence about optimization, data efficiency and systems cost, not only shallow classification. A fair study reports success and instability across seeds, depth, width and nonlinearities, with a conventional BP baseline trained comparably.

### Primary references

1. Lillicrap et al. (2016). [Random synaptic feedback weights support error backpropagation for deep learning](https://www.nature.com/articles/ncomms13276). Nature Communications.
2. Whittington & Bogacz (2017). [An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity](https://pubmed.ncbi.nlm.nih.gov/28333583/). Neural Computation.
3. Jeffrey Seely & Julian Gould (2026). [Augmented Lagrangian Predictive Coding](https://arxiv.org/html/2605.31022v1). Preprint · May 2026.

### Related topics

- [Backpropagation & biological plausibility](#article-backprop-biology) — Define the reference and biological constraints.
- [Predictive coding as local inference](#article-predictive-coding) — Compare local inference updates.
- [Remembering while learning](#article-continual-learning) — Consider learning under a changing stream.

---

**Family:** Learning & adaptation → Credit assignment → Predictive coding

<a id="article-predictive-coding"></a>

## Predictive coding as local inference

*States become optimization variables.*

**Abstract.** Reformulate a feed-forward computation as consistency constraints and inspect the distinction between inference and learning.

**At a glance:**

- Activities infer latent causes through prediction errors
- Local updates depend on the inference dynamics
- Equivalence to backpropagation requires assumptions

**Concepts:** Predictive coding; Prediction errors; Local inference.

### Free the hidden activities
A feed-forward network ordinarily sets $h_i=f_i(h_{i-1};W_i)$ in a single computation. A supervised predictive-coding construction instead treats hidden activities $h_i$ as variables and penalizes violations of these relationships:

$$
r_i=h_i-f_i(h_{i-1};W_i),\qquad
\mathcal F=\ell(y,f_L(h_{L-1};W_L))+\frac\rho2\sum_{i=1}^{L-1}\|r_i\|^2.
$$

$h_0=x$ is clamped input, $y$ the target, $W_i$ layer parameters, $\rho>0$ a penalty and $r_i$ the prediction error. This is one supervised PC formulation; the broader literature includes different generative directions, clamping schemes and approximations. [1](https://pubmed.ncbi.nlm.nih.gov/28333583/) [2](https://arxiv.org/html/2605.31022v1)

### Inference versus parameter learning
During activity inference, keep $W$ fixed and update $h_i\leftarrow h_i-\eta_h\nabla_{h_i}\mathcal F$. Each activity gradient depends on adjacent-layer consistency errors and the local function's Jacobian. After a finite inference budget, update weights using the resulting local quantities.

```python
h = forward_initialize(x, W)
for step in range(inference_budget):
    h = simultaneous_activity_step(h, W, y)
W = one_batch_aggregated_parameter_step(h, W, y)
```

“Local” here means neighboring layers in the computational graph. An implementation may use automatic differentiation to compute local derivatives; locality is not the absence of any derivative calculation.

### Why the penalty matters
At a stationary point of a finite quadratic penalty, residuals need not be zero. Therefore activity-inferred weight updates need not equal backpropagation gradients of the original feed-forward network. The relationship depends on the formulation, limits and inference assumptions. One should not turn a correspondence result into a blanket claim of exact equivalence for finite-budget nonlinear networks.

### Cost and interpretation
Inference entails repeated local computations. It can offer an interesting parallel or distributed organization without being faster on a conventional GPU. Biological plausibility also requires more than local communication: weight symmetry, timing, error coding and plasticity remain substantive questions. The Sakana page makes the iterative inference structure explicit. [3](https://pub.sakana.ai/pc-alm/)

### Experiment
On a small fixed network, compare local parameter-update vectors with exact backpropagation through cosine similarity and relative norm error as inference steps increase. Plot wall-clock cost as well. Then read [PC-ALM](#article-pc-alm), which introduces an accumulated constraint-error state rather than only a quadratic penalty.

### Primary references

1. Whittington & Bogacz (2017). [An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity](https://pubmed.ncbi.nlm.nih.gov/28333583/). Neural Computation.
2. Jeffrey Seely & Julian Gould (2026). [Augmented Lagrangian Predictive Coding](https://arxiv.org/html/2605.31022v1). Preprint · May 2026.
3. Seely & Gould · Sakana AI (2026). [Augmented Lagrangian Predictive Coding: training 1000-layer networks without backpropagation](https://pub.sakana.ai/pc-alm/). Author project page · Sep 2026.

### Related topics

- [Augmented Lagrangian predictive coding](#article-pc-alm) — Change the constraint dynamics.
- [Backpropagation & biological plausibility](#article-backprop-biology) — Compare credit-assignment assumptions.
- [JEPA: predicting representations](#article-jepa) — Contrast two meanings of prediction.

---

**Family:** Learning & adaptation → Credit assignment → Predictive coding

<a id="article-pc-alm"></a>

## Augmented Lagrangian predictive coding

*Credit assignment as a primal–dual dynamical system.*

**Abstract.** A paper-style walkthrough of PC-ALM: constraint residuals, multipliers, batched inference, theoretical scope and computational cost.

**At a glance:**

- Multipliers accumulate constraint violations
- Activities follow augmented-Lagrangian dynamics
- Linear results and nonlinear experiments have different scope

**Concepts:** PC-ALM; Augmented Lagrangian; Primal–dual dynamics.

### The constrained problem
Seely and Gould study layer-local learning by introducing independent hidden activities and constraints that recover a feed-forward network. For a network with two hidden layers and a linear readout, write

$$
h_0=x,\quad r_1=h_1-f_1(x;W_1),\quad
r_2=h_2-f_2(h_1;W_2),\quad\widehat y=W_3h_2.
$$

$x\in\mathbb R^{d_0}$ is one input; $h_1\in\mathbb R^{d_1}$ and $h_2\in\mathbb R^{d_2}$ are variable hidden activities. $W_3\in\mathbb R^{d_y\times d_2}$ maps to target dimension $d_y$. This example defines all layer roles explicitly; counting “three layers” otherwise varies by convention.

### Add a multiplier to each constraint
For squared output loss, an augmented Lagrangian is

$$
\mathcal L_\rho=
\frac12\|y-W_3h_2\|^2+
\sum_{i=1}^{2}\left[\lambda_i^\top r_i+\frac\rho2\|r_i\|^2\right].
$$

$\lambda_i$ has the same shape as $r_i$; $\rho>0$ is the penalty coefficient. Multipliers accumulate constraint violations. They are sample-specific state variables, not another set of weights shared by the dataset. The sign convention here is $r_i=h_i-f_i$; reversing the constraint reverses its multiplier convention. [1](https://arxiv.org/html/2605.31022v1)

### Activity gradients and local messages
Define $e_i=\lambda_i+\rho r_i$. With $J_2=\partial f_2/\partial h_1$, the hidden gradients in this example are

$$
\nabla_{h_1}\mathcal L_\rho=e_1-J_2^\top e_2,
\qquad
\nabla_{h_2}\mathcal L_\rho=W_3^\top(W_3h_2-y)+e_2.
$$

$J_2\in\mathbb R^{d_2\times d_1}$. The first expression combines the local consistency signal with a neighboring signal propagated through a local Jacobian. For $f_2(h)=\sigma(W_2h)$, $J_2=\operatorname{diag}(\sigma'(W_2h))W_2$. Thus the method can use nonlinear activations; a linear-network theorem is a narrower claim than the algorithm's implementation.

### One batch, step by step
1. Forward-initialize hidden activities for all examples with the current shared weights.
2. Reset each example's hidden multipliers to zero.
3. Hold weights fixed while taking an activity descent step with step size $\eta_h>0$.
4. Update multipliers using the **new** activities: $\lambda_i\leftarrow\lambda_i+\alpha r_i$, with dual step size $\alpha>0$.
5. Repeat the primal–dual cycle $T-1$ times, then take the final primal step specified by the paper.
6. Average parameter gradients across the batch and perform one parameter update.

```python
# Algorithm sketch following the paper's primal/dual/final-primal ordering.
h = forward_hidden_states(batch_x, weights)
lam = zeros_like(h)
for t in range(T - 1):
    h = primal_step_simultaneous(h, lam, weights, batch_y)
    lam = lam + alpha * constraint_residuals(h, weights)
h = primal_step_simultaneous(h, lam, weights, batch_y)
grad_W = mean_over_examples(local_weight_gradients(h, lam, weights, batch_y))
weights = weights - eta_W * grad_W
```

The code is pseudocode, not a substitute for the reference implementation. With batching, $h_i,r_i,\lambda_i$ have a leading batch dimension. No hidden-state averaging across samples is implied. All sample-wise dynamics can be vectorized; layer dependencies still determine what can run concurrently. [1](https://arxiv.org/html/2605.31022v1)

### Evidence, separated by scope
The May 2026 preprint proves convergence to the BP-gradient equilibrium under specified **linear, bias-free, fixed-weight** assumptions and stability conditions. Its nonlinear results are experiments, not the same theorem. The September project page additionally reports deep residual-MLP experiments reaching 1,000 layers on MNIST. Those results concern a particular residual architecture and training setup; they are not evidence of a 1,000-layer Transformer trained with an established speed advantage. [1](https://arxiv.org/html/2605.31022v1) [2](https://pub.sakana.ai/pc-alm/)

The original paper's short training sweeps and the later project page's five-epoch experiment must not be merged into one protocol. Nor should near-BP accuracy be paraphrased as numerical equality in every configuration.

### Computational interpretation
With dense width $d$, batch size $B$, depth $L$ and inference budget $T$, a simple local-iteration accounting scales roughly as $O(TBLd^2)$; constants, layer structure and parallel execution matter. These are repeated activity/Jacobian computations, not merely repeated ordinary forward passes. A comparison to BP needs total work, memory, communication and measured wall-clock time. The benefit studied is the propagation and locality of credit under finite inference, not a demonstrated universal accelerator for today's GPU training stack.

### Biological interpretation and open questions
The accumulated multiplier resembles an integral feedback state. This motivates a local-control interpretation, but it does not identify a biological neuron type implementing the exact algorithm. Symmetric feedback, nonlinear convergence, stability at scale and realistic neural timing remain open questions. [3](https://pubmed.ncbi.nlm.nih.gov/28333583/)

The adjacent lab solves a **two-hidden-layer scalar linear instance** with fixed weights; it shows actual primal–dual updates and BP reference gradients. It is an educational mechanism check, not a reproduction of the published datasets or accuracy curves.

### Primary references

1. Jeffrey Seely & Julian Gould (2026). [Augmented Lagrangian Predictive Coding](https://arxiv.org/html/2605.31022v1). Preprint · May 2026.
2. Seely & Gould · Sakana AI (2026). [Augmented Lagrangian Predictive Coding: training 1000-layer networks without backpropagation](https://pub.sakana.ai/pc-alm/). Author project page · Sep 2026.
3. Whittington & Bogacz (2017). [An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity](https://pubmed.ncbi.nlm.nih.gov/28333583/). Neural Computation.

### Related topics

- [Predictive coding as local inference](#article-predictive-coding) — Inspect the baseline inference scheme.
- [Feedback alignment & local objectives](#article-local-learning) — Compare other local update rules.
- [Backpropagation & biological plausibility](#article-backprop-biology) — Use backpropagation as a credit reference.

---

**Family:** Learning & adaptation → Neural dynamics

<a id="article-spiking-networks"></a>

## Spiking neural networks

*Events, membrane dynamics and surrogate derivatives.*

**Abstract.** Separate the representation of neural activity from the algorithm used to train it.

**At a glance:**

- Membrane dynamics integrate events over time
- A threshold emits discrete spikes
- Surrogate gradients approximate the threshold derivative

**Concepts:** Spiking networks; LIF; Surrogate gradients; Neuromorphic.

### A dynamical neuron
A simple discrete-time leaky integrate-and-fire model uses

$$
u_{t+1}=\beta u_t+I_t-\vartheta z_t,\qquad
z_t=\mathbf1[u_t\ge\vartheta].
$$

$u_t$ is membrane potential, $I_t$ input drive, $0\le\beta<1$ a leak factor, $\vartheta$ threshold and $z_t$ a binary spike. This is one reset convention; other models reset before or after integration or clamp to a reset potential. State and timing matter beyond an ordinary static activation.

### The gradient problem
A hard threshold has zero derivative almost everywhere and an undefined derivative at its threshold. Surrogate-gradient methods keep the forward spike but use a smooth proxy derivative during learning. Backpropagation through time can still assign credit through the resulting temporal computation. Thus “spiking” does not imply “trained without backpropagation.” [1](https://arxiv.org/abs/1901.09948)

```python
spike = (membrane >= threshold).float()    # hard forward event
# In a custom backward rule, substitute a bounded smooth derivative.
# Temporal learning may still require an unrolled computation graph.
```

### Biological and hardware axes
An event-based representation resembles one aspect of biological communication. Weight symmetry, global error objectives and unrolled gradient storage can remain biologically implausible. Feedback alignment addresses a different axis: what feedback weights carry credit. [2](https://www.nature.com/articles/ncomms13276)

Potential energy efficiency depends on hardware, event sparsity, temporal resolution, communication and the task. Simulating many dense timesteps on a general GPU need not save energy compared with an ordinary ANN. State the measured hardware and power boundary before claiming a benefit.

### Experiment
Use an event-timing task where the timing genuinely carries information. Compare against a rate-coded baseline and a conventional recurrent model. Report latency, number of timesteps, accuracy and energy measured under comparable input/output boundaries. Compare surrogate shape and spike regularization separately.

### Limits
A sparse spike raster does not reveal whether the learning rule is local. Conversely, a local plasticity rule does not guarantee that a network learns deep supervised tasks effectively. Keep neural dynamics, credit assignment and hardware implementation as distinct comparison axes.

### Primary references

1. Neftci, Mostafa & Zenke (2019). [Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948). Research paper.
2. Lillicrap et al. (2016). [Random synaptic feedback weights support error backpropagation for deep learning](https://www.nature.com/articles/ncomms13276). Nature Communications.

### Related topics

- [Backpropagation & biological plausibility](#article-backprop-biology) — Inspect biological plausibility criteria.
- [Liquid time-constant networks](#article-liquid-networks) — Compare continuous dynamics.
- [Feedback alignment & local objectives](#article-local-learning) — Consider locality of the learning rule.

---

**Family:** Learning & adaptation → Neural dynamics

<a id="article-liquid-networks"></a>

## Liquid time-constant networks

*Input-dependent dynamics in continuous time.*

**Abstract.** Understand liquid neural ODEs and the distinction between an adaptive time constant and online weight learning.

**At a glance:**

- Hidden dynamics depend on inputs and learned parameters
- Time constants describe response dynamics
- Continuous-time models still require numerical or analytic execution

**Concepts:** LTC; Liquid networks; Continuous time; CfC.

### Correct attribution
Liquid Time-constant Networks are associated with **Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus and Radu Grosu**. The cited paper was first posted in 2020 and accepted at AAAI 2021. Demis Hassabis is not an author of this work. [1](https://arxiv.org/abs/2006.04439)

### A changing effective time scale
A schematic LTC neuron follows

$$
\frac{dx(t)}{dt}=-\left[\frac1\tau+f(x(t),I(t),t;\theta)\right]x(t)+f(x(t),I(t),t;\theta)A.
$$

$x(t)$ is a state variable, $I(t)$ input, $\tau>0$ a baseline time constant, $A$ a driving level and $f$ a learned nonlinear gate. The effective time constant is $\tau_{\mathrm{eff}}=\tau/(1+\tau f)$ under this scalar form and appropriate gate assumptions. Vector networks require explicit elementwise and coupling conventions.

### What is liquid?
The state's response speed depends on the input and state through the gate. This does **not** mean the weights $\theta$ are automatically updated at inference. Continuous dynamics, recurrent state adaptation and parameter learning are three different notions.

```python
# A single illustrative Euler step, not a recommended universal solver.
dx = -(1 / tau + gate(x, u)) * x + gate(x, u) * A
x_next = x + dt * dx
```

Numerical solver choice and step size affect stability and cost. An adaptive gate does not remove the need for numerical analysis. Published results use a specific model and solver configuration; the scalar equation above is explanatory.

### Relation to other sequence models
Like an SSM, an LTC network maintains a recurrent state, but its continuous-time nonlinear gating is a particular dynamical construction. Like a spiking model, it treats time explicitly, but its state need not communicate through discrete spikes. Neither relationship makes the architectures equivalent. [2](https://arxiv.org/abs/2312.00752) [3](https://arxiv.org/abs/1901.09948)

### Research questions
Evaluate irregularly sampled trajectories and distribution shifts in input timing. Compare interpolation, numerical solver cost, stability and task error. Specify whether learning is offline or online. Do not infer general continual-learning ability from an adaptive time constant alone.

### Primary references

1. Hasani, Lechner, Amini, Rus & Grosu (2020). [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439). Paper · AAAI 2021.
2. Gu & Dao (2023). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752). Research paper.
3. Neftci, Mostafa & Zenke (2019). [Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948). Research paper.

### Related topics

- [Selective state spaces & hybrids](#article-state-space) — Compare state evolution.
- [Spiking neural networks](#article-spiking-networks) — Contrast continuous state and events.
- [Active inference](#article-active-inference) — Keep dynamics separate from control objectives.

---

**Family:** Learning & adaptation → Neuroevolution

<a id="article-neat"></a>

## NEAT: evolve weights and topology

*Historical alignment, speciation and complexification.*

**Abstract.** Inspect the direct genetic encoding and structural mutations that let an evolutionary population search networks of different shapes.

**At a glance:**

- Innovation identifiers align genes by ancestry.
- Speciation gives new structures time to optimize.
- Evolution and lifetime plasticity use different time axes.

**Concepts:** NEAT; Neuroevolution; Speciation; Topology mutation.

### Search the network as well as its weights
NeuroEvolution of Augmenting Topologies (NEAT), introduced by Kenneth Stanley and Risto Miikkulainen, searches over connection weights and network structure through population-based evolution. Candidates are evaluated, selected and varied across generations. A differentiable task loss is not required: a controller can receive a fitness score from an episode. This changes the optimization procedure; it does not turn an individual neuron's forward computation into a biological simulation. [1](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

A useful comparison asks three independent questions: what is encoded, what produces variation, and what supplies selection? Evolving a policy's weights differs from searching for an architecture and subsequently training it with gradients. Neither automatically includes synaptic plasticity during the individual's lifetime. A hybrid could combine these processes, but that additional learning rule and its evaluation cost must be specified.

### A genome records structure and ancestry
NEAT uses a direct encoding: node genes identify units, while connection genes record endpoints, weight, enabled status and an innovation identifier. Historical markings let crossover align genes descended from the same structural innovation, even when parental genomes have different lengths. An innovation identifier is bookkeeping, not a fitness score or a label for a semantic capability. [1](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

The adjacent figure lets you inspect one invented genome before and after splitting a connection. Its gene identifiers are illustrative. They show why a disabled connection can remain in the genotype while disappearing from the expressed network. The figure does not execute selection or reproduce a published run.

### Complexification changes the search space
Weight mutation changes numeric parameters; structural mutation can add a connection or split an existing connection with a new node. Starting with simple networks allows additional structure to accumulate through search. The original add-node construction disables the old edge and inserts two new edges, assigning the incoming edge weight one and retaining the old weight on the outgoing edge. An inserted nonlinear activation means this is not generally an exact function-preserving identity transformation. [2](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume21/stanley04a-html/node3.html)

For an illustrative original edge with weight 0.8 and input 0.5, its contribution is 0.4. After inserting a tanh unit with those weights, the contribution becomes $0.8\tanh(0.5)\approx0.370$. This calculation explains the distinction between preserving an inherited weight and preserving an entire function; it is not the activation convention of every NEAT implementation.

### Speciation protects opportunities, not guaranteed progress
NEAT groups compatible genomes and shares fitness within species, giving new structures time to optimize before direct competition with established structures. Compatibility uses structural differences and matching-weight differences. Speciation therefore addresses premature loss of innovation; it does not guarantee that every species improves or that useful diversity persists indefinitely. [1](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

An experiment can compare the full method with fixed topology, removed historical alignment and removed speciation. Keep evaluation budgets comparable, record unsuccessful seeds, and count environment interactions as well as generations. Population size changes the cost of a generation, so generation count alone is not an efficiency metric.

### Where this route leads
The original evidence concerned particular control benchmarks, not modern language-model pretraining. Report wall-clock cost, parameter count, task success and robustness separately. For noisy environments, reevaluate candidates on fresh episode seeds so selection cannot simply exploit a lucky rollout.

Continue to [HyperNEAT](#article-hyperneat) for an indirect encoding that generates connections from geometry. Compare with [backpropagation](#article-backprop-biology) for derivative-based credit and [continual learning](#article-continual-learning) for retention during sequential updates. Evolution across generations and adaptation within a deployed individual belong on different time axes.

### Primary references

1. Kenneth O. Stanley & Risto Miikkulainen (2002). [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf). Evolutionary Computation 10(2): 99–127.
2. Kenneth O. Stanley & Risto Miikkulainen (2004). [Competitive Coevolution through Evolutionary Complexification](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume21/stanley04a-html/node3.html). JAIR 21: 63–100 · section 3.

### Related topics

- [HyperNEAT: geometry generates connectivity](#article-hyperneat) — Move from direct to indirect encoding.
- [Backpropagation & biological plausibility](#article-backprop-biology) — Contrast evolutionary and gradient credit assignment.
- [Remembering while learning](#article-continual-learning) — Distinguish population search from adaptation.

---

**Family:** Learning & adaptation → Neuroevolution

<a id="article-hyperneat"></a>

## HyperNEAT: geometry generates connectivity

*A compact generator can describe a large substrate.*

**Abstract.** Follow coordinate queries through a CPPN to understand indirect encoding, geometric regularities and the limits of resolution transfer.

**At a glance:**

- The CPPN generates a distinct task network.
- Coordinate functions can express repeated geometric motifs.
- Compact encoding does not remove substrate execution cost.

**Concepts:** HyperNEAT; CPPN; Indirect encoding; Geometric bias.

### Encode a pattern instead of listing every connection
HyperNEAT extends neuroevolution through an indirect encoding: a compact pattern-producing network generates the weights of another network. The generated network is the **substrate**; the generator is a **compositional pattern-producing network**, or CPPN. Keeping these two networks separate is essential. A small generator can specify many substrate connections without storing one independently evolved gene for every weight. [1](https://stars.library.ucf.edu/facultybib2000/2178/)

This is representational compression, not a promise of low execution cost. Once generated, a large dense substrate can still require many operations and substantial memory. Ask separately how expensive it is to search for a generator, generate a substrate and execute that substrate.

### Coordinates become queries
Assign each substrate neuron a location. For a two-dimensional layout, a possible connection from source $(x_s,y_s)$ to target $(x_t,y_t)$ becomes a four-coordinate query:

$$
w_{st}=f_\phi(x_s,y_s,x_t,y_t).
$$

Here $f_\phi$ is the CPPN, and its output determines a candidate weight under the chosen scaling and connection-expression convention. Extra inputs such as distance or bias, and additional outputs for different connection sets, are implementation choices. The coordinates describe network geometry; they are not observations arriving from the task environment. [1](https://stars.library.ucf.edu/facultybib2000/2178/)

The interactive diagram uses two rows of three neurons. Select a source and target to inspect the corresponding coordinate query. Switching the symmetry control changes an explicitly written toy weight field. It does not evolve a CPPN, simulate a controller or measure performance.

### Why a CPPN can express regularity
CPPNs compose functions with different spatial biases. Symmetric and periodic functions can create repeated motifs, while other compositions introduce variation. Their developmental analogy concerns the genotype-to-phenotype mapping rather than a literal simulation of cell division. A coordinate-to-pattern function can describe a coherent large object through shared rules instead of independent values. [2](https://stars.library.ucf.edu/facultybib2000/7682/)

For intuition, our toy symmetric field is $w=\exp[-2(x_s-x_t)^2]$. Simultaneously reflecting both coordinates leaves the weight unchanged: replacing $(x_s,x_t)$ by $(-x_s,-x_t)$ preserves their squared difference. Adding $0.35x_t$ breaks that reflection symmetry. These hand-written functions demonstrate a geometric bias; neither is a claim about the evolved generator in the original paper.

A geometry-sensitive encoding can be a poor fit when nearby coordinates do not imply related roles. Choosing coordinates and permitted layer connections already introduces design knowledge. Compactness is useful only if its biases match useful solutions.

### Evolution acts on the generator
HyperNEAT uses NEAT to evolve CPPNs, including their weights and topology. Each candidate CPPN generates a substrate, whose task performance supplies the fitness signal. Thus an evolutionary change to one generator parameter may coherently alter many substrate connections. In direct NEAT, connection genes instead describe the task network more directly. [3](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

For the mechanism diagram, the relevant distinction is between a query-time computation that generates weights and the later execution of those weights on task inputs. Neither should be confused with gradient-based architecture optimization or online plasticity. A hybrid that learns after generation must state that extra update rule and include its cost.

### Scale the representation, then test the behavior
A coordinate function can be queried on a finer grid without increasing its own parameter count. This permits re-expression at another substrate resolution, but successful task transfer still depends on input/output semantics, scaling and evaluation. The original HyperNEAT work demonstrated particular visual-discrimination and food-gathering settings; that evidence does not establish automatic generalization to arbitrary larger networks. [1](https://stars.library.ucf.edu/facultybib2000/2178/)

Compare direct and indirect encodings at matched environment-evaluation budgets. Record generator size, expressed connections, substrate runtime and variation across seeds. Deliberately scramble coordinates as an ablation to test whether geometry supplies the claimed advantage. Return to [NEAT](#article-neat) for historical alignment and speciation, or compare [multimodal interfaces](#article-multimodal-fusion) to see another setting where the choice of representation constrains what a learner can discover.

### Primary references

1. Kenneth O. Stanley, David B. D’Ambrosio & Jason Gauci (2009). [A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks](https://stars.library.ucf.edu/facultybib2000/2178/). Artificial Life 15(2): 185–212.
2. Kenneth O. Stanley (2007). [Compositional pattern producing networks: A novel abstraction of development](https://stars.library.ucf.edu/facultybib2000/7682/). Genetic Programming and Evolvable Machines 8: 131–162.
3. Kenneth O. Stanley & Risto Miikkulainen (2002). [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf). Evolutionary Computation 10(2): 99–127.

### Related topics

- [NEAT: evolve weights and topology](#article-neat) — Evolve the pattern-producing generator.
- [From symbols to distributed representations](#article-distributed-representations) — Compare what a representation encodes.
- [Vision–language–action models](#article-vla-foundations) — Connect generated policies to embodied control.

---

**Family:** Learning & adaptation → Memory & continual learning

<a id="article-continual-learning"></a>

## Remembering while learning

*Replay, consolidation and the stability–plasticity tension.*

**Abstract.** Define forgetting operationally and compare mechanisms that protect old behavior during new learning.

**At a glance:**

- A changing task stream can cause forgetting
- Replay, regularization and isolation address different constraints
- Memory and compute budgets belong in the protocol

**Concepts:** Continual learning; Replay; EWC; Forgetting.

### A nonstationary training sequence
In continual learning, a model encounters changing data or tasks rather than one fixed shuffled dataset. Updating shared parameters for new data can reduce performance on previous data. Define an evaluation matrix $A_{t,j}$: performance on task $j$ after training through stage $t$. Forgetting should be computed from this history, with task boundaries and access to old data stated explicitly.

### Consolidate important parameters
Elastic weight consolidation (EWC) penalizes movement in parameters estimated to matter for a previous task:

$$
\mathcal L(\theta)=\mathcal L_{\mathrm{new}}(\theta)
+\frac\lambda2\sum_k F_k(\theta_k-\theta_k^*)^2.
$$

$\theta^*$ is a previous solution, $F_k$ a diagonal importance estimate related to the Fisher information, and $\lambda$ a strength. This approximation is not a complete posterior over all parameters and may constrain later learning. [1](https://arxiv.org/abs/1612.00796)

### Rehearse previous experience
Replay mixes stored or generated past examples into later training. The buffer's selection, size and sampling policy matter. Rolnick and collaborators studied replay for continual learning without task-boundary labels in their reinforcement-learning setting. That result does not eliminate the challenges of large-scale language-model updating. [2](https://arxiv.org/abs/1811.11682)

```python
for fresh_batch in stream:
    old_batch = replay_buffer.sample()
    train_step(concatenate(fresh_batch, old_batch))
    replay_buffer.add(fresh_batch)
```

### Isolate changes
Adapters or low-rank modules can isolate some task-specific parameters. They can reduce interference in a frozen backbone, but routing, capacity growth and task selection remain problems. LoRA itself is an adaptation parameterization, not a guarantee of continual learning. [3](https://arxiv.org/abs/2106.09685)

### Evaluation
Measure forward transfer, backward transfer, forgetting and new-task learning under a stated memory/computation budget. State whether task IDs are provided at inference. Keep a joint-training baseline where feasible. A model that stores one complete checkpoint per task solves a different resource problem than a fixed-capacity continual learner.

### Limits
A growing context window is not persistent parameter learning. A recurrent state that resets between sessions is not accumulated knowledge. A continually updated retrieval index is a memory-system change, distinct from weight consolidation. Continue to [memory and test-time adaptation](#article-memory-adaptation).

### Primary references

1. Kirkpatrick et al. (2016). [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796). Paper · PNAS 2017.
2. Rolnick et al. (2018). [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682). Research paper.
3. Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Research paper.

### Related topics

- [Memory, context & test-time adaptation](#article-memory-adaptation) — Separate state, retrieval and parameter updates.
- [Self-improvement needs a verifier](#article-self-improvement) — Require independent evidence of improvement.
- [NEAT: evolve weights and topology](#article-neat) — Distinguish adaptation from population search.

---

**Family:** Learning & adaptation → Memory & continual learning

<a id="article-memory-adaptation"></a>

## Memory, context & test-time adaptation

*Identify exactly what persists.*

**Abstract.** Separate external retrieval, in-context state, parameter-efficient adaptation and gradient updates at test time.

**At a glance:**

- Context, cache, retrieval and weights store different state
- Test-time training changes parameters under a reset policy
- Persistent updates require contamination and drift checks

**Concepts:** RAG; External memory; Test-time training; LoRA; Adaptation.

### Four meanings of adaptation
| Mechanism | What changes? | What survives a reset? |
| --- | --- | --- |
| In-context conditioning | Activations and cache | Nothing unless context is saved |
| Retrieval memory | External records and index | Stored records |
| Adapter training | A selected parameter set | Saved adapter weights |
| Test-time training | Parameters updated using test-time objectives | Depends on reset policy |

These mechanisms can coexist, but they answer different questions. A system can remember facts without changing its language-model weights.

### Retrieve evidence
Retrieval-augmented generation conditions generation on retrieved documents. A simplified mixture is

$$
p(y\mid x)\approx\sum_{z\in\mathcal R(x)}p_\eta(z\mid x)\,p_\theta(y\mid x,z).
$$

$\mathcal R(x)$ is a retrieved set, $\eta$ retriever parameters and $\theta$ generator parameters. Sequence-level and token-level marginalization differ in the original RAG work. A production prompt that simply concatenates top documents is a related design rather than this exact probabilistic marginalization. [1](https://arxiv.org/abs/2005.11401)

### Learn at test time
Test-time training updates parameters from an available self-supervised objective before predicting on a shifted input distribution. Sun and collaborators demonstrate the idea on visual tasks. Whether a label-free objective improves the target task depends on its relation to that task and the shift. [2](https://proceedings.mlr.press/v119/sun20b.html)

```python
adapted = copy_parameters(base_model)
for _ in range(adaptation_steps):
    adapted = update(adapted, self_supervised_loss(unlabeled_input))
prediction = adapted(unlabeled_input)
```

A copied/reset model and a continuously updated model create different evaluation protocols. Explicitly state which is used. Low-rank adaptation can restrict the update space but does not solve objective selection. [3](https://arxiv.org/abs/2106.09685)

### Evaluate persistence and interference
Test retained behavior after adaptation and after reset. Account for memory write policies, stale documents, retrieval failures and adaptation compute. For streaming updates, include forgetting diagnostics such as those motivated by replay work. [4](https://arxiv.org/abs/1811.11682)

### Limits
Retrieval can provide useful evidence without guaranteeing faithful use. Test-time updates can amplify noisy or unrepresentative inputs. More context, more stored facts and better weights should not be reported as one undifferentiated “self-learning” capability.

### Primary references

1. Lewis et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). Research paper.
2. Sun et al. (2020). [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://proceedings.mlr.press/v119/sun20b.html). Paper · ICML 2020.
3. Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Research paper.
4. Rolnick et al. (2018). [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682). Research paper.

### Related topics

- [KV-cache memory & grouped heads](#article-kv-cache) — Inspect transient inference state.
- [Remembering while learning](#article-continual-learning) — Measure persistence and forgetting.
- [Self-improvement needs a verifier](#article-self-improvement) — Evaluate a sustained update loop.

---

**Family:** Learning & adaptation → Memory & continual learning

<a id="article-self-improvement"></a>

## Self-improvement needs a verifier

*A feedback loop is only as good as its selection signal.*

**Abstract.** Distinguish output refinement, synthetic-data training and measurable improvements on held-out tasks.

**At a glance:**

- An improvement loop needs an external criterion
- Self-generated data can amplify existing errors
- Held-out evaluation must survive repeated selection

**Concepts:** Self-training; Verifiers; Self-improvement; Evaluation.

### Improve an answer or improve a model?
Self-Refine repeatedly generates feedback and revisions with fixed model weights. STaR instead generates reasoning traces, selects or rationalizes correct solutions using answer supervision, and fine-tunes on resulting data. One changes an output at inference; the other changes parameters through a training loop. [1](https://arxiv.org/abs/2203.14465) [2](https://arxiv.org/abs/2303.17651)

### A generic generate–verify–learn loop
Let $G_{\theta_k}$ generate candidates from prompts $x$, and let $V(x,y)$ be a selection score. One schematic update is

$$
\mathcal D_k=\{(x,y):y\sim G_{\theta_k}(\cdot\mid x),\;V(x,y)\ge\tau\},\qquad
\theta_{k+1}=\operatorname{Train}(\theta_k,\mathcal D_k).
$$

$\tau$ is an acceptance threshold. This formalizes a family of procedures, not the exact STaR algorithm. Selection can use known answers, unit tests, execution, a learned reward model or human judgment. Each verifier has blind spots.

```python
candidates = generate(training_prompts)
accepted = [c for c in candidates if independent_verifier(c)]
next_model = train(current_model, accepted)
evaluate(next_model, untouched_test_set)
```

### The hard part is evidence
A model judging its own output may prefer its own characteristic mistakes. Training on accepted synthetic text can amplify a biased selection signal. Repeatedly inspecting a test set can turn it into training feedback even without gradient updates. Preference-based post-training also depends on what labels reward. [3](https://arxiv.org/abs/2203.02155)

### Evaluation protocol
Freeze a genuinely held-out task suite before the loop. Track compute, external information and supervision supplied by the verifier. Include comparisons with more samples, more inference steps and ordinary supervised training at matched cost. Test whether improvement transfers beyond the verifier's original task distribution.

### Open questions
A narrow system with reliable executable feedback can improve a specific skill. That is not a demonstration of open-ended autonomous scientific progress or a general recursive-intelligence law. The interesting engineering question is where a trustworthy, informative signal can be obtained cheaply enough to drive a useful loop.

### Primary references

1. Zelikman, Wu, Mu & Goodman (2022). [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465). Research paper.
2. Madaan et al. (2023). [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651). Research paper.
3. Ouyang et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). Research paper.

### Related topics

- [Remembering while learning](#article-continual-learning) — Check retained capabilities.
- [Compression & speculative decoding](#article-compression-decoding) — Contrast verification during inference.
- [Evaluate the closed loop](#article-robot-evaluation) — Evaluate changes in an external loop.

---

# Multimodal & world models

Connect modality interfaces and objectives to predictive representations and latent world models.

**Family:** Multimodal & world models → Modality interfaces & objectives

<a id="article-multimodal-fusion"></a>

## How modalities enter a Transformer

*Tokenization is an interface; fusion is a design choice.*

**Abstract.** Compare patch projection, learned resampling, prefix concatenation and cross-attention without conflating their objectives.

**At a glance:**

- An encoder maps each modality into representations
- Prefix, query and cross-attention are different interfaces
- Resolution and token count change the compute budget

**Concepts:** Vision encoders; Cross-attention; Query resampling; Modality tokens.

### A common width does not imply a common meaning
Text IDs become learned lookup vectors. An image can be divided into patches, projected into a sequence and processed by a vision encoder. Audio may become time–frequency features or learned tokens. To enter a language backbone, these representations need a compatible width and a specified ordering/masking convention. [1](https://arxiv.org/abs/2010.11929)

For image features $Z\in\mathbb R^{p\times d_v}$ and a learned projector $P\in\mathbb R^{d_v\times d}$, define $V=ZP$. $p$ is visual-token count, $d_v$ encoder width and $d$ language width. A width match does not guarantee semantic alignment; training establishes how the backbone uses these vectors.

### Three fusion patterns
| Pattern | Interface | Main design issue |
| --- | --- | --- |
| Visual prefix | Concatenate projected visual tokens with text | Sequence cost and masking |
| Learned query bottleneck | Query a larger feature set to produce fewer tokens | Information lost by compression |
| Cross-attention | Language states query visual memory in selected layers | Extra projections and insertion points |

LLaVA provides an influential encoder–projector–language-model route. BLIP-2 uses a Q-Former between frozen pretrained components. Flamingo uses a visual resampler and gated cross-attention to connect pretrained vision and language systems. These are specific designs, not interchangeable names for “multimodal attention.” [2](https://arxiv.org/abs/2204.14198) [3](https://arxiv.org/abs/2301.12597) [4](https://arxiv.org/abs/2304.08485)

### Inspect the interface
Switch the fusion mode in the vector figure. Count which tensors join the language sequence and which remain external memory. The displayed token counts are illustrative; actual models differ in image crops, resolution, resampling and temporal handling.

```python
visual = vision_encoder(images)
visual_tokens = projector(visual)
sequence = concatenate([visual_tokens, text_embeddings], axis='positions')
answer = decoder(sequence, mask=multimodal_mask)
```

### Evidence and limits
A captioning improvement does not establish precise spatial grounding. Frozen backbones reduce trainable parameters but do not remove the cost of their forward passes. Evaluation should vary image content while preserving the question, and include cases where language priors suggest the wrong answer. Continue to [training objectives](#article-multimodal-objectives).

### Primary references

1. Dosovitskiy et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). Paper · ICLR 2021.
2. Alayrac et al. (2022). [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198). Research paper.
3. Li et al. (2023). [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597). Research paper.
4. Liu, Li, Wu & Lee (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485). Paper · NeurIPS 2023.

### Related topics

- [Alignment, generation & instruction](#article-multimodal-objectives) — Separate interface from objective.
- [Audio, video & temporal grounding](#article-temporal-modalities) — Add time and synchronization.
- [Vision–language–action models](#article-vla-foundations) — Turn perception into action conditioning.

---

**Family:** Multimodal & world models → Modality interfaces & objectives

<a id="article-multimodal-objectives"></a>

## Alignment, generation & instruction

*The loss decides what the representation must preserve.*

**Abstract.** Separate contrastive pairing, conditional generation and visual instruction tuning.

**At a glance:**

- Contrastive training aligns paired representations
- Generative training predicts an output distribution
- Instruction tuning specifies interaction behavior

**Concepts:** CLIP; Contrastive learning; Generative objectives; Instruction tuning.

### Contrastive alignment
CLIP learns image and text encoders whose normalized outputs are compared through similarity. For a batch of $B$ paired examples, one direction of a contrastive loss is

$$
\mathcal L_{I\to T}=-\frac1B\sum_i\log\frac{\exp(s(v_i,t_i)/\tau)}{\sum_j\exp(s(v_i,t_j)/\tau)}.
$$

$v_i,t_i$ are normalized image/text embeddings, $s$ their dot product and $\tau>0$ temperature. The symmetric objective also scores text-to-image retrieval. In-batch negatives create the classification problem; false negatives and biased captions can affect learning. [1](https://arxiv.org/abs/2103.00020)

### Conditional language generation
A generative vision-language objective instead scores text given visual input $v$:

$$
\mathcal L_{\mathrm{gen}}=-\sum_t\log p_\theta(y_t\mid y_{<t},v).
$$

This asks for a token distribution, not merely a paired-embedding similarity. A model can use contrastive and generative terms in different phases. BLIP-2 is a useful example of a staged bridge between pretrained vision and language components. [2](https://arxiv.org/abs/2301.12597)

### Visual instruction tuning
Instruction data specify how to answer questions about images. LLaVA studied generating multimodal instruction data and fine-tuning a vision-language assistant. Instruction following adds behavior and task formatting; it does not prove that the model's visual representation captures every answer-relevant detail. [3](https://arxiv.org/abs/2304.08485)

```python
loss = image_text_contrastive_loss(image_repr, text_repr)
# A separate generative stage may optimize answer tokens only.
loss = conditional_answer_loss(images, instructions, answers, answer_mask)
```

### Keep the axes separate
An early-fusion model can be trained generatively. A dual encoder can be trained contrastively. A query bottleneck can support several losses. The interface, attention pattern, data mixture and training objective should be separate fields in a model comparison.

### Research questions
Test compositional retrieval, fine-grained attribute swaps and image-question counterfactuals. Include resolution and visual-token budgets. Compare a language-only baseline to quantify how much the task rewards visual evidence rather than dataset priors. Report which encoders were frozen during each phase.

### Primary references

1. Radford et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). Research paper.
2. Li et al. (2023). [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597). Research paper.
3. Liu, Li, Wu & Lee (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485). Paper · NeurIPS 2023.

### Related topics

- [JEPA: predicting representations](#article-jepa) — Compare prediction in representation space.
- [How modalities enter a Transformer](#article-multimodal-fusion) — Inspect the architectural interface.
- [Diffusion & flow-based generation](#article-diffusion-flow) — Change the generative objective.

---

**Family:** Multimodal & world models → Modality interfaces & objectives

<a id="article-temporal-modalities"></a>

## Audio, video & temporal grounding

*Sampling is part of the model.*

**Abstract.** Understand why ordering, synchronization and missing information matter as much as modality adapters.

**At a glance:**

- Audio and video add temporal structure
- Frame sampling trades detail for sequence length
- Temporal grounding needs evaluation beyond recognition

**Concepts:** Audio; Video; Temporal grounding; Synchronization.

### Time is an information bottleneck
A video encoder sees sampled frames or clips, not the full continuous world. Audio frontends also discretize time. Frame rate, crop strategy and receptive field determine which events survive. A later language layer cannot recover motion that was never observed.

For a clip duration $T$ seconds sampled at $f$ frames per second with $p$ visual tokens per frame, an uncompressed visual sequence has approximately $n_v=Tfp$ tokens. With an illustrative $T=10,f=2,p=196$, this is 3,920 tokens before text and any compression. This arithmetic is not a claim about a particular released model.

### Temporal position and synchronization
Spatial patch indices and temporal indices serve different roles. Interleaving frames and text requires masks that preserve intended access. Audio/video synchronization can be represented by timestamps or aligned feature windows, but positional tags do not correct an inaccurate clock or dropped observations.

### Two concrete reference points
Whisper is an encoder–decoder speech model trained on large-scale weak supervision with task and language conditioning. It is a speech-system reference, not an example of every “native audio” LLM. Flamingo provides a visual-language architecture that can ingest images and videos through its visual interface. [1](https://arxiv.org/abs/2212.04356) [2](https://arxiv.org/abs/2204.14198)

```python
frames, times = sample_video(video, frames_per_second=2)
features = vision_encoder(frames)
tokens = add_temporal_information(features, times)
```

The pseudocode leaves the temporal representation abstract because the choice is architecture-specific. ViT-style spatial patching alone does not define video dynamics. [3](https://arxiv.org/abs/2010.11929)

### What to evaluate
Use tasks that distinguish event order, duration, state change and cross-modal correspondence. Shuffle frames or shift audio as controlled ablations. A strong clip caption can coexist with poor temporal localization. Report sampling density, input duration and whether the test answer is recoverable from one static frame.

### Bridge to action
For robotics, latency makes the observed state older than the current physical state. A language model's plausible description cannot substitute for a timestamped observation and a control loop. Continue to [closed-loop evaluation](#article-robot-evaluation).

### Primary references

1. Radford et al. (2022). [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356). Research paper.
2. Alayrac et al. (2022). [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198). Research paper.
3. Dosovitskiy et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). Paper · ICLR 2021.

### Related topics

- [Video world models & latent planning](#article-video-world-models) — Predict video representations.
- [How modalities enter a Transformer](#article-multimodal-fusion) — Inspect modality interfaces.
- [Evaluate the closed loop](#article-robot-evaluation) — Connect time to closed-loop control.

---

**Family:** Multimodal & world models → Predictive representations

<a id="article-jepa"></a>

## JEPA: predicting representations

*Learn what is missing without reconstructing every pixel.*

**Abstract.** Compare the context encoder, target encoder and predictor in I-JEPA and V-JEPA, with explicit masks, feature dimensions and collapse prevention.

**At a glance:**

- Predict target features, not necessarily pixels.
- Separate gradient training from moving-average targets.
- JEPA is a learning arrangement, not a replacement for attention.

**Concepts:** JEPA; I-JEPA; V-JEPA; Self-supervised learning; Stop-gradient.

### Predict a representation
A Joint-Embedding Predictive Architecture (JEPA) learns to predict a target's representation from a context representation. The family describes a learning arrangement, not a replacement for attention: its encoders and predictor can themselves be Transformers. I-JEPA instantiates this idea for images, predicting target-block features from an informative context block. Its masking choices encourage prediction beyond immediately adjacent texture. [1](https://arxiv.org/abs/2301.08243)

The distinction from reconstruction is the prediction target. A pixel reconstruction loss asks for an image array; a JEPA loss compares learned features. This permits the representation to discard some visual detail, but does not establish that every retained feature is useful or that all discarded detail is irrelevant. The diagram switches the target space while keeping this distinction visible.

### Context, target and predictor
Let an image or clip contain $N$ patch positions. A context mask selects $C$ and target positions form $T$, with $C\cap T=\varnothing$. A useful schematic is

$$
Z_C=f_\theta(x_C),\quad
Y_T=[f_{\bar\theta}(x)]_T,\quad
\widehat Y_T=g_\phi(Z_C,p_T).
$$

$Z_C\in\mathbb R^{|C|\times d}$ contains visible-context features; $Y_T,\widehat Y_T\in\mathbb R^{|T|\times d}$ are target and predicted features. $p_T$ identifies the requested positions. In the loss below, rows retain their original target-position labels even when stored as a compact array. The target encoder can see the full input; target positions are selected from its output. The context branch must not silently receive the hidden target pixels. Multiple target masks and normalization details belong to each implementation. [1](https://arxiv.org/abs/2301.08243) [2](https://arxiv.org/abs/2404.08471)

These are separate token-indexed feature arrays. A diagram that reduces each branch to one box is summarizing its computation, not claiming that the encoder has only one shared residual vector. Inside a Transformer encoder, every visible patch retains its own residual state.

### Why two encoder updates differ
For illustration, a feature-regression objective is

$$
\mathcal L=\frac1{|T|d}\sum_{i\in T}\sum_{k=1}^d
\rho(\widehat Y_{ik}-\operatorname{sg}(Y_{ik})),\qquad
\bar\theta\leftarrow m\bar\theta+(1-m)\theta.
$$

$\operatorname{sg}$ stops gradients through the target; $m\in[0,1)$ controls an exponential moving average. Gradient updates train the context encoder and predictor, while the target follows the moving average. The penalty $\rho$ is method-specific: V-JEPA uses absolute error. This is a family-level notation, not a claim that all JEPA variants use identical losses. Naive jointly learned feature matching admits a constant-output collapse; the asymmetric predictor, moving-average target and stop-gradient form a practical prevention strategy, not a universal theorem of successful representation learning. [2](https://arxiv.org/abs/2404.08471)

### From images to video
I-JEPA uses spatial image context and targets. V-JEPA extends masked feature prediction to video, making positions spatiotemporal and evaluating motion-sensitive representations. A hidden region need not be strictly in the future: masked video modeling and causal forecasting are different prediction tasks. V-JEPA 2 scales video representation learning, while its action-conditioned extension adds a further stage for planning. Follow [video world models](#article-video-world-models) for that distinction. [2](https://arxiv.org/abs/2404.08471) [3](https://arxiv.org/abs/2506.09985)

### Inspect what was learned
An informative comparison holds backbone size, data and compute approximately fixed and varies the prediction target or mask. Inspect feature variability, frozen-feature probes and full fine-tuning separately. A trained probe introduces supervision even when pretraining used none. Our suggested diagnostic is to vary context size and ask whether improvements transfer across appearance and motion tasks.

The official I-JEPA demonstrations visualize features using a separately trained generative decoder. Such pictures help inspect representations; they do not turn the underlying feature predictor into a pixel generator. Similarly, low feature error alone does not establish causal understanding, physical accuracy or useful planning. [4](https://github.com/facebookresearch/ijepa)

### Primary references

1. Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun & Ballas (2023). [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243). Paper · CVPR 2023.
2. Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran & Ballas (2024). [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). Research paper.
3. Assran et al. (2025). [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). Research paper.
4. Meta FAIR (2023). [I-JEPA official implementation](https://github.com/facebookresearch/ijepa). Official code.

### Related topics

- [Video world models & latent planning](#article-video-world-models) — Extend feature prediction to action-conditioned planning.
- [Alignment, generation & instruction](#article-multimodal-objectives) — Contrast alignment and predictive objectives.
- [How modalities enter a Transformer](#article-multimodal-fusion) — Separate the objective from the encoder interface.
- [Predictive coding as local inference](#article-predictive-coding) — Compare different prediction-based learning schemes.

---

**Family:** Multimodal & world models → Predictive representations

<a id="article-video-world-models"></a>

## Video world models & latent planning

*Predicting observations and choosing actions are different interfaces.*

**Abstract.** Trace action-free video representations to action-conditioned prediction, candidate rollouts and closed-loop planning, using V-JEPA 2 as a concrete example.

**At a glance:**

- Video pretraining does not yet specify control actions.
- Action-conditioned post-training needs interaction signals.
- Evaluate latent prediction and closed-loop success separately.

**Concepts:** World models; V-JEPA 2; Latent planning; Model predictive control.

### What does a world model predict?
A useful operational definition is a learned model that predicts aspects of an environment's evolution. The predicted object may be pixels, latent features or other state variables. A visually convincing video generator and a useful action-conditioned predictor therefore require different evaluations. For control, ask whether changing a proposed action produces a useful prediction of its consequences. A model can represent video well without already providing that interface.

V-JEPA learns masked video features; V-JEPA 2 develops larger-scale visual representations. V-JEPA 2-AC adds action conditioning on top of a frozen V-JEPA 2 encoder. Keep these names separate: action-free pretraining is not itself the robot controller. [1](https://arxiv.org/abs/2404.08471) [2](https://arxiv.org/abs/2506.09985)

### Observation is not intervention
Action-free video can reveal regularities of movement and appearance. It does not supply an explicit response to a command the observing agent could choose. V-JEPA 2-AC is post-trained with robot interaction data containing visual observations and end-effector state signals. “Unlabeled” here excludes additional task, success and reward annotations; it does not mean that all information about robot movement is absent. [2](https://arxiv.org/abs/2506.09985)

The official repository separates video encoders from the action-conditioned backbone and provides an example for inspecting a learned energy landscape from a robot trajectory. It also exposes probe-based evaluations. The repository also lists V-JEPA 2.1, which revises representation training; the control example here specifically concerns V-JEPA 2-AC. Recognition, prediction and control should not be collapsed into one score. [3](https://github.com/facebookresearch/vjepa2)

### Roll out candidate actions
The following notation illustrates an action-conditioned latent model; it is not the exact V-JEPA 2-AC tensor layout:

$$
z_t=f(o_{\leq t}),\quad
\widehat z_{t+k+1}=F_\psi(\widehat z_{t+k},a_{t+k},q_{t+k}),\quad
\widehat z_t=z_t.
$$

$o$ is an observation history, $z\in\mathbb R^d$ a simplified feature vector, $a\in\mathbb R^r$ an action and $q$ robot state. Real predictors can retain token arrays and longer histories. Future $q$ must be supplied by a specified transition convention or predicted; a planner cannot read future measured robot states before executing. The diagram omits this implementation detail to expose the difference between observational prediction and action-conditioned rollouts.

For an illustrative image-goal planner, encode $z_g=f(o_g)$ and solve

$$
A^*=\arg\min_{A\in\mathcal A^H}
D(\widehat z_{t+H}(A),z_g)+\lambda C(A).
$$

$A=(a_t,\ldots,a_{t+H-1})$ is a candidate action sequence, $D$ a feature discrepancy and $C$ an optional constraint/cost term. This is a generic planning objective. The illustrated candidate scores are not learned-model measurements or reproduced paper results.

### Close the loop
Execute a prefix of the selected sequence, observe again and replan. This allows new evidence to replace a potentially inaccurate imagined rollout. It also costs sensing and planning time. An action horizon, number of sampled candidates, executed prefix and observation delay are separate quantities; reducing one does not automatically improve all the others.

The V-JEPA 2 work demonstrates image-goal robot planning, with limitations including camera sensitivity and long-horizon behavior. These experiments support particular environments and tasks, not arbitrary physical understanding. A feature-space goal can also omit a requirement that matters to a user. [2](https://arxiv.org/abs/2506.09985)

### Evaluate prediction and control separately
For an exploratory experiment, first compare one-step and multi-step prediction under held-out action sequences. Then evaluate closed-loop success, recovery, latency and interventions. Record camera geometry, action units, robot/controller configuration and goal-image construction. Include a simple control baseline and report candidate-search compute.

Our diagram deliberately shows an abstract candidate comparison rather than a photorealistic generated future: its purpose is to locate the decision boundary. A promising latent rollout still needs an executable action interface and feedback. Connect this topic to [JEPA](#article-jepa), [action models](#article-action-models), and [closed-loop evaluation](#article-robot-evaluation).

### Primary references

1. Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran & Ballas (2024). [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). Research paper.
2. Assran et al. (2025). [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). Research paper.
3. Meta FAIR (2025). [V-JEPA 2 official implementation](https://github.com/facebookresearch/vjepa2). Official code.

### Related topics

- [JEPA: predicting representations](#article-jepa) — Inspect masked representation prediction.
- [Audio, video & temporal grounding](#article-temporal-modalities) — Connect video features to temporal structure.
- [Active inference](#article-active-inference) — Compare predictive control formulations.
- [Tokens, diffusion & flow for action](#article-action-models) — Contrast latent planning with action generation.
- [Evaluate the closed loop](#article-robot-evaluation) — Evaluate the executed control loop.

---

# Generation & latent computation

Explore generative constructions and computation in continuous representation spaces.

**Family:** Generation & latent computation → Generative constructions

<a id="article-diffusion-flow"></a>

## Diffusion & flow-based generation

*Change the generative process, keep the architecture question open.*

**Abstract.** Contrast Gaussian denoising, masked language diffusion and continuous transport models.

**At a glance:**

- Generation iteratively transforms noise or masked states
- Diffusion and flow matching define different training constructions
- A parallel reveal schedule is not a trained sampler

**Concepts:** Diffusion; Flow matching; Denoising; Discrete diffusion.

### Continuous denoising
A familiar Gaussian corruption is

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad\epsilon\sim\mathcal N(0,I).
$$

$x_0$ is data, $t$ a noise level and $\bar\alpha_t\in[0,1]$ the signal-retention schedule. A network may predict noise, clean data or another equivalent parameterization under stated assumptions. Sampling involves a chosen reverse-time solver. The denoiser architecture need not be a Transformer. [1](https://arxiv.org/abs/2006.11239)

### Discrete language diffusion
LLaDA instead uses a forward masking process and a Transformer trained to predict masked tokens. Iterative generation can fill positions in an order unlike left-to-right autoregression. Changing the objective and decoding procedure does not mean the attention backbone has disappeared. [2](https://arxiv.org/abs/2502.09992)

The interactive figure shows a **deterministic illustrative reveal schedule**, not samples from a trained diffusion model. Move the step control to compare left-to-right reveal with a mask-based order. A real diffusion sampler must specify confidence, remasking, conditioning and number of network evaluations.

### Continuous flow matching
For noise $x_0$ and data $x_1$, a linear interpolant is $x_t=(1-t)x_0+tx_1$. A simple flow-matching objective is

$$
\mathcal L_{\mathrm{FM}}=\mathbb E_{t,x_0,x_1}\left\|v_\theta(x_t,t)-(x_1-x_0)\right\|^2.
$$

$v_\theta$ predicts a velocity field. Sampling integrates $dx/dt=v_\theta(x,t)$. The conditional straight-line target does not imply that every learned marginal trajectory is exactly straight or that one Euler step is always accurate. [3](https://arxiv.org/abs/2209.03003)

```python
x_t = (1 - t) * noise + t * data
loss = mean_squared_error(velocity(x_t, t), data - noise)
```

### Evidence and limits
Compare generation quality against **network evaluations, total FLOPs, latency and output length**, not only the number of decoding rounds. A bidirectional denoiser can expose more context per iteration, but repeated full-sequence passes can be costly. Variable lengths and KV reuse require method-specific treatment.

### Primary references

1. Ho, Jain & Abbeel (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Research paper.
2. Nie et al. (2025). [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992). Research paper.
3. Liu, Gong & Liu (2022). [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003). Research paper.

### Related topics

- [Tokens, diffusion & flow for action](#article-action-models) — Generate action trajectories.
- [Soft prompts & continuous thought](#article-soft-latent-reasoning) — Distinguish generation from latent computation.
- [JEPA: predicting representations](#article-jepa) — Contrast output prediction spaces.

---

**Family:** Generation & latent computation → Continuous computation

<a id="article-soft-latent-reasoning"></a>

## Soft prompts & continuous thought

*Three constructions that should not share one vague name.*

**Abstract.** Distinguish optimized input vectors, distributions over vocabulary embeddings and feedback of hidden reasoning states.

**At a glance:**

- Soft prompts are trainable continuous input vectors
- Latent reasoning adds hidden-state computation
- A soft-token mixture is a separate construction

**Concepts:** Soft prompts; Latent reasoning; Continuous thought; Soft tokens.

### Soft prompts
Prompt tuning optimizes continuous vectors prepended to a frozen language model. If $P\in\mathbb R^{m\times d}$ is a learned prompt and $E(x)$ the embedded input,

$$
P^*=\arg\min_P\;\mathbb E_{(x,y)}[-\log p_{\theta\ \mathrm{fixed}}(y\mid[P;E(x)])].
$$

$m$ is prompt length and $d$ model width. These vectors need not correspond to vocabulary entries and are typically task parameters learned during training. This differs from low-rank weight adaptation, where $\Delta W=BA$ modifies a projection through trainable low-rank factors. [1](https://arxiv.org/abs/2104.08691) [3](https://arxiv.org/abs/2106.09685)

### Soft token mixtures
Another possible construction embeds a probability distribution $p\in\Delta^{V-1}$ as $v=\sum_k p_kE_k$. The simplex $\Delta^{V-1}$ contains nonnegative weights summing to one. Such a convex mixture is mathematically distinct from an arbitrary learned prompt vector; not every vector lies in the convex hull of the embedding table. This formula describes a representation option, not a claim that all latent reasoning systems use it.

### Hidden-state feedback
Coconut feeds a last hidden state back as a subsequent input embedding during latent reasoning, rather than decoding each step into a word. That state is input-dependent and computed at inference; it is not simply a fixed soft prompt. [2](https://arxiv.org/abs/2412.06769)

```python
state = model(context).last_hidden_state
for _ in range(latent_steps):
    state = model(inputs_embeds=state).last_hidden_state
# Schematic only: real implementations preserve their context and cache layout.
```

### What continuous does not imply
A continuous vector is not automatically an interpretable probability distribution over alternative thoughts. A paper's search analogy must be separated from a verified explicit search algorithm. Latent steps consume compute even if they emit no readable tokens.

### Experiment
Compare visible chain-of-thought and latent reasoning with matched total compute and training supervision. Measure generalization outside the training task family, not only token savings. Inspect failure cases and whether the model learns a meaningful stopping rule. Continue to [recurrent depth](#article-recurrent-depth).

### Primary references

1. Lester, Al-Rfou & Constant (2021). [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691). Paper · EMNLP 2021.
2. Hao et al. (2024). [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769). Paper · COLM 2025.
3. Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Research paper.

### Related topics

- [Recurrent depth & looped Transformers](#article-recurrent-depth) — Reuse depth for extra computation.
- [Diffusion & flow-based generation](#article-diffusion-flow) — Compare continuous computation and generation.
- [Memory, context & test-time adaptation](#article-memory-adaptation) — Separate prompts from persistent adaptation.
- [Token embeddings & embedding space](#article-token-embeddings) — Inspect token lookup, embedding geometry and the output decoder explicitly.

---

# Embodied intelligence

Connect perception, beliefs, action distributions, and closed-loop evaluation.

**Family:** Embodied intelligence → Policies & action representations

<a id="article-vla-foundations"></a>

## Vision–language–action models

*A language interface reaches the control loop.*

**Abstract.** Trace RT-2 and OpenVLA, then locate the boundary between semantic transfer and physical competence.

**At a glance:**

- Visual and language inputs condition an action policy
- Action representation is an architectural choice
- A policy acts inside an observation–action feedback loop

**Concepts:** VLA; RT-2; OpenVLA; Robot policies.

### From observations to actions
A VLA policy conditions actions on visual observations, language and often robot state:

$$
\pi_\theta(a_{t:t+H-1}\mid o_{\le t},\ell,q_t).
$$

$o$ denotes observations, $\ell$ a language instruction, $q_t$ proprioceptive state and $H$ an action horizon. Not every published VLA uses every conditioning variable or predicts a chunk; this notation identifies the design dimensions rather than imposing one architecture.

### Two foundational examples
RT-2 studies co-fine-tuning vision-language models with robot trajectories and web-derived vision-language tasks, representing actions through a token interface. OpenVLA provides an open 7B model built from a language backbone and visual representations, with a recipe for robot adaptation. Their reported success rates belong to their own task and embodiment evaluations. [1](https://arxiv.org/abs/2307.15818) [2](https://arxiv.org/abs/2406.09246)

### Representation is not control
A visual-language backbone can transfer object and instruction knowledge. It still needs action supervision and compatible actuation semantics. Joint positions, joint deltas, end-effector pose deltas, forces and gripper commands have different units, constraints and controllers. Normalizing an action vector is not sufficient to make robots interchangeable.

The vector figure follows visual features and language through a policy, an action representation, a low-level controller and a new observation. Toggle discrete or continuous actions to locate the changing component. It is a generic interface diagram, not an exact reproduction of RT-2, OpenVLA or π₀.

### Continuous action experts
π₀ builds on a pretrained VLM with a flow-matching action architecture. This bridges a semantic representation to continuous action chunks, rather than requiring each action coordinate to be generated as ordinary text. [3](https://arxiv.org/abs/2410.24164)

```python
observation = synchronize(cameras, robot_state)
action_chunk = policy(observation, instruction)
controller.execute(action_chunk[:execution_horizon])
```

### Limits and research questions
Generalization across objects, language, scenes and embodiments are different axes. Report dataset overlap, action normalization, fine-tuning data and intervention policy. A plausible instruction-conditioned action is not automatically feasible under contact, latency or actuator limits. Read [action representations](#article-action-models) and [closed-loop evaluation](#article-robot-evaluation).

### Primary references

1. Brohan et al. (2023). [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818). Research paper.
2. Kim et al. (2024). [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246). Research paper.
3. Black et al. (2024). [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164). Research paper.

### Related topics

- [How modalities enter a Transformer](#article-multimodal-fusion) — Inspect the perceptual interface.
- [Tokens, diffusion & flow for action](#article-action-models) — Choose an action distribution.
- [Video world models & latent planning](#article-video-world-models) — Compare prediction and planning.

---

**Family:** Embodied intelligence → Policies & action representations

<a id="article-action-models"></a>

## Tokens, diffusion & flow for action

*A multimodal action distribution deserves a structured output.*

**Abstract.** Compare discrete autoregressive actions with continuous action-chunk models and receding-horizon execution.

**At a glance:**

- Discrete tokens and continuous heads model actions differently
- Diffusion and flow can produce action chunks
- A controller may execute a prefix before observing again

**Concepts:** Action tokens; Action chunks; Diffusion policy; Flow policy.

### Discretized actions
A continuous action coordinate can be quantized to a finite set of bins and predicted as a token. This connects robot supervision to a language-model output interface, as studied in RT-2. Quantization resolution, coordinate order and autoregressive dependencies affect precision and latency. [1](https://arxiv.org/abs/2307.15818)

### Continuous action distributions
Diffusion Policy models an action sequence through conditional denoising, supporting multimodal action distributions rather than collapsing incompatible actions into their mean. π₀ uses a flow-matching action architecture on top of a VLM. These share iterative generative computation but are not identical objectives or samplers. [2](https://arxiv.org/abs/2303.04137) [3](https://arxiv.org/abs/2410.24164)

For a simple conditional flow objective, let $a_0$ be noise, $a_1$ a demonstrated action chunk and $c$ observation/instruction context:

$$
a_\tau=(1-\tau)a_0+\tau a_1,\qquad
\mathcal L=\mathbb E\|v_\theta(a_\tau,\tau,c)-(a_1-a_0)\|^2.
$$

$\tau\in[0,1]$ is generative time, not wall-clock robot time. The action horizon is a separate axis. The formula illustrates conditional flow matching; each model specifies its own parameterization and conditioning. [4](https://arxiv.org/abs/2209.03003)

### Execute a prefix, then observe again
Predict a horizon $H$, execute only $h\le H$ actions and replan with a new observation. This receding-horizon policy can react more often than executing the entire plan. Smaller $h$ increases inference demand; larger $h$ increases open-loop exposure.

```python
while task_active:
    chunk = sample_actions(policy, recent_observation)
    execute(chunk[:h])
    recent_observation = observe()
```

### What to measure
Report control frequency, inference latency distribution, chunk length, executed prefix, action units and controller behavior. Compare task success under disturbances, not only offline action reconstruction. Inference steps are not free; use matched latency or total-compute comparisons when contrasting token and flow policies.

### Limits
A generative model can represent several plausible action sequences yet still sample an unsafe one. Contact physics, calibration and feedback determine execution. A low-level controller and explicit constraints remain part of the system being evaluated.

### Primary references

1. Brohan et al. (2023). [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818). Research paper.
2. Chi et al. (2023). [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137). Paper · RSS 2023, extended 2024.
3. Black et al. (2024). [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164). Research paper.
4. Liu, Gong & Liu (2022). [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003). Research paper.

### Related topics

- [Diffusion & flow-based generation](#article-diffusion-flow) — Inspect the generative construction.
- [Vision–language–action models](#article-vla-foundations) — Place actions in a VLA policy.
- [Evaluate the closed loop](#article-robot-evaluation) — Measure the complete control loop.

---

**Family:** Embodied intelligence → Inference & control

<a id="article-active-inference"></a>

## Active inference

*Belief updating and action under a generative model.*

**Abstract.** Place variational free energy and policy selection in a probabilistic control framework, and distinguish them from supervised PC training.

**At a glance:**

- Perception updates beliefs about hidden states
- Policies can trade preference fulfilment and information gain
- A generative control framework is not one neural architecture

**Concepts:** Active inference; Free energy; Belief updates; Control.

### Infer hidden causes
Let $o$ be observed data, $s$ hidden states, $p(o,s)$ a generative model and $q(s)$ an approximate posterior. Variational free energy is

$$
F[q]=\mathbb E_q[\log q(s)-\log p(o,s)]
=D_{\mathrm{KL}}(q(s)\|p(s\mid o))-\log p(o).
$$

For a valid model with finite quantities, $F$ bounds negative log evidence. Minimizing it with respect to $q$ improves the approximate posterior; this does not make every numerical quantity called “energy” physical energy in joules. [1](https://mitpress.mit.edu/9780262045353/active-inference/)

### Act to shape future observations
Active inference extends belief updating to action or policy selection. It uses a generative model of state transitions and observations, with preferences and uncertainty contributing to evaluation of possible futures. Expected free energy is a policy-dependent quantity; its exact form and decomposition depend on the model and conventions. It should not be substituted blindly for the current-observation $F[q]$ above. [2](https://activeinference.github.io/papers/process_theory.pdf)

### Relation to predictive coding
Predictive coding can implement parts of inference under particular generative models and approximations. Active inference supplies a broader account of inference and action. PC-ALM instead studies an augmented-Lagrangian route to local credit assignment for a constrained supervised network. Shared words such as “prediction error,” “inference” and “free energy” do not make these objectives identical. [3](https://arxiv.org/html/2605.31022v1)

| Quantity | What it evaluates | Essential specification |
| --- | --- | --- |
| Variational free energy | Approximate posterior for observations | Generative model and posterior family |
| Expected free energy | Candidate futures/policies | Preferences, transitions and beliefs |
| PC constraint energy | Consistency of neural activities | Layer functions, clamping and penalties |

### An implementable starting point
Build a tiny partially observable task with two hidden states, noisy observations and two actions. Specify observation likelihoods, transition matrices and prior preferences numerically. Update beliefs by Bayes' rule before trying a neural approximation. This reveals what the framework assumes rather than attributing decisions to an undefined optimization principle.

### Evidence and limits
The cited book and process theory formalize a framework. They do not establish that all biological learning or every effective controller implements one exact algorithm. Compare against Bayesian filtering and conventional control/RL baselines at matched model knowledge and information access.

### Primary references

1. Parr, Pezzulo & Friston (2022). [Active Inference: The Free Energy Principle in Mind, Brain, and Behavior](https://mitpress.mit.edu/9780262045353/active-inference/). MIT Press book.
2. Friston et al. (2017). [Active Inference: A Process Theory](https://activeinference.github.io/papers/process_theory.pdf). Neural Computation.
3. Jeffrey Seely & Julian Gould (2026). [Augmented Lagrangian Predictive Coding](https://arxiv.org/html/2605.31022v1). Preprint · May 2026.

### Related topics

- [Predictive coding as local inference](#article-predictive-coding) — Compare inference dynamics.
- [Video world models & latent planning](#article-video-world-models) — Connect predictive state to planning.
- [Vision–language–action models](#article-vla-foundations) — Contrast routes to robot control.

---

**Family:** Embodied intelligence → Closed-loop evaluation

<a id="article-robot-evaluation"></a>

## Evaluate the closed loop

*Generalization, latency and intervention are separate axes.*

**Abstract.** Translate a model result into a reproducible robotics experiment with stated timing, action semantics and failure modes.

**At a glance:**

- Success rate depends on the task and execution protocol
- Latency and observation freshness affect the loop
- Generalization requires explicit distribution shifts

**Concepts:** Robotics evaluation; Closed-loop control; Distribution shift.

### The unit of evaluation is a rollout
Offline prediction error does not fully characterize a policy that changes its own observations through action. Small errors can move a robot into unfamiliar states. Evaluate repeated closed-loop rollouts under a specified initial-state distribution and success criterion. The VLA and Diffusion Policy papers illustrate why task and hardware definitions matter. [1](https://arxiv.org/abs/2406.09246) [2](https://arxiv.org/abs/2303.04137)

### A structured test matrix
| Axis | Controlled variation | What to record |
| --- | --- | --- |
| Perception | Lighting, occlusion, object identity | Failures attributable to observation |
| Language | Paraphrases and ambiguous references | Grounding errors |
| Dynamics | Object mass, friction, disturbances | Recovery and stability |
| Timing | Sensor age and inference delay | Missed deadlines, stale actions |
| Embodiment | Calibration and action convention | Transfer requirements |

Do not call a new object, a new instruction and a new robot the same kind of out-of-distribution test.

### Timing belongs in the specification
A simplified observation-to-action delay is

$$
\Delta=t_{\mathrm{capture}}+t_{\mathrm{transport}}+t_{\mathrm{inference}}+t_{\mathrm{dispatch}}.
$$

This additive expression assumes sequential stages; pipelined systems need a timing diagram with overlaps. At speed $v$, a simple constant-velocity displacement over delay is $v\Delta$. This is a dimensional illustration, not a sufficient collision or stability model.

### Demonstrations and interventions
State whether trajectories include failures, recovery, human takeovers and multiple embodiments. Report how often an operator intervenes and how interventions affect the success metric. π₀'s multi-platform setting is a concrete reference for the data and adaptation questions, not a universal benchmark standard. [3](https://arxiv.org/abs/2410.24164)

```python
record = {
    'observation_timestamp': sensor_time,
    'action_timestamp': execution_time,
    'intervention': intervention_occurred,
    'success': independently_scored_success,
}
```

### A reproducible report
Provide seeds or initial-state identifiers, number of trials, confidence intervals, controller version, model checkpoint, calibration and termination rules. Keep development episodes separate from final evaluation. Treat task success, safety violations and latency as separate outcomes instead of hiding them in one score.

### Limits
This atlas provides architecture and experiment guidance. Its diagrams do not run a robot controller, validate a deployment or establish real-world safety. Evaluate the whole feedback loop rather than infer physical competence from language fluency.

### Primary references

1. Kim et al. (2024). [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246). Research paper.
2. Chi et al. (2023). [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137). Paper · RSS 2023, extended 2024.
3. Black et al. (2024). [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164). Research paper.

### Related topics

- [Tokens, diffusion & flow for action](#article-action-models) — Connect chunking to execution latency.
- [Video world models & latent planning](#article-video-world-models) — Test predictions against control outcomes.
- [Self-improvement needs a verifier](#article-self-improvement) — Preserve an independent evaluation signal.

---

# Intellectual lineage

Trace the ideas that shaped modern neural architectures without imposing one reading order.

**Family:** Intellectual lineage → Representations & sequence models

<a id="article-distributed-representations"></a>

## From symbols to distributed representations

*Hinton’s family relations, then neural language models.*

**Abstract.** Trace the move from discrete identifiers to learned continuous features without claiming that vector geometry is symbolic logic.

**At a glance:**

- Learned vectors replace isolated symbol identities
- Distributed codes share statistical structure
- Embedding geometry depends on the objective

**Concepts:** Embeddings; Distributed representation; Neural language models.

### The representational shift
A discrete entity can enter a network as a one-hot vector $e_i\in\{0,1\}^{V}$, where $V$ is the number of identifiers. An embedding table $E\in\mathbb R^{d\times V}$ maps it to $v_i=Ee_i$. The category remains discrete; its internal representation is continuous and shared features can support generalization.

Hinton's 1986 work on distributed representations of concepts is the relevant historical thread for the family-relations example: learn internal representations from relational facts rather than hand-design one feature per concept. His author bibliography distinguishes this contribution from the related Rumelhart–Hinton–Williams backpropagation papers. These works should not be collapsed into one title. [1](https://www.cs.toronto.edu/~hinton/backprop.html) [2](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)

### A relational learning problem
Conceptually, a model learns a mapping

$$
(\text{person}_1,\text{relation})\longmapsto p(\text{person}_2),\qquad
\mathcal L=-\log p_\theta(\text{person}_2\mid\text{person}_1,\text{relation}).
$$

The notation is a modern abstraction of the task, not an exact reconstruction of its original circuitry. Similar hidden patterns may capture regularities shared across family members. This is evidence that learned continuous representations can support structured relational tasks; it does not establish a complete symbolic theorem prover.

### From entities to word sequences
Bengio and collaborators' 2003 neural probabilistic language model jointly learned word representations and a probability function over sequences. Shared continuous features allow one observed sequence to inform predictions for related sequences. This is a direct conceptual ancestor of learned token embeddings and neural next-token modeling, although that model was not a Transformer. [3](https://www.jmlr.org/papers/v3/bengio03a.html)

```python
context = embedding[token_ids]  # categorical lookup, differentiable parameters
features = nonlinear_network(context.flatten())
next_word_distribution = softmax(output(features))
```

### Influence, precisely stated
Distributed representation is a representational idea. Backpropagation is a credit-assignment method. Autoregressive language modeling is a probabilistic factorization. A Transformer is an architecture. Their later combination should not erase these distinctions or assign every component to one inventor.

### Reading exercise
Rebuild a small relation-completion task with a held-out set of facts. Compare training-fact memorization with generalization to unseen combinations. Control the split carefully: symmetry and inverse relations can leak answers across partitions. Then compare a learned lookup with fixed random embeddings at matched network size.

### Inspect the modern interface
For explicit lookup and unembedding shapes, weight tying, contextual geometry and a small interactive example, open [Token embeddings & embedding space](#article-token-embeddings). For reading internal states rather than input vectors, explore [Transformer observability & interpretability](#article-transformer-observability).

### Primary references

1. Geoffrey E. Hinton (1986). [Learning distributed representations of concepts](https://www.cs.toronto.edu/~hinton/backprop.html). Author bibliography.
2. Rumelhart, Hinton & Williams (1986). [Learning representations by back-propagating errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf). Research paper.
3. Bengio, Ducharme, Vincent & Jauvin (2003). [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html). JMLR.

### Related topics

- [The residual stream](#article-residual-streams) — Follow embeddings into token states.
- [JEPA: predicting representations](#article-jepa) — Predict learned representations.
- [HyperNEAT: geometry generates connectivity](#article-hyperneat) — Compare an encoding of connectivity.
- [Token embeddings & embedding space](#article-token-embeddings) — Inspect token lookup, embedding geometry and the output decoder explicitly.

---

**Family:** Intellectual lineage → Representations & sequence models

<a id="article-sequence-attention-history"></a>

## Sequence models discover attention

*From a fixed bottleneck to a learned read operation.*

**Abstract.** Place recurrent sequence modeling and learned alignment in the lineage of self-attention.

**At a glance:**

- Recurrence carries state through time
- Encoder–decoder models expose a bottleneck
- Attention gives a decoder access to source states

**Concepts:** RNN; LSTM; Seq2seq; Bahdanau attention.

### The bottleneck
Early neural sequence-to-sequence systems compressed an input sequence into a state from which a decoder produced output. Recurrence made order explicit, but long sequences stressed memory and optimization. Attention introduced an input-dependent read over a collection of encoder states rather than relying exclusively on one fixed summary. [1](https://arxiv.org/abs/1409.0473)

### Learned alignment
An additive attention mechanism can be written

$$
e_{tj}=v^\top\tanh(W_s s_{t-1}+W_h h_j),\quad
\alpha_{tj}=\frac{\exp(e_{tj})}{\sum_k\exp(e_{tk})},\quad
c_t=\sum_j\alpha_{tj}h_j.
$$

$s_{t-1}$ is the decoder state, $h_j$ an encoder state and $c_t$ its context. The projection matrices map their arguments to a common alignment width, and $v$ converts alignment features to a scalar. These weights are differentiable alignments, not hard symbolic links.

### What changed in 2017
The Transformer combined multi-head attention, position-wise networks, residual connections, normalization and positional information. It removed recurrence from the main encoder–decoder computation. The original decoder had both masked self-attention and cross-attention to an encoder; a decoder-only GPT-style architecture later retained a causal stack without that translation encoder. [2](https://arxiv.org/abs/1706.03762) [3](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

| Idea | Historical role | What remains distinct |
| --- | --- | --- |
| Recurrent state | Ordered sequence processing | Time recurrence versus depth |
| Learned alignment | Selective reads over source positions | Cross-attention versus self-attention |
| Transformer | Attention-centered parallel training | Original encoder–decoder versus decoder-only |
| Generative pretraining | Transfer from a language objective | Training recipe versus block design |

### Limits of a lineage
The diagram in this atlas is an explanatory map of relationships. Arrows denote an identified mechanism or conceptual influence; they are not a complete citation graph or a claim of sole priority. In particular, residual learning and normalization deserve their own branches. Continue to [ResNet and identity paths](#article-resnet-lineage).

### Primary references

1. Bahdanau, Cho & Bengio (2014). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). Paper · ICLR 2015.
2. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Research paper.
3. Radford, Narasimhan, Salimans & Sutskever (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). Technical report.

### Related topics

- [Attention as content-addressed computation](#article-attention) — Inspect the attention mechanism.
- [Selective state spaces & hybrids](#article-state-space) — Revisit recurrent state.
- [Recurrent depth & looped Transformers](#article-recurrent-depth) — Move recurrence into model depth.

---

**Family:** Intellectual lineage → Residual learning & scaling

<a id="article-resnet-lineage"></a>

## ResNet belongs in the main lineage

*The identity path that made depth trainable.*

**Abstract.** Understand why residual learning is one of the direct architectural foundations of the Transformer.

**At a glance:**

- Identity paths separate transport from transformation
- Residual learning preceded the Transformer
- Skip connections and Highway Networks also matter

**Concepts:** ResNet; Highway Networks; Identity paths.

### Learn a correction
ResNet popularized the parameterization $y=x+F_\theta(x)$ for very deep networks. When input and output widths agree, the skip can be the identity. With a projection skip $P$, the block is $y=Px+F_\theta(x)$ instead; its derivative is $P+J_F$, not $I+J_F$. Earlier skip connections and gated highway designs are part of the broader history, so ResNet should not be credited with every use of a bypass. [1](https://arxiv.org/abs/1512.03385)

$$
\frac{\partial\mathcal L}{\partial x}
=\left(I+J_{F_\theta}(x)\right)^\top\frac{\partial\mathcal L}{\partial y}.
$$

$J_F$ is the Jacobian of the learned residual function. The identity term supplies a direct derivative route. It does not make gradients immune to accumulation, poor scaling or unstable normalization.

### From convolutional blocks to token states
The Transformer explicitly wraps attention and feed-forward sublayers in residual connections. Replacing convolutions with attention does not erase this inheritance. Each token still carries its own vector through elementwise addition, even though attention reads from other tokens. [2](https://arxiv.org/abs/1706.03762)

### The modern extension
Hyper-Connections generalize routing across residual pathways. mHC constrains that mixing to improve stability. This is an extension of a residual-connectivity axis, not the same operation as adding more attention heads. [3](https://arxiv.org/abs/2409.19606) [4](https://arxiv.org/html/2512.24880v2)

```python
# Identity skip requires equal tensor shapes.
assert branch(x).shape == x.shape
y = x + branch(x)
```

### Evidence and limits
The original residual paper studied image recognition. Its direct architectural importance to Transformers is supported by the Transformer's own residual construction; the original ImageNet benchmark is not evidence about language-model performance. A sound historical account separates the original experiment from the later mechanism transfer.

### A useful experiment
Compare a deep plain MLP with an equal-depth residual MLP. Record loss, layerwise gradient norms and activation scale. Match widths and training budget. If you add normalization only to one model, the result no longer isolates the identity path. Read [mHC](#article-hyper-connections) after understanding this baseline.

### Primary references

1. He, Zhang, Ren & Sun (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Research paper.
2. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Research paper.
3. Zhu et al. (2024). [Hyper-Connections](https://arxiv.org/abs/2409.19606). Research paper.
4. Xie et al. (2025). [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/html/2512.24880v2). Research paper · v2 Jan 2026.

### Related topics

- [The residual stream](#article-residual-streams) — See residual learning in a decoder.
- [Hyper-connections & mHC](#article-hyper-connections) — Expand residual connectivity.
- [Backpropagation & biological plausibility](#article-backprop-biology) — Consider credit through deep graphs.

---

**Family:** Intellectual lineage → Residual learning & scaling

<a id="article-gpt-scaling"></a>

## From Transformer to language-model scaling

*Architecture meets data, compute and adaptation.*

**Abstract.** Connect GPT, empirical scaling laws, compute-optimal training and instruction tuning without treating them as one breakthrough.

**At a glance:**

- Generative pretraining supports transfer
- Scaling relationships are empirical fits
- Architecture, data and post-training are distinct axes

**Concepts:** GPT; Scaling laws; Chinchilla; LLaMA.

### Generative pretraining
GPT's 2018 report used a Transformer-based language model for pretraining followed by supervised adaptation. The contribution is not simply “inventing next-token prediction”: the report combined a scalable generative objective with transfer to language understanding tasks. [1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### Scaling as an empirical relationship
A convenient schematic fit is

$$
\mathcal L(N,D)\approx \mathcal L_\infty+aN^{-\alpha}+bD^{-\beta}.
$$

$N$ is parameter count, $D$ training tokens, $a,b,\alpha,\beta>0$ fitted quantities and $\mathcal L_\infty$ an asymptotic loss term. This expresses a modeling choice, not a universal law with fixed constants. Training compute is often approximated as proportional to $ND$ for a specified dense model family. [2](https://arxiv.org/abs/2001.08361)

Hoffmann and collaborators revisited compute allocation and found a different balance between model size and tokens under their experiments. A rule learned under fixed training compute need not minimize total serving cost, and neither paper proves that more parameters alone guarantee reasoning ability. [3](https://arxiv.org/abs/2203.15556)

### An architecture becomes a recipe
LLaMA is an important open research milestone combining a decoder with design choices such as RMSNorm, rotary positions and gated MLPs. It inherited many separate ideas. Instruction tuning and preference optimization then change how a pretrained model behaves, while leaving its high-level attention stack recognizable. [4](https://arxiv.org/abs/2302.13971) [5](https://arxiv.org/abs/2203.02155)

### An evidence-aware timeline
The atlas timeline groups by first publication year; conference years can differ. Select a milestone to inspect the primary reference and its specific contribution. It is a curated set of anchors, not an exhaustive history of language modeling.

### Research questions
Evaluate scaling claims against data mixture, deduplication, contamination control, optimizer, tokenization and inference cost. A model can improve on a benchmark through better data or post-training without an architectural invention. Keep architecture and procedure as separate entries in an experiment table.

### Primary references

1. Radford, Narasimhan, Salimans & Sutskever (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). Technical report.
2. Kaplan et al. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). Research paper.
3. Hoffmann et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). Research paper.
4. Touvron et al. (2023). [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). Research paper.
5. Ouyang et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). Research paper.

### Related topics

- [Training, prefill & decoding](#article-training-decoding) — Inspect the objectives and execution phases.
- [Normalization, position & gated MLPs](#article-normalization-position) — Unpack the decoder recipe.
- [Compression & speculative decoding](#article-compression-decoding) — Consider serving cost.

---

## Research provenance

The source registry records each primary URL, bibliographic year, source-review date, retrieved evidence identifier and the level of material inspected. Sources labeled as papers are not automatically being labeled peer-reviewed. Historical attribution follows specific mechanisms rather than one origin story. Interactive examples in the app are original educational implementations, not reproduced accuracy measurements.
