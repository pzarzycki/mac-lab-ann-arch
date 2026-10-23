## From a token ID to a vector
A tokenizer first maps text to discrete vocabulary IDs. An ID identifies a vocabulary entry, which may represent a word, subword, byte, punctuation mark, or special symbol. It is not a scalar measurement: ID 400 does not mean twice as much as ID 200. Tokenization determines the discrete interface; the embedding table determines how that interface enters the neural computation.

Choose column-vector notation. Let $V$ be vocabulary size and $d$ the residual width. The learned table is $E\in\mathbb R^{V\times d}$. Token ID $x_i$ selects a row, transposed to give $e_i=E_{x_i,:}^{\mathsf T}\in\mathbb R^d$. Equivalently, with a one-hot column $o_i\in\{0,1\}^V$, $e_i=E^{\mathsf T}o_i$. Frameworks implement a gather rather than constructing the one-hot tensor. The Transformer combines learned token representations with positional information; the exact position mechanism depends on the architecture. [1](#ref-transformer)

## Embedding space and the residual stream
The input lookup vectors occupy $\mathbb R^d$, often called the token embedding space. Later token states $h_i^{(\ell)}$ commonly retain width $d$, but become contextual: the same token ID can have different hidden states in different prompts and positions. Equal dimension does not mean that distances, directions, or semantic interpretations are constant across layers.

For learned absolute positions, an initial state might be $h_i^{(0)}=e_i+p_i$, with $p_i\in\mathbb R^d$. In a RoPE-based decoder, rotations are applied to queries and keys inside attention instead; the token lookup need not receive an additive position vector. Position, lookup identity, and contextual computation should therefore be drawn separately. The atlas's [residual-stream figure](#/article/residual-streams) retains one state per token; its [normalization and position article](#/article/normalization-position) develops those operations.

## Output embeddings, unembedding, and tying
Let $U\in\mathbb R^{V\times d}$ be an output projection, $b\in\mathbb R^V$ an optional bias, and $N_f$ the model's final normalization. Next-token scores are

$$
z_i=U N_f(h_i^{(L)})+b,\qquad p_i=\operatorname{softmax}(z_i).
$$

Each row of $U$ is an output direction associated with a vocabulary entry. **Weight tying** sets $U=E$ in this convention; untied models learn separate input and output matrices. Tying is an architectural/training choice, not a property of every language model. Press and Wolf analyze this sharing and its effects on language modeling. [2](#ref-weight-tying)

The unembedding is not generally an inverse of lookup. Multiple vectors can yield the same top token, softmax discards a common additive logit offset, and a contextual state contains more than the identity of its input token. Calling $U$ a decoder does not establish a one-to-one semantic coordinate system.

## A small geometric example
The figure uses four invented two-dimensional rows: $e_A=(1,0)$, $e_B=(0,1)$, $e_C=(-1,0)$ and $e_D=(0,-1)$. These are explanatory coordinates, not extracted model embeddings. Selecting A gives one table row. A soft mixture with coefficient $\alpha\in[0,1]$ instead gives

$$
v(\alpha)=\alpha e_A+(1-\alpha)e_B=(\alpha,1-\alpha).
$$

At $\alpha=1/2$, a tied, identity-normalization readout has logits $(1/2,1/2,-1/2,-1/2)$. A and B tie; the vector has not become a new vocabulary ID. A learned soft prompt can also lie outside the convex hull of the input table. Compare [soft prompts and continuous thought](#/article/soft-latent-reasoning).

## Geometry needs an explicit metric
Cosine similarity measures angle; Euclidean distance also responds to norm. Neither metric guarantees synonymy, truth, or a human-readable feature axis. Shared learned features were already central to neural probabilistic language modeling before Transformers. [3](#ref-bengio)

A two-dimensional projection can hide important differences in the remaining coordinates. State the source layer, normalization, centering, projection method, and reference distribution before interpreting a scatterplot. The **logit lens** asks what the output decoder reads from a hidden state, rather than finding its nearest input embedding; [layer readouts](#/article/logit-tuned-lenses) explain the distinction.

## Implementation and gradients
```python
# E: [vocabulary, width]; ids: [batch, positions]
h0 = embedding(ids)              # [batch, positions, width]
h = transformer(h0)              # separate contextual state per token
logits = lm_head(final_norm(h))  # [batch, positions, vocabulary]
```

Integer IDs do not receive gradients. Lookup parameters do, and a frozen backbone can still propagate derivatives to trainable continuous inputs. Freezing parameters is different from disabling autograd. Check the actual model's embedding scale, position scheme, normalization, weight tying, and output bias before adapting this schematic to code.
