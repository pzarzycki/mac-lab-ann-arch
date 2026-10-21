## The bottleneck
Early neural sequence-to-sequence systems compressed an input sequence into a state from which a decoder produced output. Recurrence made order explicit, but long sequences stressed memory and optimization. Attention introduced an input-dependent read over a collection of encoder states rather than relying exclusively on one fixed summary. [1](#ref-bahdanau)

## Learned alignment
An additive attention mechanism can be written

$$
e_{tj}=v^\top\tanh(W_s s_{t-1}+W_h h_j),\quad
\alpha_{tj}=\frac{\exp(e_{tj})}{\sum_k\exp(e_{tk})},\quad
c_t=\sum_j\alpha_{tj}h_j.
$$

$s_{t-1}$ is the decoder state, $h_j$ an encoder state and $c_t$ its context. The projection matrices map their arguments to a common alignment width, and $v$ converts alignment features to a scalar. These weights are differentiable alignments, not hard symbolic links.

## What changed in 2017
The Transformer combined multi-head attention, position-wise networks, residual connections, normalization and positional information. It removed recurrence from the main encoder–decoder computation. The original decoder had both masked self-attention and cross-attention to an encoder; a decoder-only GPT-style architecture later retained a causal stack without that translation encoder. [2](#ref-transformer) [3](#ref-gpt)

| Idea | Historical role | What remains distinct |
| --- | --- | --- |
| Recurrent state | Ordered sequence processing | Time recurrence versus depth |
| Learned alignment | Selective reads over source positions | Cross-attention versus self-attention |
| Transformer | Attention-centered parallel training | Original encoder–decoder versus decoder-only |
| Generative pretraining | Transfer from a language objective | Training recipe versus block design |

## Limits of a lineage
The diagram in this atlas is an explanatory map of relationships. Arrows denote an identified mechanism or conceptual influence; they are not a complete citation graph or a claim of sole priority. In particular, residual learning and normalization deserve their own branches. Continue to [ResNet and identity paths](#/article/resnet-lineage).
