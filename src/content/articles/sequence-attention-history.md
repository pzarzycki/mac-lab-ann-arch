## The bottleneck
Early neural sequence-to-sequence systems compressed an input sequence into a state from which a decoder produced output. Recurrence made order explicit, but long sequences stressed memory and optimization. Attention introduced an input-dependent read over a collection of encoder states rather than relying exclusively on one fixed summary. [1](#ref-bahdanau)

## Learned alignment
An additive attention mechanism can be written

$$
e_{tj}=v^\top\tanh(W_s s_{t-1}+W_h h_j+b),\quad
\alpha_{tj}=\frac{\exp(e_{tj})}{\sum_k\exp(e_{tk})},\quad
c_t=\sum_j\alpha_{tj}h_j.
$$

$s_{t-1}$ is the decoder state, $h_j$ an encoder state and $c_t$ its context. The projection matrices map their arguments to a common alignment width, and $v$ converts alignment features to a scalar. These weights are differentiable alignments, not hard symbolic links.

Let $j=1,\ldots,S$ index source positions and $t=1,\ldots,U$ target steps. If $h_j\in\mathbb R^{d_h}$ and $s_{t-1}\in\mathbb R^{d_s}$, then $W_h\in\mathbb R^{d_a\times d_h}$, $W_s\in\mathbb R^{d_a\times d_s}$, and $v,b\in\mathbb R^{d_a}$. The bias $b$ is an optional affine-score term. Scores and weights each have $S$ entries; context $c_t\in\mathbb R^{d_h}$ has the encoder-state width, not the alignment width.

The read closes a recurrent decoding step:

$$
s_t=f(s_{t-1},\operatorname{embed}(y_{t-1}),c_t),\qquad
p(y_t\mid y_{<t},x)=\operatorname{softmax}(g(s_t,c_t,y_{t-1})).
$$

Here $f$ is a decoder update and $g$ produces vocabulary logits; their exact implementation varies. Thus context affects both the next decoder state and the output distribution, rather than ending the computation at a weighted sum. [1](#ref-bahdanau)

## A worked read over retained values
The figure fixes scalar source states $[-1,0,1]$ and uses width-two alignment: $W_h=[1,1]^\top$, $W_s=[-1,-1]^\top$, $b=[1,-1]^\top$, $v=[1,-1]^\top$. At query $s=-1$, scores are $[1.523,0.964,0.233]$, weights $[0.541,0.310,0.149]$ and context $c=-0.392$. At $s=1$, the weights reverse and $c=0.392$. The source states have not changed; the target-step query changes which values contribute. All values are synthetic.

To expose the feedback, the toy decoder uses $s'=\tanh(0.5s+c+0.2)$, where $0.2$ stands for a fixed previous-token contribution. A two-symbol output has logits $[s',-s']$. This is a small complete calculation, not the original translation model or a claim that attention weights are symbolic word correspondences.

## What changed in 2017
The Transformer combined multi-head attention, position-wise networks, residual connections, normalization and positional information. It removed recurrence from the main encoder–decoder computation. The original decoder had both masked self-attention and cross-attention to an encoder; a decoder-only GPT-style architecture later retained a causal stack without that translation encoder. [2](#ref-transformer) [3](#ref-gpt)

In encoder–decoder cross-attention, queries come from target states while keys/values come from source states. In self-attention, queries and memory come from the same sequence, with any causal mask restricting allowed positions. Residual addition and normalization wrap individual sublayers; the figure's final strip is an ingredient comparison without serial computation arrows. [2](#ref-transformer)

| Idea | Historical role | What remains distinct |
| --- | --- | --- |
| Recurrent state | Ordered sequence processing | Time recurrence versus depth |
| Learned alignment | Selective reads over source positions | Cross-attention versus self-attention |
| Transformer | Attention-centered parallel training | Original encoder–decoder versus decoder-only |
| Generative pretraining | Transfer from a language objective | Training recipe versus block design |

## Limits of a lineage
The diagram in this atlas is an explanatory map of relationships. Arrows denote an identified mechanism or conceptual influence; they are not a complete citation graph or a claim of sole priority. In particular, residual learning and normalization deserve their own branches. Continue to [ResNet and identity paths](#/article/resnet-lineage).
