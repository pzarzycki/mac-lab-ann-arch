## What does observability mean here?
In this atlas, **Transformer observability** means making internal activations, computations, and interventions inspectable. It includes representation analysis and mechanistic interpretability. It is not a claim that a model satisfies the formal state-reconstruction observability criterion of control theory, nor is it the logging, metrics, and tracing layer of a production serving system.

An analysis starts by specifying a model checkpoint, tokenizer, prompt distribution, and behavior of interest. A question such as “where is the candidate answer available?” differs from “which component is causally necessary for this answer?” The appropriate measurement changes with the question.

## First identify the tensor and its axes
A residual activation cache may have shape $[B,T,d]$ at each layer, where $B$ is batch size, $T$ sequence positions, and $d$ hidden width. Head activations add a head axis; attention patterns use destination and source positions. A layer-by-position heatmap is not a head-by-feature map.

Record whether a hook captures a state before normalization, before attention, after attention, or after the MLP. Saving all layers, tokens, heads, and gradients can be costly. Select positions and hook sites deliberately, and keep the model in a reproducible evaluation configuration. A hook exposes a tensor; its name does not explain that tensor's role.

## Four different kinds of evidence
| View | Operation | Useful question | Main limitation |
| --- | --- | --- | --- |
| Activations and attention | Inspect cached values and routing weights | What is large or attended to? | Magnitude is not causal importance |
| Lenses and probes | Map states to vocabulary scores or supervised labels | What can this readout recover? | Decoder assumptions and probe capacity matter |
| Jacobians and attribution | Differentiate a specified downstream quantity | What changes under an infinitesimal perturbation? | Local derivative need not predict a large intervention |
| Patching and ablation | Replace or remove an internal signal, then rerun | What happens under this intervention? | Controls, interactions, and distribution shifts matter |

High probe accuracy can reflect information accessible to the probe rather than computation used by the model. Control tasks and held-out generalization help separate these possibilities. [1](#ref-probe-controls)

## Attention patterns are a starting point
For one head, the output at destination $i$ is $a_i=\sum_j A_{ij}v_j$. A large routing weight $A_{ij}$ does not determine the output effect without the corresponding value vector, output projection, other heads, and downstream computation. Two terms with large weights can cancel; a small-weight value can still matter.

Jain and Wallace give concrete counterexamples to treating attention weights as explanations in their studied NLP models. That result should not be stretched into a theorem that no attention-based analysis can ever be informative. The engineering conclusion is narrower: pair an attention pattern with a behavioral metric and an intervention before treating it as a mechanism. [2](#ref-attention-explanation)

## Direct logit attribution and its boundary
If a pre-normalization state decomposes as $h=\sum_c r_c$, a fixed linear readout gives $Uh=\sum_c Ur_c$. This motivates direct logit attribution to residual contributions. With final normalization included, the normalization scale depends on the combined state; LayerNorm additionally centers features. An exact attribution under a frozen scale is a conditional decomposition, not the effect of deleting a component and recomputing the model.

For a numerical counterexample, use RMS normalization without an epsilon or learned gain, $N(h)=h/\sqrt{(h_1^2+h_2^2)/2}$, and a one-row readout $u=(1,0)$. Let $r_a=(1,0)$ and $r_b=(0,1)$, so $h=(1,1)$ has RMS scale one and $uN(h)=1$. Under that frozen scale, $r_b$ contributes zero to this logit. Actually deleting $r_b$ leaves $(1,0)$, whose RMS scale is $1/\sqrt2$: the recomputed score is $\sqrt2\approx1.414$. Deletion changes the score by about $+0.414$, even though the fixed-scale contribution was zero. This is exact toy algebra; other normalization conventions change the numbers.

Useful analyses can therefore disagree without either being a software bug: one describes an algebraic contribution under fixed conditions; another measures a rerun after perturbing those conditions. State which quantity is being plotted. [3](#ref-tuned-lens)

## From measurement to a falsifiable account
Choose a clean prompt, a controlled alternative, and an output metric such as a target-versus-foil logit difference. Inspect readouts to propose a candidate layer, position, or feature. Then test that candidate with patching, matched controls, and additional prompts. Patching methodology can materially change localization conclusions. [4](#ref-patching-best-practices)

The diagram keeps three token streams separate and names two **post-block residual** hooks. Its synthetic cache has shape $[1,3,2]$ at each hook. Selecting a layer or position changes the cached vector; selecting a method changes the computation. The lens applies an explicit three-row unembedding. The illustrative binary probe uses $\sigma((1,-1)^{\mathsf T}h)$, with fixed coefficients standing in for a supervised fit. No labeled evaluation set is provided, so this number is a probe output, not accuracy.

For the derivative and patch views, the downstream toy metric is $m(h)=\tanh(h_1)+h_2^2$, with gradient $(\operatorname{sech}^2 h_1,2h_2)$. A patch copies a clean second coordinate increased by $0.4$ and recomputes this nonlinear metric. These are inspectable synthetic evidence types, not measurements of the omitted Transformer computation. Each method links to the article that develops its assumptions.

## Follow the question
For vocabulary readouts, compare the [logit and tuned lenses](#/article/logit-tuned-lenses). For an averaged sensitivity-derived transport, inspect the [Jacobian lens](#/article/jacobian-lens). For interventions, sparse features, and computational graphs, continue to [causal tracing and feature circuits](#/article/causal-features-circuits). Each approach exposes a different part of the evidence; a readable label is the beginning of analysis, not its conclusion.
