## A lens derived from downstream sensitivity
The **Jacobian lens (J-Lens)** uses averaged downstream derivatives to transport an intermediate residual vector toward the final-layer representation. It then applies the model's output readout. It is distinct from the training-free raw logit lens and from a translator fitted to predict the final distribution. Gurnee and collaborators introduce the method in their 2026 global-workspace study. [1](#ref-jacobian-lens)

Use column vectors. Let $h_{\ell,t}\in\mathbb R^d$ denote a source state at layer $\ell$ and position $t$. Let $h_{L,t'}\in\mathbb R^d$ denote a final-layer state at a downstream position $t'\ge t$. A local Jacobian is

$$
J_{\ell,t,t'}(x)=\frac{\partial h_{L,t'}}{\partial h_{\ell,t}}\in\mathbb R^{d\times d}.
$$

For a small perturbation $\delta h$, the first-order change is $\delta h_{L,t'}\approx J_{\ell,t,t'}(x)\delta h$. This is a derivative at a particular context, not a globally exact reconstruction of a nonlinear network.

## What gets averaged, and why dimensions matter
The paper's schematic definition averages over prompts $x$, source positions, and current-and-future target positions:

$$
\bar J_\ell=\mathbb E_{x,t,t'\ge t}\left[J_{\ell,t,t'}(x)\right],\qquad
p_\ell(h)=\operatorname{softmax}\!\left(U N_f(\bar J_\ell h)+b\right).
$$

Here $U\in\mathbb R^{V\times d}$ is the unembedding, $N_f$ final normalization, and $b$ an optional model-specific bias. The published formula omits a bias term; include one only when the model's actual readout has it. $\bar J_\ell$ is $d\times d$, whereas the composed direction matrix $U\bar J_\ell$ is $V\times d$. Neither is a vocabulary-by-vocabulary matrix. Normalization remains part of the actual readout, so the composed rows alone are not a universal exact linear decoder. [1](#ref-jacobian-lens)

The reference implementation sums cotangents over eligible target positions and averages over source positions. Sampling, aggregation, padding, and target-position weighting must be recorded to reproduce an estimator; a symbolic expectation does not settle every implementation convention. [2](#ref-jacobian-lens-code)

## A worked example of information lost by averaging
Consider the explicitly invented contexts

$$
J_A=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
J_B=\begin{bmatrix}1&0\\0&-1\end{bmatrix}.
$$

With equal weight, $\bar J=(J_A+J_B)/2=\operatorname{diag}(1,0)$. Each individual Jacobian has rank two, but their average has rank one. A perturbation along $(0,1)$ has a strong context-specific effect that cancels in the mean. A weak average direction therefore does not establish causal irrelevance in every prompt.

The interactive figure varies the fraction $q$ of context A. Then $\bar J=\operatorname{diag}(1,2q-1)$. At $q=1/2$ the second direction vanishes; at $q=1$ it survives fully. These are exact calculations for the toy matrices, not extracted LLM measurements.

## Reading is different from writing
Ranking vocabulary directions is a readout. Editing an activation along a direction, ablating it, or swapping coordinates is an intervention requiring a new forward pass. A direction can be useful for both without making a high readout score sufficient evidence of a causal mechanism. Compare a local intervention with a matched random direction, several perturbation sizes, and held-out contexts.

## J-space is not simply a low-rank matrix
The paper treats token-associated directions as a correlated, overcomplete frame and uses sparse decompositions to study active contents. This is different from declaring that the rank or leading singular vectors of $\bar J$ alone define the active workspace. A sparse representation can involve few directions even when their full matrix spans a much larger space.

The named token inventory, context distribution, estimator, and decomposition procedure all affect what becomes visible. Single-token vocabulary labels can also miss multi-token concepts. Global-workspace interpretations are empirical hypotheses about studied models, not a proof of consciousness or a theorem about every Transformer.

## Practical replication boundary
The authors provide [reference code](https://github.com/anthropics/jacobian-lens). Fit or load a lens for the exact checkpoint and hook convention, preserve its tokenizer and normalization, and compare against simple readout baselines. Computing an average transport can be expensive; applying an already computed map is a different workload. The atlas runs only the small matrix example locally, not the reference model-fitting procedure.
