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

The interactive figure varies the fraction $q$ of context A. Then $\bar J=\operatorname{diag}(1,s)$ with $s=2q-1$. At $q=1/2$ the second direction vanishes; at $q=1$ it survives fully. Its declared four-token unembedding has rows $u_A=(1,0)$, $u_B=(0,1)$, $u_C=(1,1)$ and $u_D=(0,-1)$. With identity final normalization, transporting $h=(1,1)$ gives $\bar Jh=(1,s)$ and vocabulary logits $(1,s,1+s,-s)$. The displayed probabilities therefore genuinely change with $q$. These are exact toy calculations, not extracted LLM measurements.

## Reading is different from writing
Ranking vocabulary directions is a readout. Editing an activation along a direction, ablating it, or swapping coordinates is an intervention requiring a new forward pass. A direction can be useful for both without making a high readout score sufficient evidence of a causal mechanism. Compare a local intervention with a matched random direction, several perturbation sizes, and held-out contexts.

## J-space is not simply a low-rank matrix
The paper treats token-associated directions as a correlated, overcomplete frame and uses sparse decompositions to study active contents. This is different from declaring that the rank or leading singular vectors of $\bar J$ alone define the active workspace. A sparse representation can involve few directions even when their full matrix spans a much larger space.

More explicitly, let $D_\ell\in\mathbb R^{d\times V}$ contain token-associated J-lens directions as columns. Choose an allowable sparsity $k$ and approximate a source activation by

$$
a^*\in\arg\min_{a\ge0,\,\|a\|_0\le k}\|h-D_\ell a\|_2^2.
$$

Here $\|a\|_0$ counts nonzero coefficients, not a vector norm measuring magnitude. The reconstruction $D_\ell a^*$ and remainder $h-D_\ell a^*$ depend on this constraint and the fitting procedure. The source uses gradient pursuit for its large dictionary; exhaustive support search is practical only for a tiny toy. [1](#ref-jacobian-lens)

In the figure's identity-normalization example, use the transposed rows of $U\bar J$ as the source frame: $A=(1,0)$, $B=(0,s)$, $C=(1,s)$ and $D=(0,-s)$. There are four labeled directions in two dimensions. The figure enumerates supports of size at most one or two, solves nonnegative least squares on each support, and picks the smallest reconstruction error, breaking ties in favor of fewer active coefficients. At $q=.75$, $s=.5$, the source vector $h=(1,1)$ can be reconstructed as $C+B$, while the best single direction leaves a nonzero error. At $q=1$, $h=C=A+B$: correlated directions permit different valid decompositions. At $q=.5$, all four directions lie on the x-axis and every reconstruction leaves y-error one. An ordinary rank-one x-axis projection also leaves error one; that fixed linear subspace is a different object from a union of sparse nonnegative cones.

The named token inventory, context distribution, estimator, and decomposition procedure all affect what becomes visible. Single-token vocabulary labels can also miss multi-token concepts. Global-workspace interpretations are empirical hypotheses about studied models, not a proof of consciousness or a theorem about every Transformer.

## Practical replication boundary
The authors provide [reference code](https://github.com/anthropics/jacobian-lens). Fit or load a lens for the exact checkpoint and hook convention, preserve its tokenizer and normalization, and compare against simple readout baselines. Computing an average transport can be expensive; applying an already computed map is a different workload. The atlas runs only the small matrix example locally, not the reference model-fitting procedure.
