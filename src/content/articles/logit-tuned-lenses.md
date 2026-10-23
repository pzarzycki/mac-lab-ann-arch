## Reading a hidden state through the output head
A decoder produces one hidden vector $h_i^{(\ell)}\in\mathbb R^d$ for each position $i$ and layer $\ell$. Let $N_f$ be its final LayerNorm or RMSNorm, $U\in\mathbb R^{V\times d}$ its unembedding, and $b\in\mathbb R^V$ an optional output bias. The actual final prediction uses $U N_f(h_i^{(L)})+b$.

The **logit lens**, introduced by nostalgebraist in 2020 and developed in the tuned-lens literature, applies that final readout prematurely:

$$
p_{\ell,i}^{\mathrm{LL}}=\operatorname{softmax}\!\left(U N_f(h_i^{(\ell)})+b\right).
$$

It requires no separately trained translator. Its interpretation depends on intermediate states being compatible with the final decoder's coordinates and normalization. [1](#ref-tuned-lens)

## What a layer-by-position grid says
Rows can index layers and columns token positions. A cell may display the top decoded token, a chosen token's rank, probability, or a logit difference. These are different views: a top-1 token can change after an arbitrarily small score change when two candidates are close. A rank can improve even while absolute probability falls.

In an autoregressive model, the actual output at input position $i$ predicts position $i+1$. A readout attached to the token “is” is not automatically a reconstruction of “is.” Middle-layer lens distributions are auxiliary measurements, not additional tokens that the model actually generated. Always label the input position and prediction alignment.

## The tuned lens learns a translator
A **tuned lens** adds a learned affine transformation per layer, $T_\ell(h)=A_\ell h+c_\ell$, before the frozen final readout:

$$
p_{\ell,i}^{\mathrm{TL}}=\operatorname{softmax}\!\left(U N_f(T_\ell(h_i^{(\ell)}))+b\right).
$$

The translators are fitted to match the frozen model's final output distribution, commonly using a distribution-matching loss. This differs from training the language model itself. The purpose is a more reliable intermediate prediction readout, not a literal recording of every internal concept. [1](#ref-tuned-lens)

```python
# Schematic column-vector convention; adapt to the model's actual hook sites.
h = cached_residual[layer, position]  # [width]
p_logit = softmax(U @ final_norm(h) + bias)
p_tuned = softmax(U @ final_norm(A[layer] @ h + c[layer]) + bias)
# Train the translator on held-out activations; keep backbone and decoder fixed.
```

## A worked coordinate-mismatch example
The interactive figure uses a synthetic two-dimensional model, with identity normalization and no bias. Its vocabulary directions are $u_A=(1,0)$, $u_B=(0,1)$ and $u_C=(-1,-1)$. Take the intermediate state $h=(0,1)$. Raw decoding gives logits $(0,1,-1)$, favoring B.

Now define a coordinate translator

$$
T=\begin{bmatrix}0&1\\1&0\end{bmatrix}.
$$

The translated state is $Th=(1,0)$, giving logits $(1,0,-1)$ and favoring A. The same stored activation can produce a different vocabulary readout under a different decoder. The figure does not fit a translator or reproduce the paper's measurements; it exposes these chosen matrices directly.

A learned translator should earn its interpretation through held-out prediction and control experiments. Merely finding one matrix that gives an appealing word proves little about the model.

## Prediction quality and causal fidelity
A translator can predict the eventual output while overlooking a transient representation used to compute it. Conversely, a raw logit lens can fail to decode meaningful information because the geometry is mismatched. Neither failure demonstrates that the underlying state contains no relevant information.

Compare readouts on unseen prompts; inspect more than the winning token; test multiple normalizations and hook sites only when justified. Use interventions to examine whether a decoded direction affects the behavior of interest. Patching evidence depends on its corruption, metric, and granularity. [2](#ref-patching-best-practices)

## Relation to the Jacobian lens
The tuned lens estimates translators through a prediction objective. The Jacobian lens constructs layer maps from averaged derivatives, including a specified distribution of source and downstream positions. These choices answer related but distinct questions. The [Jacobian-lens article](#/article/jacobian-lens) gives its exact dimensions and a toy example of cancellation during averaging. Avoid interpreting any lens's softmax as a calibrated probability that a model “has a thought”; it is a vocabulary distribution produced by the selected readout.
