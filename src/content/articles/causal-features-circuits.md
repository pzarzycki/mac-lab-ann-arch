## Intervene on an explicit computational site
Activation patching replaces an internal signal in one run with a cached signal from another, then measures the downstream output. Suppose clean prompt $x_c$ supports answer A and a controlled alternative $x_a$ changes the relevant information. At a specified layer, position, and component, replace

$$
h_{\ell,t}(x_a)\leftarrow h_{\ell,t}(x_c).
$$

The rest of the alternative run then continues. This can test a localization hypothesis under the chosen intervention. It does not automatically identify a unique or minimal circuit. Noising, denoising, ablation baselines, redundancy, and nonlinear interactions require separate interpretation. [1](#ref-patching-interpretation)

## A metric with a visible denominator
A useful behavioral quantity is the answer-versus-foil logit difference $m=z_A-z_B$. For illustration, define

$$
R=\frac{m_{\mathrm{patched}}-m_{\mathrm{alternative}}}{m_{\mathrm{clean}}-m_{\mathrm{alternative}}}.
$$

The figure uses invented clean and alternative margins 4 and 0. Its selectable patches produce 1, 3, or 5, giving $R=0.25$, $0.75$, or $1.25$. The last case illustrates that normalized recovery can exceed one; it is not intrinsically a percentage of “the explanation.” If clean and alternative margins nearly coincide, this ratio is unstable or undefined. Always report the raw margins too.

Changes to corruption methods and metrics can lead to different localization results. Repeat the analysis over multiple controlled examples and include uncertainty, rather than presenting a single successful patch as a complete account. [2](#ref-patching-best-practices)

## From activations to candidate features
A neuron index is a coordinate, not necessarily one concept. A representation can distribute a feature across coordinates and can superpose multiple features. A sparse-autoencoder-style dictionary describes a state approximately as

$$
f=\operatorname{ReLU}(W_e(h-b_d)+b_e),\qquad
\hat h=W_d f+b_d.
$$

Here $f\in\mathbb R^m$ contains feature coefficients, $m$ can exceed residual width $d$, and $W_d\in\mathbb R^{d\times m}$ contains decoder directions. A generic reconstruction-plus-sparsity objective is $\|h-\hat h\|_2^2+\lambda\|f\|_1$. This is one illustrative objective, not a specification of every SAE training recipe.

When patching feature coefficients, preserve the reconstruction error where the intervention protocol calls for it: $h'=W_d f'+b_d+(h-\hat h)$. Otherwise, differences can reflect deleting unexplained activation content rather than changing the chosen feature alone. Decoder directions can be correlated; a label attached to one feature is an empirical summary, not its exact mathematical meaning.

## Transcoders and attribution graphs
An SAE reconstructs an activation. A transcoder instead approximates a component's input-to-output transformation using sparse features. Cross-layer variants can connect these approximations across depth. The 2025 circuit-tracing work builds prompt-specific attribution graphs using a replacement model and validates proposed mechanisms with interventions. Its graph is an approximate analytical object, with a faithfulness boundary to the original network. [3](#ref-circuit-tracing)

A graph edge should therefore identify the quantity being approximated: a local derivative, an attributed contribution under frozen attention/normalization conditions, or a measured effect from rerunning a patch. These are not interchangeable merely because all can be drawn as arrows.

## Useful neighboring techniques
**Path patching** constrains which source-to-destination route is tested. **Attribution patching** uses a gradient-based approximation to rank possible interventions before expensive reruns. **Causal tracing** is a family of corruption-and-restoration experiments. **Steering** adds a chosen activation direction and tests a behavior change. Each requires a precise intervention site and an evaluation distribution; none removes the need to validate the hypothesis on the original model.

The diagram deliberately keeps clean and alternative token streams separate and draws the replacement as a distinct cross-run connection. The displayed recovery values come from the toy margins above; no model is loaded in the browser.

## A reproducible investigation
Specify the checkpoint, prompts, token alignment, hook site, raw metric, replacement rule, and any frozen components. Use exploratory prompts to find candidates and separate prompts to test them. Compare the original model with any replacement or reconstructed model. Retain failures and alternative explanations: a mechanism that works only for one hand-selected sentence is a narrower result than a circuit that generalizes across a controlled prompt family.
