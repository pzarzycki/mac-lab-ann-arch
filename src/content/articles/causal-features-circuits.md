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

The figure's separate **recovery metric** view retains an invented comparison: clean margin 4, alternative 0, patched 1, 3 or 5 give $R=0.25$, $0.75$ or $1.25$. Its sliders also expose the denominator, including the undefined zero case. These chosen margins are a metric lesson, independent of the executable toy network in the activation-patch view. The overshoot case illustrates that normalized recovery can exceed one; it is not intrinsically a percentage of “the explanation.” If clean and alternative margins nearly coincide, this ratio is unstable or undefined. Always report the raw margins too.

Changes to corruption methods and metrics can lead to different localization results. Repeat the analysis over multiple controlled examples and include uncertainty, rather than presenting a single successful patch as a complete account. [2](#ref-patching-best-practices)

## An executed patch with a typed hook
The activation-patch view uses one token and three explicit vector sites:

$$
h_0=(x,1-x),\quad h_1=(h_{0,1}+h_{0,2},\ h_{0,1}-h_{0,2}),\quad
h_2=(\tanh h_{1,1},\ h_{1,2}+\tfrac12h_{1,2}^2),\quad m=2h_{2,1}+h_{2,2}.
$$

Clean input $x=1$ yields $h_1=(1,1)$ and $m\approx3.023$; alternative input $x=0$ yields $h_1=(1,-1)$ and $m\approx1.023$. Site A copies only the first coordinate of $h_0$: the patched input state is $(1,1)$, not a natural $h_0(x)$, and rerunning produces $h_1=(2,0)$ and $m=2\tanh2\approx1.928$. Site B copies the clean second coordinate of $h_1$, recovering the clean margin. Site C copies the first coordinate of $h_2$, which was already identical in both runs, so the intervention has no effect. Each site identifies a token, block or branch, and coordinate. Copying the entire vector would answer a different question.

Changing clean $x$ recomputes all these states and outcomes. Near $x=0$ the clean and alternative metrics become close, exposing the normalization problem in an actual calculation. The deliberately tiny graph isolates replacement semantics; it is not evidence for a circuit in a language model.

## From activations to candidate features
A neuron index is a coordinate, not necessarily one concept. A representation can distribute a feature across coordinates and can superpose multiple features. A sparse-autoencoder-style dictionary describes a state approximately as

$$
f=\operatorname{ReLU}(W_e(h-b_d)+b_e),\qquad
\hat h=W_d f+b_d.
$$

Here $f\in\mathbb R^m$ contains feature coefficients, $m$ can exceed residual width $d$, and $W_d\in\mathbb R^{d\times m}$ contains decoder directions. A generic reconstruction-plus-sparsity objective is $\|h-\hat h\|_2^2+\lambda\|f\|_1$. This is one illustrative objective, not a specification of every SAE training recipe.

When patching feature coefficients, preserve the reconstruction error where the intervention protocol calls for it: $h'=W_d f'+b_d+(h-\hat h)$. Otherwise, differences can reflect deleting unexplained activation content rather than changing the chosen feature alone. Decoder directions can be correlated; a label attached to one feature is an empirical summary, not its exact mathematical meaning.

For the **feature and circuit** view, choose $h=(1.2,.7)$, decoder columns $(1,0),(0,1),(1,1)$, zero bias and coefficients $f=(.8,.3,.2)$. Then $\hat h=(1,.5)$ and error $\epsilon=(.2,.2)$. Setting $f_1'=0$ while retaining the error gives $h'=(.4,.7)$. Dropping the error instead gives $(.2,.5)$: both coordinates change, introducing a confound beyond the intended first-feature edit. This is a hand-specified decoder, not an SAE trained in the browser.

Send $h=(a,b)$ through the declared margin $m(h)=2\tanh a+b+\tfrac12b^2$. The candidate edge from feature 1 through $a$ has local slope $2\operatorname{sech}^2(1.2)$. Its dashed edge predicts a first-order change; the solid validation path computes the finite edited margin exactly. A large edit can disagree with the local estimate even in this simple graph. The checkbox makes the reconstruction-error confound directly inspectable.

## Transcoders and attribution graphs
An SAE reconstructs an activation. A transcoder instead approximates a component's input-to-output transformation using sparse features. Cross-layer variants can connect these approximations across depth. The 2025 circuit-tracing work builds prompt-specific attribution graphs using a replacement model and validates proposed mechanisms with interventions. Its graph is an approximate analytical object, with a faithfulness boundary to the original network. [3](#ref-circuit-tracing)

A graph edge should therefore identify the quantity being approximated: a local derivative, an attributed contribution under frozen attention/normalization conditions, or a measured effect from rerunning a patch. These are not interchangeable merely because all can be drawn as arrows.

## Useful neighboring techniques
**Path patching** constrains which source-to-destination route is tested. **Attribution patching** uses a gradient-based approximation to rank possible interventions before expensive reruns. **Causal tracing** is a family of corruption-and-restoration experiments. **Steering** adds a chosen activation direction and tests a behavior change. Each requires a precise intervention site and an evaluation distribution; none removes the need to validate the hypothesis on the original model.

The diagram keeps clean and alternative runs separate and draws copying as a distinct cross-run connection. Activation and feature views rerun their declared toy computations; the metric view uses independently chosen margins. No language model, SAE training process or fitted replacement model is loaded in the browser.

## A reproducible investigation
Specify the checkpoint, prompts, token alignment, hook site, raw metric, replacement rule, and any frozen components. Use exploratory prompts to find candidates and separate prompts to test them. Compare the original model with any replacement or reconstructed model. Retain failures and alternative explanations: a mechanism that works only for one hand-selected sentence is a narrower result than a circuit that generalizes across a controlled prompt family.
